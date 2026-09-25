"""
MODUS 4: Depot-Analyse.

Eigene Optionspositionen als Kombos aus 1 bis 4 Beinen erfassen (gemeinsamer
Verfall UND gemeinsames Instrument je Kombo), dauerhaft speichern
(data/depot.json, wird beim Start automatisch wieder geladen) und mit
derselben Logik bewerten wie die Iron-Condor-Suche ("Rechnung 2", siehe
condor_search.weighted_pnl_stats).

MULTI-INSTRUMENT: das Depot ist EIN gemeinsames Depot fuer ALLE Instrumente -
jede Kombo traegt ein 'instrument'-Feld (siehe depot_engine.py, Migration
alter Depots auf "spx"). Die Depot-LISTE zeigt alle Kombos aller Instrumente.
Die 'Analyse' weiter unten (Analog-Tage-Simulation, Ausstiegspreise, P&L-Kurve)
beschraenkt sich pro Aufruf bewusst auf Kombos DES AKTUELL OBEN GEWAEHLTEN
INSTRUMENTS (dashboard.py-Instrument-Auswaehler): Kombos anderer Instrumente
werden dabei uebersprungen (mit Hinweis, welches Instrument dafuer oben
ausgewaehlt werden muss). Grund: die Analog-Suche und die Kurshistorie sind
pro Seitenaufbau nur fuer GENAU EIN Instrument geladen (das, was dashboard.py
gerade an alle Modi durchreicht) - mehrere Instrumente gleichzeitig
auszuwerten wuerde mehrere Kurshistorien/Trainingstabellen parallel laden,
das ist (noch) nicht Teil dieses Umbauschritts.

BEWERTUNG (Analog-Tage-Simulation), unveraendert gegenueber vorher:
    Historische PROZENTUALE Kursbewegungen der Analog-Tage (gewichtet, siehe
    analog_matching.find_analogs) werden auf den heutigen Spot projiziert:
    S_d = Spot_heute * (1 + Bewegung_d). Die Kombo wird bis Verfall gehalten:
      P&L_d (seit Einstieg)  = entry_net + V(S_d)
      Restgewinn_d (ab jetzt) = V(S_d) - Marktwert_heute      (Marktwert = Mid je Bein)
    Kennzahlen EXAKT wie in der Iron-Condor-Suche (condor_search.weighted_pnl_stats):
    E[P&L], Rendite/Risiko % (= E[P&L] / Max-Loss), HAC-t-Stat (Bartlett,
    Kalenderzeit), Train/Val-Split, Voll-Verlust-Anteil, 5%-Quantil relativ
    zum Max-Loss, Gewinnrate. Risikobezug (Nenner) passt sich der P&L-Basis
    an: bei 'Restgewinn' kommt der Max-Loss aus einer hypothetisch heute-zum-
    Mid-eroeffneten Kombo, sonst aus der echten historischen Kombo. Kombos mit
    unbegrenztem Risiko (z.B. nackter Short Call) liefern dafuer n/a statt
    eines falschen/undefinierten Werts.

AUSSTIEGSPREISE HEUTE: Black-76 je Bein mit der impliziten Vola aus der
Kette (sticky strike), ohne Vola-Aenderung bei Kursbewegung - Break-even und
Stop-Level HEUTE (mit Zeitwert) sowie Break-even/Max-Gewinn/-Verlust BEI
VERFALL (exakt, bezogen auf den tatsaechlichen historischen Einstieg).
"""
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import depot_engine as dl
import dashboard_livechain
import market_data
from instruments import InstrumentConfig, INSTRUMENTS, DEFAULT_INSTRUMENT
from chain_providers.base import UnsupportedChainError
from black76 import forward_price
from chain_delta import black76_price
from analog_matching import available_t_days
from condor_search import analog_moves, weighted_pnl_stats

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEPOT_FILE = DATA_DIR / "depot.json"

_NY = "America/New_York"
_IC_DEFAULTS = [("put", "long"), ("put", "short"), ("call", "short"), ("call", "long")]
_GENERIC_DEFAULTS = [("put", "short"), ("call", "short"), ("put", "long"), ("call", "long")]
_LEVEL_LO, _LEVEL_HI = 0.5, 1.5     # Suchbereich der Ausstiegspreise: +-50 % um den Spot


# ───────────────────────── Hilfsfunktionen ─────────────────────────

def _dte_exact(expiry_str: str) -> float:
    """Restlaufzeit in Tagen bis Verfall 16:00 ET (PM-Settlement-Naeherung,
    wie in der Kettenansicht - gilt fuer alle aktuell live-fähigen Instrumente,
    die durchweg US-boersengehandelt sind, siehe instruments.py)."""
    now = pd.Timestamp.now(tz=_NY)
    exp = pd.Timestamp(f"{expiry_str} 16:00", tz=_NY)
    return (exp - now).total_seconds() / 86400.0


def _norm_dates(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series)
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s.dt.normalize()


def _lookup_leg(enriched: pd.DataFrame | None, expiry_str: str, leg: dict):
    """(mid, implizite Vola) des Beins aus der heutigen Kette - exakter Treffer
    auf Verfall/Typ/Strike, sonst (None, None)."""
    if enriched is None or enriched.empty:
        return None, None
    exp = pd.Timestamp(expiry_str).normalize()
    m = ((_norm_dates(enriched["expiry"]) == exp)
         & (enriched["option_type"] == leg["option_type"])
         & np.isclose(enriched["strike"].astype(float), leg["strike"]))
    sub = enriched[m]
    if sub.empty:
        return None, None
    r = sub.iloc[0]
    return float(r["mid"]), float(r["implied_vol_mkt"])


def _leg_text(leg: dict) -> str:
    side = "L" if leg["side"] == "long" else "S"
    typ = "P" if leg["option_type"] == "put" else "C"
    q = f"{leg['qty']:g}x " if leg["qty"] != 1 else ""
    return f"{side} {q}{leg['strike']:g}{typ}"


def _combo_label(c: dict) -> str:
    return f"{c['name']} ({c['expiry']})"


def _instr_label(c: dict) -> str:
    key = c.get("instrument", DEFAULT_INSTRUMENT)
    instr = INSTRUMENTS.get(key)
    return instr.label if instr else key


def _commit(combos: list[dict]) -> None:
    """Speichert das Depot in Session UND auf Platte."""
    st.session_state["depot_combos"] = combos
    try:
        dl.save_depot(DEPOT_FILE, combos)
    except OSError as e:
        st.error(f"Depot konnte NICHT gespeichert werden ({e}) - Aenderung gilt nur bis zum "
                 "Schliessen des Programms!")


def add_combo_to_depot(name: str, expiry: str, legs: list[dict], note: str = "",
                       instrument: str = DEFAULT_INSTRUMENT) -> dict:
    """OEFFENTLICHE Schnittstelle fuer ANDERE Module (z.B. die 'Zum Depot
    hinzufuegen'-Buttons in der Analog-Tabelle und der Iron-Condor-Suche in
    dashboard_livechain.py): legt eine neue Kombo FUER 'instrument' an, haengt
    sie an das AKTUELLE (gemeinsame) Depot an und speichert sofort. Liest das
    Depot dafuer frisch von der Platte statt blind aus st.session_state - der
    Aufrufer muss diesen Modus in der laufenden Sitzung nicht vorher geoeffnet
    haben. Gibt die gespeicherte Kombo zurueck; wirft ValueError bei einer
    ungueltigen Kombo (dann wird NICHTS gespeichert)."""
    combo = dl.make_combo(name, expiry, legs, note, instrument=instrument)
    errs = dl.validate_combo(combo)
    if errs:
        raise ValueError(" ".join(errs))
    combos, _warn = dl.load_depot(DEPOT_FILE)
    combos = combos + [combo]
    _commit(combos)
    st.session_state.pop("depot_results", None)   # alte Analyse bezieht sich nicht auf die neue Kombo
    return combo


def _init_state() -> None:
    if "depot_combos" not in st.session_state:
        combos, warn = dl.load_depot(DEPOT_FILE)
        st.session_state["depot_combos"] = combos
        st.session_state["depot_load_warning"] = warn


def _make_pnl_today(combo: dict, ivs: list[float], dte: float, r: float, q: float):
    """P&L-Funktion HEUTE (relativ zur heutigen Netto-Praemie) in Abhaengigkeit
    vom Spot: Black-76 je Bein mit fester Vola je Bein. Akzeptiert/liefert
    numpy-Arrays. 'combo' muss die HEUTE-Kombo sein (entry_price = heutiger
    Preis), bzw. hier die tatsaechliche Kombo mit historischem Einstieg -
    siehe Aufrufer."""
    T = dte / 365.0
    factor = forward_price(1.0, r, q, T)        # F ist proportional zu S
    net = dl.entry_net(combo)
    legs = combo["legs"]

    def f(S):
        S = np.asarray(S, dtype=float)
        F = S * factor
        total = np.full_like(S, net)
        for leg, iv in zip(legs, ivs):
            p = black76_price(F, leg["strike"], T, iv, r, leg["option_type"] == "put")
            total = total + dl._sign(leg) * leg["qty"] * p
        return total
    return f


def _hypothetical_today_combo(combo: dict, mids: list[float]) -> dict | None:
    """Kopie der Kombo mit entry_price je Bein = HEUTIGER Mid-Preis. Das ist
    der Risikobezugspunkt fuer 'Restgewinn' (P&L ab heute bis Verfall) -
    ANALOG zu 'Kredit'/'Max-Loss' in der Iron-Condor-Suche, die ebenfalls auf
    heutigen Preisen basieren, nicht auf dem historischen Einstieg. None, wenn
    nicht fuer jedes Bein ein Mid-Preis vorliegt."""
    if any(m is None for m in mids):
        return None
    legs = [{**leg, "entry_price": mid} for leg, mid in zip(combo["legs"], mids)]
    return {**combo, "legs": legs}


def _zone_info(f, target: float, spot: float) -> dict:
    """Zerlegt [lo, hi] in Zonen 'P&L >= target' / '< target' und beschreibt
    die Zone um den Spot. lower/upper = Kurs, unter/ueber dem P&L < target
    wird (None = im Suchbereich kein Level)."""
    lo, hi = spot * _LEVEL_LO, spot * _LEVEL_HI
    roots = dl.find_levels(f, target, lo, hi)
    pts = [lo] + roots + [hi]
    zones = []
    for a, b in zip(pts[:-1], pts[1:]):
        ok = float(f(np.array([(a + b) / 2.0]))[0]) >= target
        if zones and zones[-1][2] == ok:
            zones[-1] = (zones[-1][0], b, ok)
        else:
            zones.append((a, b, ok))
    here = next((z for z in zones if z[0] <= spot <= z[1]), zones[-1])
    profit_zones = [(a, b) for a, b, ok in zones if ok]
    return {
        "in_zone": here[2],
        "lower": here[0] if (here[2] and here[0] > lo + 1e-9) else None,
        "upper": here[1] if (here[2] and here[1] < hi - 1e-9) else None,
        "profit_zones": profit_zones,
    }


def _fmt_level(x: float | None, spot: float) -> str:
    return "-" if x is None else f"{x:,.0f} ({(x / spot - 1) * 100:+.1f} %)"


def _fmt_zone_text(zi: dict, spot: float, what: str) -> str:
    if not zi["in_zone"]:
        zs = "; ".join(f"{a:,.0f} bis {b:,.0f}" for a, b in zi["profit_zones"]) or "keine im Suchbereich"
        return f"BEREITS unter der {what}-Schwelle. Wieder darueber bei: {zs}"
    parts = []
    if zi["lower"] is not None:
        parts.append(f"unter {_fmt_level(zi['lower'], spot)}")
    if zi["upper"] is not None:
        parts.append(f"ueber {_fmt_level(zi['upper'], spot)}")
    return " oder ".join(parts) if parts else "kein Level im Bereich +-50 % um den Spot"


# ───────────────────────── Marktdaten (geteilter Snapshot mit 'Heutige Kette') ─────────────────────────

def _needed_max_dte(combos: list[dict]) -> float:
    return max([_dte_exact(c["expiry"]) for c in combos] + [1.0]) + 2.0


def _get_market(instr: InstrumentConfig, combos_for_instr: list[dict], r_free: float,
                dividend_yield: float, force: bool = False) -> dict | None:
    """Gemeinsamer Marktdaten-Snapshot FUER DIESES INSTRUMENT (siehe
    market_data.py) - dieselbe Kette wie im Tab 'Heutige Kette' fuer dasselbe
    Instrument, wird NICHT pro Tab neu geladen. None (mit Hinweis/
    Fehlermeldung), falls das Instrument keine Live-Kette hat (chain_provider
    None, z.B. Nikkei/EuroStoxx50 aktuell) oder der Abruf fehlschlaegt."""
    needed = _needed_max_dte(combos_for_instr) if combos_for_instr else market_data.DEFAULT_MAX_DTE
    snap = None if force else market_data.current(instr, needed)
    if snap is None:
        with st.spinner(f"Lade aktuellen Spot und Optionskette fuer {instr.label}..."):
            try:
                snap = market_data.get(instr, float(r_free), float(dividend_yield), needed, force=force)
            except UnsupportedChainError as e:
                st.info(str(e))
                return None
            except Exception as e:
                st.error(f"Marktdaten fuer {instr.label} konnten nicht geladen werden: {e}")
                snap = market_data.current(instr, 0.0)
    return snap


# ───────────────────────── Eingabe / Verwaltung ─────────────────────────

def _render_add_form(daily_df: pd.DataFrame, active_instr: InstrumentConfig) -> None:
    st.subheader("Kombo hinzufuegen")
    valid_close = daily_df["close"].dropna() if "close" in daily_df.columns else pd.Series(dtype=float)
    last_close = float(valid_close.iloc[-1]) if len(valid_close) else 100.0
    default_strike = float(round(last_close / 5.0) * 5.0)

    instrument_keys = list(INSTRUMENTS.keys())
    n_legs = st.selectbox("Anzahl Beine der Kombo", [1, 2, 3, 4], index=3, key="depot_n_legs",
                          help="4 = z.B. Iron Condor, 2 = Spread/Strangle, 1 = Einzeloption.")
    defaults = _IC_DEFAULTS if n_legs == 4 else _GENERIC_DEFAULTS

    with st.form(f"depot_add_form_{n_legs}"):
        c0, c1, c2 = st.columns([1.3, 2, 1])
        with c0:
            instrument_key = st.selectbox(
                "Instrument", instrument_keys, index=instrument_keys.index(active_instr.key),
                format_func=lambda k: INSTRUMENTS[k].label, key="depot_new_instrument",
                help="Voreingestellt ist das oben aktive Instrument - bei Bedarf hier fuer diese "
                     "Kombo umstellen (das Depot ist gemeinsam fuer alle Instrumente).",
            )
        with c1:
            name = st.text_input("Name der Kombo", placeholder="z.B. IC Oktober")
        with c2:
            expiry = st.date_input("Verfall (gilt fuer alle Beine)", value=date.today() + timedelta(days=7))
        st.caption("Pro Bein: Typ · Richtung · Strike · Menge · Einstiegspreis "
                   "(Punkte je Option, NICHT mal Multiplikator).")
        legs = []
        for k in range(n_legs):
            t0, s0 = defaults[k]
            cols = st.columns([1, 1, 1.3, 0.8, 1.2])
            typ = cols[0].selectbox(f"Bein {k + 1}: Typ", ["put", "call"], index=0 if t0 == "put" else 1,
                                    key=f"depot_new_{n_legs}_{k}_t")
            side = cols[1].selectbox("Richtung", ["long", "short"], index=0 if s0 == "long" else 1,
                                     key=f"depot_new_{n_legs}_{k}_s")
            strike = cols[2].number_input("Strike", min_value=0.0, value=default_strike, step=5.0,
                                          key=f"depot_new_{n_legs}_{k}_k")
            qty = cols[3].number_input("Menge", min_value=0.0, value=1.0, step=1.0,
                                       key=f"depot_new_{n_legs}_{k}_q")
            price = cols[4].number_input("Einstiegspreis", min_value=0.0, value=0.0, step=0.05,
                                         key=f"depot_new_{n_legs}_{k}_p")
            legs.append({"option_type": typ, "side": side, "strike": strike, "qty": qty, "entry_price": price})
        note = st.text_input("Notiz (optional)")
        submitted = st.form_submit_button("Zum Depot hinzufuegen", type="primary")

    if submitted:
        combo = dl.make_combo(name, expiry.isoformat(), legs, note, instrument=instrument_key)
        errs = dl.validate_combo(combo)
        if errs:
            st.error("Nicht gespeichert: " + " ".join(errs))
        else:
            _commit(st.session_state["depot_combos"] + [combo])
            st.success(f"'{combo['name']}' ({INSTRUMENTS[instrument_key].label}) gespeichert.")


def _render_depot_list() -> None:
    combos = st.session_state["depot_combos"]
    st.subheader(f"Depot ({len(combos)} Kombos, alle Instrumente)")
    st.caption(f"Gespeichert in `{DEPOT_FILE}` - wird beim naechsten Start automatisch wieder geladen.")
    if not combos:
        st.info("Noch keine Positionen. Oben eine Kombo hinzufuegen.")
        return

    rows = []
    for c in combos:
        valid = not dl.validate_combo(c)
        dte = _dte_exact(c["expiry"]) if valid else np.nan
        net = dl.entry_net(c)
        mult = dl.multiplier_for(c)
        instr_key = c.get("instrument", DEFAULT_INSTRUMENT)
        currency = INSTRUMENTS[instr_key].currency if instr_key in INSTRUMENTS else "?"
        rows.append({
            "Kombo": c["name"], "Instrument": _instr_label(c), "Verfall": c["expiry"],
            "Rest (Tage)": dte, "Beine": " · ".join(_leg_text(l) for l in c["legs"]),
            "Netto-Einstieg (Pkt)": net, "Netto-Einstieg (Geldwert)": net * mult,
            "Waehrung": currency,
            "Status": "abgelaufen" if (pd.notna(dte) and dte <= 0) else ("Fehler" if pd.isna(dte) else "offen"),
        })
    st.dataframe(pd.DataFrame(rows).style.format(
        {"Rest (Tage)": "{:.1f}", "Netto-Einstieg (Pkt)": "{:+.2f}", "Netto-Einstieg (Geldwert)": "{:+,.0f}"},
        na_rep="n/a"), use_container_width=True, hide_index=True)
    st.caption("Netto-Einstieg: positiv = Kredit erhalten, negativ = Debit bezahlt. 'Geldwert' nutzt den "
               "instrumentenspezifischen Multiplikator (siehe instruments.py) - KEINE Waehrungsumrechnung, "
               "die Betraege bei EUR/JPY-Instrumenten sind in der jeweiligen Landeswaehrung.")

    instrument_keys = list(INSTRUMENTS.keys())
    for c in list(combos):
        cid = c["id"]
        with st.expander(f"Bearbeiten / Loeschen: {_combo_label(c)}  [{_instr_label(c)}]"):
            e0, e1, e2 = st.columns([1.3, 2, 1])
            with e0:
                cur_instr = c.get("instrument", DEFAULT_INSTRUMENT)
                new_instr = st.selectbox(
                    "Instrument", instrument_keys,
                    index=instrument_keys.index(cur_instr) if cur_instr in instrument_keys else 0,
                    format_func=lambda k: INSTRUMENTS[k].label, key=f"depot_instr_{cid}",
                )
            with e1:
                new_name = st.text_input("Name", value=c["name"], key=f"depot_name_{cid}")
            with e2:
                try:
                    exp_default = datetime.strptime(c["expiry"], "%Y-%m-%d").date()
                except ValueError:
                    exp_default = date.today()
                new_exp = st.date_input("Verfall", value=exp_default, key=f"depot_exp_{cid}")
            legs_df = pd.DataFrame(c["legs"])[["option_type", "side", "strike", "qty", "entry_price"]]
            edited = st.data_editor(
                legs_df, num_rows="fixed", hide_index=True, use_container_width=True, key=f"depot_legs_{cid}",
                column_config={
                    "option_type": st.column_config.SelectboxColumn("Typ", options=["put", "call"], required=True),
                    "side": st.column_config.SelectboxColumn("Richtung", options=["long", "short"], required=True),
                    "strike": st.column_config.NumberColumn("Strike", min_value=0.0, step=5.0, required=True),
                    "qty": st.column_config.NumberColumn("Menge", min_value=0.0, step=1.0, required=True),
                    "entry_price": st.column_config.NumberColumn("Einstiegspreis", min_value=0.0, step=0.05,
                                                                 required=True),
                },
            )
            b1, b2 = st.columns(2)
            if b1.button("Aenderungen speichern", key=f"depot_save_{cid}"):
                upd = dl.make_combo(new_name, new_exp.isoformat(), edited.to_dict("records"),
                                    c.get("note", ""), cid, c.get("created"), instrument=new_instr)
                errs = dl.validate_combo(upd)
                if errs:
                    st.error("Nicht gespeichert: " + " ".join(errs))
                else:
                    _commit([upd if x["id"] == cid else x for x in st.session_state["depot_combos"]])
                    st.session_state.pop("depot_results", None)
                    st.rerun()
            if b2.button("Kombo loeschen", key=f"depot_del_{cid}"):
                _commit([x for x in st.session_state["depot_combos"] if x["id"] != cid])
                st.session_state.pop("depot_results", None)
                st.rerun()

    st.download_button("Depot als JSON herunterladen (Sicherung)",
                       data=DEPOT_FILE.read_text(encoding="utf-8") if DEPOT_FILE.exists() else "{}",
                       file_name="depot.json", mime="application/json")


# ───────────────────────── Analyse (nur Kombos des AKTIVEN Instruments) ─────────────────────────

def _analyze(daily_df: pd.DataFrame, combos: list[dict], market: dict,
            instr: InstrumentConfig, p: dict) -> dict | None:
    """Bewertet jede Kombo (bereits auf 'instr' gefiltert, siehe render()) mit
    DERSELBEN Rechnung wie die Iron-Condor-Suche (condor_search.weighted_pnl_stats) -
    eine Funktion je Kennzahl, kein Nachbau."""
    spot = market["spot"]
    todo, skipped = [], []
    for c in combos:
        errs = dl.validate_combo(c)
        dte_now = _dte_exact(c["expiry"]) if not errs else None
        if errs or dte_now is None or dte_now <= 0:
            skipped.append(f"{_combo_label(c)}: {'abgelaufen' if not errs else ' '.join(errs)}")
        else:
            todo.append((c, dte_now))
    if not todo:
        st.warning(f"Keine offene, gueltige Kombo fuer {instr.label} zum Analysieren.")
        return None

    ctx = dashboard_livechain._find_today_analogs(daily_df, 7.0, p["max_distance"], p["bandwidth"], p["min_gap"], instr)
    if ctx is None:
        return None
    valid_t = available_t_days(ctx["training_table"])
    moves_cache: dict[int, pd.DataFrame] = {}
    n_unbounded = 0

    results = []
    for c, dte in todo:
        notes = []
        T_used = min(valid_t, key=lambda t: abs(t - dte))
        if abs(T_used - dte) > 1.0:
            notes.append(f"Restlaufzeit {dte:.1f} d, Analog-Bewegungen fuer T={T_used} d")
        if T_used not in moves_cache:
            moves_cache[T_used] = analog_moves(ctx["analogs"], ctx["training_table"], T_used, ctx["query_date"])
        moves = moves_cache[T_used]
        if moves.empty:
            skipped.append(f"{_combo_label(c)}: keine Analog-Tage fuer T={T_used}")
            continue

        w = moves["weight"].to_numpy(dtype=float)
        S_d = spot * (1.0 + moves["ret"].to_numpy(dtype=float))
        net = dl.entry_net(c)
        V_d = dl.expiry_value(c, S_d)
        pnl_entry = net + V_d

        # ── Marktdaten je Bein (Mid fuer Marktwert, Vola fuer Ausstiegspreise) ──
        mids, ivs, fallback_iv = [], [], False
        for leg in c["legs"]:
            mid, iv = _lookup_leg(market["enriched"], c["expiry"], leg)
            mids.append(mid)
            if iv is None or not iv > 0:
                iv, fallback_iv = p["fallback_vol"], True
            ivs.append(max(iv, 0.01))
        mark = None
        today_combo = None
        if all(m is not None for m in mids):
            mark = float(sum(dl._sign(l) * l["qty"] * m for l, m in zip(c["legs"], mids)))
            today_combo = _hypothetical_today_combo(c, mids)
        if mark is None:
            notes.append("kein Marktpreis in der Kette -> Marktwert/Restgewinn n/a")
        if fallback_iv:
            notes.append("Ersatz-Vola fuer mind. ein Bein")

        rest = (V_d - mark) if mark is not None else None

        # Basis + Risikobezug MUESSEN zusammenpassen: mit Marktpreis ->
        # Restgewinn ab heute, Risiko aus der heute-eroeffneten Kombo; ohne
        # Marktpreis -> P&L seit historischem Einstieg, Risiko aus der
        # tatsaechlichen Kombo.
        if today_combo is not None:
            basis = rest
            risk_profile = dl.payoff_profile(today_combo)
        else:
            basis = pnl_entry
            risk_profile = dl.payoff_profile(c)
        max_loss = risk_profile["max_loss"]

        if not np.isfinite(max_loss):
            n_unbounded += 1
            notes.append(
                "unbegrenztes Verlustrisiko (ab jetzigem Bezugspunkt) -> 'Rendite/"
                "Risiko %', 'Voll-Verlust %' und '5%-Quantil (R/R)' nicht definiert (n/a)"
            )

        train_mask = (moves["entry_date"] < moves["entry_date"].median()).to_numpy()
        hold_days = int(moves["hold_days"].max())
        stats = weighted_pnl_stats(basis, w, moves["entry_date"].to_numpy(), hold_days,
                                   train_mask, max_loss, tail_quantile=0.05)

        prof = dl.payoff_profile(c)   # bleibt: exakte Verfallskennzahlen ab dem TATSAECHLICHEN Einstieg
        f_today = _make_pnl_today(c, ivs, dte, market["r_free"], market["dividend_yield"])
        stop_target = -p["stop_pct"] / 100.0 * abs(net) if abs(net) > 0 else None
        zi_be = _zone_info(f_today, 0.0, spot)
        zi_stop = _zone_info(f_today, stop_target, spot) if stop_target is not None else None

        grid = np.linspace(spot * 0.8, spot * 1.2, 401)
        chart_df = pd.concat([
            pd.DataFrame({"Spot": grid, "P&L (Pkt)": f_today(grid), "Linie": "heute (Modell)"}),
            pd.DataFrame({"Spot": grid, "P&L (Pkt)": dl.pnl_at_expiry(c, grid), "Linie": "bei Verfall"}),
        ])

        results.append({
            "id": c["id"], "name": _combo_label(c), "expiry": c["expiry"], "dte": dte, "T_used": T_used,
            "net": net, "mark": mark,
            "unreal": (net + mark) if mark is not None else None,
            "t_basis": "Restgewinn" if today_combo is not None else "P&L seit Einstieg",
            "mean": stats["mean"], "ror": stats["ror"], "tstat": stats["tstat"],
            "e_train": stats["e_train"], "e_val": stats["e_val"],
            "full_loss_share": stats["full_loss_share"], "n_full_loss": stats["n_full_loss"],
            "q_tail": stats["q_tail"], "win_rate_pct": stats["win_rate_pct"],
            "max_loss": prof["max_loss"], "max_profit": prof["max_profit"],
            "be_expiry": prof["breakevens"],
            "zi_be": zi_be, "zi_stop": zi_stop, "stop_target": stop_target,
            "pnl_now_model": float(f_today(np.array([spot]))[0]),
            "chart": chart_df, "notes": notes,
        })
    return {"results": results, "skipped": skipped, "spot": spot,
            "n_days": {t: len(m) for t, m in moves_cache.items()},
            "sig": dl_signature(combos), "stop_pct": p["stop_pct"], "n_unbounded": n_unbounded,
            "instrument": instr.key, "multiplier": instr.multiplier, "currency": instr.currency}


def dl_signature(combos: list[dict]) -> str:
    import json
    return json.dumps(combos, sort_keys=True)


def _yn(flag: bool) -> str:
    return "✅" if flag else "❌"


def _render_results(res: dict) -> None:
    spot = res["spot"]
    mult = res.get("multiplier", dl.MULTIPLIER)
    ccy = res.get("currency", "USD")
    rows = res["results"]
    for s in res["skipped"]:
        st.caption(f"⚠️ Uebersprungen - {s}")

    tot_unreal = sum(r["unreal"] for r in rows if r["unreal"] is not None)
    tot_mean = sum(r["mean"] for r in rows)
    m1, m2 = st.columns(2)
    m1.metric(f"Summe unrealisiert heute ({ccy})", f"{tot_unreal * mult:+,.0f}")
    m2.metric(f"Summe E[P&L] ({ccy})", f"{tot_mean * mult:+,.0f}")
    st.caption(
        "E[P&L] nutzt dieselbe Rechnung wie die Iron-Condor-Suche (Teil C, 'Heutige Kette'): "
        "gewichteter Mittelwert ueber die Analog-Tage, HAC-t-Stat (Bartlett, Kalenderzeit), "
        "Train/Val-Split, Voll-Verlust-Anteil und 5%-Quantil relativ zum Max-Verlust. Basis "
        "siehe Spalte 'Basis': 'Restgewinn' (Marktpreis vorhanden, Bezugspunkt = heutiger "
        "Mid-Preis, wie 'Kredit' bei der Condor-Suche) oder 'P&L seit Einstieg' (kein "
        "Marktpreis verfuegbar, Bezugspunkt = tatsaechlicher historischer Einstieg). Additiv "
        "ueber Kombos DIESES Instruments, auch bei unterschiedlichem Verfall/T_used."
    )
    if res.get("n_unbounded", 0) > 0:
        st.warning(
            f"{res['n_unbounded']} von {len(rows)} Kombo(s) haben unbegrenztes Verlustrisiko "
            "ab dem jeweiligen Risikobezugspunkt (z.B. mehr Short- als Long-Calls) - fuer diese "
            "sind 'Rendite/Risiko %', 'Voll-Verlust %' und '5%-Quantil (R/R)' n/a."
        )

    tbl = pd.DataFrame([{
        "Kombo": r["name"], "Rest (Tage)": r["dte"], "T (Analog)": r["T_used"],
        "Basis": r["t_basis"],
        "Netto-Einstieg (Pkt)": r["net"], "Marktwert heute (Pkt)": r["mark"],
        "Unrealisiert (Pkt)": r["unreal"],
        "E[P&L] (Pkt)": r["mean"],
        "Rendite/Risiko %": r["ror"] * 100.0 if pd.notna(r["ror"]) else np.nan,
        "t (HAC)": r["tstat"],
        "E Train": r["e_train"], "E Val": r["e_val"],
        "Voll-Verlust %": r["full_loss_share"] * 100.0 if pd.notna(r["full_loss_share"]) else np.nan,
        "Voll-Verlust-Tage (n)": r["n_full_loss"] if r["n_full_loss"] is not None else np.nan,
        "5%-Quantil (R/R)": r["q_tail"],
        "Gewinnrate %": r["win_rate_pct"],
        "Max-Verlust bei Verfall (Pkt)": r["max_loss"] if np.isfinite(r["max_loss"]) else np.nan,
        "Break-even Verfall": " / ".join(f"{b:,.0f}" for b in r["be_expiry"]) or "-",
    } for r in rows])
    fmt = {
        "Rest (Tage)": "{:.1f}", "T (Analog)": "{:.0f}",
        "Netto-Einstieg (Pkt)": "{:+.2f}", "Marktwert heute (Pkt)": "{:+.2f}",
        "Unrealisiert (Pkt)": "{:+.2f}",
        "E[P&L] (Pkt)": "{:+.2f}", "Rendite/Risiko %": "{:+.1f}",
        "t (HAC)": "{:.2f}", "E Train": "{:+.2f}", "E Val": "{:+.2f}",
        "Voll-Verlust %": "{:.1f}", "Voll-Verlust-Tage (n)": "{:.0f}",
        "5%-Quantil (R/R)": "{:.2f}", "Gewinnrate %": "{:.1f}",
        "Max-Verlust bei Verfall (Pkt)": "{:,.1f}",
    }
    st.dataframe(tbl.style.format(fmt, na_rep="n/a"), use_container_width=True, hide_index=True)
    st.caption("Netto-Einstieg: positiv = Kredit erhalten, negativ = Debit bezahlt.")

    st.subheader("Ausstiegspreise HEUTE")
    st.caption(
        f"Aktueller Spot: {spot:,.2f} {ccy}. 'Unrentabel heute ab' = Kurs, ab dem die Kombo HEUTE "
        "(mit Zeitwert; Black-76 je Bein mit der impliziten Vola aus der Kette, ohne "
        f"Vola-Aenderung) im Minus waere. 'Stop-Level' = Kurs, bei dem der Verlust "
        f"{res['stop_pct']:g} % des Netto-Einstiegs erreicht. 'Break-even bei Verfall' ist exakt "
        "(Netto-Einstieg + Payoff). Suchbereich +-50 % um den Spot."
    )
    exit_rows = []
    for r in rows:
        exit_rows.append({
            "Kombo": r["name"],
            "Heute im Plus (Modell)": "ja" if r["pnl_now_model"] >= 0 else "nein",
            "Unrentabel heute ab": _fmt_zone_text(r["zi_be"], spot, "Break-even"),
            "Stop-Level heute": (_fmt_zone_text(r["zi_stop"], spot, "Stop")
                                 if r["zi_stop"] is not None else "n/a (Netto-Einstieg 0)"),
            "Break-even bei Verfall (exakt)": " / ".join(
                f"{b:,.0f} ({(b / spot - 1) * 100:+.1f} %)" for b in r["be_expiry"]) or "-",
            "Hinweise": "; ".join(r["notes"]),
        })
    st.dataframe(pd.DataFrame(exit_rows), use_container_width=True, hide_index=True)

    st.subheader("P&L-Kurve")
    names = [r["name"] for r in rows]
    pick = st.selectbox("Kombo", names, key="depot_chart_pick")
    r = rows[names.index(pick)]
    base = alt.Chart(r["chart"]).mark_line().encode(
        x=alt.X("Spot:Q", scale=alt.Scale(zero=False)), y="P&L (Pkt):Q",
        color=alt.Color("Linie:N"), strokeDash="Linie:N",
        tooltip=["Spot:Q", "P&L (Pkt):Q", "Linie:N"],
    )
    spot_rule = alt.Chart(pd.DataFrame({"Spot": [spot]})).mark_rule(color="gray", strokeDash=[4, 4]).encode(x="Spot:Q")
    layers = [base, spot_rule,
              alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color="gray").encode(y="y:Q")]
    if r["stop_target"] is not None:
        layers.append(alt.Chart(pd.DataFrame({"y": [r["stop_target"]]}))
                      .mark_rule(color="#d33", strokeDash=[2, 2]).encode(y="y:Q"))
    chart = layers[0]
    for lay in layers[1:]:
        chart = chart + lay
    st.altair_chart(chart.properties(height=350).interactive(), use_container_width=True)
    st.caption(
        "Beide Linien zeigen das P&L gegenueber dem tatsaechlichen historischen Einstieg: "
        "'heute (Modell)' = wenn der Kurs jetzt springt und man sofort schliesst (mit Zeitwert), "
        "'bei Verfall' = Netto-Einstieg + Payoff. Rote gestrichelte Linie = Stop-Grenze. "
        f"Basis des t-Stats oben ('{r['t_basis']}'): HAC-korrigiert (Bartlett, Kalenderzeit) - "
        "dieselbe Methode wie in der Iron-Condor-Suche. Die Analog-Bewertung ist eine "
        "historische Ruecksimulation, keine Prognose."
    )


# ───────────────────────── Einstiegspunkt ─────────────────────────

def render(df, daily_df, has_resolution, instr: InstrumentConfig) -> None:
    """Oeffentlicher Einstiegspunkt - bekommt jetzt zusaetzlich das oben
    gewaehlte Instrument (siehe dashboard.py). Das Depot selbst bleibt
    instrumentuebergreifend; 'Analyse' unten filtert auf instr.key."""
    st.header("Depot-Analyse")
    _init_state()
    if st.session_state.get("depot_load_warning"):
        st.warning(st.session_state["depot_load_warning"])

    all_combos = st.session_state["depot_combos"]
    other_instruments = sorted({c.get("instrument", DEFAULT_INSTRUMENT) for c in all_combos} - {instr.key})

    with st.sidebar:
        st.divider()
        st.header("Depot-Analyse")
        st.caption(f"Aktives Instrument (oben gewaehlt): **{instr.label}**")
        r_free = st.number_input("Risikoloser Zins", 0.0, 0.20, 0.0, 0.005, format="%.3f", key="depot_r")
        dividend_yield = st.number_input(
            "Dividenden-/Convenience-Yield", 0.0, 0.10, float(instr.dividend_yield), 0.001,
            format="%.3f", key=f"depot_q_{instr.key}",
            help="Voreingestellt aus instruments.py fuer dieses Instrument, hier pro Analyse ueberschreibbar.",
        )
        fallback_vol = st.number_input("Ersatz-Vola (%), falls Strike nicht in der Kette", 1.0, 200.0, 15.0, 1.0,
                                       key="depot_fbvol") / 100.0
        stop_pct = st.number_input("Stop-Level: Verlust in % des Netto-Einstiegs", 1.0, 1000.0, 100.0, 10.0,
                                   key="depot_stop")
        st.caption("Analog-Suche (wie im Tab 'Heutige Kette'):")
        max_distance = st.number_input("Max. Distanz", 0.01, 100.0, 4.0, 0.5, key="depot_maxdist")
        bandwidth = st.number_input("Kernel-Bandbreite σ", 0.01, 50.0, 1.0, 0.1, key="depot_bw")
        min_gap = st.number_input("Mindestabstand zu heute (Handelstage)", 0, 60, 5, key="depot_gap")
        reload_market = st.button("Marktdaten neu laden", key="depot_reload")

    daily_df = daily_df.sort_values("date")
    _render_add_form(daily_df, instr)
    st.divider()
    _render_depot_list()

    if other_instruments:
        other_labels = ", ".join(INSTRUMENTS[k].label for k in other_instruments if k in INSTRUMENTS)
        st.info(
            f"Das Depot enthaelt auch Kombos in anderen Instrumenten ({other_labels}) - die "
            f"Analyse unten wertet nur Kombos des oben aktiven Instruments ({instr.label}) aus. "
            "Zum Analysieren der anderen Kombos oben das jeweilige Instrument auswaehlen."
        )

    combos = [c for c in st.session_state["depot_combos"] if c.get("instrument", DEFAULT_INSTRUMENT) == instr.key]
    if not combos:
        st.info(f"Keine Kombos fuer {instr.label} im Depot.")
        return
    st.divider()
    st.subheader(f"Analyse ({instr.label})")

    market = _get_market(instr, combos, r_free, dividend_yield, force=reload_market)
    if market is None:
        return
    st.caption(f"Spot {market['spot']:,.2f} {instr.currency} · Kette abgerufen "
               f"{market['fetch_ts'].tz_convert(_NY).strftime('%Y-%m-%d %H:%M:%S')} ET (~15 Min. verzoegert).")

    params = dict(max_distance=float(max_distance), bandwidth=float(bandwidth), min_gap=int(min_gap),
                  fallback_vol=float(fallback_vol), stop_pct=float(stop_pct))
    if st.button("Depot analysieren", type="primary", key="depot_run"):
        res = _analyze(daily_df, combos, market, instr, params)
        if res is not None:
            st.session_state["depot_results"] = res

    res = st.session_state.get("depot_results")
    if res is None:
        st.info("'Depot analysieren' klicken.")
    elif res.get("instrument") != instr.key:
        st.info("Die letzte Analyse bezog sich auf ein anderes Instrument - bitte erneut analysieren.")
    elif res["sig"] != dl_signature(combos):
        st.info("Das Depot wurde seit der letzten Analyse geaendert - bitte erneut analysieren.")
    else:
        _render_results(res)