"""
MODUS 4: Depot-Analyse.

Eigene Optionspositionen als Kombos aus 1 bis 4 Beinen erfassen (gemeinsamer Verfall),
dauerhaft speichern (data/depot.json, wird beim Start automatisch wieder geladen).

BEWERTUNG ZU HEUTIGEN MARKTPREISEN (kein Einstiegspreis-Eingabefeld mehr):
  Jede Kombo wird so bewertet, als waere sie HEUTE zu den aktuellen Mid-Preisen der Kette
  eroeffnet (wie 'Kredit' in der Iron-Condor-Suche). Die Netto-Praemie heute
      = Summe: short +Mid*Menge, long -Mid*Menge        (Kredit positiv, Debit negativ)
  ist der Bezugspunkt fuer alle Kennzahlen. Fehlt ein Strike in der Kette, wird der Preis dieses
  Beins per Black-76 mit der 'Ersatz-Vola' aus der Sidebar geschaetzt und die Zeile markiert.
  Ein frueher eingegebener Einstiegspreis (alte depot.json) wird ignoriert.

  ANALOG-TAGE-SIMULATION:
    Historische PROZENTUALE Kursbewegungen der Analog-Tage (gewichtet, siehe
    analog_matching.find_analogs) werden auf den heutigen Spot projiziert:
    S_d = Spot_heute * (1 + Bewegung_d). Die Kombo wird bis Verfall gehalten:
      P&L_d = Netto-Praemie_heute + V(S_d),   V(S) = Summe: long +intrinsic*qty, short -intrinsic*qty.
    Kennzahlen: E[P&L bis Verfall], Wahrscheinlichkeit im Plus, gewichtetes 5%-Quantil,
    HAC-t-Stat (Bartlett, Kalenderzeit). Kein Vola-Modell in dieser Bewertung.

  AUSSTIEGSPREISE HEUTE:
    Wer HEUTE zu Marktpreisen einsteigt, hat bei aktuellem Spot definitionsgemaess P&L = 0 - ein
    'Break-even heute' waere daher immer der aktuelle Spot selbst. Sinnvoll ist stattdessen ein
    STOP-LEVEL: der Kurs, bei dem die Kombo HEUTE (mit Zeitwert) X % ihrer Netto-Praemie an
    Verlust erreicht. Dafuer wird jedes Bein mit Black-76 bei der Restlaufzeit neu bewertet, mit
    der impliziten Vola des Beins aus der Kette (sticky strike, keine Vola-Aenderung). Exakt und
    modellfrei: Break-even sowie Max-Gewinn/-Verlust BEI VERFALL (aus Netto-Praemie heute + Payoff).
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
from black76 import forward_price
from chain_delta import black76_price
from analog_matching import available_t_days
from condor_search import (
    CondorConfig, analog_moves, weighted_pnl_stats, criteria_flags, analog_robustness,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEPOT_FILE = DATA_DIR / "depot.json"

_NY = "America/New_York"
_IC_DEFAULTS = [("put", "long"), ("put", "short"), ("call", "short"), ("call", "long")]
_GENERIC_DEFAULTS = [("put", "short"), ("call", "short"), ("put", "long"), ("call", "long")]
_LEVEL_LO, _LEVEL_HI = 0.5, 1.5     # Suchbereich der Ausstiegspreise: +-50 % um den Spot


# ───────────────────────── Hilfsfunktionen ─────────────────────────

def _dte_exact(expiry_str: str) -> float:
    """Restlaufzeit in Tagen bis Verfall 16:00 ET (PM-Settlement, wie in der Kettenansicht)."""
    now = pd.Timestamp.now(tz=_NY)
    exp = pd.Timestamp(f"{expiry_str} 16:00", tz=_NY)
    return (exp - now).total_seconds() / 86400.0


def _norm_dates(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series)
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s.dt.normalize()


def _lookup_leg(enriched: pd.DataFrame | None, expiry_str: str, leg: dict) -> dict | None:
    """Mid/Bid/Ask und implizite Vola des Beins aus der heutigen Kette - exakter Treffer auf
    Verfall/Typ/Strike, sonst None."""
    if enriched is None or enriched.empty:
        return None
    exp = pd.Timestamp(expiry_str).normalize()
    m = ((_norm_dates(enriched["expiry"]) == exp)
         & (enriched["option_type"] == leg["option_type"])
         & np.isclose(enriched["strike"].astype(float), leg["strike"]))
    sub = enriched[m]
    if sub.empty:
        return None
    r = sub.iloc[0]

    def num(col):
        v = r[col] if col in r.index else None
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v if np.isfinite(v) else None
    return {"mid": num("mid"), "bid": num("bid"), "ask": num("ask"), "iv": num("implied_vol_mkt")}


# Standardwerte, falls der Tab 'Heutige Kette' noch nie geoeffnet wurde. Sonst gelten IMMER die dort
# eingestellten Werte (session_state 'shared_analog_params'/'shared_condor_criteria'), damit Depot-
# Analyse und Iron-Condor-Suche auf demselben Analog-Set, denselben Preisen und denselben Schwellen
# rechnen - sonst waeren die Zahlen nie vergleichbar.
_DEFAULT_ANALOG = dict(max_distance=4.0, bandwidth=1.0, min_gap=5)
_DEFAULT_CRITERIA = dict(basis="Bid/Ask (echt)", t_min=3.0, max_full=3.0, tail_floor=-0.75)


def _shared_settings() -> tuple[dict, dict]:
    analog = {**_DEFAULT_ANALOG, **st.session_state.get("shared_analog_params", {})}
    crit = {**_DEFAULT_CRITERIA, **st.session_state.get("shared_condor_criteria", {})}
    return analog, crit


def _dte_for(combo: dict, market: dict | None) -> float:
    """Restlaufzeit wie in der Kettenansicht (dte_exact der Kette, Stand Kettenabruf); ohne Treffer
    eigene Rechnung bis Verfall 16:00 ET. So waehlen Depot und Condor-Suche dieselbe Analog-Laufzeit."""
    if market is not None and not market["enriched"].empty and "dte_exact" in market["enriched"].columns:
        en = market["enriched"]
        m = _norm_dates(en["expiry"]) == pd.Timestamp(combo["expiry"]).normalize()
        vals = en.loc[m, "dte_exact"]
        if len(vals):
            return float(vals.iloc[0])
    return _dte_exact(combo["expiry"])


def _leg_market(combo: dict, dte: float, market: dict, fallback_vol: float, basis: str = "bidask") -> list[dict]:
    """Je Bein: Preis fuer eine ERÖFFNUNG HEUTE, Vola fuer die Ausstiegspreise und Quelle.
    basis 'bidask': short zum Bid, long zum Ask (wie die Iron-Condor-Suche); fehlt Bid/Ask, gilt
    Mid (markiert). basis 'mid': Mid. Fehlt der Strike in der Kette: Black-76-Preis mit Ersatz-Vola."""
    out = []
    T = dte / 365.0
    F = forward_price(market["spot"], market["r_free"], market["dividend_yield"], T)
    for leg in combo["legs"]:
        row = _lookup_leg(market["enriched"], combo["expiry"], leg)
        if row is not None and row["mid"] is not None and row["mid"] > 0 and row["iv"] is not None and row["iv"] > 0:
            price, src = row["mid"], "Markt (Mid)"
            if basis == "bidask":
                px = row["bid"] if leg["side"] == "short" else row["ask"]
                if px is not None and px > 0:
                    price, src = px, "Markt (Bid/Ask)"
                else:
                    src = "Markt (Mid, Bid/Ask fehlt)"
            out.append({"price": price, "iv": max(row["iv"], 0.01), "source": src})
        else:
            vol = max(fallback_vol, 0.01)
            price = float(black76_price(F, leg["strike"], T, vol, market["r_free"], leg["option_type"] == "put"))
            out.append({"price": price, "iv": vol, "source": "Modell"})
    return out


def _today_combo(combo: dict, legs_mkt: list[dict]) -> dict:
    """Kopie der Kombo mit entry_price je Bein = heutiger Preis (Bezugspunkt fuer alle Kennzahlen)."""
    legs = [{**leg, "entry_price": m["price"]} for leg, m in zip(combo["legs"], legs_mkt)]
    return {**combo, "legs": legs}


def _source_text(legs_mkt: list[dict]) -> str:
    n_model = sum(1 for m in legs_mkt if m["source"] == "Modell")
    srcs = sorted({m["source"] for m in legs_mkt if m["source"] != "Modell"})
    text = " + ".join(srcs)
    if n_model:
        text = (text + " · " if text else "") + f"{n_model} von {len(legs_mkt)} Bein(en) Modellpreis (Ersatz-Vola)"
    return text


def _leg_text(leg: dict) -> str:
    side = "L" if leg["side"] == "long" else "S"
    typ = "P" if leg["option_type"] == "put" else "C"
    q = f"{leg['qty']:g}x " if leg["qty"] != 1 else ""
    return f"{side} {q}{leg['strike']:g}{typ}"


def _combo_label(c: dict) -> str:
    return f"{c['name']} ({c['expiry']})"


def _commit(combos: list[dict]) -> None:
    """Speichert das Depot in Session UND auf Platte."""
    st.session_state["depot_combos"] = combos
    try:
        dl.save_depot(DEPOT_FILE, combos)
    except OSError as e:
        st.error(f"Depot konnte NICHT gespeichert werden ({e}) - Aenderung gilt nur bis zum "
                 "Schliessen des Programms!")


def add_combo_to_depot(name: str, expiry: str, legs: list[dict], note: str = "") -> dict:
    """OEFFENTLICHE Schnittstelle fuer ANDERE Module (z.B. die 'Zum Depot hinzufuegen'-Buttons in
    der Analog-Tabelle und der Iron-Condor-Suche in dashboard_livechain.py): legt eine neue Kombo
    an, haengt sie an das AKTUELLE Depot an und speichert sofort. Liest das Depot dafuer frisch von
    der Platte statt blind aus st.session_state - der Aufrufer muss diesen Modus in der laufenden
    Sitzung nicht vorher geoeffnet haben. Gibt die gespeicherte Kombo zurueck; wirft ValueError bei
    einer ungueltigen Kombo (dann wird NICHTS gespeichert)."""
    combo = dl.make_combo(name, expiry, legs, note)
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
    """P&L-Funktion HEUTE (relativ zur heutigen Netto-Praemie) in Abhaengigkeit vom Spot:
    Black-76 je Bein mit fester Vola je Bein. Akzeptiert/liefert numpy-Arrays.
    'combo' muss die HEUTE-Kombo sein (entry_price = heutiger Preis)."""
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


def _zone_info(f, target: float, spot: float) -> dict:
    """Zerlegt [lo, hi] in Zonen 'P&L >= target' / '< target' und beschreibt die Zone um den Spot.
    lower/upper = Kurs, unter/ueber dem P&L < target wird (None = im Suchbereich kein Level)."""
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


# ───────────────────────── Marktdaten ─────────────────────────

def _needed_max_dte(combos: list[dict]) -> float:
    return max([_dte_exact(c["expiry"]) for c in combos] + [1.0]) + 2.0


def _get_market(combos: list[dict], r_free: float, dividend_yield: float, force: bool = False) -> dict | None:
    """Gemeinsamer Marktdaten-Snapshot (siehe market_data.py) - dieselbe Kette wie im Tab
    'Heutige Kette', wird NICHT pro Tab neu geladen. Laedt nur, wenn noch keiner existiert, er
    veraltet ist, die Kombos einen laengeren DTE-Bereich brauchen oder 'neu laden' geklickt wurde."""
    needed = _needed_max_dte(combos)
    snap = None if force else market_data.current(needed)
    if snap is None:
        with st.spinner("Lade aktuellen Spot und Optionskette..."):
            try:
                snap = market_data.get(float(r_free), float(dividend_yield), needed, force=force)
            except Exception as e:
                st.error(f"Marktdaten konnten nicht geladen werden: {e}")
                snap = market_data.current(0.0)
    return snap


# ───────────────────────── Eingabe / Verwaltung ─────────────────────────

def _render_add_form(daily_df: pd.DataFrame) -> None:
    st.subheader("Kombo hinzufuegen")
    # letzter GUELTIGER Schlusskurs (nicht einfach die letzte Zeile - die kann NaN sein, z.B. ein
    # unvollstaendiger/nicht gehandelter heutiger Balken) - sonst stuerzt round(NaN) weiter unten ab.
    valid_close = daily_df["close"].dropna() if "close" in daily_df.columns else pd.Series(dtype=float)
    last_close = float(valid_close.iloc[-1]) if len(valid_close) else 7000.0
    default_strike = float(round(last_close / 5.0) * 5.0)

    n_legs = st.selectbox("Anzahl Beine der Kombo", [1, 2, 3, 4], index=3, key="depot_n_legs",
                          help="4 = z.B. Iron Condor, 2 = Spread/Strangle, 1 = Einzeloption.")
    defaults = _IC_DEFAULTS if n_legs == 4 else _GENERIC_DEFAULTS

    with st.form(f"depot_add_form_{n_legs}"):
        c1, c2 = st.columns([2, 1])
        with c1:
            name = st.text_input("Name der Kombo", placeholder="z.B. IC Oktober")
        with c2:
            expiry = st.date_input("Verfall (gilt fuer alle Beine)", value=date.today() + timedelta(days=7))
        st.caption("Pro Bein: Typ · Richtung · Strike · Menge. Der Preis wird automatisch aus der "
                   "aktuellen Optionskette bestimmt - kein Einstiegspreis noetig.")
        legs = []
        for k in range(n_legs):
            t0, s0 = defaults[k]
            cols = st.columns([1, 1, 1.4, 0.9])
            typ = cols[0].selectbox(f"Bein {k + 1}: Typ", ["put", "call"], index=0 if t0 == "put" else 1,
                                    key=f"depot_new_{n_legs}_{k}_t")
            side = cols[1].selectbox("Richtung", ["long", "short"], index=0 if s0 == "long" else 1,
                                     key=f"depot_new_{n_legs}_{k}_s")
            strike = cols[2].number_input("Strike", min_value=0.0, value=default_strike, step=5.0,
                                          key=f"depot_new_{n_legs}_{k}_k")
            qty = cols[3].number_input("Menge", min_value=0.0, value=1.0, step=1.0,
                                       key=f"depot_new_{n_legs}_{k}_q")
            legs.append({"option_type": typ, "side": side, "strike": strike, "qty": qty, "entry_price": 0.0})
        note = st.text_input("Notiz (optional)")
        submitted = st.form_submit_button("Zum Depot hinzufuegen", type="primary")

    if submitted:
        combo = dl.make_combo(name, expiry.isoformat(), legs, note)
        errs = dl.validate_combo(combo)
        if errs:
            st.error("Nicht gespeichert: " + " ".join(errs))
        else:
            _commit(st.session_state["depot_combos"] + [combo])
            st.session_state.pop("depot_results", None)
            st.session_state["depot_flash"] = f"'{combo['name']}' gespeichert."
            st.rerun()


def _render_depot_list(market: dict | None, fallback_vol: float, basis: str = "bidask") -> None:
    combos = st.session_state["depot_combos"]
    st.subheader(f"Depot ({len(combos)} Kombos)")
    st.caption(f"Gespeichert in `{DEPOT_FILE}` - wird beim naechsten Start automatisch wieder geladen.")
    if not combos:
        st.info("Noch keine Positionen. Oben eine Kombo hinzufuegen.")
        return

    rows = []
    for c in combos:
        valid = not dl.validate_combo(c)
        dte = _dte_exact(c["expiry"]) if valid else np.nan
        net, src = np.nan, "n/a (keine Marktdaten)"
        if valid and dte > 0 and market is not None:
            lm = _leg_market(c, _dte_for(c, market), market, fallback_vol, basis)
            net = dl.entry_net(_today_combo(c, lm))
            src = _source_text(lm)
        rows.append({
            "Kombo": c["name"], "Verfall": c["expiry"],
            "Rest (Tage)": dte, "Beine": " · ".join(_leg_text(l) for l in c["legs"]),
            "Netto-Praemie heute (Pkt)": net, "Netto-Praemie heute ($)": net * dl.MULTIPLIER,
            "Preisquelle": src,
            "Status": "abgelaufen" if (pd.notna(dte) and dte <= 0) else ("Fehler" if pd.isna(dte) else "offen"),
        })
    st.dataframe(pd.DataFrame(rows).style.format(
        {"Rest (Tage)": "{:.1f}", "Netto-Praemie heute (Pkt)": "{:+.2f}", "Netto-Praemie heute ($)": "{:+,.0f}"},
        na_rep="n/a"), use_container_width=True, hide_index=True)
    st.caption("Netto-Praemie heute (aus den aktuellen Mid-Preisen): positiv = Kredit, negativ = Debit, "
               "wenn die Kombo heute eroeffnet wuerde.")

    for c in list(combos):
        cid = c["id"]
        with st.expander(f"Bearbeiten / Loeschen: {_combo_label(c)}"):
            e1, e2 = st.columns([2, 1])
            with e1:
                new_name = st.text_input("Name", value=c["name"], key=f"depot_name_{cid}")
            with e2:
                try:
                    exp_default = datetime.strptime(c["expiry"], "%Y-%m-%d").date()
                except ValueError:
                    exp_default = date.today()
                new_exp = st.date_input("Verfall", value=exp_default, key=f"depot_exp_{cid}")
            legs_df = pd.DataFrame(c["legs"])[["option_type", "side", "strike", "qty"]]
            edited = st.data_editor(
                legs_df, num_rows="fixed", hide_index=True, use_container_width=True, key=f"depot_legs_{cid}",
                column_config={
                    "option_type": st.column_config.SelectboxColumn("Typ", options=["put", "call"], required=True),
                    "side": st.column_config.SelectboxColumn("Richtung", options=["long", "short"], required=True),
                    "strike": st.column_config.NumberColumn("Strike", min_value=0.0, step=5.0, required=True),
                    "qty": st.column_config.NumberColumn("Menge", min_value=0.0, step=1.0, required=True),
                },
            )
            b1, b2 = st.columns(2)
            if b1.button("Aenderungen speichern", key=f"depot_save_{cid}"):
                recs = [{**r, "entry_price": 0.0} for r in edited.to_dict("records")]
                upd = dl.make_combo(new_name, new_exp.isoformat(), recs, c.get("note", ""), cid, c.get("created"))
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


# ───────────────────────── Analyse ─────────────────────────

def _analyze(daily_df: pd.DataFrame, combos: list[dict], market: dict, p: dict) -> dict | None:
    """Bewertet jede Kombo mit DERSELBEN Rechnung wie die Iron-Condor-Suche (condor_search.
    weighted_pnl_stats/criteria_flags) und die Analog-Tabelle (analog_robustness) - eine Funktion je
    Kennzahl, kein Nachbau. Analog-Set, Preisbasis und Schwellen kommen aus dem Tab 'Heutige Kette'."""
    spot = market["spot"]
    crit = CondorConfig(t_min=p["t_min"], max_full_loss_share=p["max_full"] / 100.0, tail_floor=p["tail_floor"])
    todo, skipped = [], []
    for c in combos:
        errs = dl.validate_combo(c)
        dte_now = _dte_exact(c["expiry"]) if not errs else None
        if errs or dte_now is None or dte_now <= 0:
            skipped.append(f"{_combo_label(c)}: {'abgelaufen' if not errs else ' '.join(errs)}")
        else:
            todo.append((c, _dte_for(c, market)))
    if not todo:
        st.warning("Keine offene, gueltige Kombo zum Analysieren.")
        return None

    ctx = dashboard_livechain._find_today_analogs(daily_df, 7.0, p["max_distance"], p["bandwidth"], p["min_gap"])
    if ctx is None:
        return None
    valid_t = available_t_days(ctx["training_table"])
    moves_cache: dict[int, pd.DataFrame] = {}

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

        # ── Bezugspunkt: HEUTIGE Preise einer Eroeffnung (Bid/Ask oder Mid; sonst Modellpreis) ──
        lm = _leg_market(c, dte, market, p["fallback_vol"], p["basis_key"])
        tc = _today_combo(c, lm)
        net = dl.entry_net(tc)                                   # Netto-Praemie heute (Kredit +)
        if any(m["source"] == "Modell" for m in lm):
            notes.append("kein Marktpreis in der Kette fuer: " + ", ".join(
                _leg_text(l) for l, m in zip(c["legs"], lm) if m["source"] == "Modell") + " -> Modellpreis (Ersatz-Vola)")
        ivs = [m["iv"] for m in lm]

        # ── Analog-Tage-Simulation bis Verfall ──
        w = moves["weight"].to_numpy(dtype=float)
        S_d = spot * (1.0 + moves["ret"].to_numpy(dtype=float))
        pnl = net + dl.expiry_value(c, S_d)

        prof = dl.payoff_profile(tc)
        max_loss = prof["max_loss"] if np.isfinite(prof["max_loss"]) else None      # None = unbegrenzt
        train_mask = (moves["entry_date"] < moves["entry_date"].median()).to_numpy()
        stats = weighted_pnl_stats(pnl, w, moves["entry_date"].to_numpy(), int(moves["hold_days"].max()),
                                   train_mask, max_loss, crit.tail_quantile)
        flags = criteria_flags(stats, crit)
        rb = analog_robustness(moves, pnl)
        if max_loss is None:
            notes.append("unbegrenztes Verlustrisiko -> 'Rendite/Risiko', 'Voll-Verlust' und '5%-Quantil (R/R)' "
                         "nicht definiert; Tail-Kriterium gilt als NICHT bestanden")

        # ── Ausstiegspreise heute (Black-76, sticky strike), auf 0 am aktuellen Spot normiert ──
        f_raw = _make_pnl_today(tc, ivs, dte, market["r_free"], market["dividend_yield"])
        f0 = float(f_raw(np.array([spot]))[0])
        f_today = lambda S, f_raw=f_raw, f0=f0: f_raw(S) - f0
        stop_target = -p["stop_pct"] / 100.0 * abs(net) if abs(net) > 0 else None
        zi_stop = _zone_info(f_today, stop_target, spot) if stop_target is not None else None

        grid = np.linspace(spot * 0.8, spot * 1.2, 401)
        chart_df = pd.concat([
            pd.DataFrame({"Spot": grid, "P&L (Pkt)": f_today(grid), "Linie": "heute (Modell)"}),
            pd.DataFrame({"Spot": grid, "P&L (Pkt)": dl.pnl_at_expiry(tc, grid), "Linie": "bei Verfall"}),
        ])

        results.append({
            "id": c["id"], "name": _combo_label(c), "expiry": c["expiry"], "dte": dte, "T_used": T_used,
            "net": net, "source": _source_text(lm),
            "stats": stats, "flags": flags, "rb": rb,
            "q5_pnl": dl.weighted_quantile(pnl, w, 0.05),
            "max_loss": prof["max_loss"], "max_profit": prof["max_profit"], "be_expiry": prof["breakevens"],
            "zi_stop": zi_stop, "stop_target": stop_target, "chart": chart_df, "notes": notes,
        })
    return {"results": results, "skipped": skipped, "spot": spot,
            "n_days": {t: len(m) for t, m in moves_cache.items()},
            "sig": dl_signature(combos), "stop_pct": p["stop_pct"], "settings": _settings_sig(p)}


def _settings_sig(p: dict) -> tuple:
    return (p["max_distance"], p["bandwidth"], p["min_gap"], p["basis_key"], p["t_min"], p["max_full"],
            p["tail_floor"], p["fallback_vol"], p["stop_pct"])


def dl_signature(combos: list[dict]) -> str:
    import json
    return json.dumps([{k: v for k, v in c.items() if k != "note"} for c in combos], sort_keys=True)


def _yn(flag: bool) -> str:
    return "✅" if flag else "❌"


def _render_results(res: dict, crit: dict, analog: dict) -> None:
    spot = res["spot"]
    rows = res["results"]
    for s in res["skipped"]:
        st.caption(f"⚠️ Uebersprungen - {s}")

    tot_net = sum(r["net"] for r in rows)
    tot_e = sum(r["stats"]["mean"] for r in rows)
    n_ok = sum(1 for r in rows if r["flags"]["c_base"])
    m1, m2, m3 = st.columns(3)
    m1.metric("Summe Netto-Praemie heute ($)", f"{tot_net * dl.MULTIPLIER:+,.0f}")
    m2.metric("Summe E[P&L] ($)", f"{tot_e * dl.MULTIPLIER:+,.0f}")
    m3.metric("Kombos, die alle Kriterien bestehen", f"{n_ok} von {len(rows)}")
    st.caption(
        "Alle Kombos sind zu HEUTIGEN Preisen einer Eroeffnung bewertet und mit DERSELBEN Rechnung wie in der "
        "Iron-Condor-Suche und der Analog-Tabelle (gemeinsame Funktionen): E[P&L] = gewichteter Mittelwert "
        "ueber die Analog-Tage, t-Stat HAC-korrigiert, Train/Val = aeltere/juengere Haelfte der Analog-Tage. "
        f"Analog-Set (Max. Distanz {analog['max_distance']:g}, σ {analog['bandwidth']:g}, Mindestabstand "
        f"{analog['min_gap']}) und Kriterien (t > {crit['t_min']:g}, Voll-Verlust ≤ {crit['max_full']:g} %, "
        f"5%-Quantil ≥ {crit['tail_floor']:g}, Preisbasis {crit['basis']}) stammen aus dem Tab 'Heutige "
        "Kette' - dort aendern. Kosten und Spread sind bei Mid nicht abgezogen."
    )

    def num(v):
        return np.nan if v is None else v

    rows_df, state_in = [], []
    for r in rows:
        s, f, rb = r["stats"], r["flags"], r["rb"]
        unbounded = not np.isfinite(r["max_loss"])
        rows_df.append({
            "Kombo": r["name"], "Rest (Tage)": r["dte"], "T (Analog)": r["T_used"], "Preisquelle": r["source"],
            "Netto-Praemie (Pkt)": r["net"],
            "Max-Loss (Pkt)": np.nan if unbounded else r["max_loss"],
            "Max-Gewinn (Pkt)": r["max_profit"] if np.isfinite(r["max_profit"]) else np.nan,
            "E[P&L] (Pkt)": s["mean"], "Rendite/Risiko %": s["ror"] * 100.0 if np.isfinite(s["ror"]) else np.nan,
            "t (HAC)": s["tstat"], "E Train": s["e_train"], "E Val": s["e_val"],
            "Voll-Verlust %": s["full_loss_share"] * 100.0 if np.isfinite(s["full_loss_share"]) else np.nan,
            "Voll-Verlust-Tage (n)": num(s["n_full_loss"]),
            "5%-Quantil (R/R)": s["q_tail"], "5%-Quantil P&L (Pkt)": r["q5_pnl"],
            "Gewinnrate %": s["win_rate_pct"], "Effektive n (HAC)": rb["n_eff_hac"],
            "CI ≠ 0": _yn(rb["ci_excludes_zero_full"]),
            "Perioden-Match": rb["subperiod_matching"], "Perioden gesamt": rb["subperiod_nonempty"],
            "t-Stat (Val)": rb["tstat_val"], "Robust (Train+Val)": _yn(rb["robust_train_val"]),
            "Train>0": _yn(f["c_train"]), "Val>0": _yn(f["c_val"]), "t": _yn(f["c_t"]),
            "Tail": "n/a" if unbounded else _yn(f["c_tail"]),
            "Alle Kriterien": _yn(f["c_base"]),
        })
        state_in.append({"tstat_full": rb["tstat_full"], "tstat_val": rb["tstat_val"],
                         "subperiod_matching": rb["subperiod_matching"],
                         "ci_excludes_zero_full": rb["ci_excludes_zero_full"],
                         "robust_train_val": rb["robust_train_val"],
                         "robust_train_val_neg": rb["robust_train_val_neg"]})
    tbl = pd.DataFrame(rows_df)
    state = dashboard_livechain._robustness_row_state(pd.DataFrame(state_in))
    fmt = {"Rest (Tage)": "{:.1f}", "T (Analog)": "{:.0f}", "Netto-Praemie (Pkt)": "{:+.2f}",
           "Max-Loss (Pkt)": "{:,.1f}", "Max-Gewinn (Pkt)": "{:,.1f}", "E[P&L] (Pkt)": "{:+.2f}",
           "Rendite/Risiko %": "{:+.1f}", "t (HAC)": "{:.2f}", "E Train": "{:+.2f}", "E Val": "{:+.2f}",
           "Voll-Verlust %": "{:.2f}", "Voll-Verlust-Tage (n)": "{:.0f}", "5%-Quantil (R/R)": "{:.2f}",
           "5%-Quantil P&L (Pkt)": "{:+.2f}", "Gewinnrate %": "{:.1f}", "Effektive n (HAC)": "{:.1f}",
           "Perioden-Match": "{:.0f}", "Perioden gesamt": "{:.0f}", "t-Stat (Val)": "{:.2f}"}
    st.caption(
        "**Zeilenfarben (wie in der Analog-Tabelle):** 🟩 grün = alle Robustheitskriterien positiv "
        f"(t-Stat gesamt > {dashboard_livechain._ROBUST_TSTAT_THRESHOLD:g}, CI ≠ 0, Perioden-Match ≥ "
        f"{dashboard_livechain._ROBUST_MIN_PERIODS_MATCH}/6, t-Stat Val > {dashboard_livechain._ROBUST_TSTAT_THRESHOLD:g}, "
        "Robust Train+Val), 🟥 rot = spiegelbildlich negativ (Kombo hat robust NEGATIVEN Erwartungswert). "
        "'Alle Kriterien' = Train>0, Val>0, t und Tail wie in der Iron-Condor-Suche."
    )
    st.dataframe(
        tbl.style.apply(lambda r: [dashboard_livechain._ROW_STYLE.get(state.iloc[r.name], "")] * len(r), axis=1)
        .format(fmt, na_rep="n/a"), use_container_width=True, hide_index=True)

    st.subheader("Ausstiegspreise HEUTE")
    st.caption(
        f"Aktueller Spot: {spot:,.2f}. Bezugspunkt ist der heutige Marktpreis: bei aktuellem Spot ist "
        "das P&L per Definition 0, ein 'Break-even heute' waere also der Spot selbst. Deshalb: "
        f"'Stop-Level heute' = Kurs, bei dem die Kombo HEUTE (mit Zeitwert; Black-76 je Bein mit der "
        f"impliziten Vola aus der Kette, ohne Vola-Aenderung) {res['stop_pct']:g} % ihrer Netto-Praemie "
        "verloren haette. 'Break-even bei Verfall' ist exakt (Netto-Praemie heute + Payoff). "
        "Suchbereich +-50 % um den Spot."
    )
    exit_rows = []
    for r in rows:
        exit_rows.append({
            "Kombo": r["name"],
            "Stop-Level heute (Verlust = X % der Netto-Praemie)":
                (_fmt_zone_text(r["zi_stop"], spot, "Stop") if r["zi_stop"] is not None else "n/a (Netto-Praemie 0)"),
            "Break-even bei Verfall (exakt)": " / ".join(
                f"{b:,.0f} ({(b / spot - 1) * 100:+.1f} %)" for b in r["be_expiry"]) or "-",
            "Hinweise": "; ".join(r["notes"]),
        })
    st.dataframe(pd.DataFrame(exit_rows), use_container_width=True, hide_index=True)

    st.subheader("P&L-Kurve (relativ zu den heutigen Marktpreisen)")
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
        "Beide Linien zeigen das P&L gegenueber einer Eroeffnung HEUTE: 'heute (Modell)' = wenn der Kurs jetzt "
        "springt und man sofort schliesst (mit Zeitwert; am Spot auf 0 normiert), 'bei Verfall' = Netto-Praemie "
        "heute + Payoff. Rote gestrichelte Linie = Stop-Grenze. Die Analog-Bewertung ist eine historische "
        "Ruecksimulation, keine Prognose."
    )


# ───────────────────────── Einstiegspunkt ─────────────────────────

def render(df, daily_df, has_resolution) -> None:
    st.header("Depot-Analyse")
    _init_state()
    if st.session_state.get("depot_load_warning"):
        st.warning(st.session_state["depot_load_warning"])
    flash = st.session_state.pop("depot_flash", None)
    if flash:
        st.success(flash)

    with st.sidebar:
        st.divider()
        st.header("Depot-Analyse")
        r_free = st.number_input("Risikoloser Zins", 0.0, 0.20, 0.0, 0.005, format="%.3f", key="depot_r")
        dividend_yield = st.number_input("Dividendenrendite", 0.0, 0.10, 0.015, 0.001, format="%.3f", key="depot_q")
        fallback_vol = st.number_input("Ersatz-Vola (%), falls Strike nicht in der Kette", 1.0, 200.0, 15.0, 1.0,
                                       key="depot_fbvol") / 100.0
        stop_pct = st.number_input("Stop-Level: Verlust in % der heutigen Netto-Praemie", 1.0, 1000.0, 100.0, 10.0,
                                   key="depot_stop")
        analog, crit = _shared_settings()
        st.caption(
            f"Analog-Suche und Kriterien wie im Tab 'Heutige Kette' (dort aendern): Max. Distanz "
            f"{analog['max_distance']:g}, σ {analog['bandwidth']:g}, Mindestabstand {analog['min_gap']}; "
            f"t > {crit['t_min']:g}, Voll-Verlust ≤ {crit['max_full']:g} %, 5%-Quantil ≥ {crit['tail_floor']:g}, "
            f"Preisbasis {crit['basis']}."
        )
        reload_market = st.button("Marktdaten neu laden", key="depot_reload")

    daily_df = daily_df.sort_values("date")
    combos = st.session_state["depot_combos"]

    # Marktdaten: gemeinsamer Snapshot (laedt nur, falls noetig - nicht bei jedem Tab-Wechsel)
    market = _get_market(combos, r_free, dividend_yield, force=reload_market) if combos else market_data.current(0.0)

    _render_add_form(daily_df)
    st.divider()
    analog, crit = _shared_settings()
    basis_key = "mid" if crit["basis"].startswith("Mid") else "bidask"
    _render_depot_list(market, float(fallback_vol), basis_key)

    combos = st.session_state["depot_combos"]
    if not combos:
        return
    st.divider()
    st.subheader("Analyse")

    if market is None:
        st.info("Marktdaten fehlen - siehe Fehlermeldung oben, dann 'Marktdaten neu laden' in der Sidebar.")
        return
    st.caption(f"Spot {market['spot']:,.2f} · Kette abgerufen "
               f"{market['fetch_ts'].tz_convert(_NY).strftime('%Y-%m-%d %H:%M:%S')} ET (~15 Min. verzoegert).")

    params = dict(max_distance=float(analog["max_distance"]), bandwidth=float(analog["bandwidth"]),
                  min_gap=int(analog["min_gap"]), fallback_vol=float(fallback_vol), stop_pct=float(stop_pct),
                  t_min=float(crit["t_min"]), max_full=float(crit["max_full"]),
                  tail_floor=float(crit["tail_floor"]), basis_key=basis_key)
    if st.button("Depot analysieren", type="primary", key="depot_run"):
        res = _analyze(daily_df, combos, market, params)
        if res is not None:
            st.session_state["depot_results"] = res

    res = st.session_state.get("depot_results")
    if res is None:
        st.info("'Depot analysieren' klicken.")
    elif res["sig"] != dl_signature(combos):
        st.info("Das Depot wurde seit der letzten Analyse geaendert - bitte erneut analysieren.")
    elif res.get("settings") != _settings_sig(params):
        st.info("Einstellungen (Analog-Set, Kriterien, Preisbasis, Stop) haben sich geaendert - bitte erneut analysieren.")
    else:
        _render_results(res, crit, analog)