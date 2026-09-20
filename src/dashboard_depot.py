"""
MODUS 4: Depot-Analyse.

Eigene Optionspositionen als Kombos aus 1 bis 4 Beinen erfassen (gemeinsamer Verfall),
dauerhaft speichern (data/depot.json, wird beim Start automatisch wieder geladen) und
mit derselben Logik bewerten wie die Iron-Condor-Suche ("Rechnung 2"):

  BEWERTUNG (Analog-Tage-Simulation):
    Historische PROZENTUALE Kursbewegungen der Analog-Tage (gewichtet, siehe
    analog_matching.find_analogs) werden auf den heutigen Spot projiziert:
    S_d = Spot_heute * (1 + Bewegung_d). Die Kombo wird bis Verfall gehalten:
      P&L_d (seit Einstieg)  = entry_net + V(S_d)
      Restgewinn_d (ab jetzt) = V(S_d) - Marktwert_heute      (Marktwert = Mid je Bein)
    V(S) = Summe: long +intrinsic*qty, short -intrinsic*qty.
    Kennzahlen: E[P&L bis Verfall], Wahrscheinlichkeit im Plus, gewichtetes 5%-Quantil,
    E[Restgewinn] mit HAC-t-Stat (Vergleich 'halten' gegen 'heute zum Mid schliessen').
    Kein Vola-Modell in dieser Bewertung.

  AUSSTIEGSPREISE HEUTE ("ab welchem Kurs wird die Kombo heute unrentabel?"):
    Hier BRAUCHT es ein Modell, denn heute ist noch Zeitwert im Spiel: jedes Bein wird
    mit Black-76 bei der Restlaufzeit neu bewertet, mit der impliziten Vola des jeweiligen
    Beins aus der heutigen Kette (Vola je Strike bleibt fest - 'sticky strike'), ohne
    Vola-Aenderung bei Kursbewegung. Gesucht: Spotwerte S mit
      P&L_heute(S) = entry_net + Marktwert_modell(S) = 0        (Break-even heute)
      P&L_heute(S) = -X% * |entry_net|                            (Stop-Level, X einstellbar)
    Zusaetzlich exakt und modellfrei: Break-even und Max-Gewinn/-Verlust BEI VERFALL.
    Ohne Chain-Vola (Strike fehlt in der Kette) wird die 'Ersatz-Vola' aus der Sidebar
    genommen und die Zeile entsprechend markiert.
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
from black76 import forward_price
from chain_delta import ChainDeltaConfig, enrich_chain_with_delta, black76_price
from fetch_today_chain import fetch_today_chain, fetch_today_spot
from analog_matching import weighted_reliability_summary, available_t_days
from condor_search import analog_moves

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


def _lookup_leg(enriched: pd.DataFrame | None, expiry_str: str, leg: dict):
    """(mid, implizite Vola) des Beins aus der heutigen Kette - exakter Treffer auf
    Verfall/Typ/Strike, sonst (None, None)."""
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


def _commit(combos: list[dict]) -> None:
    """Speichert das Depot in Session UND auf Platte."""
    st.session_state["depot_combos"] = combos
    try:
        dl.save_depot(DEPOT_FILE, combos)
    except OSError as e:
        st.error(f"Depot konnte NICHT gespeichert werden ({e}) - Aenderung gilt nur bis zum "
                 "Schliessen des Programms!")


def _init_state() -> None:
    if "depot_combos" not in st.session_state:
        combos, warn = dl.load_depot(DEPOT_FILE)
        st.session_state["depot_combos"] = combos
        st.session_state["depot_load_warning"] = warn


def _make_pnl_today(combo: dict, ivs: list[float], dte: float, r: float, q: float):
    """P&L-Funktion HEUTE (seit Einstieg) in Abhaengigkeit vom Spot: Black-76 je Bein
    mit fester Vola je Bein. Akzeptiert/liefert numpy-Arrays."""
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

def _load_market(combos: list[dict], r_free: float, dividend_yield: float) -> None:
    max_dte = max([_dte_exact(c["expiry"]) for c in combos] + [1.0]) + 2.0
    with st.spinner("Lade aktuellen Spot und Optionskette..."):
        try:
            fetch_ts = pd.Timestamp.now(tz="UTC")
            spot = fetch_today_spot()
            chain = fetch_today_chain(min_dte=0.0, max_dte=float(max_dte))
            enriched = enrich_chain_with_delta(
                chain, spot,
                ChainDeltaConfig(min_abs_delta=0.0001, r_free=float(r_free),
                                 dividend_yield=float(dividend_yield)),
                as_of=fetch_ts,
            )
        except Exception as e:
            st.error(f"Marktdaten konnten nicht geladen werden: {e}")
            return
    st.session_state["depot_market"] = {
        "spot": float(spot), "enriched": enriched, "fetch_ts": fetch_ts,
        "r_free": float(r_free), "dividend_yield": float(dividend_yield),
    }


# ───────────────────────── Eingabe / Verwaltung ─────────────────────────

def _render_add_form(daily_df: pd.DataFrame) -> None:
    st.subheader("Kombo hinzufuegen")
    last_close = float(daily_df["close"].iloc[-1]) if "close" in daily_df.columns and len(daily_df) else 7000.0
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
        st.caption("Pro Bein: Typ · Richtung · Strike · Menge · Einstiegspreis (Punkte je Option, NICHT mal 100)")
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
        combo = dl.make_combo(name, expiry.isoformat(), legs, note)
        errs = dl.validate_combo(combo)
        if errs:
            st.error("Nicht gespeichert: " + " ".join(errs))
        else:
            _commit(st.session_state["depot_combos"] + [combo])
            st.success(f"'{combo['name']}' gespeichert.")


def _render_depot_list() -> None:
    combos = st.session_state["depot_combos"]
    st.subheader(f"Depot ({len(combos)} Kombos)")
    st.caption(f"Gespeichert in `{DEPOT_FILE}` - wird beim naechsten Start automatisch wieder geladen.")
    if not combos:
        st.info("Noch keine Positionen. Oben eine Kombo hinzufuegen.")
        return

    rows = []
    for c in combos:
        dte = _dte_exact(c["expiry"]) if not dl.validate_combo(c) else np.nan
        net = dl.entry_net(c)
        rows.append({
            "Kombo": c["name"], "Verfall": c["expiry"],
            "Rest (Tage)": dte, "Beine": " · ".join(_leg_text(l) for l in c["legs"]),
            "Netto-Einstieg (Pkt)": net, "Netto-Einstieg ($)": net * dl.MULTIPLIER,
            "Status": "abgelaufen" if (pd.notna(dte) and dte <= 0) else ("Fehler" if pd.isna(dte) else "offen"),
        })
    st.dataframe(pd.DataFrame(rows).style.format(
        {"Rest (Tage)": "{:.1f}", "Netto-Einstieg (Pkt)": "{:+.2f}", "Netto-Einstieg ($)": "{:+,.0f}"}, na_rep="n/a"),
        use_container_width=True, hide_index=True)
    st.caption("Netto-Einstieg: positiv = Kredit erhalten, negativ = Debit bezahlt.")

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
                                    c.get("note", ""), cid, c.get("created"))
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
    spot, enriched = market["spot"], market["enriched"]
    todo, skipped = [], []
    for c in combos:
        errs = dl.validate_combo(c)
        dte = _dte_exact(c["expiry"]) if not errs else None
        if errs or dte is None or dte <= 0:
            skipped.append(f"{_combo_label(c)}: {'abgelaufen' if not errs else ' '.join(errs)}")
        else:
            todo.append((c, dte))
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

        w = moves["weight"].to_numpy(dtype=float)
        S_d = spot * (1.0 + moves["ret"].to_numpy(dtype=float))
        net = dl.entry_net(c)
        V_d = dl.expiry_value(c, S_d)
        pnl_entry = net + V_d

        # Marktdaten je Bein (Mid fuer Marktwert, Vola fuer Ausstiegspreise)
        mids, ivs, fallback_iv = [], [], False
        for leg in c["legs"]:
            mid, iv = _lookup_leg(enriched, c["expiry"], leg)
            mids.append(mid)
            if iv is None or not iv > 0:
                iv, fallback_iv = p["fallback_vol"], True
            ivs.append(max(iv, 0.01))
        mark = None
        if all(m is not None for m in mids):
            mark = float(sum(dl._sign(l) * l["qty"] * m for l, m in zip(c["legs"], mids)))
        if mark is None:
            notes.append("kein Marktpreis in der Kette -> Marktwert/Restgewinn n/a")
        if fallback_iv:
            notes.append("Ersatz-Vola fuer mind. ein Bein")

        rest = (V_d - mark) if mark is not None else None
        basis = rest if rest is not None else pnl_entry
        ev = moves[["entry_date", "hold_days", "weight"]].copy()
        ev["pnl_depot_combo"] = basis
        rel = weighted_reliability_summary(ev, "combo", 6, "pnl_depot")

        prof = dl.payoff_profile(c)
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
            "E_entry": float(np.sum(w * pnl_entry) / w.sum()),
            "p_plus": float(w[pnl_entry >= 0].sum() / w.sum() * 100.0),
            "q5": dl.weighted_quantile(pnl_entry, w, 0.05),
            "E_rest": float(np.sum(w * rest) / w.sum()) if rest is not None else None,
            "t_rest": rel["tstat"], "n_eff_hac": rel["n_eff_hac"],
            "t_basis": "Restgewinn" if rest is not None else "P&L seit Einstieg",
            "max_loss": prof["max_loss"], "max_profit": prof["max_profit"],
            "be_expiry": prof["breakevens"],
            "zi_be": zi_be, "zi_stop": zi_stop, "stop_target": stop_target,
            "pnl_now_model": float(f_today(np.array([spot]))[0]),
            "chart": chart_df, "notes": notes,
        })
    return {"results": results, "skipped": skipped, "spot": spot, "n_days": {t: len(m) for t, m in moves_cache.items()},
            "sig": dl_signature(combos), "stop_pct": p["stop_pct"]}


def dl_signature(combos: list[dict]) -> str:
    import json
    return json.dumps(combos, sort_keys=True)


def _render_results(res: dict) -> None:
    spot = res["spot"]
    rows = res["results"]
    for s in res["skipped"]:
        st.caption(f"⚠️ Uebersprungen - {s}")

    tot_unreal = sum(r["unreal"] for r in rows if r["unreal"] is not None)
    tot_e = sum(r["E_entry"] for r in rows)
    tot_rest = sum(r["E_rest"] for r in rows if r["E_rest"] is not None)
    m1, m2, m3 = st.columns(3)
    m1.metric("Summe unrealisiert heute ($)", f"{tot_unreal * dl.MULTIPLIER:+,.0f}")
    m2.metric("Summe E[P&L bis Verfall] ($)", f"{tot_e * dl.MULTIPLIER:+,.0f}")
    m3.metric("Summe E[Restgewinn] ($)", f"{tot_rest * dl.MULTIPLIER:+,.0f}")
    st.caption("Erwartungswerte sind additiv ueber Kombos (auch bei unterschiedlichem Verfall). "
               "Positives E[Restgewinn] = im Schnitt lohnt Halten mehr als heute zum Mid zu schliessen "
               "(vor Spread/Kosten).")

    tbl = pd.DataFrame([{
        "Kombo": r["name"], "Rest (Tage)": r["dte"], "T (Analog)": r["T_used"],
        "Netto-Einstieg (Pkt)": r["net"], "Marktwert heute (Pkt)": r["mark"],
        "Unrealisiert (Pkt)": r["unreal"],
        "E[P&L bis Verfall] (Pkt)": r["E_entry"], "Im Plus bei Verfall %": r["p_plus"],
        "5%-Quantil P&L (Pkt)": r["q5"],
        "E[Restgewinn] (Pkt)": r["E_rest"], "t (HAC)": r["t_rest"],
        "Max-Verlust bei Verfall (Pkt)": r["max_loss"],
        "Break-even Verfall": " / ".join(f"{b:,.0f}" for b in r["be_expiry"]) or "-",
    } for r in rows])
    fmt = {"Rest (Tage)": "{:.1f}", "T (Analog)": "{:.0f}", "Netto-Einstieg (Pkt)": "{:+.2f}",
           "Marktwert heute (Pkt)": "{:+.2f}", "Unrealisiert (Pkt)": "{:+.2f}",
           "E[P&L bis Verfall] (Pkt)": "{:+.2f}", "Im Plus bei Verfall %": "{:.1f}",
           "5%-Quantil P&L (Pkt)": "{:+.2f}", "E[Restgewinn] (Pkt)": "{:+.2f}", "t (HAC)": "{:.2f}",
           "Max-Verlust bei Verfall (Pkt)": "{:,.1f}"}
    st.dataframe(tbl.style.format(fmt, na_rep="n/a"), use_container_width=True, hide_index=True)

    st.subheader("Ausstiegspreise HEUTE")
    st.caption(
        f"Aktueller Spot: {spot:,.2f}. 'Break-even heute' = Kurs, ab dem die Kombo HEUTE (mit Zeitwert, Modell: "
        "Black-76 je Bein mit der impliziten Vola aus der Kette, ohne Vola-Aenderung) im Minus waere. "
        f"'Stop-Level' = Kurs, bei dem der Verlust {res['stop_pct']:g} % des Netto-Einstiegs erreicht. "
        "Suchbereich +-50 % um den Spot."
    )
    exit_rows = []
    for r in rows:
        exit_rows.append({
            "Kombo": r["name"],
            "Heute im Plus (Modell)": "ja" if r["pnl_now_model"] >= 0 else "nein",
            "Unrentabel heute ab": _fmt_zone_text(r["zi_be"], spot, "Break-even"),
            "Stop-Level heute": (_fmt_zone_text(r["zi_stop"], spot, "Stop") if r["zi_stop"] is not None else "n/a (Netto-Einstieg 0)"),
            "Break-even bei Verfall (exakt)": " / ".join(f"{b:,.0f} ({(b / spot - 1) * 100:+.1f} %)" for b in r["be_expiry"]) or "-",
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
    zero_rule = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color="gray").encode(y="y:Q")
    st.altair_chart((base + spot_rule + zero_rule).properties(height=350).interactive(), use_container_width=True)
    st.caption(f"Basis des t-Stats ('{r['t_basis']}'): HAC-korrigiert, n_eff(HAC)={r['n_eff_hac']:.0f}. "
               "Die Analog-Bewertung ist eine historische Ruecksimulation, keine Prognose.")


# ───────────────────────── Einstiegspunkt ─────────────────────────

def render(df, daily_df, has_resolution) -> None:
    st.header("Depot-Analyse")
    _init_state()
    if st.session_state.get("depot_load_warning"):
        st.warning(st.session_state["depot_load_warning"])

    with st.sidebar:
        st.divider()
        st.header("Depot-Analyse")
        r_free = st.number_input("Risikoloser Zins", 0.0, 0.20, 0.0, 0.005, format="%.3f", key="depot_r")
        dividend_yield = st.number_input("Dividendenrendite", 0.0, 0.10, 0.015, 0.001, format="%.3f", key="depot_q")
        fallback_vol = st.number_input("Ersatz-Vola (%), falls Strike nicht in der Kette", 1.0, 200.0, 15.0, 1.0,
                                       key="depot_fbvol") / 100.0
        stop_pct = st.number_input("Stop-Level: Verlust in % des Netto-Einstiegs", 1.0, 1000.0, 100.0, 10.0,
                                   key="depot_stop")
        st.caption("Analog-Suche (wie in der Kettenansicht):")
        max_distance = st.number_input("Max. Distanz", 0.01, 100.0, 4.0, 0.5, key="depot_maxdist")
        bandwidth = st.number_input("Kernel-Bandbreite σ", 0.01, 50.0, 1.0, 0.1, key="depot_bw")
        min_gap = st.number_input("Mindestabstand zu heute (Handelstage)", 0, 60, 5, key="depot_gap")
        reload_market = st.button("Marktdaten neu laden", key="depot_reload")

    daily_df = daily_df.sort_values("date")
    _render_add_form(daily_df)
    st.divider()
    _render_depot_list()

    combos = st.session_state["depot_combos"]
    if not combos:
        return
    st.divider()
    st.subheader("Analyse")

    if reload_market or "depot_market" not in st.session_state:
        _load_market(combos, r_free, dividend_yield)
    market = st.session_state.get("depot_market")
    if market is None:
        st.info("Marktdaten fehlen - siehe Fehlermeldung oben, dann 'Marktdaten neu laden' in der Sidebar.")
        return
    st.caption(f"Spot {market['spot']:,.2f} · Kette abgerufen "
               f"{market['fetch_ts'].tz_convert(_NY).strftime('%Y-%m-%d %H:%M:%S')} ET (~15 Min. verzoegert).")

    params = dict(max_distance=float(max_distance), bandwidth=float(bandwidth), min_gap=int(min_gap),
                  fallback_vol=float(fallback_vol), stop_pct=float(stop_pct))
    if st.button("Depot analysieren", type="primary", key="depot_run"):
        res = _analyze(daily_df, combos, market, params)
        if res is not None:
            st.session_state["depot_results"] = res

    res = st.session_state.get("depot_results")
    if res is None:
        st.info("'Depot analysieren' klicken.")
    elif res["sig"] != dl_signature(combos):
        st.info("Das Depot wurde seit der letzten Analyse geaendert - bitte erneut analysieren.")
    else:
        _render_results(res)