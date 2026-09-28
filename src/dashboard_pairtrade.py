"""
MODUS 5: Pair-Trade-Suche (Backend: pair_trade_search.py).

Lead-Instrument bestimmt den Aehnlichkeits-Raum (Analog-Tage, wie in
dashboard_livechain.py); Follower wird an DENSELBEN Kalendertagen
mitbetrachtet (siehe pair_trade_search.pair_analog_moves). Bewusst SCHLANK:
Expiry je Instrument = naechstliegende Restlaufzeit, DTE-Naeherung ueber
Kalendertage (keine PM-Settlement-Feinheiten wie in dashboard_livechain.py),
kein Caching zwischen Reruns - jeder Klick rechnet neu. Fuer die
Performance-Grenze des Grids siehe pair_trade_search.PairTradeConfig.

Distanz-Raster bewusst GROB (siehe pair_trade_search.py, Abschnitt
PERFORMANCE-WARNUNG) - mehr Zeilen je Tabelle lassen die Paar-Anzahl
quadratisch wachsen.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from instruments import INSTRUMENTS, InstrumentConfig, get_instrument
import market_data
from chain_delta import build_symmetric_distance_table
from dashboard_common import load_data
from dashboard_livechain import _find_today_analogs, _ensure_training_table, _bounds_cached
from pair_trade_search import PairTradeConfig, pair_analog_moves, evaluate_pair_grid

_DEFAULT_DISTANCES = "0,1,2,3,5,8,12"


def _distance_table_for(instr: InstrumentConfig, daily_df: pd.DataFrame, distances: list[float],
                        r_free: float, dividend_yield: float):
    """Kette laden + symmetrische Distanz-Tabelle fuer EIN Instrument bauen
    (identisches Prinzip zu dashboard_livechain.render() Teil A, hier kompakt
    fuer Lead+Follower wiederverwendet). Expiry = naechstliegende verfuegbare.
    Rueckgabe: (table, spot, expiry, dte_calendar_days) oder (None, None, None, None)."""
    snap = market_data.get(instr, r_free, dividend_yield, 90.0)
    enriched = snap["enriched"]
    if enriched.empty:
        return None, None, None, None
    expiry = pd.Timestamp(enriched["expiry"].min())
    dte_days = max((expiry - pd.Timestamp.now()).days, 1)
    T_years = dte_days / 365.0
    bounds = _bounds_cached(daily_df, max(2, dte_days), 252, instr)
    hv_valid = bounds["hv"].dropna()
    if hv_valid.empty:
        return None, None, None, None
    flat_hv = float(hv_valid.iloc[-1])
    table = build_symmetric_distance_table(
        enriched, expiry, snap["spot"], flat_hv, T_years, r_free, dividend_yield, distances,
    )
    return table, snap["spot"], expiry, dte_days


def render(df, daily_df, has_resolution, instr: InstrumentConfig) -> None:
    st.header("Pair-Trade-Suche (zwei Instrumente)")
    st.caption(
        "Lead-Instrument bestimmt den Aehnlichkeits-Raum (Analog-Tage); das Follower-"
        "Instrument wird an DENSELBEN historischen Kalendertagen mitbewertet - siehe "
        "pair_trade_search.py. Schlanke Version: Expiry je Seite = naechstliegende "
        "verfuegbare, DTE als Kalendertage-Naeherung."
    )

    keys = list(INSTRUMENTS.keys())
    with st.sidebar:
        st.divider()
        st.header("Pair-Trade")
        lead_key = st.selectbox(
            "Lead-Instrument", keys, index=keys.index(instr.key),
            format_func=lambda k: INSTRUMENTS[k].label, key="pt_lead",
        )
        follower_key = st.selectbox(
            "Follower-Instrument", [k for k in keys if k != lead_key],
            format_func=lambda k: INSTRUMENTS[k].label, key="pt_follower",
        )
        max_distance = st.number_input("Max. Distanz (Lead-Aehnlichkeit)", 0.01, 100.0, 4.0, 0.5, key="pt_maxdist")
        bandwidth = st.number_input("Kernel-Bandbreite σ", 0.01, 50.0, 1.0, 0.1, key="pt_bw")
        min_gap = st.number_input("Mindestabstand (Handelstage)", 0, 60, 5, key="pt_gap")
        dist_input = st.text_input("Distanz-Raster (%, grob halten)", _DEFAULT_DISTANCES, key="pt_dist")
        t_min = st.number_input("Min. t-Stat (HAC)", 0.0, 10.0, 3.0, 0.1, key="pt_tmin")
        run_btn = st.button("Pair-Trade-Suche ausfuehren", type="primary")

    lead_instr, follower_instr = get_instrument(lead_key), get_instrument(follower_key)
    missing = [i.label for i in (lead_instr, follower_instr) if i.chain_provider is None]
    if missing:
        st.warning(f"Beide Instrumente brauchen eine Live-Kette - fehlt bei: {', '.join(missing)}.")
        return

    _ensure_training_table(lead_instr)
    if not run_btn:
        st.info("Parameter links einstellen und 'Pair-Trade-Suche ausfuehren' klicken.")
        return

    try:
        distances = [float(x) for x in dist_input.split(",") if x.strip()]
    except ValueError:
        st.error("Distanz-Raster ungueltig (Format z.B. '0,1,2,5,10').")
        return

    lead_daily, follower_daily = load_data(lead_instr), load_data(follower_instr)
    if lead_daily.empty or follower_daily.empty:
        st.error("Kursdaten fuer Lead oder Follower fehlen - erst `python src/data_fetch.py` ausfuehren.")
        return
    if "resolution" in lead_daily.columns:
        lead_daily = lead_daily[lead_daily["resolution"] == "daily"]
    if "resolution" in follower_daily.columns:
        follower_daily = follower_daily[follower_daily["resolution"] == "daily"]

    with st.spinner("Baue Lead-Kette und suche Analog-Tage..."):
        lead_table, lead_spot, lead_exp, lead_dte = _distance_table_for(
            lead_instr, lead_daily, distances, 0.0, lead_instr.dividend_yield,
        )
        if lead_table is None:
            st.error(f"Kette/HV fuer {lead_instr.label} nicht verfuegbar.")
            return
        ctx = _find_today_analogs(lead_daily, float(lead_dte), max_distance, bandwidth, min_gap, lead_instr)
    if ctx is None:
        return

    with st.spinner(f"Baue Follower-Kette ({follower_instr.label})..."):
        follower_table, follower_spot, follower_exp, _ = _distance_table_for(
            follower_instr, follower_daily, distances, 0.0, follower_instr.dividend_yield,
        )
    if follower_table is None:
        st.error(f"Kette/HV fuer {follower_instr.label} nicht verfuegbar.")
        return

    moves = pair_analog_moves(ctx["analogs"], ctx["training_table"], follower_daily,
                              ctx["T_days_used"], ctx["query_date"])
    if moves.empty:
        st.error("Keine gemeinsamen Lead+Follower-Analog-Tage mit gueltiger Bewegung gefunden.")
        return
    st.caption(
        f"{len(moves)} gemeinsame Analog-Tage nutzbar (T_days={ctx['T_days_used']}, "
        f"{moves.attrs.get('n_no_follower_match', 0)} ohne Follower-Kurs verworfen) · "
        f"Lead-Expiry {lead_exp.date()} · Follower-Expiry {follower_exp.date()}."
    )

    cfg = PairTradeConfig(t_min=float(t_min))
    try:
        result = evaluate_pair_grid(lead_table, follower_table, moves, lead_spot, follower_spot, cfg)
    except ValueError as e:
        st.error(str(e))
        return
    if result.empty:
        st.warning("Keine gueltige Lead/Follower-Kombination gefunden.")
        return

    st.success(f"{int(result['passes'].sum())} von {len(result)} Kombinationen bestehen alle Kriterien.")
    show = result[[
        "structure", "lead_short_distance", "lead_wing_distance",
        "follower_short_distance", "follower_wing_distance",
        "net_cashflow", "max_loss", "mean", "ror", "tstat", "e_train", "e_val", "passes",
    ]].rename(columns={
        "structure": "Struktur", "lead_short_distance": "Lead Short %", "lead_wing_distance": "Lead Fluegel %",
        "follower_short_distance": "Follower Short %", "follower_wing_distance": "Follower Fluegel %",
        "net_cashflow": "Netto-Cashflow", "max_loss": "Max-Loss", "mean": "E[P&L]",
        "ror": "Rendite/Risiko", "tstat": "t (HAC)", "e_train": "E Train", "e_val": "E Val", "passes": "Kandidat",
    })
    show["Rendite/Risiko"] = show["Rendite/Risiko"] * 100.0
    st.dataframe(
        show.head(300).style.format({
            "Netto-Cashflow": "{:+.2f}", "Max-Loss": "{:.2f}", "E[P&L]": "{:+.2f}",
            "Rendite/Risiko": "{:+.1f}", "t (HAC)": "{:.2f}", "E Train": "{:+.2f}", "E Val": "{:+.2f}",
        }, na_rep="n/a"),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "'passes'/'Kandidat' = Basis-Kriterien (Train>0, Val>0, t>Schwelle, Tail) UND Lead-Plateau "
        "erfuellt (siehe pair_trade_search.PairTradeConfig). Kombinierter Max-Loss ist eine "
        "KONSERVATIVE additive Schranke (Lead-Max-Loss + Follower-Max-Loss, keine Korrelations-"
        "Optimierung ueber den gemeinsamen Kursraum) - siehe Docstring von evaluate_pair_grid()."
    )