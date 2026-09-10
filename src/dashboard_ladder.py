"""
MODUS 1: Iron-Condor-/Multi-Leg-Ladder-Backtest.

Ein einziger Einstiegspunkt fuer dashboard.py: render(df, daily_df,
has_resolution). Rendert sowohl den eigenen Sidebar-Abschnitt als auch den
Hauptbereich - dashboard.py ruft diese Funktion NUR auf, wenn dieser Modus
oben in der Reiterleiste aktiv ist (siehe dashboard.py), es entsteht also
keine Ueberschneidung mit der Sidebar von dashboard_analog.py.

Aufbau (von oben nach unten in dieser Datei):
  1. _sidebar_legs()          - die 4 Bein-Eingabefelder
  2. _sidebar_params()        - restliche Parameter (Laufzeit, Exit, Vol-Surface, Fixer Rang)
  3. _max_consecutive() + _check_data_quality() - Datenqualitaets-Warnung
  4. _render_price_preview()  - Live-Preisvorschau der aktuell konfigurierten Struktur
  5. _run_backtest()          - fuehrt den Backtest aus, speichert Ergebnis in st.session_state
  6. _render_results()        - Kennzahlen, Equity-Kurve, Reliability, Trade-Tabelle
  7. render()                 - Oeffentlicher Einstiegspunkt, ruft 1-6 in der richtigen Reihenfolge
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_lightweight_charts import renderLightweightCharts

sys.path.insert(0, str(Path(__file__).resolve().parent))

from black76 import VolSurfaceConfig, strike_from_delta, leg_price
from backtest import (
    IronCondorConfig, LegConfig, FixedRankConfig, run_daily_ladder, summarize_trades,
    compute_hv, compute_rank_target_vol,
)
from stats import reliability_summary
from dashboard_common import metric_box

_DEFAULT_LEGS = [
    ("Bein 1", "put", "long", 0.05),
    ("Bein 2", "put", "short", 0.16),
    ("Bein 3", "call", "short", 0.16),
    ("Bein 4", "call", "long", 0.05),
]


def _sidebar_legs(default_spot: float) -> list[LegConfig]:
    """Die 4 Bein-Eingabefelder (Typ/Richtung/Delta ODER fixer Strike/Menge)."""
    st.subheader("4 Beine (je Put/Call, Long/Short, Delta ODER fixer Strike)")
    st.caption(
        "Default = klassischer Iron Condor. Menge=0 setzen, um ein Bein zu "
        "deaktivieren. Statt eines Ziel-Deltas (Strike wird taeglich neu "
        "berechnet) kann je Bein auch ein FESTER Strike-Preis eingegeben "
        "werden (bleibt an jedem Entry-Tag unveraendert)."
    )
    hc1, hc2, hc3, hc4, hc5 = st.columns([1, 1, 1.1, 1, 1])
    hc1.caption("Typ"); hc2.caption("Richtung"); hc3.caption("Basis")
    hc4.caption("Delta / Strike"); hc5.caption("Menge")

    legs = []
    for label, def_type, def_side, def_delta in _DEFAULT_LEGS:
        st.markdown(f"**{label}**")
        lc1, lc2, lc3, lc4, lc5 = st.columns([1, 1, 1.1, 1, 1])
        option_type = lc1.selectbox(
            "Typ", ["put", "call"], index=["put", "call"].index(def_type),
            key=f"type_{label}", label_visibility="collapsed",
        )
        side = lc2.selectbox(
            "Richtung", ["long", "short"], index=["long", "short"].index(def_side),
            key=f"side_{label}", label_visibility="collapsed",
        )
        basis = lc3.selectbox(
            "Basis", ["Delta", "Fixer Strike"], key=f"basis_{label}", label_visibility="collapsed",
        )
        if basis == "Fixer Strike":
            fixed_strike = lc4.number_input(
                "Strike", min_value=0.0, value=round(default_spot, 0), step=1.0,
                key=f"strike_{label}", label_visibility="collapsed",
            )
            delta, strike_mode = def_delta, "fixed"   # delta bleibt belegt, wird ignoriert
        else:
            delta = lc4.number_input(
                "Delta", 0.01, 0.99, def_delta, 0.01,
                key=f"delta_{label}", label_visibility="collapsed",
            )
            fixed_strike, strike_mode = None, "delta"
        qty = lc5.number_input(
            "Menge", 0.0, 50.0, 1.0, 1.0, key=f"qty_{label}", label_visibility="collapsed",
        )
        legs.append(LegConfig(
            label=label, option_type=option_type, side=side, delta=delta, qty=qty,
            strike_mode=strike_mode, fixed_strike=fixed_strike,
        ))
    return legs


def _sidebar_params(T_days_default: int = 7) -> dict:
    """Restliche Sidebar-Parameter: Laufzeit/HV, vorzeitiger Ausstieg,
    Vol-Oberflaeche, Backtest-Modus 'Fixer IV-Rang'. Gibt alles als dict
    zurueck (statt vieler Einzelrueckgaben), damit render() uebersichtlich bleibt."""
    T_days = st.slider("Laufzeit T (Tage)", 2, 60, T_days_default)
    hv_length = st.slider("HV-Periode (Tage)", 5, 90, 30)

    st.subheader("Vorzeitiger Ausstieg")
    close_on_touch = st.checkbox("Bei Short-Strike-Beruehrung schliessen", value=False)
    use_early_exit = st.checkbox("Nach X Tagen zwangsweise schliessen", value=False)
    early_exit_days = st.slider(
        "Ausstieg nach X Tagen", 1, 59, min(T_days - 1, 5), disabled=not use_early_exit,
    )

    st.subheader("Vol-Oberflaeche")
    use_vol_surface = st.checkbox("Skew + Term Structure verwenden", value=True)

    st.subheader("Backtest-Modus: Fixer IV-Rang (hypothetisch)")
    st.caption(
        "Beantwortet: 'Wie haette JEDER Trade performt, waere er IMMER zum "
        "selben (heutigen) IV-Rang eroeffnet/bewertet worden?' Rechnet an "
        "jedem Bar zurueck, welche Vola noetig gewesen waere, um genau den "
        "Ziel-Rang im damals gueltigen rollierenden HV-Min/Max-Fenster zu "
        "erreichen - ersetzt current_iv komplett."
    )
    use_fixed_rank = st.checkbox("Aktivieren: Immer zum festen Ziel-IV-Rang handeln", value=False)
    target_rank_pct = st.slider(
        "Ziel-IV-Rang (%)", -50.0, 200.0, 50.0, 5.0, disabled=not use_fixed_rank,
        help="50 = immer genau in der Mitte zwischen HV-Min und HV-Max des Lookback-"
             "Fensters. Werte ausserhalb 0-100 extrapolieren bewusst.",
    )
    rc1, rc2 = st.columns(2)
    rank_hv_length = rc1.number_input("HV-Periode fuer Rang (Tage)", 2, 365, 30, disabled=not use_fixed_rank)
    rank_lookback_days = rc2.number_input("Lookback fuer Rang (Tage)", 10, 5000, 252, disabled=not use_fixed_rank)

    return dict(
        T_days=T_days, hv_length=hv_length,
        close_on_touch=close_on_touch, use_early_exit=use_early_exit, early_exit_days=early_exit_days,
        use_vol_surface=use_vol_surface,
        use_fixed_rank=use_fixed_rank, target_rank_pct=target_rank_pct,
        rank_hv_length=int(rank_hv_length), rank_lookback_days=int(rank_lookback_days),
    )


def _max_consecutive(mask: pd.Series) -> int:
    """Laengste Serie von aufeinanderfolgenden True-Werten in mask."""
    if not mask.any():
        return 0
    groups = (~mask).cumsum()
    return int(mask.groupby(groups).sum().max())


def _check_data_quality(daily_df: pd.DataFrame) -> None:
    """Warnt vor Serien exakt identischer Schlusskurse - starkes Signal fuer
    fehlerhafte/eingefrorene Daten (Stale Cache, Merge-Bug in data_fetch.py),
    NICHT fuer einen Fehler in HV-/Backtest-Formeln. Erklaert u.a. auffaellig
    identische entry_iv/pnl_norm-Werte, wenn ein rollierendes HV-Fenster
    vollstaendig innerhalb einer solchen Dup-Serie liegt."""
    dup_mask = daily_df["close"].diff().eq(0.0)
    max_dup_run = _max_consecutive(dup_mask)
    if max_dup_run < 3:
        return
    last_dup_date = daily_df.loc[
        dup_mask[::-1].idxmax() if dup_mask.any() else daily_df.index[-1], "date"
    ]
    st.warning(
        f"⚠️ Datenqualitaet: {max_dup_run} aufeinanderfolgende Handelstage mit "
        "EXAKT identischem Schlusskurs gefunden "
        f"(zuletzt um den {pd.Timestamp(last_dup_date).date()}). Fuer echte "
        "SPX-Kurse ist das extrem unwahrscheinlich und deutet auf einen "
        "Stale-Cache- oder Datenabruf-Fehler hin. Empfehlung: "
        "`python src/data_fetch.py` mit Internetverbindung erneut ausfuehren."
    )


def _render_price_preview(daily_df: pd.DataFrame, legs: list[LegConfig], params: dict, r_free: float) -> None:
    """Zeigt, was die aktuell links konfigurierte 4-Bein-Struktur JETZT
    kosten wuerde (Strike, Theoriepreis je Bein, Netto-Credit der Kombo) -
    rein informativ, unabhaengig davon ob der Backtest schon gelaufen ist."""
    st.subheader("Options-Preisvorschau (aktuell): Preise je Strike + Kombo gesamt")
    st.caption(
        "Theoriepreise (Black-76) der links konfigurierten Beine auf Basis des "
        "letzten Schlusskurses, der aktuellen IV-Quelle und der eingestellten "
        "Laufzeit T. Rein informativ - unabhaengig vom Backtest-Ergebnis unten."
    )

    S_last = float(daily_df["close"].iloc[-1])
    T_years = params["T_days"] / 365.0

    hv_series = compute_hv(daily_df, params["hv_length"])
    current_hv = hv_series.iloc[-1] if len(hv_series) else np.nan

    if params["use_fixed_rank"]:
        rank_series = compute_rank_target_vol(
            daily_df, params["rank_hv_length"], params["rank_lookback_days"], params["target_rank_pct"],
        )
        iv_preview = rank_series.iloc[-1] if len(rank_series) else np.nan
        iv_source_label = f"Fixer IV-Rang ({params['target_rank_pct']:.0f}%)"
    else:
        iv_preview = current_hv
        iv_source_label = f"HV ({params['hv_length']}d)"

    if pd.isna(iv_preview) or iv_preview <= 0.0:
        st.warning(
            "Aktuelle IV-Quelle liefert keinen gueltigen Wert (z.B. zu wenig "
            "Historie fuer die gewaehlte HV-/Lookback-Periode) - keine Preisvorschau moeglich."
        )
        return

    vc = VolSurfaceConfig(use_vol_surface=params["use_vol_surface"])
    rows, short_sum, long_sum = [], 0.0, 0.0
    for leg in legs:
        if leg.uses_fixed_strike:
            K, basis_label = leg.fixed_strike, "Fix"
        else:
            K = strike_from_delta(S_last, leg.delta, iv_preview, T_years, is_put=leg.is_put)
            basis_label = f"Δ{leg.delta:.2f}"
        premium_per_contract = leg_price(K, leg.is_put, T_years, S_last, iv_preview, r_free, vc)
        premium_total = premium_per_contract * leg.qty
        if leg.is_long:
            long_sum += premium_total
        else:
            short_sum += premium_total
        rows.append({
            "Bein": leg.label, "Typ": "Put" if leg.is_put else "Call",
            "Richtung": "Long" if leg.is_long else "Short", "Basis": basis_label,
            "Menge": leg.qty, "Strike": round(K, 2),
            "Preis / Kontrakt": round(premium_per_contract, 4),
            "Preis gesamt": round(premium_total, 4),
            "Cashflow": ("-" if leg.is_long else "+") + f"{premium_total:.4f}",
        })

    net_credit = short_sum - long_sum
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("Spot (letzter Schlusskurs)", f"{S_last:,.2f}")
    pc2.metric(f"Aktive IV-Quelle: {iv_source_label}", f"{iv_preview * 100:.2f}%")
    pc3.metric("Short-Praemien (Summe)", f"+{short_sum:.4f}")
    pc4.metric(
        "Netto-Credit der Kombo (gesamt)", f"{net_credit:+.4f}",
        help="Summe der Short-Praemien minus Summe der Long-Praemien (entry_net, "
             "gleiche Definition wie im Backtest).",
    )
    st.caption(
        f"Long-Praemien (Summe): -{long_sum:.4f}  |  "
        f"Datenstand: {daily_df['date'].iloc[-1].date()}  |  "
        f"Laufzeit T={params['T_days']} Tage ({T_years*365:.0f} Kalendertage)."
    )


def _run_backtest(df: pd.DataFrame, legs: list[LegConfig], params: dict, n_daily_total: int) -> None:
    """Fuehrt run_daily_ladder() aus und legt Ergebnis + Beschriftung in
    st.session_state ab (ueberlebt den naechsten Rerun, z.B. beim
    Umschalten der Trade-Tabelle-Sortierung)."""
    cfg = IronCondorConfig(
        T_days=params["T_days"], hv_length=params["hv_length"], legs=legs,
        close_on_touch=params["close_on_touch"],
        use_early_exit=params["use_early_exit"], early_exit_days=params["early_exit_days"],
        vol_surface=VolSurfaceConfig(use_vol_surface=params["use_vol_surface"]),
        fixed_rank=FixedRankConfig(
            enabled=params["use_fixed_rank"], target_rank_pct=params["target_rank_pct"],
            hv_length=params["rank_hv_length"], lookback_days=params["rank_lookback_days"],
        ),
    )
    with st.spinner(f"Backtest laeuft ({n_daily_total} Handelstage)..."):
        trades = run_daily_ladder(df, cfg)

    if trades.empty:
        st.warning(
            "Keine Trades erzeugt - Parameter pruefen (z.B. HV-Periode/Lookback "
            "zu lang fuer die Datenmenge, oder Ziel-Rang fuehrt zu ungueltiger Vola)."
        )
        return

    st.session_state["ic_trades"] = trades
    leg_summary = " / ".join(
        f"{l.option_type[0].upper()}{'+' if l.is_long else '-'}"
        + (f"K{l.fixed_strike:.0f}" if l.uses_fixed_strike else f"{l.delta}")
        for l in legs
    )
    rank_label = f", Fixer Rang={params['target_rank_pct']:.0f}%" if params["use_fixed_rank"] else ""
    st.session_state["ic_cfg_label"] = (
        f"T={params['T_days']}d, HV={params['hv_length']}d, Beine: {leg_summary}{rank_label}"
    )


def _to_equity_points(trades: pd.DataFrame) -> tuple[list[dict], pd.Series]:
    """Baut die Datenpunkte fuer die Equity-Kurve (kumulierter P&L),
    TAGESAGGREGIERT (siehe dort im Docstring: lightweight-charts braucht pro
    Serie strikt aufsteigende, eindeutige Zeitpunkte - an einem Kalendertag
    koennen aber mehrere Positionen gleichzeitig schliessen)."""
    daily_pnl = trades.groupby(trades["exit_date"].dt.normalize())["net_pnl"].sum().sort_index()
    equity = daily_pnl.cumsum()
    points = [{"time": ts.strftime("%Y-%m-%d"), "value": float(val)} for ts, val in equity.items()]
    return points, equity


def _render_results() -> None:
    """Kennzahlen, Equity-Kurve, Reliability und Trade-Tabelle fuer den
    zuletzt gelaufenen Backtest (aus st.session_state)."""
    trades = st.session_state["ic_trades"]
    summary = summarize_trades(trades)
    rel = reliability_summary(trades, nw_lags=20, n_subperiods=6)

    st.caption(f"Letzter Lauf: {st.session_state.get('ic_cfg_label', '')}")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        metric_box("Trades", f"{summary['n_trades']:,}", "neutral")
    with m2:
        win_rate = summary["win_rate_pct"]
        metric_box("Gewinnrate", f"{win_rate:.1f}%", "good" if win_rate >= 50.0 else "bad")
    with m3:
        pf = summary["profit_factor"]
        metric_box("Profit Factor", "n/a" if pd.isna(pf) else f"{pf:.2f}",
                    "neutral" if pd.isna(pf) else ("good" if pf > 1.0 else "bad"))
    with m4:
        avg_pnl = summary["avg_pnl"]
        metric_box(
            "Erwartungswert / Trade", f"{avg_pnl:+.4f}", "good" if avg_pnl > 0.0 else "bad",
            help_text="Arithmetisches Mittel des Netto-P&L ueber alle abgeschlossenen Trades, in Punkten.",
        )
    with m5:
        avg_pnl_norm = summary["avg_pnl_norm_pct"]
        metric_box("Ø norm. P&L/Trade", f"{avg_pnl_norm:+.2f}%", "good" if avg_pnl_norm > 0.0 else "bad")
    with m6:
        metric_box("Ø genutzte IV (Entry)", f"{summary['avg_entry_iv_pct']:.1f}%", "neutral")

    if summary.get("n_unlimited_risk", 0) > 0:
        st.warning(
            f"{summary['n_unlimited_risk']} von {summary['n_trades']} Trades haben "
            "unbegrenztes Verlustrisiko (mehr Short- als Long-Calls) - fuer diese ist "
            "'normierter P&L' nicht definiert (NaN) und wird oben automatisch ausgeschlossen."
        )

    st.subheader("Equity-Kurve (kumulierter P&L)")
    equity_points, equity_vals = _to_equity_points(trades)
    equity_chart_options = {
        "height": 400,
        "layout": {"background": {"type": "solid", "color": "#0e1117"}, "textColor": "#d1d4dc"},
        "grid": {
            "vertLines": {"color": "rgba(42, 46, 57, 0.5)"},
            "horzLines": {"color": "rgba(42, 46, 57, 0.5)"},
        },
        "timeScale": {"borderColor": "rgba(197, 203, 206, 0.4)"},
        "rightPriceScale": {"borderColor": "rgba(197, 203, 206, 0.4)"},
        "handleScroll": True, "handleScale": True,
    }
    equity_series = [{
        "type": "Area", "data": equity_points,
        "options": {
            "lineColor": "#2962FF", "topColor": "rgba(41, 98, 255, 0.28)",
            "bottomColor": "rgba(41, 98, 255, 0.0)", "lineWidth": 2,
        },
    }]
    renderLightweightCharts([{"chart": equity_chart_options, "series": equity_series}], key="equity_chart")
    st.caption(
        f"{len(equity_points)} Tages-Datenpunkte "
        f"({equity_vals.index.min().date()} bis {equity_vals.index.max().date()}) - "
        "frei zoom-/pannbar ueber die gesamte Backtest-Historie."
    )

    st.subheader("Reliability (Newey-West, autokorrelationsrobust)")
    sig_status = "good" if rel["ci_excludes_zero"] else "bad"
    r1, r2, r3 = st.columns(3)
    with r1:
        metric_box("t-Statistik", f"{rel['tstat']:.2f}" if pd.notna(rel["tstat"]) else "n/a",
                    sig_status if pd.notna(rel["tstat"]) else "neutral")
    with r2:
        ci_text = (f"{rel['ci_low']*100:+.2f}% bis {rel['ci_high']*100:+.2f}%"
                   if pd.notna(rel["ci_low"]) else "n/a")
        metric_box("95%-Konfidenzintervall", ci_text, sig_status if pd.notna(rel["ci_low"]) else "neutral")
    with r3:
        if rel["subperiod_nonempty"] > 0:
            ratio = rel["subperiod_matching"] / rel["subperiod_nonempty"]
            consistency_status = "good" if ratio >= 0.5 else "bad"
        else:
            consistency_status = "neutral"
        metric_box(
            "Perioden-Konsistenz", f"{rel['subperiod_matching']}/{rel['subperiod_nonempty']}",
            consistency_status,
            help_text="Anzahl Teilperioden mit gleichem Vorzeichen wie der Gesamt-Mittelwert.",
        )
    if rel["ci_excludes_zero"]:
        st.success("Konfidenzintervall schliesst 0 aus - statistisch robuster Edge in dieser Konfiguration.")
    else:
        st.info("Konfidenzintervall schliesst 0 nicht aus - kein robust nachweisbarer Edge.")

    with st.expander(f"Alle Trades anzeigen ({len(trades)})"):
        st.dataframe(
            trades[["entry_date", "exit_date", "exit_reason", "entry_iv", "net_pnl", "pnl_norm", "hold_days", "touched"]]
            .sort_values("entry_date", ascending=False),
            use_container_width=True,
        )


def render(df: pd.DataFrame, daily_df: pd.DataFrame, has_resolution: bool) -> None:
    """Oeffentlicher Einstiegspunkt fuer diesen Modus - wird von dashboard.py
    nur aufgerufen, wenn 'Iron Condor Backtest' oben aktiv ist."""
    st.header("Iron Condor Backtest")
    st.caption(
        "Taegliche ueberlappende Positionen (Ladder-Modus), IV-Quelle = "
        "historische Volatilitaet (HV). Laeuft nur auf echten Tagesdaten "
        "(ab 1927-12-30)."
    )

    n_daily_total = (df["resolution"] == "daily").sum() if has_resolution else len(df)
    if n_daily_total < 60:
        st.warning(
            "Zu wenige echte Tagesdaten fuer einen sinnvollen Backtest "
            f"(nur {n_daily_total} gefunden). Fuehre `python src/data_fetch.py` "
            "mit Internetverbindung aus, um echte yfinance-Daten zu laden."
        )
        return

    default_spot = float(df["close"].iloc[-1])

    with st.sidebar:
        st.divider()
        st.header("Iron Condor Parameter")
        legs = _sidebar_legs(default_spot)
        params = _sidebar_params()
        run_backtest = st.button("Backtest ausfuehren", type="primary")

    r_free = 0.0  # Black-76: 0 = korrekt fuer Futures-Optionen

    daily_df = daily_df.sort_values("date")
    _check_data_quality(daily_df)
    _render_price_preview(daily_df, legs, params, r_free)

    if run_backtest:
        _run_backtest(df, legs, params, n_daily_total)

    if "ic_trades" in st.session_state:
        _render_results()
    else:
        st.info("Parameter links einstellen und 'Backtest ausfuehren' klicken.")