"""
Lokales Dashboard - laeuft nur auf deinem Rechner.

Start:
    streamlit run src/dashboard.py

Zeigt aktuell einen TradingView-artigen Candlestick-Chart der SPX-Historie
(aus data/spx_daily.parquet, siehe data_fetch.py). Das ist bewusst das
Grundgeruest: hier kommen spaeter IV-Rank-Verlauf, Bucket-Klassifikation,
Trade-Empfehlungen etc. als weitere Panels/Tabs dazu.
"""

from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_lightweight_charts import renderLightweightCharts

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent))

from black76 import VolSurfaceConfig
from backtest import IronCondorConfig, LegConfig, FixedRankConfig, run_daily_ladder, summarize_trades
from stats import reliability_summary

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "spx_daily.parquet"

st.set_page_config(page_title="SPX Options Dashboard", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        return pd.DataFrame()
    df = pd.read_parquet(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def to_candles(df: pd.DataFrame) -> list[dict]:
    """Konvertiert in das von lightweight-charts erwartete Format."""
    out = []
    for row in df.itertuples():
        out.append({
            "time": row.date.strftime("%Y-%m-%d"),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
        })
    return out


st.title("SPX Options Dashboard")
st.caption("Lokales Analyse-Tool - laeuft nur bei dir auf localhost")

df = load_data()

if df.empty:
    st.warning(
        "Noch keine Daten gefunden. Einmal im Terminal ausfuehren:\n\n"
        "`python src/data_fetch.py`\n\n"
        "und diese Seite danach neu laden."
    )
    st.stop()

# ── Sidebar: einfache Steuerung, spaeter Platz fuer Bucket-Filter etc. ──
has_resolution = "resolution" in df.columns
with st.sidebar:
    st.header("Ansicht")
    n_days = st.slider("Anzahl Tage anzeigen", 60, len(df), min(750, len(df)))
    if has_resolution:
        only_real_daily = st.checkbox(
            "Nur echte Tagesdaten (vor 1927-12-30 ausblenden)",
            value=False,
            help="Vor Ende 1927 gibt es keine echten taeglichen OHLC-Daten fuer den "
                 "S&P - die Reihe davor ist aus Robert Shillers Monatsdaten (seit 1871) "
                 "auf Handelstage vorwaertsgefuellt (open=high=low=close).",
        )
    else:
        only_real_daily = False
    st.divider()
    st.caption(f"Letzter Datenstand: {df['date'].max().date()}")
    st.caption(f"{len(df)} Zeilen geladen (ab {df['date'].min().date()})")

plot_df = df
if has_resolution and only_real_daily:
    plot_df = df[df["resolution"] == "daily"]
    if plot_df.empty:
        st.warning(
            "Keine echten Tagesdaten vorhanden (yfinance vermutlich nicht erreicht). "
            "Zeige stattdessen die volle Reihe inkl. monatlich interpolierter Werte."
        )
        plot_df = df

view = plot_df.tail(n_days)

if has_resolution:
    n_monthly_in_view = (view["resolution"] == "monthly_interpolated").sum()
    if n_monthly_in_view > 0:
        st.info(
            f"{n_monthly_in_view} von {len(view)} angezeigten Tagen sind monatlich "
            f"interpolierte Werte (kein echter Tages-OHLC verfuegbar) - siehe "
            f"Checkbox in der Sidebar."
        )

col1, col2, col3 = st.columns(3)
col1.metric("Letzter Schlusskurs", f"{view['close'].iloc[-1]:,.2f}")
day_change = view["close"].iloc[-1] - view["close"].iloc[-2]
day_change_pct = day_change / view["close"].iloc[-2] * 100
col2.metric("Veraenderung (1 Tag)", f"{day_change:+.2f}", f"{day_change_pct:+.2f}%")
col3.metric("Angezeigter Zeitraum", f"{view['date'].iloc[0].date()} bis {view['date'].iloc[-1].date()}")

# ── Candlestick-Chart (TradingView Lightweight Charts) ──
chart_options = {
    "height": 500,
    "layout": {
        "background": {"type": "solid", "color": "#0e1117"},
        "textColor": "#d1d4dc",
    },
    "grid": {
        "vertLines": {"color": "rgba(42, 46, 57, 0.5)"},
        "horzLines": {"color": "rgba(42, 46, 57, 0.5)"},
    },
    "timeScale": {"borderColor": "rgba(197, 203, 206, 0.4)"},
    "rightPriceScale": {"borderColor": "rgba(197, 203, 206, 0.4)"},
}

series = [{
    "type": "Candlestick",
    "data": to_candles(view),
    "options": {
        "upColor": "#26a69a",
        "downColor": "#ef5350",
        "borderVisible": False,
        "wickUpColor": "#26a69a",
        "wickDownColor": "#ef5350",
    },
}]

renderLightweightCharts([{"chart": chart_options, "series": series}], key="spx_chart")

if has_resolution and view["date"].min() < pd.Timestamp("1928-01-01"):
    st.caption(
        "Hinweis: Kerzen vor 1927-12-30 zeigen keine echte Tagesspanne "
        "(open=high=low=close), da nur monatliche Kursdaten existieren."
    )

st.divider()
st.caption(
    "Naechste Ausbaustufen: IV-Rank-Panel, Bucket-Klassifikation je Handelstag, "
    "Trade-Empfehlungstabelle (Short/Long-Bein-Reliability)."
)

# ═══════════════════════════════════════════════════════════════════════════
#  IRON CONDOR BACKTEST (portiert aus dem Pine-Skript)
# ═══════════════════════════════════════════════════════════════════════════
st.header("Iron Condor Backtest")
st.caption(
    "Taegliche ueberlappende Positionen (Ladder-Modus), IV-Quelle = "
    "historische Volatilitaet (HV). Laeuft nur auf echten Tagesdaten "
    "(ab 1927-12-30) - monatlich interpolierte Tage werden automatisch "
    "ausgeschlossen."
)

n_daily_total = (df["resolution"] == "daily").sum() if has_resolution else len(df)
if n_daily_total < 60:
    st.warning(
        "Zu wenige echte Tagesdaten fuer einen sinnvollen Backtest "
        f"(nur {n_daily_total} gefunden). Fuehre `python src/data_fetch.py` "
        "mit Internetverbindung aus, um echte yfinance-Daten zu laden."
    )
else:
    with st.sidebar:
        st.divider()
        st.header("Iron Condor Parameter")
        T_days = st.slider("Laufzeit T (Tage)", 2, 60, 7)
        hv_length = st.slider("HV-Periode (Tage)", 5, 90, 30)

        st.subheader("4 Beine (je Put/Call, Long/Short, Delta)")
        st.caption(
            "Default entspricht dem klassischen Iron Condor "
            "(Put long / Put short / Call short / Call long). Beliebig "
            "aenderbar - z.B. Menge=0 setzen, um ein Bein zu deaktivieren."
        )
        hc1, hc2, hc3, hc4 = st.columns(4)
        hc1.caption("Typ")
        hc2.caption("Richtung")
        hc3.caption("Delta")
        hc4.caption("Menge")
        default_legs = [
            ("Bein 1", "put", "long", 0.05),
            ("Bein 2", "put", "short", 0.16),
            ("Bein 3", "call", "short", 0.16),
            ("Bein 4", "call", "long", 0.05),
        ]
        legs = []
        for label, def_type, def_side, def_delta in default_legs:
            st.markdown(f"**{label}**")
            lc1, lc2, lc3, lc4 = st.columns(4)
            option_type = lc1.selectbox(
                "Typ", ["put", "call"], index=["put", "call"].index(def_type),
                key=f"type_{label}", label_visibility="collapsed",
            )
            side = lc2.selectbox(
                "Richtung", ["long", "short"], index=["long", "short"].index(def_side),
                key=f"side_{label}", label_visibility="collapsed",
            )
            delta = lc3.number_input(
                "Delta", 0.01, 0.99, def_delta, 0.01,
                key=f"delta_{label}", label_visibility="collapsed",
            )
            qty = lc4.number_input(
                "Menge", 0.0, 50.0, 1.0, 1.0,
                key=f"qty_{label}", label_visibility="collapsed",
            )
            legs.append(LegConfig(label=label, option_type=option_type, side=side, delta=delta, qty=qty))

        st.subheader("Vorzeitiger Ausstieg")
        close_on_touch = st.checkbox("Bei Short-Strike-Beruehrung schliessen", value=False)
        use_early_exit = st.checkbox("Nach X Tagen zwangsweise schliessen", value=False)
        early_exit_days = st.slider("Ausstieg nach X Tagen", 1, 59, min(T_days - 1, 5), disabled=not use_early_exit)

        st.subheader("Vol-Oberflaeche")
        use_vol_surface = st.checkbox("Skew + Term Structure verwenden", value=True)

        st.subheader("Backtest-Modus: Fixer IV-Rang (hypothetisch)")
        st.caption(
            "Beantwortet: 'Wie haette JEDER Trade performt, waere er IMMER "
            "zum selben (heutigen) IV-Rang eroeffnet/bewertet worden?' "
            "Rechnet an jedem Bar zurueck, welche Vola noetig gewesen waere, "
            "um genau den Ziel-Rang im damals gueltigen rollierenden "
            "HV-Min/Max-Fenster zu erreichen - ersetzt current_iv komplett "
            "(wirkt auf Strike-Auswahl UND Bewertung)."
        )
        use_fixed_rank = st.checkbox("Aktivieren: Immer zum festen Ziel-IV-Rang handeln", value=False)
        target_rank_pct = st.slider(
            "Ziel-IV-Rang (%)", -50.0, 200.0, 50.0, 5.0, disabled=not use_fixed_rank,
            help="50 = immer genau in der Mitte zwischen dem HV-Min und HV-Max des Lookback-Fensters. "
                 "Werte ausserhalb 0-100 extrapolieren bewusst ueber das historische Fenster hinaus.",
        )
        rc1, rc2 = st.columns(2)
        rank_hv_length = rc1.number_input(
            "HV-Periode fuer Rang (Tage)", 2, 365, 30, disabled=not use_fixed_rank,
        )
        rank_lookback_days = rc2.number_input(
            "Lookback fuer Rang (Tage)", 10, 5000, 252, disabled=not use_fixed_rank,
        )

        run_backtest = st.button("Backtest ausfuehren", type="primary")

    if run_backtest:
        cfg = IronCondorConfig(
            T_days=T_days,
            hv_length=hv_length,
            legs=legs,
            close_on_touch=close_on_touch,
            use_early_exit=use_early_exit, early_exit_days=early_exit_days,
            vol_surface=VolSurfaceConfig(use_vol_surface=use_vol_surface),
            fixed_rank=FixedRankConfig(
                enabled=use_fixed_rank, target_rank_pct=target_rank_pct,
                hv_length=int(rank_hv_length), lookback_days=int(rank_lookback_days),
            ),
        )

        with st.spinner(f"Backtest laeuft ({n_daily_total} Handelstage)..."):
            trades = run_daily_ladder(df, cfg)

        if trades.empty:
            st.warning(
                "Keine Trades erzeugt - Parameter pruefen (z.B. HV-Periode/Lookback "
                "zu lang fuer die Datenmenge, oder Ziel-Rang fuehrt zu ungueltiger Vola)."
            )
        else:
            st.session_state["ic_trades"] = trades
            leg_summary = " / ".join(f"{l.option_type[0].upper()}{'+' if l.is_long else '-'}{l.delta}" for l in legs)
            rank_label = f", Fixer Rang={target_rank_pct:.0f}%" if use_fixed_rank else ""
            st.session_state["ic_cfg_label"] = f"T={T_days}d, HV={hv_length}d, Beine: {leg_summary}{rank_label}"

    if "ic_trades" in st.session_state:
        trades = st.session_state["ic_trades"]
        summary = summarize_trades(trades)
        rel = reliability_summary(trades, nw_lags=20, n_subperiods=6)

        st.caption(f"Letzter Lauf: {st.session_state.get('ic_cfg_label', '')}")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Trades", f"{summary['n_trades']:,}")
        m2.metric("Gewinnrate", f"{summary['win_rate_pct']:.1f}%")
        m3.metric("Profit Factor", f"{summary['profit_factor']:.2f}" if pd.notna(summary["profit_factor"]) else "n/a")
        m4.metric("Ø norm. P&L/Trade", f"{summary['avg_pnl_norm_pct']:+.2f}%")
        m5.metric("Ø genutzte IV (Entry)", f"{summary['avg_entry_iv_pct']:.1f}%")

        if summary.get("n_unlimited_risk", 0) > 0:
            st.warning(
                f"{summary['n_unlimited_risk']} von {summary['n_trades']} Trades haben "
                "unbegrenztes Verlustrisiko (mehr Short- als Long-Calls) - fuer diese "
                "ist 'normierter P&L' nicht definiert (NaN) und wird bei den obigen "
                "Kennzahlen automatisch ausgeschlossen."
            )

        st.subheader("Reliability (Newey-West, autokorrelationsrobust)")
        r1, r2, r3 = st.columns(3)
        r1.metric("t-Statistik", f"{rel['tstat']:.2f}" if pd.notna(rel["tstat"]) else "n/a")
        ci_text = (f"{rel['ci_low']*100:+.2f}% bis {rel['ci_high']*100:+.2f}%"
                   if pd.notna(rel["ci_low"]) else "n/a")
        r2.metric("95%-Konfidenzintervall", ci_text)
        r3.metric(
            "Perioden-Konsistenz",
            f"{rel['subperiod_matching']}/{rel['subperiod_nonempty']}",
            help="Anzahl Teilperioden mit gleichem Vorzeichen wie der Gesamt-Mittelwert.",
        )
        if rel["ci_excludes_zero"]:
            st.success("Konfidenzintervall schliesst 0 aus - statistisch robuster Edge in dieser Konfiguration.")
        else:
            st.info("Konfidenzintervall schliesst 0 nicht aus - kein robust nachweisbarer Edge.")

        st.subheader("Equity-Kurve (kumulierter P&L)")
        eq = trades.sort_values("exit_date").copy()
        eq["cumulative"] = eq["net_pnl"].cumsum()

        equity_series = [{
            "time": row.exit_date.strftime("%Y-%m-%d"),
            "value": float(row.cumulative),
        } for row in eq.itertuples()]

        eq_chart_options = {
            "height": 350,
            "layout": {"background": {"type": "solid", "color": "#0e1117"}, "textColor": "#d1d4dc"},
            "grid": {
                "vertLines": {"color": "rgba(42, 46, 57, 0.5)"},
                "horzLines": {"color": "rgba(42, 46, 57, 0.5)"},
            },
            "timeScale": {"borderColor": "rgba(197, 203, 206, 0.4)"},
            "rightPriceScale": {"borderColor": "rgba(197, 203, 206, 0.4)"},
        }
        eq_series = [{
            "type": "Line",
            "data": equity_series,
            "options": {"color": "#26a69a", "lineWidth": 2},
        }]
        renderLightweightCharts([{"chart": eq_chart_options, "series": eq_series}], key="equity_chart")

        with st.expander(f"Alle Trades anzeigen ({len(trades)})"):
            st.dataframe(
                trades[["entry_date", "exit_date", "exit_reason", "entry_iv", "net_pnl", "pnl_norm", "hold_days", "touched"]]
                .sort_values("entry_date", ascending=False),
                use_container_width=True,
            )
    else:
        st.info("Parameter links einstellen und 'Backtest ausfuehren' klicken.")
