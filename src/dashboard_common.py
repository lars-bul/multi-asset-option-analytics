"""
Gemeinsame Bausteine fuer dashboard.py und die beiden Modus-Module
(dashboard_ladder.py = Iron Condor Backtest, dashboard_analog.py =
Analog-Tage-Matching).

Hier liegt NUR, was BEIDE Modi brauchen: Daten laden, die immer sichtbare
Candlestick-Uebersicht (samt ihrem eigenen Sidebar-Abschnitt "Ansicht"),
und kleine Anzeige-Helfer. Modus-spezifische Logik gehoert NICHT hierher -
wer nur einen Modus lesen/aendern will, muss dieses Modul zwar importieren,
aber nie inhaltlich anfassen.
"""
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_lightweight_charts import renderLightweightCharts

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "spx_daily.parquet"
TRAINING_TABLE_FILE = Path(__file__).resolve().parent.parent / "data" / "training_table.parquet"


def metric_box(label: str, value: str, status: str = "neutral", help_text: str | None = None) -> None:
    """Rendert eine Kennzahl als farbig hinterlegte Box statt st.metric.

    status: "good" (gruen) / "bad" (rot) / "neutral" (kein Werturteil moeglich,
    z.B. reine Anzahl oder Info-Kennzahl).
    """
    colors = {
        "good": ("rgba(38, 166, 90, 0.16)", "rgba(38, 166, 90, 0.65)"),
        "bad": ("rgba(239, 83, 80, 0.16)", "rgba(239, 83, 80, 0.65)"),
        "neutral": ("rgba(255, 255, 255, 0.04)", "rgba(197, 203, 206, 0.25)"),
    }
    bg, border = colors.get(status, colors["neutral"])
    help_html = (
        f'<div style="font-size:0.72rem;color:#444;margin-top:4px;line-height:1.3;">{help_text}</div>'
        if help_text else ""
    )
    # WICHTIG: HTML als EINE Zeile ohne fuehrende Einrueckung bauen. Streamlit/
    # CommonMark interpretiert 4+ Leerzeichen am Zeilenanfang als Code-Block
    # (auch mit unsafe_allow_html=True) - sonst landet das schliessende </div>
    # als sichtbarer Text in einer eigenen grauen Box statt den Block zu
    # schliessen. Textfarben bewusst dunkel: die Box ist immer hell/transluzent
    # getint, hell-auf-hell (helles Theme) waere sonst unlesbar.
    html = (
        f'<div style="background:{bg};border:1px solid {border};border-radius:10px;'
        f'padding:10px 14px 12px 14px;margin-bottom:6px;height:100%;">'
        f'<div style="font-size:0.8rem;color:#555;margin-bottom:2px;">{label}</div>'
        f'<div style="font-size:1.55rem;font-weight:700;color:#111;line-height:1.2;">{value}</div>'
        f'{help_html}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


@st.cache_data
def load_data() -> pd.DataFrame:
    """Laedt data/spx_daily.parquet (siehe data_fetch.py)."""
    if not DATA_FILE.exists():
        return pd.DataFrame()
    df = pd.read_parquet(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


@st.cache_data
def load_training_table() -> pd.DataFrame:
    """Laedt data/training_table.parquet (Grundlage fuers Analog-Matching,
    siehe build_dataset.py / labels.py)."""
    if not TRAINING_TABLE_FILE.exists():
        return pd.DataFrame()
    tt = pd.read_parquet(TRAINING_TABLE_FILE)
    tt["entry_date"] = pd.to_datetime(tt["entry_date"])
    tt["exit_date"] = pd.to_datetime(tt["exit_date"])
    return tt


def to_candles(df: pd.DataFrame) -> list[dict]:
    """Konvertiert in das von lightweight-charts erwartete Candle-Format."""
    return [
        {
            "time": row.date.strftime("%Y-%m-%d"),
            "open": float(row.open), "high": float(row.high),
            "low": float(row.low), "close": float(row.close),
        }
        for row in df.itertuples()
    ]


def render_overview(df: pd.DataFrame, has_resolution: bool) -> pd.DataFrame:
    """Rendert den Teil, der IMMER sichtbar ist, egal welcher Modus oben
    gewaehlt ist: Sidebar-Abschnitt "Ansicht" (Zeitraum-Slider) + Candlestick-
    Chart + Kennzahlen zum letzten Schlusskurs.

    Rueckgabe: 'daily_df' (nur echte Tagesdaten, sortiert nach Datum) - das
    ist die Datengrundlage, die BEIDE Modi fuer ihre Backtests verwenden
    (unabhaengig vom "Anzahl Tage anzeigen"-Slider, der NUR den Chart hier
    betrifft, nicht die Backtest-Zeitraeume)."""
    with st.sidebar:
        st.header("Ansicht")
        n_days = st.slider("Anzahl Tage anzeigen", 60, len(df), min(750, len(df)))
        if has_resolution:
            only_real_daily = st.checkbox(
                "Nur echte Tagesdaten (vor 1927-12-30 ausblenden)", value=False,
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

    chart_options = {
        "height": 500,
        "layout": {"background": {"type": "solid", "color": "#0e1117"}, "textColor": "#d1d4dc"},
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
            "upColor": "#26a69a", "downColor": "#ef5350", "borderVisible": False,
            "wickUpColor": "#26a69a", "wickDownColor": "#ef5350",
        },
    }]
    renderLightweightCharts([{"chart": chart_options, "series": series}], key="spx_chart")

    if has_resolution and view["date"].min() < pd.Timestamp("1928-01-01"):
        st.caption(
            "Hinweis: Kerzen vor 1927-12-30 zeigen keine echte Tagesspanne "
            "(open=high=low=close), da nur monatliche Kursdaten existieren."
        )

    daily_df = df[df["resolution"] == "daily"] if has_resolution else df
    return daily_df.sort_values("date")