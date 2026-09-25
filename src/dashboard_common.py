"""
Gemeinsame Bausteine fuer dashboard.py und die Modus-Module (dashboard_ladder.py
= Iron Condor Backtest, dashboard_analog.py = Analog-Tage-Matching,
dashboard_livechain.py = Heutige Kette, dashboard_depot.py = Depot-Analyse).

Hier liegt NUR, was mehrere Modi brauchen: Daten laden, die immer sichtbare
Candlestick-Uebersicht (samt ihrem eigenen Sidebar-Abschnitt "Ansicht"),
und kleine Anzeige-Helfer. Modus-spezifische Logik gehoert NICHT hierher -
wer nur einen Modus lesen/aendern will, muss dieses Modul zwar importieren,
aber nie inhaltlich anfassen.

MULTI-INSTRUMENT-UMBAU: load_data()/load_training_table()/render_overview()
nehmen jetzt zusaetzlich ein Instrument (instruments.InstrumentConfig)
entgegen, statt fest data/spx_daily.parquet bzw. data/training_table.parquet
zu laden. Der Dateiname kommt ueber instruments.data_file(instr, kind, DATA_DIR) -
z.B. data/gold_daily.parquet, data/oil_training_table.parquet. Alles andere
(Chart-Aufbau, NaN-Schutz, Kennzahlen) ist inhaltlich unveraendert gegenueber
dem Original; lediglich die beiden SPX-spezifischen Hinweise (Shiller-
Monatsdaten vor 1927-12-30) werden jetzt nur noch angezeigt, wenn
instr.has_shiller_history True ist (aktuell nur SPX) - fuer alle anderen
Instrumente waeren sie irrefuehrend, da deren 'resolution'-Spalte (falls
vorhanden) nie 'monthly_interpolated' enthaelt.

KORREKTUR (NaN-Schutz, unveraendert gegenueber dem Original): Die letzte(n)
Zeile(n) von df/daily_df koennen einen NaN-Schlusskurs haben (z.B. ein
heutiger Balken, der noch nicht befuellt ist, ein Feiertag ohne echten Handel,
oder ein fehlgeschlagener data_fetch-Lauf). Vorher fuehrte das zu 'Letzter
Schlusskurs nan' und - weil NaN kein gueltiges JSON ist - zu einem
SyntaxError im Candlestick-Chart. Jetzt:
  - Die Kopfkennzahlen (Schlusskurs, Tagesveraenderung) nehmen den letzten
    GUELTIGEN (nicht-NaN) Schlusskurs, nicht blind die letzte Zeile.
  - to_candles() laesst Zeilen mit NaN in open/high/low/close aus - eine
    einzelne kaputte Zeile blockiert nicht mehr den ganzen Chart.
"""
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_lightweight_charts import renderLightweightCharts

from instruments import InstrumentConfig, data_file

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


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
def load_data(instr: InstrumentConfig) -> pd.DataFrame:
    """Laedt data/{instr.key}_daily.parquet (siehe data_fetch.py). Leeres
    DataFrame, falls die Datei fuer dieses Instrument noch nicht existiert
    (z.B. `python src/data_fetch.py {instr.key}` noch nicht ausgefuehrt)."""
    f = data_file(instr, "daily", DATA_DIR)
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_parquet(f)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


@st.cache_data
def load_training_table(instr: InstrumentConfig) -> pd.DataFrame:
    """Laedt data/{instr.key}_training_table.parquet (Grundlage fuers
    Analog-Matching, siehe build_dataset.py / labels.py). Leeres DataFrame,
    falls sie fuer dieses Instrument noch nicht gebaut wurde - build_dataset.py
    ist aktuell noch SPX-spezifisch fest verdrahtet (siehe dashboard_livechain.
    _ensure_training_table fuer den Hinweistext dazu)."""
    f = data_file(instr, "training_table", DATA_DIR)
    if not f.exists():
        return pd.DataFrame()
    tt = pd.read_parquet(f)
    tt["entry_date"] = pd.to_datetime(tt["entry_date"])
    tt["exit_date"] = pd.to_datetime(tt["exit_date"])
    return tt


def to_candles(df: pd.DataFrame) -> list[dict]:
    """Konvertiert in das von lightweight-charts erwartete Candle-Format.

    Zeilen mit NaN in open/high/low/close werden ausgelassen: NaN ist kein
    gueltiges JSON (weder 'NaN' noch 'null' passt zum Zahlenfeld, das die
    Chart-Bibliothek erwartet) - eine einzelne solche Zeile hat vorher den
    kompletten Chart mit einem JSON-SyntaxError zum Absturz gebracht."""
    valid = df.dropna(subset=["open", "high", "low", "close"])
    return [
        {
            "time": row.date.strftime("%Y-%m-%d"),
            "open": float(row.open), "high": float(row.high),
            "low": float(row.low), "close": float(row.close),
        }
        for row in valid.itertuples()
    ]


def _last_valid_closes(view: pd.DataFrame, n: int = 2) -> pd.Series:
    """Die letzten n GUELTIGEN (nicht-NaN) Schlusskurse in 'view', chronologisch
    aufsteigend - Grundlage fuer Kopfkennzahlen, die nicht blind auf die
    zuletzt vorhandene ZEILE (die NaN sein kann) zugreifen duerfen."""
    return view["close"].dropna().tail(n)


def render_overview(df: pd.DataFrame, has_resolution: bool, instr: InstrumentConfig) -> pd.DataFrame:
    """Rendert den Teil, der IMMER sichtbar ist, egal welcher Modus oben
    gewaehlt ist: Sidebar-Abschnitt "Ansicht" (Zeitraum-Slider) + Candlestick-
    Chart + Kennzahlen zum letzten Schlusskurs - jetzt fuer das gewaehlte
    Instrument 'instr'. Widget-Keys sind pro Instrument getrennt, damit ein
    Instrumentenwechsel nicht mit dem Slider-/Checkbox-Zustand des vorherigen
    Instruments kollidiert (die Kurshistorien sind unterschiedlich lang).

    Rueckgabe: 'daily_df' (nur echte Tagesdaten, sortiert nach Datum) - das
    ist die Datengrundlage, die alle Modi fuer ihre Backtests verwenden
    (unabhaengig vom "Anzahl Tage anzeigen"-Slider, der NUR den Chart hier
    betrifft, nicht die Backtest-Zeitraeume)."""
    with st.sidebar:
        st.header("Ansicht")
        st.caption(f"Instrument: **{instr.label}** ({instr.currency})")
        n_days = st.slider(
            "Anzahl Tage anzeigen", 60, len(df), min(750, len(df)), key=f"n_days_{instr.key}",
        )
        # Die Shiller-Monatsreihe (vor 1927-12-30) gibt es NUR bei SPX (siehe
        # instruments.py, has_shiller_history) - bei anderen Instrumenten waere
        # diese Checkbox/der Hinweis unten irrefuehrend.
        if has_resolution and instr.has_shiller_history:
            only_real_daily = st.checkbox(
                "Nur echte Tagesdaten (vor 1927-12-30 ausblenden)", value=False,
                key=f"only_real_daily_{instr.key}",
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

    if has_resolution and instr.has_shiller_history:
        n_monthly_in_view = (view["resolution"] == "monthly_interpolated").sum()
        if n_monthly_in_view > 0:
            st.info(
                f"{n_monthly_in_view} von {len(view)} angezeigten Tagen sind monatlich "
                f"interpolierte Werte (kein echter Tages-OHLC verfuegbar) - siehe "
                f"Checkbox in der Sidebar."
            )

    # KORREKTUR: letzter GUELTIGER Schlusskurs statt blind view['close'].iloc[-1] - die zuletzt
    # vorhandene ZEILE kann NaN sein (unvollstaendiger heutiger Balken, Feiertag ohne echten
    # Handel, fehlgeschlagener data_fetch-Lauf). Fehlen ganz gueltige Werte (z.B. n_days sehr
    # klein und ausgerechnet die sichtbaren Tage sind alle NaN), wird eine Warnung gezeigt statt
    # eines Absturzes/„nan“ in der Anzeige.
    last_valid = _last_valid_closes(view, n=2)
    col1, col2, col3 = st.columns(3)
    if len(last_valid) >= 1:
        last_close = float(last_valid.iloc[-1])
        last_date = view.loc[last_valid.index[-1], "date"]
        stale = last_date != view["date"].iloc[-1]
        col1.metric(f"Letzter Schlusskurs ({instr.currency})", f"{last_close:,.2f}")
        if stale:
            col1.caption(f"⚠️ Stand {last_date.date()} - neuere Tage ohne Kurs.")
        if len(last_valid) >= 2:
            prev_close = float(last_valid.iloc[-2])
            day_change = last_close - prev_close
            day_change_pct = day_change / prev_close * 100 if prev_close else float("nan")
            col2.metric("Veraenderung (1 Tag)", f"{day_change:+.2f}", f"{day_change_pct:+.2f}%")
        else:
            col2.metric("Veraenderung (1 Tag)", "n/a")
    else:
        col1.metric("Letzter Schlusskurs", "n/a")
        col2.metric("Veraenderung (1 Tag)", "n/a")
        st.warning(
            "Keine gueltigen Schlusskurse im angezeigten Zeitraum (alle Werte NaN) - "
            "'Anzahl Tage anzeigen' vergroessern oder data_fetch.py pruefen/neu ausfuehren."
        )
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
    candles = to_candles(view)
    n_dropped = len(view) - len(candles)
    if n_dropped > 0:
        st.caption(
            f"⚠️ {n_dropped} von {len(view)} angezeigten Tagen haben einen unvollstaendigen "
            "Kurs (NaN in Open/High/Low/Close) und wurden im Chart ausgelassen."
        )
    series = [{
        "type": "Candlestick",
        "data": candles,
        "options": {
            "upColor": "#26a69a", "downColor": "#ef5350", "borderVisible": False,
            "wickUpColor": "#26a69a", "wickDownColor": "#ef5350",
        },
    }]
    # Key ist instrumentenspezifisch (vorher fest "spx_chart"), damit die
    # Chart-Komponente beim Instrumentenwechsel sauber neu aufgebaut wird.
    renderLightweightCharts([{"chart": chart_options, "series": series}], key=f"chart_{instr.key}")

    if has_resolution and instr.has_shiller_history and view["date"].min() < pd.Timestamp("1928-01-01"):
        st.caption(
            "Hinweis: Kerzen vor 1927-12-30 zeigen keine echte Tagesspanne "
            "(open=high=low=close), da nur monatliche Kursdaten existieren."
        )

    daily_df = df[df["resolution"] == "daily"] if has_resolution else df
    return daily_df.sort_values("date")