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

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_lightweight_charts import renderLightweightCharts

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent))

from black76 import VolSurfaceConfig, strike_from_delta, leg_price
from backtest import (
    IronCondorConfig, LegConfig, FixedRankConfig, run_daily_ladder, summarize_trades,
    compute_hv, compute_rank_target_vol,
)
from stats import reliability_summary
from analog_matching import (
    AnalogConfig, SimilaritySpaceConfig, recommend_for_today, available_t_days,
)

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "spx_daily.parquet"

st.set_page_config(page_title="SPX Options Dashboard", layout="wide")


def metric_box(label: str, value: str, status: str = "neutral", help_text: str | None = None):
    """Rendert eine Kennzahl als farbig hinterlegte Box statt st.metric.

    status:
      - "good"    -> gruen hinterlegt (Kennzahl spricht fuer die Strategie)
      - "bad"     -> rot hinterlegt (Kennzahl spricht gegen die Strategie)
      - "neutral" -> neutrale Standardfarbe (kein klares Werturteil moeglich/
                     sinnvoll, z.B. reine Anzahl oder Info-Kennzahl)
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
    # WICHTIG: das HTML muss als EINE Zeile ohne fuehrende Einrueckung gebaut
    # werden. Streamlit/CommonMark interpretiert 4+ Leerzeichen am Zeilenanfang
    # als eingerueckten Code-Block - das passiert VOR der HTML-Verarbeitung
    # (auch mit unsafe_allow_html=True) und hat vorher dazu gefuehrt, dass das
    # schliessende </div> als literaler Text in einer eigenen grauen Box
    # angezeigt wurde, statt den div-Block sauber zu schliessen.
    #
    # Textfarben bewusst DUNKEL (statt hellgrau/weiss): die Box selbst ist
    # immer ein helles, transluzentes Gruen/Rot/Grau-Tint - das funktioniert
    # sowohl im hellen als auch im dunklen Streamlit-Theme. Helle Schrift war
    # nur im dunklen Theme lesbar und ist im hellen Theme (weisser Hintergrund)
    # praktisch unsichtbar.
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
    if not DATA_FILE.exists():
        return pd.DataFrame()
    df = pd.read_parquet(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


TRAINING_TABLE_FILE = Path(__file__).resolve().parent.parent / "data" / "training_table.parquet"


@st.cache_data
def load_training_table() -> pd.DataFrame:
    """Laedt data/training_table.parquet fuer das Analog-Tage-Matching."""
    if not TRAINING_TABLE_FILE.exists():
        return pd.DataFrame()
    tt = pd.read_parquet(TRAINING_TABLE_FILE)
    tt["entry_date"] = pd.to_datetime(tt["entry_date"])
    tt["exit_date"] = pd.to_datetime(tt["exit_date"])
    return tt


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


def to_equity_points(trades: pd.DataFrame) -> tuple[list[dict], pd.Series]:
    """Baut die Datenpunkte fuer die Equity-Kurve (kumulierter P&L).

    WICHTIG - Tagesaggregation statt Punkt-je-Trade:
    lightweight-charts erwartet pro Serie STRIKT aufsteigende, EINDEUTIGE
    Zeitpunkte. Bei der taeglichen Ladder-Strategie koennen an einem
    Kalendertag aber mehrere Positionen gleichzeitig schliessen (z.B. eine
    per Stop-Loss und eine andere per regulaerem Verfall, beide mit
    unterschiedlichem Entry-Datum) - das wuerde zu doppelten Zeitstempeln
    fuehren. Deshalb wird der Netto-P&L zuerst pro Kalendertag summiert und
    erst danach kumuliert.

    Das reduziert nebenbei auch die Punktzahl deutlich (ein Punkt pro
    Handelstag statt einer pro Trade) - zusammen mit der Umstellung von
    st.line_chart (Vega-Lite, wird bei vielen tausend Punkten spuerbar
    langsam und laesst sich nur eingeschraenkt zoomen) auf die bereits fuer
    den Candlestick-Chart genutzte, Canvas-basierte TradingView-Lightweight-
    Charts-Bibliothek ist die Kurve so ueber die GESAMTE Backtest-Historie
    fluessig zoom- und pannbar, nicht nur ueber ein Fenster von ~10 Jahren."""
    daily_pnl = trades.groupby(trades["exit_date"].dt.normalize())["net_pnl"].sum().sort_index()
    equity = daily_pnl.cumsum()
    points = [
        {"time": ts.strftime("%Y-%m-%d"), "value": float(val)}
        for ts, val in equity.items()
    ]
    return points, equity


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
    # Grober Referenz-Spot NUR fuer sinnvolle Default-Werte bei der Eingabe
    # eines fixen Strikes in der Sidebar unten (Delta-Modus braucht das nicht).
    _default_spot = float(df["close"].iloc[-1])

    with st.sidebar:
        st.divider()
        st.header("Iron Condor Parameter")
        T_days = st.slider("Laufzeit T (Tage)", 2, 60, 7)
        hv_length = st.slider("HV-Periode (Tage)", 5, 90, 30)

        st.subheader("4 Beine (je Put/Call, Long/Short, Delta ODER fixer Strike)")
        st.caption(
            "Default entspricht dem klassischen Iron Condor "
            "(Put long / Put short / Call short / Call long). Beliebig "
            "aenderbar - z.B. Menge=0 setzen, um ein Bein zu deaktivieren. "
            "Je Bein kann statt eines Ziel-Deltas (Strike wird dann taeglich "
            "neu berechnet) auch ein FESTER Strike-Preis eingegeben werden "
            "(bleibt an jedem Entry-Tag unveraendert)."
        )
        hc1, hc2, hc3, hc4, hc5 = st.columns([1, 1, 1.1, 1, 1])
        hc1.caption("Typ")
        hc2.caption("Richtung")
        hc3.caption("Basis")
        hc4.caption("Delta / Strike")
        hc5.caption("Menge")
        default_legs = [
            ("Bein 1", "put", "long", 0.05),
            ("Bein 2", "put", "short", 0.16),
            ("Bein 3", "call", "short", 0.16),
            ("Bein 4", "call", "long", 0.05),
        ]
        legs = []
        for label, def_type, def_side, def_delta in default_legs:
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
                "Basis", ["Delta", "Fixer Strike"],
                key=f"basis_{label}", label_visibility="collapsed",
            )
            if basis == "Fixer Strike":
                fixed_strike = lc4.number_input(
                    "Strike", min_value=0.0, value=round(_default_spot, 0), step=1.0,
                    key=f"strike_{label}", label_visibility="collapsed",
                )
                delta = def_delta  # bleibt im LegConfig belegt, wird aber ignoriert
                strike_mode = "fixed"
            else:
                delta = lc4.number_input(
                    "Delta", 0.01, 0.99, def_delta, 0.01,
                    key=f"delta_{label}", label_visibility="collapsed",
                )
                fixed_strike = None
                strike_mode = "delta"
            qty = lc5.number_input(
                "Menge", 0.0, 50.0, 1.0, 1.0,
                key=f"qty_{label}", label_visibility="collapsed",
            )
            legs.append(LegConfig(
                label=label, option_type=option_type, side=side, delta=delta, qty=qty,
                strike_mode=strike_mode, fixed_strike=fixed_strike,
            ))

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

    r_free = 0.0  # identisch zum bisherigen Default (Black-76: 0 = korrekt fuer Futures-Optionen)

    # ═════════════════════════════════════════════════════════════════════
    #  OPTIONS-PREISVORSCHAU: EINZELPREISE JE STRIKE + GESAMTPREIS DER KOMBO
    #  Zeigt, unabhaengig vom eigentlichen Backtest, was die aktuell links
    #  konfigurierte 4-Bein-Struktur JETZT kosten wuerde: Strike, Theoriepreis
    #  je Bein (Black-76, inkl. Skew/Term-Structure falls aktiviert) und den
    #  Netto-Credit/Debit der gesamten Kombo. Basis: letzter Schlusskurs +
    #  aktuell gewaehlte IV-Quelle (HV oder - falls aktiviert - der aus dem
    #  Ziel-IV-Rang zurueckgerechnete Wert), jeweils mit der oben gewaehlten
    #  Laufzeit T_days. Aktualisiert sich live mit jeder Sidebar-Aenderung,
    #  unabhaengig davon, ob der Backtest bereits gelaufen ist. Beine mit
    #  fixem Strike (statt Delta) verwenden hier direkt den eingegebenen
    #  Strike-Preis, ohne Delta->Strike-Umrechnung.
    # ═════════════════════════════════════════════════════════════════════
    st.subheader("Options-Preisvorschau (aktuell): Preise je Strike + Kombo gesamt")
    st.caption(
        "Theoriepreise (Black-76) der 4 links konfigurierten Beine auf Basis des "
        "letzten Schlusskurses, der aktuellen IV-Quelle und der eingestellten "
        "Laufzeit T. Rein informativ - unabhaengig vom Backtest-Ergebnis unten."
    )

    daily_df = df[df["resolution"] == "daily"] if has_resolution else df
    daily_df = daily_df.sort_values("date")

    # ── Datenqualitaets-Check: laengste Serie exakt identischer Schlusskurse ──
    #  Echte SPX-Tagesschlusskurse sind praktisch nie ueber mehrere Tage exakt
    #  bit-identisch. Taucht so eine Serie auf, ist das ein starkes Signal fuer
    #  fehlerhafte/eingefrorene Daten (z.B. yfinance liefert zuletzt denselben
    #  Preis mehrfach, oder ein Caching-/Merge-Bug in data_fetch.py dupliziert
    #  Zeilen) - NICHT fuer einen Fehler in HV-/Backtest-Formeln. Das erklaert
    #  u.a. auffaellig identische entry_iv/pnl_norm-Werte in der Trade-Tabelle,
    #  wenn ein rollierendes HV-Fenster (z.B. 7 Tage) vollstaendig innerhalb
    #  einer solchen Dup-Serie liegt.
    def _max_consecutive(mask: pd.Series) -> int:
        if not mask.any():
            return 0
        groups = (~mask).cumsum()
        return int(mask.groupby(groups).sum().max())

    _dup_mask = daily_df["close"].diff().eq(0.0)
    _max_dup_run = _max_consecutive(_dup_mask)
    if _max_dup_run >= 3:
        _last_dup_date = daily_df.loc[_dup_mask[::-1].idxmax() if _dup_mask.any() else daily_df.index[-1], "date"]
        st.warning(
            f"⚠️ Datenqualitaet: {_max_dup_run} aufeinanderfolgende Handelstage mit "
            "EXAKT identischem Schlusskurs in den geladenen Tagesdaten gefunden "
            f"(zuletzt um den {pd.Timestamp(_last_dup_date).date()}). Fuer echte "
            "SPX-Kurse ist das extrem unwahrscheinlich und deutet auf einen "
            "Stale-Cache- oder Datenabruf-Fehler hin. Das verzerrt HV-Berechnung, "
            "Strike-Auswahl und alle Trade-Kennzahlen (u.a. auffaellig aehnliche "
            "'entry_iv'/'pnl_norm'-Werte in der Trade-Tabelle unten). Empfehlung: "
            "`python src/data_fetch.py` mit Internetverbindung erneut ausfuehren, "
            "um data/spx_daily.parquet frisch zu erzeugen."
        )

    S_last = float(daily_df["close"].iloc[-1])
    T_years_preview = T_days / 365.0

    hv_series_preview = compute_hv(daily_df, hv_length)
    current_hv_preview = hv_series_preview.iloc[-1] if len(hv_series_preview) else np.nan

    if use_fixed_rank:
        rank_series_preview = compute_rank_target_vol(
            daily_df, int(rank_hv_length), int(rank_lookback_days), target_rank_pct,
        )
        iv_preview = rank_series_preview.iloc[-1] if len(rank_series_preview) else np.nan
        iv_source_label = f"Fixer IV-Rang ({target_rank_pct:.0f}%)"
    else:
        iv_preview = current_hv_preview
        iv_source_label = f"HV ({hv_length}d)"

    if pd.isna(iv_preview) or iv_preview <= 0.0:
        st.warning(
            "Aktuelle IV-Quelle liefert keinen gueltigen Wert (z.B. zu wenig "
            "Historie fuer die gewaehlte HV-/Lookback-Periode) - keine "
            "Preisvorschau moeglich."
        )
    else:
        vc_preview = VolSurfaceConfig(use_vol_surface=use_vol_surface)

        preview_rows = []
        short_sum = 0.0
        long_sum = 0.0
        for leg in legs:
            if leg.uses_fixed_strike:
                K = leg.fixed_strike
                basis_label = "Fix"
            else:
                K = strike_from_delta(S_last, leg.delta, iv_preview, T_years_preview, is_put=leg.is_put)
                basis_label = f"Δ{leg.delta:.2f}"
            premium_per_contract = leg_price(
                K, leg.is_put, T_years_preview, S_last, iv_preview, r_free, vc_preview,
            )
            premium_total = premium_per_contract * leg.qty

            if leg.is_long:
                long_sum += premium_total
            else:
                short_sum += premium_total

            preview_rows.append({
                "Bein": leg.label,
                "Typ": "Put" if leg.is_put else "Call",
                "Richtung": "Long" if leg.is_long else "Short",
                "Basis": basis_label,
                "Menge": leg.qty,
                "Strike": round(K, 2),
                "Preis / Kontrakt": round(premium_per_contract, 4),
                "Preis gesamt": round(premium_total, 4),
                "Cashflow": ("-" if leg.is_long else "+") + f"{premium_total:.4f}",
            })

        net_credit = short_sum - long_sum

        preview_df = pd.DataFrame(preview_rows)
        st.dataframe(
            preview_df[["Bein", "Typ", "Richtung", "Basis", "Menge", "Strike",
                        "Preis / Kontrakt", "Preis gesamt", "Cashflow"]],
            use_container_width=True, hide_index=True,
        )

        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("Spot (letzter Schlusskurs)", f"{S_last:,.2f}")
        pc2.metric(f"Aktive IV-Quelle: {iv_source_label}", f"{iv_preview * 100:.2f}%")
        pc3.metric("Short-Praemien (Summe)", f"+{short_sum:.4f}")
        pc4.metric(
            "Netto-Credit der Kombo (gesamt)",
            f"{net_credit:+.4f}",
            help="Summe der Short-Praemien minus Summe der Long-Praemien "
                 "(entry_net, gleiche Definition wie im Backtest).",
        )
        st.caption(
            f"Long-Praemien (Summe): -{long_sum:.4f}  |  "
            f"Datenstand: {daily_df['date'].iloc[-1].date()}  |  "
            f"Laufzeit T={T_days} Tage ({T_years_preview*365:.0f} Kalendertage)."
        )

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
            leg_summary = " / ".join(
                f"{l.option_type[0].upper()}{'+' if l.is_long else '-'}"
                + (f"K{l.fixed_strike:.0f}" if l.uses_fixed_strike else f"{l.delta}")
                for l in legs
            )
            rank_label = f", Fixer Rang={target_rank_pct:.0f}%" if use_fixed_rank else ""
            st.session_state["ic_cfg_label"] = f"T={T_days}d, HV={hv_length}d, Beine: {leg_summary}{rank_label}"

    if "ic_trades" in st.session_state:
        trades = st.session_state["ic_trades"]
        summary = summarize_trades(trades)
        rel = reliability_summary(trades, nw_lags=20, n_subperiods=6)

        st.caption(f"Letzter Lauf: {st.session_state.get('ic_cfg_label', '')}")

        # ── Kennzahlen, farbig hinterlegt je nachdem ob gut/schlecht ──
        #  Faerbung ist bewusst nur dort binaer gruen/rot, wo eine Kennzahl
        #  ein klares Werturteil zulaesst (>0, >1, statistische Signifikanz
        #  etc.). Reine Mengen-/Info-Kennzahlen (Anzahl Trades, genutzte IV)
        #  bleiben neutral, da es dafuer kein 'gut' oder 'schlecht' gibt.
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            metric_box("Trades", f"{summary['n_trades']:,}", "neutral")
        with m2:
            win_rate = summary["win_rate_pct"]
            metric_box("Gewinnrate", f"{win_rate:.1f}%", "good" if win_rate >= 50.0 else "bad")
        with m3:
            pf = summary["profit_factor"]
            if pd.isna(pf):
                metric_box("Profit Factor", "n/a", "neutral")
            else:
                metric_box("Profit Factor", f"{pf:.2f}", "good" if pf > 1.0 else "bad")
        with m4:
            avg_pnl = summary["avg_pnl"]
            metric_box(
                "Erwartungswert / Trade", f"{avg_pnl:+.4f}",
                "good" if avg_pnl > 0.0 else "bad",
                help_text="Arithmetisches Mittel des Netto-P&L ueber alle "
                          "abgeschlossenen Trades, in absoluten Punkten.",
            )
        with m5:
            avg_pnl_norm = summary["avg_pnl_norm_pct"]
            metric_box(
                "Ø norm. P&L/Trade", f"{avg_pnl_norm:+.2f}%",
                "good" if avg_pnl_norm > 0.0 else "bad",
            )
        with m6:
            metric_box("Ø genutzte IV (Entry)", f"{summary['avg_entry_iv_pct']:.1f}%", "neutral")

        if summary.get("n_unlimited_risk", 0) > 0:
            st.warning(
                f"{summary['n_unlimited_risk']} von {summary['n_trades']} Trades haben "
                "unbegrenztes Verlustrisiko (mehr Short- als Long-Calls) - fuer diese "
                "ist 'normierter P&L' nicht definiert (NaN) und wird bei den obigen "
                "Kennzahlen automatisch ausgeschlossen."
            )

        # ─────────────────────────────────────────────────────────────────
        #  EQUITY-KURVE (kumulierter P&L)
        #  Siehe Docstring von to_equity_points() fuer Details/Begruendung:
        #  Tagesaggregation (statt Punkt-je-Trade) + TradingView Lightweight
        #  Charts (statt st.line_chart) - dadurch ueber die GESAMTE Historie
        #  fluessig zoom-/pannbar statt nur ueber ein Fenster von ~10 Jahren.
        # ─────────────────────────────────────────────────────────────────
        st.subheader("Equity-Kurve (kumulierter P&L)")

        equity_points, equity_series_vals = to_equity_points(trades)

        equity_chart_options = {
            "height": 400,
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
            # explizit aktiviert (sind zwar Default bei lightweight-charts,
            # aber so ist unmissverstaendlich klar, dass Zoom/Pan ueber die
            # komplette geladene Reihe gewuenscht sind, nicht nur ein
            # eingeschraenktes Fenster):
            "handleScroll": True,
            "handleScale": True,
        }

        equity_chart_series = [{
            "type": "Area",
            "data": equity_points,
            "options": {
                "lineColor": "#2962FF",
                "topColor": "rgba(41, 98, 255, 0.28)",
                "bottomColor": "rgba(41, 98, 255, 0.0)",
                "lineWidth": 2,
            },
        }]

        renderLightweightCharts(
            [{"chart": equity_chart_options, "series": equity_chart_series}],
            key="equity_chart",
        )
        st.caption(
            f"{len(equity_points)} Tages-Datenpunkte "
            f"({equity_series_vals.index.min().date()} bis {equity_series_vals.index.max().date()}) - "
            "frei zoom-/pannbar ueber die gesamte Backtest-Historie "
            "(Mausrad = Zoom, Ziehen = Pan, Doppelklick auf die Zeitachse = volle Ansicht)."
        )

        st.subheader("Reliability (Newey-West, autokorrelationsrobust)")
        # Signifikanz-Status (t-Stat und Konfidenzintervall sind zwei
        # Ausdrucksformen desselben Tests -> selbe Faerbung).
        sig_status = "good" if rel["ci_excludes_zero"] else "bad"
        r1, r2, r3 = st.columns(3)
        with r1:
            tstat_status = sig_status if pd.notna(rel["tstat"]) else "neutral"
            metric_box(
                "t-Statistik", f"{rel['tstat']:.2f}" if pd.notna(rel["tstat"]) else "n/a",
                tstat_status,
            )
        with r2:
            ci_text = (f"{rel['ci_low']*100:+.2f}% bis {rel['ci_high']*100:+.2f}%"
                       if pd.notna(rel["ci_low"]) else "n/a")
            ci_status = sig_status if pd.notna(rel["ci_low"]) else "neutral"
            metric_box("95%-Konfidenzintervall", ci_text, ci_status)
        with r3:
            if rel["subperiod_nonempty"] > 0:
                ratio = rel["subperiod_matching"] / rel["subperiod_nonempty"]
                consistency_status = "good" if ratio >= 0.5 else "bad"
            else:
                consistency_status = "neutral"
            metric_box(
                "Perioden-Konsistenz",
                f"{rel['subperiod_matching']}/{rel['subperiod_nonempty']}",
                consistency_status,
                help_text="Anzahl Teilperioden mit gleichem Vorzeichen wie der "
                          "Gesamt-Mittelwert.",
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
    else:
        st.info("Parameter links einstellen und 'Backtest ausfuehren' klicken.")

# ═══════════════════════════════════════════════════════════════════════════
#  ANALOG-TAGE-MATCHING (Grundlage: dieselbe Rang-Skalierung wie oben bei
#  "Fixer IV-Rang", nur nicht auf die GESAMTE Historie angewendet, sondern
#  gezielt auf die K Handelstage, die einem gewaehlten Tag ("heute") in
#  Bezug auf Vola-Regime/Momentum/Tail-Risiko am aehnlichsten waren - siehe
#  src/analog_matching.py fuer die Details/Docstrings.)
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
st.header("Analog-Tage-Matching")
st.caption(
    "Findet die K historisch aehnlichsten Handelstage zu einem gewaehlten Tag "
    "(PCA-Distanz ueber die taeglichen Features aus features.py) und wertet "
    "auf genau diesen Tagen einen Short- und Long-Strangle aus - rang-skaliert "
    "auf den heutigen HV-Rang (dieselbe Grundidee wie beim 'Fixer IV-Rang'-"
    "Backtest oben, hier aber nicht auf jeden Handelstag angewendet, sondern "
    "gezielt nur auf die K aehnlichsten Analog-Tage). Per Grid-Search wird "
    "daraus das robusteste Put-/Call-Delta-Paar fuer die SHORT-Beine (Bein 2 "
    "'Short Put' / Bein 3 'Short Call') des Iron Condor gesucht - inkl. "
    "Train/Validierungs-Split gegen Mehrfachvergleichs-Zufallstreffer."
)

training_table = load_training_table()

if training_table.empty:
    st.warning(
        "Noch keine Trainingstabelle gefunden. Einmal im Terminal ausfuehren:\n\n"
        "`python src/build_dataset.py`\n\n"
        "und diese Seite danach neu laden. Das baut die Feature-Matrix + "
        "Strangle-Labels, die das Analog-Matching als Grundlage braucht "
        "(siehe build_dataset.py, labels.py)."
    )
else:
    valid_t_days = available_t_days(training_table)

    with st.sidebar:
        st.divider()
        st.header("Analog-Tage-Matching")
        query_date_input = st.date_input(
            "Tag, fuer den Analog-Tage gesucht werden ('heute')",
            value=daily_df["date"].max().date(),
            min_value=daily_df["date"].min().date(),
            max_value=daily_df["date"].max().date(),
            help="Standardmaessig der letzte verfuegbare Handelstag. Ein "
                 "frueheres Datum erlaubt, das Verfahren selbst rueckblickend "
                 "zu pruefen - es werden dabei garantiert nur Analog-Tage VOR "
                 "diesem Datum verwendet (siehe analog_matching.py, Kausalitaet).",
        )
        analog_k = st.slider(
            "Anzahl Analog-Tage (k)", 50, 3000, 1000, 50,
            help="Wie viele der historisch aehnlichsten Handelstage verwendet "
                 "werden. Mehr Tage -> stabilere Statistik, aber steigendes "
                 "Risiko, dass sich die Analog-Tage in wenigen historischen "
                 "Ereignissen (z.B. einer einzelnen Krise) haeufen - siehe "
                 "Clustering-Warnung unten im Ergebnis.",
        )
        analog_min_gap = st.number_input(
            "Mindestabstand zum gewaehlten Tag (Handelstage)", 0, 60, 5,
            help="Vermeidet triviale, fast identische Nachbartage direkt vor "
                 "dem gewaehlten Tag als 'Analog'.",
        )
        analog_T_days = st.selectbox(
            "Laufzeit T fuer die Analog-Bewertung (Tage)", valid_t_days,
            index=valid_t_days.index(7) if 7 in valid_t_days else 0,
            help="Nur Laufzeiten verfuegbar, fuer die build_dataset.py "
                 "Strangle-Labels erzeugt hat (labels.StrangleLabelConfig."
                 "T_days_list) - unabhaengig von der Laufzeit T oben beim "
                 "Iron-Condor-Backtest.",
        )
        analog_hv_length = st.number_input("HV-Periode fuer den Rang (Tage)", 2, 365, 30)
        analog_rank_lookback = st.number_input("Lookback fuer den Rang (Tage)", 10, 5000, 252)

        st.caption("Delta-Raster fuer die Grid-Search (Put x Call):")
        delta_grid_options = [0.05, 0.10, 0.16, 0.20, 0.25, 0.30, 0.35, 0.40]
        put_delta_grid = st.multiselect(
            "Put-Deltas", delta_grid_options, default=[0.10, 0.16, 0.20, 0.25, 0.30],
        )
        call_delta_grid = st.multiselect(
            "Call-Deltas", delta_grid_options, default=[0.10, 0.16, 0.20, 0.25, 0.30],
        )

        run_analog = st.button("Analog-Matching ausfuehren", type="primary")

    if run_analog:
        if not put_delta_grid or not call_delta_grid:
            st.error("Bitte mindestens ein Put-Delta und ein Call-Delta im Raster auswaehlen.")
        else:
            try:
                with st.spinner(f"Suche {analog_k} Analog-Tage und werte "
                                 f"{len(put_delta_grid) * len(call_delta_grid)} Delta-Kombinationen aus..."):
                    result = recommend_for_today(
                        daily_df, training_table,
                        query_date=pd.Timestamp(query_date_input),
                        T_days=int(analog_T_days),
                        put_delta_grid=put_delta_grid,
                        call_delta_grid=call_delta_grid,
                        analog_cfg=AnalogConfig(k=int(analog_k), min_gap_days=int(analog_min_gap)),
                        space_cfg=SimilaritySpaceConfig(),
                        hv_length=int(analog_hv_length),
                        rank_lookback_days=int(analog_rank_lookback),
                        vol_surface=VolSurfaceConfig(),
                    )
                st.session_state["analog_result"] = result
            except ValueError as e:
                st.error(f"Analog-Matching konnte nicht ausgefuehrt werden: {e}")

    if "analog_result" in st.session_state:
        result = st.session_state["analog_result"]

        a1, a2, a3, a4 = st.columns(4)
        with a1:
            metric_box("Gewaehlter Tag", f"{result['query_date'].date()}", "neutral")
        with a2:
            metric_box("Heutiger HV-Rang", f"{result['today_rank']*100:.1f}%", "neutral",
                       help_text="Position der aktuellen HV im rollierenden Min/Max-Fenster - "
                                 "identische Definition wie beim 'Fixer IV-Rang'-Backtest oben.")
        with a3:
            metric_box("Gefundene Analog-Tage", f"{result['n_analogs']:,}", "neutral")
        with a4:
            metric_box("PCA-Komponenten", f"{result['n_pca_components']}", "neutral",
                       help_text="Anzahl Hauptkomponenten, die den Aehnlichkeits-Raum aufspannen "
                                 "(so viele wie fuer 80% erklaerte Varianz noetig).")

        clustering = result["clustering"]
        frac = clustering["frac_within_gap"]
        if frac >= 0.30:
            st.warning(
                f"⚠️ Clustering-Hinweis: {frac*100:.0f}% der Analog-Tage liegen (chronologisch "
                f"sortiert) hoechstens {clustering['gap_days_threshold']} Kalendertage auseinander "
                f"(Median-Abstand: {clustering['median_gap_days']:.0f} Tage). Vola-Regime dauern "
                "typischerweise mehrere Tage bis Wochen an - ein grosser Teil der 'unabhaengigen' "
                "Analog-Tage kann daher aus wenigen historischen Ereignissen stammen. Die "
                "Newey-West-Korrektur unten beruecksichtigt Autokorrelation, ersetzt bei starkem "
                "Clustering aber keine gesunde Skepsis gegenueber der Trade-Anzahl."
            )

        grid_result = result["grid_result"]
        if grid_result.empty:
            st.info(
                "Keine gueltige Delta-Kombination auf dem Analog-Set bewertbar "
                "(z.B. weil target_vol durchgehend ungueltig war). Parameter pruefen."
            )
        else:
            st.subheader("Robusteste Leg-Paare (Short- und Long-Strangle)")
            st.caption(
                "'robust' = Konfidenzintervall schliesst 0 aus UND gleiches (positives) "
                "Vorzeichen in BEIDEN chronologischen Haelften des Analog-Sets (Train/Val) - "
                "sortiert nach Validierungs-t-Statistik."
            )

            display_cols = {
                "put_delta": "Put-Δ", "call_delta": "Call-Δ", "side": "Richtung",
                "n_val": "n (Val)", "val_mean_pnl_norm": "Ø norm. P&L (Val)",
                "val_tstat": "t-Stat (Val)", "val_ci_excludes_zero": "CI ≠ 0",
                "val_subperiod_matching": "Perioden-Match", "val_subperiod_nonempty": "Perioden gesamt",
                "robust_both_halves": "Robust (Train+Val)",
            }
            show_df = grid_result[list(display_cols.keys())].rename(columns=display_cols).copy()
            show_df["Ø norm. P&L (Val)"] = (show_df["Ø norm. P&L (Val)"] * 100).round(2).astype(str) + "%"
            show_df["Richtung"] = show_df["Richtung"].map({"short": "Short Strangle", "long": "Long Strangle"})
            st.dataframe(show_df, use_container_width=True, hide_index=True)

            robust_rows = grid_result[grid_result["robust_both_halves"]]
            if robust_rows.empty:
                st.info(
                    "Keine Delta-Kombination war in BEIDEN Haelften des Analog-Sets robust "
                    "signifikant - kein belastbarer Edge fuer diese Konfiguration erkennbar."
                )
            else:
                best = robust_rows.iloc[0]
                side_label = "Short Strangle" if best["side"] == "short" else "Long Strangle"
                st.success(
                    f"Bestes robustes Leg-Paar: **{side_label}**, Put-Δ={best['put_delta']:.2f} / "
                    f"Call-Δ={best['call_delta']:.2f} (Validierung: Ø norm. P&L "
                    f"{best['val_mean_pnl_norm']*100:+.2f}%, t={best['val_tstat']:.2f})."
                )
                if best["side"] == "short":
                    st.caption(
                        f"Uebertragen auf den Iron-Condor-Backtest oben: Bein 2 (Short Put) mit "
                        f"Delta={best['put_delta']:.2f} und Bein 3 (Short Call) mit "
                        f"Delta={best['call_delta']:.2f} einstellen - die Wings (Bein 1/4) sind "
                        "hiervon unabhaengig und muessen separat gewaehlt werden, da das "
                        "Analog-Matching nur den ungehedgten 2-Bein-Strangle-Kern bewertet."
                    )
                else:
                    st.caption(
                        "Die Long-Richtung war hier robuster als Short - das entspraeche eher "
                        "einem 'Long Strangle' (Bein 2/3 auf 'long' statt 'short' stellen) statt "
                        "eines klassischen Iron Condor, der auf kassierten Praemien basiert."
                    )

        with st.expander(f"Alle gefundenen Analog-Tage anzeigen ({len(result['analogs'])})"):
            st.dataframe(
                result["analogs"].assign(analog_date=result["analogs"]["analog_date"].dt.date)
                .sort_values("distance"),
                use_container_width=True, hide_index=True,
            )
    else:
        st.info("Parameter links einstellen und 'Analog-Matching ausfuehren' klicken.")