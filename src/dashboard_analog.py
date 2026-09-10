"""
MODUS 2: Analog-Tage-Matching.

Ein einziger Einstiegspunkt fuer dashboard.py: render(df, daily_df,
has_resolution). Rendert sowohl den eigenen Sidebar-Abschnitt als auch den
Hauptbereich - dashboard.py ruft diese Funktion NUR auf, wenn dieser Modus
oben in der Reiterleiste aktiv ist, es entsteht also keine Ueberschneidung
mit der Sidebar von dashboard_ladder.py.

IV-Bewertungsniveau: IMMER rang-basiert (siehe analog_matching.py, Punkt 8
im Moduldokstring) - identisches Prinzip zum "Fixer IV-Rang"-Modus im Iron
Condor Backtest (dashboard_ladder.py/backtest.FixedRankConfig): man gibt
einen Ziel-HV-Rang (%) vor, und fuer JEDEN Analog-Tag wird rollierend
zurueckgerechnet, welche IV an GENAU DIESEM Tag noetig gewesen waere, um
exakt diesen Rang zu erreichen. Es gibt bewusst KEIN Eingabefeld fuer eine
absolute IV mehr (frueher "Heutige IV (%)") - der Rang allein bestimmt die
Bewertung, konsistent ueber alle Analog-Tage hinweg.

Aufbau (von oben nach unten in dieser Datei):
  1. _sidebar_params()     - alle Eingabefelder in der Sidebar, gibt ein dict zurueck
  2. _run_matching()       - fuehrt recommend_for_today() aus, speichert Ergebnis in st.session_state
  3. _render_grid_result() - Tabelle der robustesten Leg-Paare + "bestes robustes Paar"-Hinweis
  4. _render_detail_view() - volle Kennzahlen fuer EINE ausgewaehlte Delta-Kombination
  5. _render_result()      - Kopfkennzahlen + Clustering-Warnung + ruft 3+4 auf
  6. render()               - Oeffentlicher Einstiegspunkt, ruft 1-5 in der richtigen Reihenfolge
"""
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from black76 import VolSurfaceConfig
from stats import reliability_summary
from analog_matching import (
    AnalogConfig, SimilaritySpaceConfig, recommend_for_today, available_t_days,
    evaluate_analog_set, summarize_analog_trades, hv_rank_bounds, target_vol_for_rank,
)
from dashboard_common import metric_box, load_training_table


def _sidebar_params(daily_df: pd.DataFrame, valid_t_days: list[int]) -> dict:
    """Alle Eingabefelder fuer diesen Modus. Gibt ein dict zurueck statt
    vieler Einzelwerte, damit render() uebersichtlich bleibt."""
    query_date_input = st.date_input(
        "Tag, fuer den Analog-Tage gesucht werden ('heute')",
        value=daily_df["date"].max().date(),
        min_value=daily_df["date"].min().date(),
        max_value=daily_df["date"].max().date(),
        help="Standardmaessig der letzte verfuegbare Handelstag. Ein frueheres "
             "Datum erlaubt, das Verfahren selbst rueckblickend zu pruefen - es "
             "werden dabei nur Analog-Tage VOR diesem Datum verwendet.",
    )
    analog_k = st.slider(
        "Anzahl Analog-Tage (k)", 50, 3000, 1000, 50,
        help="Mehr Tage -> stabilere Statistik, aber steigendes Risiko, dass sich "
             "die Analog-Tage in wenigen historischen Ereignissen haeufen (siehe "
             "Clustering-Warnung im Ergebnis).",
    )
    analog_min_gap = st.number_input(
        "Mindestabstand zum gewaehlten Tag (Handelstage)", 0, 60, 5,
        help="Vermeidet triviale, fast identische Nachbartage als 'Analog'.",
    )
    analog_T_days = st.selectbox(
        "Laufzeit T fuer die Analog-Bewertung (Tage)", valid_t_days,
        index=valid_t_days.index(7) if 7 in valid_t_days else 0,
        help="Nur Laufzeiten verfuegbar, fuer die build_dataset.py Strangle-Labels "
             "erzeugt hat - unabhaengig von der Laufzeit im Iron-Condor-Backtest.",
    )

    st.subheader("Bewertungsniveau (IV)")
    st.caption(
        "Fuer JEDEN Analog-Tag wird rollierend zurueckgerechnet, welche IV an "
        "GENAU DIESEM Tag noetig gewesen waere, um den hier eingestellten "
        "HV-Rang zu erreichen - identisches Prinzip zum 'Fixer IV-Rang'-Modus "
        "im Iron Condor Backtest oben. Beispiel: Rang 40% bedeutet an einem "
        "ruhigen Markttag eine niedrige IV, in einer Vola-Krise (z.B. "
        "Finanzkrise 2008) bedeutet DERSELBE Rang 40% dagegen eine deutlich "
        "hoehere IV, weil dort das ganze HV-Min/Max-Fenster hoeher liegt."
    )
    target_rank_pct = st.slider(
        "Ziel-HV-Rang (%)", -50.0, 200.0, 50.0, 5.0,
        help="50 = immer genau in der Mitte zwischen HV-Min und HV-Max des "
             "jeweiligen Lookback-Fensters JEDES Analog-Tages. Werte ausserhalb "
             "0-100 extrapolieren bewusst ueber das historische Fenster hinaus.",
    )
    hv_length = st.number_input(
        "HV-Periode fuer das Rang-Fenster (Tage)", 2, 365, 30,
        help="Historische Volatilitaetsperiode, aus der das rollierende "
             "HV-Min/Max-Fenster je Analog-Tag gebaut wird.",
    )
    rank_lookback = st.number_input(
        "Lookback fuer den Rang (Tage)", 10, 5000, 252,
        help="Laenge des rollierenden Fensters, in dem der Ziel-Rang je "
             "Analog-Tag gesucht wird.",
    )

    # Rein informative Vorschau: was wuerde der eingestellte Ziel-Rang HEUTE
    # (am gewaehlten Tag) als absolute IV bedeuten? Fliesst NICHT in die
    # Bewertung ein (die passiert pro Analog-Tag einzeln, siehe oben) -
    # dient nur als Sanity-Check/Referenzpunkt fuer die Eingabe.
    bounds_preview = hv_rank_bounds(daily_df, int(hv_length), int(rank_lookback))
    today_bounds_rows = bounds_preview.loc[bounds_preview["date"] == pd.Timestamp(query_date_input)]
    if len(today_bounds_rows):
        b = today_bounds_rows.iloc[0]
        implied_iv_today = target_vol_for_rank(b["hv_min"], b["hv_max"], target_rank_pct / 100.0)
        if pd.notna(implied_iv_today):
            st.caption(
                f"Am {query_date_input} entspraeche Rang {target_rank_pct:.0f}% "
                f"einer IV von **{implied_iv_today*100:.2f}%** "
                f"(aktuelle HV({int(hv_length)}d): {b['hv']*100:.2f}%)."
            )
        else:
            st.caption(
                "Rang-Umrechnung am gewaehlten Tag n/a (zu wenig Historie im "
                "Lookback-Fenster)."
            )
    else:
        st.caption("Rang-Umrechnung am gewaehlten Tag n/a (kein HV-Wert verfuegbar).")

    st.caption("Delta-Raster fuer die Grid-Search (Put x Call):")
    delta_grid_options = [0.05, 0.10, 0.16, 0.20, 0.25, 0.30, 0.35, 0.40]
    put_delta_grid = st.multiselect("Put-Deltas", delta_grid_options, default=[0.10, 0.16, 0.20, 0.25, 0.30])
    call_delta_grid = st.multiselect("Call-Deltas", delta_grid_options, default=[0.10, 0.16, 0.20, 0.25, 0.30])

    run_analog = st.button("Analog-Matching ausfuehren", type="primary")

    return dict(
        query_date=pd.Timestamp(query_date_input), analog_k=int(analog_k), analog_min_gap=int(analog_min_gap),
        T_days=int(analog_T_days), target_rank_pct=float(target_rank_pct),
        hv_length=int(hv_length), rank_lookback=int(rank_lookback),
        put_delta_grid=put_delta_grid, call_delta_grid=call_delta_grid,
        run=run_analog,
    )


def _run_matching(daily_df: pd.DataFrame, training_table: pd.DataFrame, p: dict) -> None:
    """Fuehrt recommend_for_today() aus und legt das Ergebnis in
    st.session_state ab (ueberlebt den naechsten Rerun, z.B. beim Wechseln
    der Detail-Kombination)."""
    if not p["put_delta_grid"] or not p["call_delta_grid"]:
        st.error("Bitte mindestens ein Put-Delta und ein Call-Delta im Raster auswaehlen.")
        return
    try:
        with st.spinner(
            f"Suche {p['analog_k']} Analog-Tage und werte "
            f"{len(p['put_delta_grid']) * len(p['call_delta_grid'])} Delta-Kombinationen aus..."
        ):
            result = recommend_for_today(
                daily_df, training_table,
                query_date=p["query_date"], T_days=p["T_days"],
                put_delta_grid=p["put_delta_grid"], call_delta_grid=p["call_delta_grid"],
                analog_cfg=AnalogConfig(k=p["analog_k"], min_gap_days=p["analog_min_gap"]),
                space_cfg=SimilaritySpaceConfig(),
                hv_length=p["hv_length"], rank_lookback_days=p["rank_lookback"],
                vol_surface=VolSurfaceConfig(), target_rank_pct=p["target_rank_pct"],
            )
        st.session_state["analog_result"] = result
    except ValueError as e:
        st.error(f"Analog-Matching konnte nicht ausgefuehrt werden: {e}")


def _render_grid_result(grid_result: pd.DataFrame) -> None:
    """Screening-Tabelle: alle Put/Call-Delta x Short/Long-Kombinationen,
    sortiert nach Validierungs-t-Statistik, plus Hinweis auf die beste
    robuste Kombination (falls vorhanden)."""
    if grid_result.empty:
        st.info(
            "Keine gueltige Delta-Kombination auf dem Analog-Set bewertbar "
            "(z.B. weil target_vol durchgehend ungueltig war). Parameter pruefen."
        )
        return

    st.subheader("Robusteste Leg-Paare (Short- und Long-Strangle)")
    st.caption(
        "'robust' = Konfidenzintervall schliesst 0 aus UND gleiches (positives) "
        "Vorzeichen in BEIDEN chronologischen Haelften des Analog-Sets (Train/Val)."
    )
    display_cols = {
        "put_delta": "Put-Δ", "call_delta": "Call-Δ", "side": "Richtung",
        "n_val": "n (Val)", "val_mean_pnl_norm": "Ø norm. P&L (Val)", "val_tstat": "t-Stat (Val)",
        "val_ci_excludes_zero": "CI ≠ 0", "val_subperiod_matching": "Perioden-Match",
        "val_subperiod_nonempty": "Perioden gesamt", "robust_both_halves": "Robust (Train+Val)",
    }
    show_df = grid_result[list(display_cols.keys())].rename(columns=display_cols).copy()
    show_df["Ø norm. P&L (Val)"] = show_df["Ø norm. P&L (Val)"].mul(100).round(2).astype(str) + "%"
    show_df["Richtung"] = show_df["Richtung"].map({"short": "Short Strangle", "long": "Long Strangle"})
    st.dataframe(show_df, use_container_width=True, hide_index=True)

    robust_rows = grid_result[grid_result["robust_both_halves"]]
    if robust_rows.empty:
        st.info(
            "Keine Delta-Kombination war in BEIDEN Haelften des Analog-Sets robust "
            "signifikant - kein belastbarer Edge fuer diese Konfiguration erkennbar."
        )
        return
    best = robust_rows.iloc[0]
    side_label = "Short Strangle" if best["side"] == "short" else "Long Strangle"
    st.success(
        f"Bestes robustes Leg-Paar: **{side_label}**, Put-Δ={best['put_delta']:.2f} / "
        f"Call-Δ={best['call_delta']:.2f} (Validierung: Ø norm. P&L "
        f"{best['val_mean_pnl_norm']*100:+.2f}%, t={best['val_tstat']:.2f})."
    )


def _render_cum_pnl_chart(cum_series: pd.Series) -> None:
    """Rendert die kumulierte P&L-Kurve als STEP-Chart (Treppe): flach
    zwischen zwei Analog-Tagen, Sprung GENAU am naechsten Trade.

    WARUM ueberhaupt ein Step-Chart: Analog-Tage koennen Monate bis Jahre
    auseinanderliegen (anders als beim Ladder-Backtest, wo taeglich ein
    Trade schliesst). Eine normale Linie (st.line_chart) verbindet zwei weit
    auseinanderliegende Punkte mit einer GERADEN, schraegen Linie - das sah
    aus wie eine kontinuierliche Rampe, obwohl sich der P&L in Wirklichkeit
    nur GENAU am Analog-Tag aendert und dazwischen konstant bleibt.

    WARUM Altair statt einer taeglichen Forward-Fill-Reihe: eine fruehere
    Version hat das ueber eine taeglich geforward-fillte Reihe geloest - das
    erzeugt aber ueber die gesamte SPX-Historie (bis 1871) zehntausende
    Punkte und machte den Chart beim Zoomen/Verschieben spuerbar traege.
    Altairs `interpolate='step-after'` zeichnet dieselbe Treppenform direkt
    beim Rendern, OHNE die Datenpunkte kuenstlich aufzublaehen - die
    Punktzahl bleibt so klein wie die Anzahl der tatsaechlichen Analog-Trades
    (typischerweise Hunderte bis wenige Tausend, nicht Zehntausende)."""
    if cum_series.empty:
        st.info("Keine Datenpunkte fuer die kumulierte P&L-Kurve.")
        return
    chart_df = cum_series.rename("kum_pnl_norm").reset_index()
    chart_df.columns = ["entry_date", "kum_pnl_norm"]
    chart = (
        alt.Chart(chart_df)
        .mark_line(interpolate="step-after", color="#2962FF")
        .encode(
            x=alt.X("entry_date:T", title="Entry-Datum"),
            y=alt.Y("kum_pnl_norm:Q", title="kumulierter norm. P&L"),
            tooltip=["entry_date:T", alt.Tooltip("kum_pnl_norm:Q", format="+.3f")],
        )
        .properties(height=350)
        .interactive()   # Zoom/Pan bleibt fluessig, da nur ~n Analog-Trades statt Tausender Kalendertage geplottet werden
    )
    st.altair_chart(chart, use_container_width=True)


def _render_detail_view(result: dict, grid_result: pd.DataFrame) -> None:
    """Volle Kennzahlen (Newey-West-Reliability, Equity-Konsistenz-Check,
    Trade-Tabelle) fuer EINE ausgewaehlte Delta-Kombination auf dem
    KOMPLETTEN Analog-Set (nicht nur Train/Val wie beim Screening)."""
    st.subheader("Detailansicht: volle Kennzahlen fuer eine Kombination")

    combo_labels = [
        f"{row.put_delta:.2f} / {row.call_delta:.2f} - "
        f"{'Short' if row.side == 'short' else 'Long'} Strangle"
        + (" ✅" if row.robust_both_halves else "")
        for row in grid_result.itertuples()
    ]
    robust_positions = [pos for pos, row in enumerate(grid_result.itertuples()) if row.robust_both_halves]
    default_pos = robust_positions[0] if robust_positions else 0

    chosen_label = st.selectbox("Delta-Kombination fuer die Detailansicht", combo_labels, index=default_pos)
    chosen_row = grid_result.iloc[combo_labels.index(chosen_label)]

    with st.spinner("Berechne volle Kennzahlen auf dem kompletten Analog-Set..."):
        full_eval = evaluate_analog_set(
            result["analogs"], result["training_table"], result["bounds"], result["T_days"],
            float(chosen_row["put_delta"]), float(chosen_row["call_delta"]),
            result["r_free"], result["dividend_yield"], result["vol_surface"],
            vol_mode="rank", today_rank=result["today_rank"],
        )

    side = chosen_row["side"]
    if full_eval.empty:
        st.info("Keine gueltigen Trades fuer diese Kombination auf dem kompletten Analog-Set.")
        return

    trade_summary = summarize_analog_trades(full_eval, side)
    sub = full_eval.rename(columns={f"net_pnl_{side}": "net_pnl", f"pnl_norm_{side}": "pnl_norm"})
    rel_full = reliability_summary(sub, nw_lags=None, n_subperiods=6)

    st.caption(
        f"{'Short' if side == 'short' else 'Long'} Strangle, "
        f"Put-Δ={chosen_row['put_delta']:.2f} / Call-Δ={chosen_row['call_delta']:.2f}, "
        f"T={result['T_days']}d, Ziel-HV-Rang: {result['target_rank_pct']:.0f}%, "
        f"n={trade_summary['n_trades']} Analog-Tage (komplettes Set)."
    )

    d1, d2, d3, d4, d5, d6 = st.columns(6)
    with d1:
        metric_box("Trades", f"{trade_summary['n_trades']:,}", "neutral")
    with d2:
        wr = trade_summary["win_rate_pct"]
        metric_box("Gewinnrate", f"{wr:.1f}%", "good" if wr >= 50.0 else "bad")
    with d3:
        pf = trade_summary["profit_factor"]
        metric_box("Profit Factor", "n/a" if pd.isna(pf) else f"{pf:.2f}",
                    "neutral" if pd.isna(pf) else ("good" if pf > 1.0 else "bad"))
    with d4:
        avg_pnl = trade_summary["avg_pnl"]
        metric_box("Erwartungswert / Trade", f"{avg_pnl:+.4f}", "good" if avg_pnl > 0.0 else "bad")
    with d5:
        avg_pnl_norm = trade_summary["avg_pnl_norm_pct"]
        metric_box("Ø norm. P&L/Trade", f"{avg_pnl_norm:+.2f}%", "good" if avg_pnl_norm > 0.0 else "bad")
    with d6:
        metric_box(
            "Beste / Schlechteste Trade",
            f"{trade_summary['best_trade']:+.2f} / {trade_summary['worst_trade']:+.2f}", "neutral",
        )

    st.markdown("**Reliability (Newey-West, autokorrelationsrobust) - komplettes Analog-Set**")
    sig_status = "good" if rel_full["ci_excludes_zero"] else "bad"
    r1, r2, r3 = st.columns(3)
    with r1:
        metric_box("t-Statistik", f"{rel_full['tstat']:.2f}" if pd.notna(rel_full["tstat"]) else "n/a",
                    sig_status if pd.notna(rel_full["tstat"]) else "neutral")
    with r2:
        ci_text = (f"{rel_full['ci_low']*100:+.2f}% bis {rel_full['ci_high']*100:+.2f}%"
                   if pd.notna(rel_full["ci_low"]) else "n/a")
        metric_box("95%-Konfidenzintervall", ci_text, sig_status if pd.notna(rel_full["ci_low"]) else "neutral")
    with r3:
        if rel_full["subperiod_nonempty"] > 0:
            ratio = rel_full["subperiod_matching"] / rel_full["subperiod_nonempty"]
            consistency_status = "good" if ratio >= 0.5 else "bad"
        else:
            consistency_status = "neutral"
        metric_box("Perioden-Konsistenz",
                    f"{rel_full['subperiod_matching']}/{rel_full['subperiod_nonempty']}", consistency_status)

    st.markdown("**Kumulierter norm. P&L ueber die chronologisch sortierten Analog-Tage**")
    st.caption(
        "Konsistenz-Check ueber die Handelsgeschichte hinweg - KEINE reale Equity-Kurve "
        "(die Analog-Tage stammen aus unterschiedlichen, nicht-aufeinanderfolgenden Epochen)."
    )
    cum_series = full_eval.sort_values("entry_date").set_index("entry_date")[f"pnl_norm_{side}"].cumsum()
    _render_cum_pnl_chart(cum_series)

    with st.expander(f"Alle Trades dieser Kombination anzeigen ({len(full_eval)})"):
        show_trades = full_eval[[
            "entry_date", "hold_days", "distance", "target_vol", "K_put", "K_call",
            "entry_premium", f"net_pnl_{side}", f"pnl_norm_{side}",
        ]].rename(columns={f"net_pnl_{side}": "net_pnl", f"pnl_norm_{side}": "pnl_norm"})
        st.dataframe(
            show_trades.assign(entry_date=show_trades["entry_date"].dt.date).sort_values("entry_date", ascending=False),
            use_container_width=True, hide_index=True,
        )

    if side == "short":
        st.caption(
            f"Uebertragen auf den Iron-Condor-Backtest: Bein 2 (Short Put) mit "
            f"Delta={chosen_row['put_delta']:.2f} und Bein 3 (Short Call) mit "
            f"Delta={chosen_row['call_delta']:.2f} einstellen - die Wings (Bein 1/4) "
            "sind hiervon unabhaengig und muessen separat gewaehlt werden."
        )
    else:
        st.caption(
            "Die Long-Richtung war hier gewaehlt - das entspraeche eher einem "
            "'Long Strangle' (Bein 2/3 auf 'long' statt 'short' stellen) statt eines "
            "klassischen Iron Condor."
        )


def _render_result() -> None:
    """Kopfkennzahlen, Clustering-Warnung, Screening-Tabelle, Detailansicht
    und die Liste aller gefundenen Analog-Tage fuer das zuletzt gelaufene
    Matching (aus st.session_state)."""
    result = st.session_state["analog_result"]

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        metric_box("Gewaehlter Tag", f"{result['query_date'].date()}", "neutral")
    with a2:
        metric_box(
            "Ziel-HV-Rang", f"{result['target_rank_pct']:.0f}%", "neutral",
            help_text="Fuer jeden Analog-Tag rollierend zurueckgerechnete IV, die an "
                      "genau diesem Tag exakt diesen Rang erzeugt haette.",
        )
    with a3:
        metric_box("Gefundene Analog-Tage", f"{result['n_analogs']:,}", "neutral")
    with a4:
        metric_box(
            "PCA-Komponenten", f"{result['n_pca_components']}", "neutral",
            help_text="Anzahl Hauptkomponenten, die den Aehnlichkeits-Raum aufspannen "
                      "(so viele wie fuer 80% erklaerte Varianz noetig).",
        )

    clustering = result["clustering"]
    if clustering["frac_within_gap"] >= 0.30:
        st.warning(
            f"⚠️ Clustering-Hinweis: {clustering['frac_within_gap']*100:.0f}% der Analog-Tage "
            f"liegen (chronologisch sortiert) hoechstens {clustering['gap_days_threshold']} "
            f"Kalendertage auseinander (Median-Abstand: {clustering['median_gap_days']:.0f} Tage). "
            "Vola-Regime dauern typischerweise mehrere Tage bis Wochen an - ein grosser Teil "
            "der 'unabhaengigen' Analog-Tage kann daher aus wenigen historischen Ereignissen "
            "stammen. Die Newey-West-Korrektur unten federt das ab, ersetzt aber keine gesunde "
            "Skepsis gegenueber der Trade-Anzahl."
        )

    grid_result = result["grid_result"]
    _render_grid_result(grid_result)
    if not grid_result.empty:
        _render_detail_view(result, grid_result)

    with st.expander(f"Alle gefundenen Analog-Tage anzeigen ({len(result['analogs'])})"):
        st.dataframe(
            result["analogs"].assign(analog_date=result["analogs"]["analog_date"].dt.date).sort_values("distance"),
            use_container_width=True, hide_index=True,
        )


def render(df: pd.DataFrame, daily_df: pd.DataFrame, has_resolution: bool) -> None:
    """Oeffentlicher Einstiegspunkt fuer diesen Modus - wird von dashboard.py
    nur aufgerufen, wenn 'Analog-Tage-Matching' oben aktiv ist."""
    st.header("Analog-Tage-Matching")
    st.caption(
        "Findet die K historisch aehnlichsten Handelstage zu einem gewaehlten Tag "
        "(PCA-Distanz ueber die taeglichen Features) und wertet auf genau diesen "
        "Tagen einen Short- und Long-Strangle aus. Das Bewertungsniveau ist ein "
        "vorgegebener HV-Rang, rollierend pro Analog-Tag in eine passende IV "
        "umgerechnet. Per Grid-Search wird daraus das robusteste Put-/Call-Delta-"
        "Paar fuer die Short-Beine (Bein 2/3) des Iron Condor gesucht - inkl. "
        "Train/Validierungs-Split gegen Mehrfachvergleich-Zufallstreffer."
    )

    training_table = load_training_table()
    if training_table.empty:
        st.warning(
            "Noch keine Trainingstabelle gefunden. Einmal im Terminal ausfuehren:\n\n"
            "`python src/build_dataset.py`\n\n"
            "und diese Seite danach neu laden."
        )
        return

    daily_df = daily_df.sort_values("date")
    valid_t_days = available_t_days(training_table)

    with st.sidebar:
        st.divider()
        st.header("Analog-Tage-Matching")
        p = _sidebar_params(daily_df, valid_t_days)

    if p["run"]:
        _run_matching(daily_df, training_table, p)

    if "analog_result" in st.session_state:
        _render_result()
    else:
        st.info("Parameter links einstellen und 'Analog-Matching ausfuehren' klicken.")