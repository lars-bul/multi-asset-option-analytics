"""
MODUS 3: Heutige Kette - symmetrische Abstands-Ansicht + Analog-Tage-Simulation.

TEIL A (unveraendert): zeigt die AKTUELLE SPX-Optionskette als EINE Tabelle,
je Ziel-ABSTAND vom Spot (in %) eine Zeile, Put/Call-Strike unabhaengig auf
den naechstgelegenen real vorhandenen Strike gerundet - siehe
chain_delta.build_symmetric_distance_table.

TEIL B (NEU): fuehrt fuer JEDE Zeile der Tabelle einen Analog-Tage-Lauf aus
(identisches Prinzip wie dashboard_analog.py) und haengt die Ergebnis-
Kennzahlen als zusaetzliche Spalten an. Wichtige Design-Entscheidungen
(mit dem Nutzer abgestimmt):

  1. Ziel-HV-Rang je Zeile = die in Teil A bereits berechnete Spalte
     'HV-Rang % (dieser Kombi)' - KEIN eigener Slider, jede Zeile hat ihren
     eigenen Rang.
  2. Put-/Call-Delta je Zeile = die 'Modell-Delta (flache HV)'-Spalten aus
     Teil A - NICHT die echten Marktdeltas (die waeren mit Skew behaftet
     und wuerden bei solve_strike_delta_consistent() einen anderen Strike
     treffen als in der Tabelle gezeigt).
  3. vol_surface = VolSurfaceConfig(use_vol_surface=False) - Skew fuer den
     Analog-Lauf explizit AUS, damit die Delta-Definition durchgaengig
     dieselbe (flache) bleibt wie in Punkt 2.
  4. T_days fuer evaluate_analog_set() MUSS einer der in der vorab gebauten
     training_table verfuegbaren Werte sein (siehe labels.py/build_dataset.py)
     - die tatsaechliche Restlaufzeit (dte_exact) wird auf den naechst-
     gelegenen verfuegbaren Wert GERUNDET (siehe available_t_days()).
  5. 'Heute' fuer die Analog-Suche: waehrend eines laufenden Handelstags
     enthaelt daily_df noch keinen abgeschlossenen Balken fuer heute (montags
     waere das sonst Freitag, 3 Tage alt). Es wird daher ein synthetischer
     Balken (open=high=low=close=heutiger Eroeffnungskurs) an eine KOPIE von
     daily_df angehaengt (siehe _build_daily_df_with_today()).
  6. Performance: die Aehnlichkeits-Raum-PCA und die Analog-Tage-Suche
     (find_analogs) haengen NICHT von Delta/Rang ab - werden daher NUR
     EINMAL pro Klick berechnet, nicht pro Zeile. Trotzdem bleibt die
     eigentliche Bewertung (Strike-Solve je Analog-Tag x Zeile) ein
     Python-Loop - bei vielen Zeilen x vielen Analog-Tagen kann das mehrere
     Minuten dauern, daher expliziter Button statt Auto-Run.
  7. Je Zeile: Kennzahlen auf dem KOMPLETTEN Analog-Set (n, Gewinnrate,
     Profit Factor, Ø norm. P&L, t-Stat, CI≠0, Perioden-Konsistenz X/6)
     PLUS Train/Val-Split-Robustheit (t-Stat Val, Robust-Flag) - identisches
     Kriterium wie leg_pair_grid_search() in analog_matching.py, aber ohne
     die dreifache Neuberechnung: evaluate_analog_set() wird nur EINMAL auf
     dem kompletten Set aufgerufen, Train/Val wird danach aus dem fertigen
     Ergebnis chronologisch aufgeteilt (mathematisch identisch, da jede
     Analog-Tag-Bewertung unabhaengig von den anderen ist).
  8. NEU - 'Erwarteter Gewinn Short' (in Punkten): Ø norm. P&L% ist eine
     relative Kennzahl (normiert auf die historische Analog-Entry-Praemie
     JEDES einzelnen Analog-Tages, siehe reprice_leg_pair() -
     pnl_norm = net_pnl / entry_premium). Fuer einen Vergleich ueber
     Zeilen/Strikes hinweg, der auf die HEUTIGE (reale) Praemiengroesse
     bezogen ist, wird diese relative Kennzahl mit der HEUTIGEN Entry-
     Praemie DIESER Zeile (mid_put + mid_call aus Teil A) multipliziert:

         erwarteter Gewinn Short (Pkt) = avg_pnl_norm_pct/100 * (mid_put+mid_call)

     avg_pnl_norm_pct wird hier IMMER aus Short-Sicht berechnet
     (summarize_analog_trades(..., "short")) - diese Spalte bewertet daher
     ausschliesslich das Short-Bein (die Praemien-Seite des Condors), NICHT
     die Wings. Fuer die Wing-/Long-Auswahl und v.a. fuer die STREUUNG/
     Konsistenz der KOMBINIERTEN Position (das eigentliche 'wie geradlinig
     laeuft die Kapitalkurve') den Iron-Condor-Ladder-Backtest verwenden
     (dashboard_ladder.py) - dort wird Short+Long GEMEINSAM pro Analog-/
     Handelstag bewertet und liefert eine echte, korrelationskonsistente
     Equity-Kurve (die Streuung der Kombination ist NICHT einfach die Summe
     der Einzel-Streuungen, da beide Beine am selben Tag korreliert sind).
  9. NEU - Auto-Load: die Kette wird beim ERSTEN Aufruf dieses Tabs
     automatisch geladen (kein Klick auf 'Heutige Kette laden' mehr noetig,
     siehe render()). Einmal geladene Daten (Kette, Distanz-Tabelle,
     Analog-Ergebnisse) liegen in st.session_state und bleiben daher auch
     nach einem Wechsel zu einem anderen Tab/Modus erhalten - Streamlit
     raeumt st.session_state NICHT auf, nur weil ein anderer Reiter aktiv
     ist (siehe dashboard.py-Docstring: es rendert zwar nur der aktive
     Modus, aber session_state ist global fuer die gesamte Browser-Sitzung).
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from black76 import VolSurfaceConfig
from fetch_today_chain import fetch_today_chain, fetch_today_spot, fetch_today_open
from chain_delta import ChainDeltaConfig, enrich_chain_with_delta, build_symmetric_distance_table
from analog_matching import (
    AnalogConfig, SimilaritySpaceConfig, hv_rank_bounds, fit_similarity_space,
    transform_query, find_analogs, evaluate_analog_set, summarize_analog_trades,
    available_t_days, _reliability_for_side,
)
from features import FeatureConfig, build_feature_matrix
from dashboard_common import load_training_table


def _build_daily_df_with_today(daily_df: pd.DataFrame):
    """Haengt einen synthetischen 'Heute'-Tagesbalken an eine KOPIE von
    daily_df an (open=high=low=close=heutiger Eroeffnungskurs), damit die
    Analog-Suche waehrend eines laufenden Handelstags nicht auf den letzten
    ABGESCHLOSSENEN Tag zurueckfaellt (z.B. montags auf Freitag, 3 Tage alt).
    Dieselbe open=high=low=close-Konvention wie fuer die interpolierten
    Shiller-Monatsdaten im Rest des Projekts (siehe dashboard_common.py).

    Faellt auf den letzten echten Schlusskurs zurueck (used_fallback=True),
    falls kein Eroeffnungskurs verfuegbar ist (vor Handelsbeginn/Wochenende).

    Rueckgabe: (daily_df_erweitert, query_date, used_fallback)."""
    today = pd.Timestamp(date.today())
    if (daily_df["date"] == today).any():
        return daily_df.copy(), today, False

    open_price = fetch_today_open()
    if open_price is None:
        return daily_df.copy(), daily_df["date"].max(), True

    new_row = pd.DataFrame([{
        "date": today, "open": open_price, "high": open_price,
        "low": open_price, "close": open_price,
    }])
    if "resolution" in daily_df.columns:
        new_row["resolution"] = "daily"
    out = pd.concat([daily_df, new_row], ignore_index=True).sort_values("date").reset_index(drop=True)
    return out, today, False


def _run_bulk_analog(table: pd.DataFrame, daily_df: pd.DataFrame,
                      chosen_dte_exact: float, r_free: float, dividend_yield: float,
                      hv_length: int, rank_lookback: int,
                      analog_max_distance: float, analog_min_gap: int) -> pd.DataFrame | None:
    """Fuehrt den Analog-Tage-Lauf fuer JEDE Zeile von 'table' aus (siehe
    Moduldokstring Punkt 1-8). Gibt None zurueck (mit st.error) bei
    Abbruch, sonst ein DataFrame mit einer Zeile je distance_pct."""
    training_table = load_training_table()
    if training_table.empty:
        st.error(
            "Keine Trainingstabelle gefunden - einmal im Terminal "
            "`python src/build_dataset.py` ausfuehren und die Seite neu laden."
        )
        return None

    valid_t_days = available_t_days(training_table)
    T_days_used = min(valid_t_days, key=lambda t: abs(t - chosen_dte_exact))
    if abs(T_days_used - chosen_dte_exact) > 1.0:
        st.info(
            f"Restlaufzeit {chosen_dte_exact:.2f} Tage wird fuer die Analog-Bewertung "
            f"auf das naechstgelegene verfuegbare T_days={T_days_used} gerundet "
            f"(verfuegbar: {valid_t_days})."
        )

    daily_df_today, query_date, used_fallback = _build_daily_df_with_today(daily_df)
    if used_fallback:
        st.warning(
            f"Kein heutiger Eroeffnungskurs verfuegbar (vor Handelsbeginn/Wochenende?) "
            f"- Analog-Suche faellt auf den letzten echten Handelstag "
            f"({pd.Timestamp(query_date).date()}) zurueck."
        )

    with st.spinner("Baue Aehnlichkeits-Raum und suche Analog-Tage (einmalig fuer alle Zeilen)..."):
        try:
            feature_df = build_feature_matrix(daily_df_today, FeatureConfig())
            if query_date not in set(feature_df["date"]):
                st.error(
                    f"{pd.Timestamp(query_date).date()} hat keine vollstaendigen "
                    "Features (Warmup-Fenster noch nicht voll?)."
                )
                return None
            space = fit_similarity_space(
                feature_df, SimilaritySpaceConfig(), max_date=query_date + pd.Timedelta(days=1),
            )
            query_row = feature_df.loc[feature_df["date"] == query_date].iloc[0]
            query_score = transform_query(space, query_row)
            analogs = find_analogs(
                query_date, query_score, space,
                AnalogConfig(max_distance=analog_max_distance, min_gap_days=analog_min_gap),
            )
            bounds = hv_rank_bounds(daily_df_today, hv_length, rank_lookback)
        except ValueError as e:
            st.error(f"Analog-Suche fehlgeschlagen: {e}")
            return None

    st.caption(f"{len(analogs)} Analog-Tage gefunden (einmalig ermittelt, gilt fuer alle Zeilen unten).")

    # Skew fuer den Analog-Lauf explizit AUS (siehe Moduldokstring Punkt 3) -
    # damit die Delta-Definition dieselbe (flache) bleibt wie in den
    # Modell-Delta-Spalten von Teil A.
    vol_surface = VolSurfaceConfig(use_vol_surface=False)

    progress = st.progress(0.0)
    status = st.empty()
    results = []
    n = len(table)
    for i, row in enumerate(table.itertuples()):
        status.text(f"Analog-Lauf {i + 1}/{n} (Ziel-Abstand {row.distance_pct:g}%)...")
        progress.progress((i + 1) / n)

        put_delta = row.model_delta_put_pct / 100.0 if pd.notna(row.model_delta_put_pct) else None
        call_delta = row.model_delta_call_pct / 100.0 if pd.notna(row.model_delta_call_pct) else None
        today_rank = row.hv_rank_pct / 100.0 if pd.notna(row.hv_rank_pct) else None

        # NEU (siehe Moduldokstring Punkt 8): heutige (reale) Entry-Praemie
        # DIESER Zeile - Grundlage fuer die 'erwarteter Gewinn'-Spalten
        # unten. Kann NaN sein, falls put_row/call_row in Teil A fuer diese
        # Zeile nicht gefunden wurden (siehe build_symmetric_distance_table) -
        # dann bleiben auch die erwarteter-Gewinn-Spalten NaN.
        entry_premium_today = None
        if pd.notna(row.mid_put) and pd.notna(row.mid_call):
            entry_premium_today = float(row.mid_put) + float(row.mid_call)

        # GUARD (Fix): bei sehr weiten Zielabstaenden (siehe Distanz-Raster
        # in der Sidebar, per Default bis 40%) kann das reale Modell-Delta
        # numerisch auf EXAKT 0.0 runden (echte Zahl, kein None - der obige
        # None-Check greift hier also NICHT). Ein Ziel-Delta von exakt 0
        # entspricht rechnerisch einem Strike bei unendlich -
        # solve_strike_delta_consistent()/leg_price() liefern dafuer keine
        # sinnvolle Loesung mehr (Absturz mit 'ValueError: expected a
        # positive input, got 0.0' tief in black76.py). Solche Zeilen daher
        # genauso behandeln wie die bereits vorhandenen None-Faelle -
        # ueberspringen statt den kompletten Bulk-Lauf abzubrechen.
        if (put_delta is None or call_delta is None or today_rank is None
                or put_delta <= 0.0 or call_delta <= 0.0):
            results.append({
                "distance_pct": row.distance_pct, "analog_n": None,
                "entry_premium_today": entry_premium_today,
                "expected_profit_short": None,
            })
            continue

        # ZUSAETZLICHER GUARD: selbst mit obigem Filter kann eine einzelne
        # Zeile aus einem anderen numerischen Grenzfall heraus fehlschlagen
        # (z.B. ein einzelner entarteter Analog-Tag). Ein try/except HIER
        # (statt den Fehler bis dashboard.py durchschlagen zu lassen) sorgt
        # dafuer, dass nur DIESE Zeile als 'n/a' markiert wird und der
        # Bulk-Lauf fuer alle anderen Zeilen trotzdem fertig laeuft.
        try:
            full_eval = evaluate_analog_set(
                analogs, training_table, bounds, T_days_used, put_delta, call_delta,
                r_free, dividend_yield, vol_surface, vol_mode="rank", today_rank=today_rank,
            )
        except (ValueError, ZeroDivisionError) as e:
            st.caption(
                f"⚠️ Ziel-Abstand {row.distance_pct:g}%: Bewertung numerisch "
                f"fehlgeschlagen ({e}) - Zeile wird als n/a markiert, Rest laeuft weiter."
            )
            results.append({
                "distance_pct": row.distance_pct, "analog_n": None,
                "entry_premium_today": entry_premium_today,
                "expected_profit_short": None,
            })
            continue

        if full_eval.empty:
            results.append({
                "distance_pct": row.distance_pct, "analog_n": 0,
                "entry_premium_today": entry_premium_today,
                "expected_profit_short": None,
            })
            continue

        trade_summary = summarize_analog_trades(full_eval, "short")
        rel_full = _reliability_for_side(full_eval, "short", 6)

        median_date = full_eval["entry_date"].median()
        train_eval = full_eval[full_eval["entry_date"] < median_date]
        val_eval = full_eval[full_eval["entry_date"] >= median_date]
        train_rel = _reliability_for_side(train_eval, "short", 4)
        val_rel = _reliability_for_side(val_eval, "short", 4)

        robust = False
        if train_rel["n_trades"] > 0 and val_rel["n_trades"] > 0:
            robust = bool(
                train_rel["ci_excludes_zero"] and val_rel["ci_excludes_zero"]
                and np.sign(train_rel["mean_pnl_norm"]) == np.sign(val_rel["mean_pnl_norm"])
                and np.sign(train_rel["mean_pnl_norm"]) > 0.0
            )

        # NEU (siehe Moduldokstring Punkt 8): avg_pnl_norm_pct ist IMMER aus
        # Short-Sicht (summarize_analog_trades(full_eval, "short")) - mit
        # der heutigen Entry-Praemie DIESER Zeile multipliziert ergibt das
        # den erwarteten Gewinn in PUNKTEN, vergleichbar ueber Zeilen
        # hinweg (statt der relativen %-Kennzahl, die auf die historische
        # Analog-Praemie JEDES einzelnen Tages normiert ist). Fuer 'Long'
        # wird das Vorzeichen gedreht (pnl_norm_long = -pnl_norm_short exakt,
        # siehe reprice_leg_pair()-Docstring) - keine Neuberechnung noetig.
        expected_profit_short = None
        if entry_premium_today is not None and pd.notna(trade_summary["avg_pnl_norm_pct"]):
            avg_pnl_norm_frac = trade_summary["avg_pnl_norm_pct"] / 100.0
            expected_profit_short = avg_pnl_norm_frac * entry_premium_today

        results.append({
            "distance_pct": row.distance_pct,
            "analog_n": trade_summary["n_trades"],
            "win_rate_pct": trade_summary["win_rate_pct"],
            "profit_factor": trade_summary["profit_factor"],
            "avg_pnl_norm_pct": trade_summary["avg_pnl_norm_pct"],
            "entry_premium_today": entry_premium_today,
            "expected_profit_short": expected_profit_short,
            "tstat_full": rel_full["tstat"],
            "ci_excludes_zero_full": rel_full["ci_excludes_zero"],
            "subperiod_matching": rel_full["subperiod_matching"],
            "subperiod_nonempty": rel_full["subperiod_nonempty"],
            "tstat_val": val_rel["tstat"],
            "robust_train_val": robust,
        })

    status.empty()
    progress.empty()

    out = pd.DataFrame(results)
    out["T_days_used"] = T_days_used
    out["n_analogs_total"] = len(analogs)
    return out


def render(df, daily_df, has_resolution) -> None:
    st.header("Heutige Kette - symmetrische Abstands-Ansicht")
    st.caption(
        "Laedt die aktuelle SPX-Kette (Yahoo, ~15 Min. verzoegert). Put- und "
        "Call-Strike werden je Zeile unabhaengig auf den naechstgelegenen realen "
        "Strike im gleichen prozentualen Abstand vom Spot gerundet - dadurch "
        "IMMER symmetrisch um den Spot. Die HV-Periode fuer Modell-Delta und "
        "HV-Rang wird automatisch an die gewaehlte Restlaufzeit gekoppelt "
        "(z.B. 7.5 Tage Restlaufzeit -> HV(7.5)-Fenster)."
    )

    with st.sidebar:
        st.divider()
        st.header("Heutige Kette")
        min_dte = st.number_input("Min. DTE (fraktional)", 0.0, 365.0, 0.0, 0.5)
        max_dte = st.number_input("Max. DTE (fraktional)", 1.0, 365.0, 90.0, 1.0)
        min_abs_delta = st.number_input(
            "Minimales |Delta| (Filter, echtes Marktdelta)", 0.0001, 0.5, 0.001, 0.001, format="%.4f",
            help="ACHTUNG: dieser Filter wird VOR dem Tabellenbau angewendet und "
                 "kappt damit effektiv, wie weit die Ziel-Abstaende (%) tatsaechlich "
                 "reichen koennen. Fuer die Abstands-Ansicht eher klein wählen "
                 "(Default 0.001 = 0.1%).",
        )
        st.caption("Abstands-Raster (Ziel-% vom Spot) fuer die Zeilen:")
        distance_grid_input = st.text_input(
            "Ziel-Abstaende (%, kommagetrennt)",
            "0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.5,4,5,6,8,10",
        )
        rank_lookback = st.number_input(
            "Lookback fuer HV-Min/Max-Rangfenster (Tage)", 30, 5000, 252,
            help="Laenge des rollierenden Fensters, in dem hv_min/hv_max fuer den "
                 "HV-Rang der Kombi-IV gesucht wird - identisches Prinzip zu "
                 "backtest.FixedRankConfig.lookback_days.",
        )
        r_free = st.number_input("Risikoloser Zins", 0.0, 0.20, 0.0, 0.005, format="%.3f")
        dividend_yield = st.number_input("Dividendenrendite", 0.0, 0.10, 0.015, 0.001, format="%.3f")
        reload_chain = st.button(
            "Heutige Kette neu laden", type="primary",
            help="Beim ERSTEN Oeffnen dieses Tabs wird die Kette automatisch geladen "
                 "(siehe unten) - dieser Button ist nur fuer ein manuelles Neuladen "
                 "noetig, z.B. nach Aenderung von DTE-Filter/Delta-Filter oder um "
                 "einen aktuelleren Snapshot zu holen.",
        )

    # AUTO-LOAD (Fix): frueher musste 'Heutige Kette laden' bei JEDEM
    # Tab-Wechsel erneut geklickt werden, bevor ueberhaupt etwas zu sehen
    # war - das wirkte wie 'alles verschwindet beim Tab-Wechsel'. Jetzt wird
    # beim ALLERERSTEN Aufruf dieses Tabs (kein 'livechain_enriched' in
    # st.session_state) automatisch geladen, mit den aktuellen (Default-)
    # Sidebar-Werten - kein Klick noetig. Ist einmal geladen, bleibt der
    # Stand in st.session_state erhalten (siehe Modul-Docstring: Streamlit
    # behaelt session_state ueber Tab-/Modus-Wechsel hinweg bei) - ein
    # spaeterer Tab-Wechsel loescht NICHTS mehr. Der Button oben dient nur
    # noch dem BEWUSSTEN manuellen Neuladen (z.B. nach Parameteraenderung
    # oder fuer einen frischeren Snapshot).
    needs_initial_load = "livechain_enriched" not in st.session_state
    if reload_chain or needs_initial_load:
        with st.spinner("Lade heutige Kette und berechne Delta/IV je Serie..."):
            try:
                spot = fetch_today_spot()
                chain = fetch_today_chain(min_dte=float(min_dte), max_dte=float(max_dte))
                enriched = enrich_chain_with_delta(
                    chain, spot,
                    ChainDeltaConfig(min_abs_delta=float(min_abs_delta),
                                      r_free=float(r_free), dividend_yield=float(dividend_yield)),
                )
            except Exception as e:
                st.error(f"Laden fehlgeschlagen: {e}")
                if needs_initial_load:
                    return
                # Bei einem manuellen Neuladen NICHT den ganzen Tab abbrechen -
                # der vorherige (funktionierende) Stand in st.session_state
                # bleibt dann einfach unveraendert sichtbar.
            else:
                st.session_state["livechain_spot"] = spot
                st.session_state["livechain_enriched"] = enriched
                st.session_state["livechain_r_free"] = float(r_free)
                st.session_state["livechain_dividend_yield"] = float(dividend_yield)
                st.session_state["livechain_rank_lookback"] = int(rank_lookback)

    if "livechain_enriched" not in st.session_state:
        st.info("Kette konnte noch nicht geladen werden - siehe Fehlermeldung oben.")
        return

    spot = st.session_state["livechain_spot"]
    enriched = st.session_state["livechain_enriched"]
    r_free = st.session_state["livechain_r_free"]
    dividend_yield = st.session_state["livechain_dividend_yield"]
    rank_lookback = st.session_state["livechain_rank_lookback"]
    st.metric("Aktueller Spot (SPX)", f"{spot:,.2f}")

    if enriched.empty:
        st.warning(
            "Keine Serien nach dem Delta-Filter uebrig - min. |Delta| verringern "
            "oder DTE-Fenster pruefen."
        )
        return

    if "dte_exact" not in enriched.columns:
        st.warning(
            "Spalte 'dte_exact' fehlt - alte fetch_today_chain.py? App neu starten "
            "und 'Heutige Kette laden' erneut klicken. Bis dahin Fallback auf 'dte'."
        )
        enriched = enriched.copy()
        enriched["dte_exact"] = enriched["dte"].astype(float)

    # ── Expiry-Auswahl ZUERST (Datum + exakte fraktionale Resttage) - die
    # HV-Periode haengt davon ab, muss also DANACH berechnet werden. ──
    expiry_info = (
        enriched[["expiry", "dte", "dte_exact"]]
        .drop_duplicates()
        .sort_values("expiry")
    )
    expiry_labels = [
        f"{pd.Timestamp(row.expiry).date().isoformat()}  (Restlaufzeit: {row.dte_exact:.1f} Tage)"
        for row in expiry_info.itertuples()
    ]
    chosen_label = st.selectbox("Expiry", expiry_labels)
    chosen_expiry = expiry_info.iloc[expiry_labels.index(chosen_label)]["expiry"]
    chosen_dte_exact = float(expiry_info.iloc[expiry_labels.index(chosen_label)]["dte_exact"])
    T_years = chosen_dte_exact / 365.0
    st.caption(
        f"Exakte Restlaufzeit: **{chosen_dte_exact:.2f} Tage** (Annahme "
        "PM-Settlement 16:00 ET - bei klassischen monatlichen AM-settled "
        "Serien ca. 6.5h zu hoch)."
    )

    # ── HV-Periode IMMER = Restlaufzeit (auf ganze Handelstage gerundet,
    # da compute_hv()/hv_rank_bounds() ein ganzzahliges Rolling-Fenster
    # brauchen - die fraktionale Restlaufzeit selbst geht weiterhin exakt
    # in T_years fuer Black-76 ein, siehe Moduldokstring). ──
    hv_length = max(2, round(chosen_dte_exact))
    st.caption(
        f"→ HV-Periode automatisch auf **{hv_length} Handelstage** gesetzt "
        f"(gerundete Restlaufzeit von {chosen_dte_exact:.2f} Tagen)."
    )

    bounds = hv_rank_bounds(daily_df, hv_length, int(rank_lookback))
    if bounds.empty or bounds["hv"].dropna().empty:
        st.error(
            f"HV({hv_length}) konnte nicht berechnet werden - zu wenig Historie "
            "in daily_df."
        )
        return
    today_bounds = bounds.dropna(subset=["hv"]).iloc[-1]
    flat_hv = float(today_bounds["hv"])
    hv_min = float(today_bounds["hv_min"]) if pd.notna(today_bounds["hv_min"]) else None
    hv_max = float(today_bounds["hv_max"]) if pd.notna(today_bounds["hv_max"]) else None

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(f"Flache HV({hv_length}d, = Restlaufzeit)", f"{flat_hv*100:.2f}%")
    with c2:
        st.metric("HV-Min im Rangfenster", f"{hv_min*100:.2f}%" if hv_min is not None else "n/a")
    with c3:
        st.metric("HV-Max im Rangfenster", f"{hv_max*100:.2f}%" if hv_max is not None else "n/a")
    if hv_min is None or hv_max is None or hv_max <= hv_min:
        st.warning(
            "HV-Min/Max fuer das Rangfenster nicht verfuegbar (zu wenig Historie im "
            "Lookback) - hv_rank_pct wird in der Tabelle 'n/a' bleiben. Lookback "
            "verkleinern oder mehr Kurshistorie laden."
        )

    try:
        distance_targets_pct = [float(x.strip()) for x in distance_grid_input.split(",") if x.strip()]
    except ValueError:
        st.error("Ziel-Abstaende konnten nicht geparst werden - Format z.B. '0.5,1,2,5,10'.")
        return

    table = build_symmetric_distance_table(
        enriched, chosen_expiry, spot, flat_hv, T_years,
        r_free, dividend_yield, distance_targets_pct, hv_min, hv_max,
    )

    # Grenzbereich erkennen: wiederholt sich strike_put/strike_call gegenueber
    # der VORHERIGEN Zeile (Distanz-Grid ist aufsteigend sortiert), ist die
    # Kette (nach Filter) an dieser Stelle "ausgegangen".
    put_stuck = table["strike_put"].astype(float).diff().eq(0)
    call_stuck = table["strike_call"].astype(float).diff().eq(0)
    if put_stuck.any() or call_stuck.any():
        stuck_from_put = table.loc[put_stuck, "distance_pct"].min() if put_stuck.any() else None
        stuck_from_call = table.loc[call_stuck, "distance_pct"].min() if call_stuck.any() else None
        parts = []
        if stuck_from_put is not None:
            parts.append(f"Put ab Ziel-Abstand {stuck_from_put:g}%")
        if stuck_from_call is not None:
            parts.append(f"Call ab Ziel-Abstand {stuck_from_call:g}%")
        st.warning(
            "Randbereich erreicht - Zeilen wiederholen den letzten verfuegbaren "
            "Strike: " + " und ".join(parts) + ". Ursache meist der 'Minimales "
            "|Delta|'-Filter in der Sidebar - dort verkleinern, um weiter OTM zu kommen."
        )

    show = table.rename(columns={
        "distance_pct": "Ziel-Abstand % v. Spot",
        "model_delta_put_pct": "Δ Put % (Modell, flache HV)",
        "model_delta_call_pct": "Δ Call % (Modell, flache HV)",
        "iv_pair_pct": "IV Kombi % (Markt, Paket-Solve)",
        "hv_rank_pct": "HV-Rang % (dieser Kombi)",
        "strike_put": "Strike Put (real)", "strike_call": "Strike Call (real)",
        "mid_put": "Mid Put", "mid_call": "Mid Call",
        "iv_put_pct": "IV Put % (echt)", "iv_call_pct": "IV Call % (echt)",
        "delta_put_pct": "Δ Put % (echt)", "delta_call_pct": "Δ Call % (echt)",
    })[[
        "Ziel-Abstand % v. Spot",
        "Δ Put % (Modell, flache HV)", "Δ Call % (Modell, flache HV)",
        "IV Kombi % (Markt, Paket-Solve)", "HV-Rang % (dieser Kombi)",
        "Strike Put (real)", "Strike Call (real)",
        "Mid Put", "Mid Call",
        "IV Put % (echt)", "IV Call % (echt)",
        "Δ Put % (echt)", "Δ Call % (echt)",
    ]]
    st.caption(
        "'Ziel-Δ' ist die Suchvorgabe (Zeilenlabel). 'Δ Put/Call (echt)' sind die "
        "TATSAECHLICHEN, aus dem realen Mid-Preis per Black-76 zurueckgerechneten "
        "Deltas der jeweils gewaehlten Option - koennen leicht vom Ziel abweichen, "
        "da nur reale (diskrete) Strikes existieren. 'IV Kombi' = die EINE Vola, "
        "die - in Black-76 fuer Put+Call GEMEINSAM eingesetzt - exakt die Summe "
        "beider echter Mid-Preise reproduziert (kein Mittelwert der Einzel-IVs). "
        "'HV-Rang' = wo diese Kombi-IV im HV-Min/Max-Fenster oben liegt."
    )
    st.dataframe(
        show.style.format({
            "Ziel-Abstand % v. Spot": "{:.2f}",
            "Strike Put (real)": "{:.0f}", "Strike Call (real)": "{:.0f}",
            "Mid Put": "{:.2f}", "Mid Call": "{:.2f}",
            "IV Put % (echt)": "{:.2f}", "IV Call % (echt)": "{:.2f}",
            "Δ Put % (echt)": "{:.2f}", "Δ Call % (echt)": "{:.2f}",
            "Δ Put % (Modell, flache HV)": "{:.2f}", "Δ Call % (Modell, flache HV)": "{:.2f}",
            "IV Kombi % (Markt, Paket-Solve)": "{:.2f}", "HV-Rang % (dieser Kombi)": "{:.1f}",
        }, na_rep="n/a"),
        use_container_width=True, hide_index=True,
    )

    st.session_state["livechain_table"] = table
    st.session_state["livechain_chosen_expiry"] = chosen_expiry
    st.session_state["livechain_chosen_dte_exact"] = chosen_dte_exact
    st.session_state["livechain_flat_hv"] = flat_hv
    st.session_state["livechain_hv_min"] = hv_min
    st.session_state["livechain_hv_max"] = hv_max

    with st.expander("Rohdaten dieser Expiry (alle Serien nach Delta-Filter)"):
        st.dataframe(
            enriched[enriched["expiry"] == chosen_expiry][
                ["strike", "option_type", "bid", "ask", "mid", "implied_vol_mkt", "delta_mkt"]
            ].sort_values(["option_type", "strike"]),
            use_container_width=True, hide_index=True,
        )

    # ── TEIL B: Analog-Tage-Simulation je Zeile (siehe Moduldokstring) ──
    st.divider()
    st.subheader("Analog-Tage-Simulation je Zeile")
    st.caption(
        "Fuehrt fuer JEDE Zeile oben (= reale Delta-Kombination) einen Analog-Tage-"
        "Lauf aus - Ziel-HV-Rang je Zeile = 'HV-Rang % (dieser Kombi)' aus der "
        "Tabelle oben, Put-/Call-Delta = die Modell-Delta-Spalten (flache HV, "
        "Skew AUS). Analog-Tage werden NUR EINMAL gesucht (identisch fuer alle "
        "Zeilen), danach je Zeile neu bewertet. Bei vielen Analog-Tagen x vielen "
        "Zeilen kann das mehrere Minuten dauern - Fortschrittsbalken zeigt den Stand."
    )
    st.caption(
        "'Erw. Gewinn Short' (Punkte) = Ø norm. P&L% dieser Zeile multipliziert mit "
        "der HEUTIGEN Entry-Praemie dieser Zeile (Mid Put + Mid Call oben) - damit "
        "auf die reale, heutige Praemiengroesse bezogen statt nur relativ zur "
        "historischen Analog-Praemie. Bewertet NUR das Short-Bein (Praemien-Seite), "
        "NICHT die Wings. Fuer die Wing-Auswahl und vor allem fuer die echte "
        "kombinierte Equity-Kurve (Short+Long GEMEINSAM, inkl. Korrelation zwischen "
        "den Beinen) den Iron-Condor-Ladder-Backtest verwenden."
    )
    ac1, ac2 = st.columns(2)
    with ac1:
        analog_max_distance = st.number_input(
            "Maximale Ähnlichkeitsdistanz", min_value=0.01, max_value=100.0, value=3.0, step=0.1,
        )
    with ac2:
        analog_min_gap = st.number_input(
            "Mindestabstand zu heute (Handelstage)", min_value=0, max_value=60, value=5,
        )

    run_bulk = st.button("Analog-Läufe für alle Zeilen starten")
    if run_bulk:
        analog_results = _run_bulk_analog(
            table, daily_df, chosen_dte_exact, r_free, dividend_yield,
            hv_length, int(rank_lookback), float(analog_max_distance), int(analog_min_gap),
        )
        if analog_results is not None:
            st.session_state["livechain_analog_results"] = analog_results
            st.session_state["livechain_analog_key"] = (chosen_expiry, tuple(distance_targets_pct))

    stored = st.session_state.get("livechain_analog_results")
    stored_key = st.session_state.get("livechain_analog_key")
    if stored is not None and stored_key == (chosen_expiry, tuple(distance_targets_pct)):
        merged = table.merge(stored, on="distance_pct", how="left")
        show2 = merged.rename(columns={
            "distance_pct": "Ziel-Abstand % v. Spot",
            "analog_n": "Analog-Trades (n)",
            "win_rate_pct": "Gewinnrate %",
            "profit_factor": "Profit Factor",
            "avg_pnl_norm_pct": "Ø norm. P&L %",
            "entry_premium_today": "Entry-Prämie heute (Pkt)",
            "expected_profit_short": "Erw. Gewinn Short (Pkt)",
            "tstat_full": "t-Stat (gesamt)",
            "ci_excludes_zero_full": "CI ≠ 0 (gesamt)",
            "subperiod_matching": "Perioden-Match",
            "subperiod_nonempty": "Perioden gesamt",
            "tstat_val": "t-Stat (Val)",
            "robust_train_val": "Robust (Train+Val)",
        })[[
            "Ziel-Abstand % v. Spot", "Analog-Trades (n)", "Gewinnrate %", "Profit Factor",
            "Ø norm. P&L %", "Entry-Prämie heute (Pkt)",
            "Erw. Gewinn Short (Pkt)",
            "t-Stat (gesamt)", "CI ≠ 0 (gesamt)",
            "Perioden-Match", "Perioden gesamt", "t-Stat (Val)", "Robust (Train+Val)",
        ]]
        st.caption(
            f"T_days fuer die Analog-Bewertung: "
            f"{int(stored['T_days_used'].iloc[0]) if len(stored) else 'n/a'} "
            f"(gerundet von {chosen_dte_exact:.2f} Tagen Restlaufzeit) · "
            f"{int(stored['n_analogs_total'].iloc[0]) if len(stored) else 0} Analog-Tage gefunden. "
            "'Robust (Train+Val)' = Konfidenzintervall schliesst 0 aus UND gleiches "
            "positives Vorzeichen in BEIDEN chronologischen Haelften des Analog-Sets."
        )
        st.dataframe(
            show2.style.format({
                "Ziel-Abstand % v. Spot": "{:.2f}",
                "Analog-Trades (n)": "{:.0f}",
                "Gewinnrate %": "{:.1f}",
                "Profit Factor": "{:.2f}",
                "Ø norm. P&L %": "{:+.2f}",
                "Entry-Prämie heute (Pkt)": "{:.2f}",
                "Erw. Gewinn Short (Pkt)": "{:+.2f}",
                "t-Stat (gesamt)": "{:.2f}",
                "Perioden-Match": "{:.0f}",
                "Perioden gesamt": "{:.0f}",
                "t-Stat (Val)": "{:.2f}",
            }, na_rep="n/a"),
            use_container_width=True, hide_index=True,
        )
    elif stored is not None:
        st.info("Tabelle/Expiry hat sich seit dem letzten Analog-Lauf geaendert - bitte erneut starten.")