"""
MODUS 3: Heutige Kette - symmetrische Abstands-Ansicht + Analog-Tage-Simulation.

TEIL A (unveraendert): zeigt die AKTUELLE SPX-Optionskette als EINE Tabelle,
je Ziel-ABSTAND vom Spot (in %) eine Zeile, Put/Call-Strike unabhaengig auf
den naechstgelegenen real vorhandenen Strike gerundet - siehe
chain_delta.build_symmetric_distance_table.

TEIL B: fuehrt fuer JEDE Zeile der Tabelle einen Analog-Tage-Lauf aus
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
  8. 'Erwarteter Gewinn Short' (in Punkten): Ø norm. P&L% ist eine
     relative Kennzahl (normiert auf die historische Analog-Entry-Praemie
     JEDES einzelnen Analog-Tages, siehe reprice_leg_pair() -
     pnl_norm = net_pnl / entry_premium). Fuer einen Vergleich ueber
     Zeilen/Strikes hinweg, der auf die HEUTIGE (reale) Praemiengroesse
     bezogen ist, wird diese relative Kennzahl mit der HEUTIGEN Entry-
     Praemie DIESER Zeile (mid_put + mid_call aus Teil A) multipliziert:

         erwarteter Gewinn Short (Pkt) = avg_pnl_norm_pct/100 * (mid_put+mid_call)

     avg_pnl_norm_pct wird hier IMMER aus Short-Sicht berechnet
     (summarize_analog_trades_weighted(..., "short")) - diese Spalte
     bewertet daher ausschliesslich das Short-Bein (die Praemien-Seite des
     Condors), NICHT die Wings. Fuer die Wing-/Long-Auswahl und v.a. fuer
     die STREUUNG/Konsistenz der KOMBINIERTEN Position (das eigentliche
     'wie geradlinig laeuft die Kapitalkurve') den Iron-Condor-Ladder-
     Backtest verwenden (dashboard_ladder.py) - dort wird Short+Long
     GEMEINSAM pro Analog-/Handelstag bewertet und liefert eine echte,
     korrelationskonsistente Equity-Kurve.
  9. Auto-Load: die Kette wird beim ERSTEN Aufruf dieses Tabs automatisch
     geladen (kein Klick auf 'Heutige Kette laden' mehr noetig, siehe
     render()). Einmal geladene Daten (Kette, Distanz-Tabelle, Analog-
     Ergebnisse) liegen in st.session_state und bleiben daher auch nach
     einem Wechsel zu einem anderen Tab/Modus erhalten.
  10. Staleness-Diagnose je Zeile (Quote-Alter Put/Call): siehe
      chain_delta.py-Moduldokstring - dort wird bereits `quote_age_minutes`
      je Einzelserie und `quote_age_put_min`/`quote_age_call_min` je Zeile
      der Distanz-Tabelle berechnet. Ab einem Schwellenwert
      (_QUOTE_AGE_WARNING_MINUTES) wird eine explizite Warnung mit den
      betroffenen Zeilen angezeigt.
  11. 'Preis fuer PF=3' (Punkte): beantwortet die Frage "Welcher
      (einheitliche) Preis waere noetig gewesen, damit sich auf dem
      KOMPLETTEN Analog-Set (Short-Seite), ausgewertet mit den ECHTEN,
      heute in der Optionskette sichtbaren Strikes dieser Zeile, ein
      Profit Factor von genau 3 ergeben haette?" - siehe
      _payout_at_todays_scale()/_required_premium_for_profit_factor() fuer
      die Skalen-Korrektur (prozentuale statt absolute Kursbewegung).

  12. NEU - Kontinuierliche Kernel-Gewichtung statt hartem max_distance-
      Cutoff (siehe analog_matching.py, Punkt 12 im dortigen Moduldokstring
      fuer die vollstaendige Begruendung). AUSLOESER: in ausgepraegten
      Krisenphasen (Marktzustand liegt im Aehnlichkeits-Raum sehr
      ungewoehnlich, es gibt keine nahen historischen Entsprechungen) fiel
      find_analogs() bei der bisherigen harten Schwelle (dist <= max_distance)
      haeufig komplett leer aus - eine ValueError brach den GESAMTEN
      Bulk-Lauf ab, und die Tabelle 'Analog-Tage-Simulation je Zeile' blieb
      unsichtbar/leer, obwohl es durchaus (wenn auch weit entfernte)
      Vergleichstage gab.

      FIX: _run_bulk_analog() ruft find_analogs() jetzt mit
      use_kernel_weighting=True auf. Jeder Kandidat bekommt ein STETIGES
      Gewicht exp(-(distance/bandwidth)**2) statt einer Ja/Nein-Mitgliedschaft,
      UND es werden per AnalogConfig.min_candidates IMMER mindestens ein paar
      der naechstgelegenen Tage aufgenommen, egal wie weit sie tatsaechlich
      entfernt sind. Dadurch laedt die Tabelle jetzt auch in einer
      Krisenphase mit nur wenigen/weit entfernten Vergleichstagen zuverlaessig -
      die (dann sehr geringe) Verlaesslichkeit dieser Zeile zeigt sich
      stattdessen TRANSPARENT in der neuen Spalte 'Effektive n (Kish)'
      (siehe analog_matching.effective_sample_size()): faellt n_eff deutlich
      unter die rohe Trade-Zahl, steckt kaum echte Vergleichsinformation
      dahinter - die Zeile ist dann zwar sichtbar, aber mit Vorsicht zu
      geniessen, statt komplett zu fehlen. Alle Kennzahlen (Gewinnrate,
      Profit Factor, Ø norm. P&L, t-Stat, 'Preis fuer PF=3', ...) werden
      entsprechend GEWICHTET berechnet (siehe
      analog_matching.summarize_analog_trades_weighted()/
      weighted_reliability_summary() sowie die lokal angepassten
      _payout_at_todays_scale()/_required_premium_for_profit_factor()).

      Modus 2 (dashboard_analog.py) ist von dieser Aenderung NICHT betroffen
      (nutzt weiterhin use_kernel_weighting=False, siehe dort).

  13. NEU - Zeilenfaerbung der Analog-Tabelle (gruen/rot) nach
      Robustheitskriterien + Strike-Spalten (real) direkt hinter
      'Ziel-Abstand % v. Spot'.
        gruen: t-Stat (gesamt) > 2, CI ≠ 0 (gesamt), Perioden-Match >= 5/6,
               t-Stat (Val) > 2, Robust (Train+Val) == True.
        rot:   spiegelbildlich - t-Stat (gesamt) < -2, CI ≠ 0 (gesamt),
               Perioden-Match >= 5/6, t-Stat (Val) < -2 und ein GESPIEGELTES
               Robust-Flag (robust_train_val_neg: beide Haelften signifikant,
               gleiches NEGATIVES Vorzeichen). Noetig, weil 'robust_train_val'
               per Definition nur positive Edges erfasst (Vorzeichen > 0) und
               bei einer negativen Zeile daher immer False waere.
      Alle anderen Zeilen bleiben ungefaerbt. Die sichtbare Spalte
      'Robust (Train+Val)' bleibt unveraendert (nur positives Kriterium).
      Die Strike-Werte stammen aus 'strike_put'/'strike_call' der
      Distanz-Tabelle aus Teil A (dieselben realen Strikes, die im
      Analog-Lauf ausgewertet werden). Siehe _robustness_row_state().
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
    transform_query, find_analogs, evaluate_analog_set,
    summarize_analog_trades_weighted, weighted_reliability_summary,
    available_t_days,
)
from features import FeatureConfig, build_feature_matrix
from dashboard_common import load_training_table

# Schwellenwert (Minuten) fuer die Quote-Alter-Warnung in Teil A - bewusst
# als Modul-Konstante statt Sidebar-Eingabe, um die Sidebar nicht mit einem
# weiteren, selten geaenderten Parameter zu ueberladen. Bei Bedarf hier
# anpassen (siehe Moduldokstring Punkt 10).
_QUOTE_AGE_WARNING_MINUTES = 60.0

# NEU (siehe Moduldokstring Punkt 12) - Mindestanzahl an Analog-Tagen, die
# find_analogs() IMMER aufnimmt, unabhaengig von Distanz/Kernel-Gewicht.
# Verhindert einen kompletten Ausfall der Tabelle in Marktphasen ohne nahe
# historische Entsprechung (z.B. scharfe Krisen). Bewusst als Modul-Konstante
# statt Sidebar-Eingabe - selten geaendert, und ein zu kleiner Wert wuerde
# genau das Problem wieder einfuehren, das dieser Fix beheben soll.
_ANALOG_MIN_CANDIDATES = 20

# Schwellenwert fuer die 'geringe effektive Datenbasis'-Warnung unterhalb
# der Analog-Ergebnistabelle (siehe Moduldokstring Punkt 12) - orientiert
# sich an der bestehenden "< 30 Analog-Tage"-Warnung in dashboard_analog.py.
_LOW_N_EFF_WARNING = 10.0

# NEU (siehe Moduldokstring Punkt 13) - Schwellen/Farben fuer die
# Zeilenfaerbung der Analog-Tabelle.
_ROBUST_TSTAT_THRESHOLD = 2.0
_ROBUST_MIN_PERIODS_MATCH = 5   # mindestens 5 von 6 Teilperioden

_ROW_STYLE = {
    "green": "background-color: rgba(46, 160, 67, 0.30)",
    "red": "background-color: rgba(218, 54, 51, 0.30)",
}


def _robustness_row_state(df: pd.DataFrame) -> pd.Series:
    """Klassifiziert jede Zeile als 'green' (alle Robustheitskriterien positiv
    erfuellt), 'red' (alle spiegelbildlich negativ erfuellt) oder '' (Rest).

    green: t(gesamt) > 2, CI != 0, Perioden-Match >= 5, t(Val) > 2, Robust (Train+Val)
    red:   t(gesamt) < -2, CI != 0, Perioden-Match >= 5, t(Val) < -2,
           gespiegeltes Robust-Flag (beide Haelften signifikant, gleiches NEGATIVES Vorzeichen)

    NaN-Werte (uebersprungene Zeilen) ergeben bei jedem Vergleich False -> ''."""
    T = _ROBUST_TSTAT_THRESHOLD
    tstat_full = pd.to_numeric(df["tstat_full"], errors="coerce")
    tstat_val = pd.to_numeric(df["tstat_val"], errors="coerce")
    periods_ok = pd.to_numeric(df["subperiod_matching"], errors="coerce") >= _ROBUST_MIN_PERIODS_MATCH
    ci_ok = df["ci_excludes_zero_full"].eq(True)
    robust_pos = df["robust_train_val"].eq(True)
    # Fallback, falls Ergebnisse aus einem aelteren Lauf (ohne neue Spalte) in session_state liegen
    robust_neg = df.get("robust_train_val_neg", pd.Series(False, index=df.index)).eq(True)

    green = (tstat_full > T) & ci_ok & periods_ok & (tstat_val > T) & robust_pos
    red = (tstat_full < -T) & ci_ok & periods_ok & (tstat_val < -T) & robust_neg

    state = pd.Series("", index=df.index)
    state.loc[green] = "green"
    state.loc[red] = "red"
    return state


def _required_premium_for_profit_factor(payout: np.ndarray, weights: np.ndarray,
                                         target_pf: float = 3.0,
                                         max_iter: int = 60) -> float | None:
    """Bisektion (siehe Moduldokstring Punkt 11): gesuchter EINHEITLICHER
    Entry-Preis x, bei dem sich - gegen die uebergebenen Payouts - ein
    Profit Factor von genau target_pf ergeben wuerde.

    PF(x) = GrossProfit(x) / GrossLoss(x) mit net_pnl_i(x) = x - payout_i
    ist monoton nicht-fallend in x (mit steigendem x werden mehr Tage zu
    Gewinnern, und die Gewinnmargen wachsen, waehrend Verlustmargen
    schrumpfen) - deshalb Bisektion statt eines Gradientenverfahrens,
    robuster gegen die stueckweise-lineare, an den payout_i-Werten
    geknickte Struktur von PF(x).

    NEU (siehe Moduldokstring Punkt 12): `weights` sind die Kernel-Gewichte
    (siehe analog_matching.find_analogs()) der zugehoerigen Analog-Tage -
    PF(x) wird jetzt GEWICHTET berechnet (jeder Analog-Tag zaehlt mit
    seinem Gewicht statt gleich), konsistent zur Ablösung des harten
    max_distance-Cutoffs an der Quelle.

    WICHTIG: `payout` muss bereits auf EINE konsistente Skala gebracht sein
    (siehe _payout_at_todays_scale() - NICHT direkt die rohen, ueber
    Jahrzehnte unterschiedlicher Index-Niveaus gemischten
    entry_premium_i/net_pnl_i-Werte aus evaluate_analog_set() verwenden,
    siehe Moduldokstring Punkt 11 fuer die Begruendung).

    Sonderfall: sind ALLE Payouts 0 (Struktur ist auf dem kompletten
    Analog-Set nie ins Geld gelaufen), ist JEDER Preis >= 0 bereits
    verlustfrei (PF technisch nicht definiert/unendlich) - dann wird 0.0
    zurueckgegeben ('jeder Preis reicht bereits'). Gibt None zurueck, falls
    die obere Bisektions-Grenze (numerisch) nicht ausreicht (praktisch so
    gut wie nie, da PF(x)->unendlich fuer x->unendlich), oder falls keine
    gueltigen (gewichteten) Payouts vorliegen."""
    payout = np.asarray(payout, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = ~np.isnan(payout) & ~np.isnan(weights) & (weights > 0.0)
    payout, weights = payout[mask], weights[mask]
    if len(payout) == 0:
        return None
    if not np.any(payout > 0.0):
        return 0.0

    def pf_at(x: float) -> float:
        net = x - payout
        win = net >= 0.0
        gross_profit = np.sum(weights[win] * net[win])
        gross_loss = -np.sum(weights[~win] * net[~win])
        if gross_loss <= 0.0:
            return np.inf
        return gross_profit / gross_loss

    lo, hi = 0.0, float(payout.max()) * 10.0 + 1.0
    if pf_at(hi) < target_pf:
        return None

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        if pf_at(mid) < target_pf:
            lo = mid
        else:
            hi = mid
    return hi


def _payout_at_todays_scale(entry_S: np.ndarray, exit_S: np.ndarray, weights: np.ndarray,
                             spot_today: float, strike_put: float, strike_call: float
                             ) -> tuple[np.ndarray, np.ndarray]:
    """KORREKTUR (siehe Moduldokstring Punkt 11): loest das Skalenproblem
    einer Analog-Suche, die - bei ausreichend grosser max_distance/
    min_candidates-Garantie (siehe Punkt 12) - die KOMPLETTE SPX-Historie
    seit 1927 umfassen kann. Der Index stand dort bei voellig
    unterschiedlichen absoluten Niveaus (z.B. ~5 Punkte 1932 vs. ~7500
    Punkte heute) - eine direkte Verwendung der historischen,
    per-era-spezifischen Modell-Strikes/Payouts (aus evaluate_analog_set(),
    dort fuer die Delta-Treffgenauigkeit auf JEDES Analog-Tages EIGENEN
    Spot-Stand berechnet) wuerde bei einer Preis-Schwellenwert-Berechnung
    wie hier automatisch die betragsmaessig grossen, MODERNEN Analog-Tage
    massiv staerker gewichten als alte - unabhaengig davon, ob sie relativ
    gesehen ueberhaupt schlechter liefen (das genau war der Fehler in der
    ersten Version dieser Funktion).

    Fix: statt der historischen absoluten Kursbewegung wird nur die
    PROZENTUALE Kursbewegung jedes Analog-Tages uebernommen
    (exit_S/entry_S - 1) und auf den HEUTIGEN Spot projiziert - das ergibt
    einen hypothetischen 'heutigen' Exit-Spot je Analog-Tag, alle auf
    EXAKT DERSELBEN (heutigen) Skala. Ausgewertet wird dagegen mit den
    ECHTEN, in der Optionskette sichtbaren Strikes (strike_put/strike_call
    aus Teil A) - NICHT mit den historischen, per-era-spezifischen
    Modell-Strikes aus evaluate_analog_set(). Ergebnis: ein Payout je
    Analog-Tag, alle in HEUTIGEN Punkten, direkt vergleichbar mit dem
    HEUTE am Markt verfuegbaren Preis (Mid Put + Mid Call).

    NEU (siehe Moduldokstring Punkt 12): gibt zusaetzlich die zu jedem
    Payout gehoerenden Kernel-Gewichte zurueck (gleiche Maskierung wie
    entry_S/exit_S), damit 'Preis fuer PF=3' konsistent zur
    kontinuierlichen Gewichtung berechnet werden kann statt jeden
    Analog-Tag gleich zu zaehlen."""
    entry_S = np.asarray(entry_S, dtype=float)
    exit_S = np.asarray(exit_S, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = (entry_S > 0.0) & ~np.isnan(entry_S) & ~np.isnan(exit_S) & ~np.isnan(weights)
    entry_S, exit_S, weights = entry_S[mask], exit_S[mask], weights[mask]
    if len(entry_S) == 0:
        return np.array([]), np.array([])

    pct_return = exit_S / entry_S - 1.0
    hyp_exit_spot = spot_today * (1.0 + pct_return)
    payout = np.maximum(strike_put - hyp_exit_spot, 0.0) + np.maximum(hyp_exit_spot - strike_call, 0.0)
    return payout, weights


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
                      analog_max_distance: float, analog_bandwidth: float, analog_min_gap: int,
                      spot_today: float) -> pd.DataFrame | None:
    """Fuehrt den Analog-Tage-Lauf fuer JEDE Zeile von 'table' aus (siehe
    Moduldokstring Punkt 1-8). Gibt None zurueck (mit st.error) bei
    Abbruch, sonst ein DataFrame mit einer Zeile je distance_pct.

    NEU (siehe Moduldokstring Punkt 12): find_analogs() wird mit
    use_kernel_weighting=True aufgerufen - jeder Analog-Tag bekommt ein
    stetiges Kernel-Gewicht statt eines harten Ja/Nein, und mindestens
    _ANALOG_MIN_CANDIDATES Tage werden IMMER aufgenommen (auch in
    Krisenphasen ohne nahe historische Entsprechung). Alle Kennzahlen
    unten werden entsprechend GEWICHTET berechnet (summarize_analog_trades_
    weighted()/weighted_reliability_summary()) - 'analog_n_eff' (Kish)
    macht sichtbar, wenn eine Zeile trotz vorhandener Zeilenzahl kaum echte
    Vergleichsinformation traegt.

    NEU (siehe Moduldokstring Punkt 13): zusaetzlich 'robust_train_val_neg'
    (gespiegeltes Robust-Flag fuer negative Edges) fuer die Rotfaerbung."""
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
            # NEU (siehe Moduldokstring Punkt 12): kontinuierliche Kernel-
            # Gewichtung + Mindest-Kandidaten-Garantie statt hartem Cutoff -
            # verhindert einen Totalausfall in Krisenphasen ohne nahe
            # historische Entsprechung.
            analogs = find_analogs(
                query_date, query_score, space,
                AnalogConfig(
                    max_distance=analog_max_distance,
                    min_gap_days=analog_min_gap,
                    use_kernel_weighting=True,
                    bandwidth=analog_bandwidth,
                    min_candidates=_ANALOG_MIN_CANDIDATES,
                ),
            )
            bounds = hv_rank_bounds(daily_df_today, hv_length, rank_lookback)
        except ValueError as e:
            st.error(f"Analog-Suche fehlgeschlagen: {e}")
            return None

    st.caption(
        f"{len(analogs)} Analog-Tage gefunden (einmalig ermittelt, gilt fuer alle Zeilen unten). "
        "Jeder Tag geht mit einem stetigen Kernel-Gewicht ein (siehe 'Effektive n (Kish)' unten) "
        "statt einer harten Ja/Nein-Aufnahme."
    )

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

        # (siehe Moduldokstring Punkt 8): heutige (reale) Entry-Praemie
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
                "distance_pct": row.distance_pct, "analog_n": None, "analog_n_eff": None,
                "entry_premium_today": entry_premium_today,
                "expected_profit_short": None,
                "required_premium_pf3": None,
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
                "distance_pct": row.distance_pct, "analog_n": None, "analog_n_eff": None,
                "entry_premium_today": entry_premium_today,
                "expected_profit_short": None,
                "required_premium_pf3": None,
            })
            continue

        if full_eval.empty:
            results.append({
                "distance_pct": row.distance_pct, "analog_n": 0, "analog_n_eff": 0.0,
                "entry_premium_today": entry_premium_today,
                "expected_profit_short": None,
                "required_premium_pf3": None,
            })
            continue

        # NEU (siehe Moduldokstring Punkt 12): GEWICHTET statt gleichgewichtet.
        trade_summary = summarize_analog_trades_weighted(full_eval, "short")
        rel_full = weighted_reliability_summary(full_eval, "short", 6)

        median_date = full_eval["entry_date"].median()
        train_eval = full_eval[full_eval["entry_date"] < median_date]
        val_eval = full_eval[full_eval["entry_date"] >= median_date]
        train_rel = weighted_reliability_summary(train_eval, "short", 4)
        val_rel = weighted_reliability_summary(val_eval, "short", 4)

        robust = False
        robust_neg = False   # NEU (Punkt 13): gespiegeltes Flag fuer negative Edges (Rotfaerbung)
        if train_rel["n_eff"] >= 2.0 and val_rel["n_eff"] >= 2.0:
            robust = bool(
                train_rel["ci_excludes_zero"] and val_rel["ci_excludes_zero"]
                and np.sign(train_rel["mean_pnl_norm"]) == np.sign(val_rel["mean_pnl_norm"])
                and np.sign(train_rel["mean_pnl_norm"]) > 0.0
            )
            robust_neg = bool(
                train_rel["ci_excludes_zero"] and val_rel["ci_excludes_zero"]
                and np.sign(train_rel["mean_pnl_norm"]) == np.sign(val_rel["mean_pnl_norm"])
                and np.sign(train_rel["mean_pnl_norm"]) < 0.0
            )

        # (siehe Moduldokstring Punkt 8): avg_pnl_norm_pct ist IMMER aus
        # Short-Sicht (summarize_analog_trades_weighted(full_eval, "short")) -
        # mit der heutigen Entry-Praemie DIESER Zeile multipliziert ergibt
        # das den erwarteten Gewinn in PUNKTEN, vergleichbar ueber Zeilen
        # hinweg (statt der relativen %-Kennzahl, die auf die historische
        # Analog-Praemie JEDES einzelnen Tages normiert ist).
        expected_profit_short = None
        if entry_premium_today is not None and pd.notna(trade_summary["avg_pnl_norm_pct"]):
            avg_pnl_norm_frac = trade_summary["avg_pnl_norm_pct"] / 100.0
            expected_profit_short = avg_pnl_norm_frac * entry_premium_today

        # (siehe Moduldokstring Punkt 11/12): historische PROZENTUALE
        # Kursbewegungen (nicht absolute Punkte!) jedes Analog-Tages auf
        # den HEUTIGEN Spot projizieren und gegen die ECHTEN, heute in der
        # Optionskette sichtbaren Strikes auswerten - GEWICHTET mit dem
        # Kernel-Gewicht jedes Analog-Tages.
        payout, payout_weights = _payout_at_todays_scale(
            full_eval["entry_S"].to_numpy(), full_eval["exit_S"].to_numpy(),
            full_eval["weight"].to_numpy(), spot_today,
            float(row.strike_put), float(row.strike_call),
        )
        required_premium_pf3 = _required_premium_for_profit_factor(payout, payout_weights, target_pf=3.0)

        results.append({
            "distance_pct": row.distance_pct,
            "analog_n": trade_summary["n_trades"],
            "analog_n_eff": trade_summary["n_eff"],
            "win_rate_pct": trade_summary["win_rate_pct"],
            "profit_factor": trade_summary["profit_factor"],
            "avg_pnl_norm_pct": trade_summary["avg_pnl_norm_pct"],
            "entry_premium_today": entry_premium_today,
            "expected_profit_short": expected_profit_short,
            "required_premium_pf3": required_premium_pf3,
            "tstat_full": rel_full["tstat"],
            "ci_excludes_zero_full": rel_full["ci_excludes_zero"],
            "subperiod_matching": rel_full["subperiod_matching"],
            "subperiod_nonempty": rel_full["subperiod_nonempty"],
            "tstat_val": val_rel["tstat"],
            "robust_train_val": robust,
            "robust_train_val_neg": robust_neg,   # NEU (Punkt 13)
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
                # NEU (Staleness-Diagnose, siehe Moduldokstring Punkt 10):
                # Fetch-Zeitpunkt EXPLIZIT festhalten und als `as_of` an
                # enrich_chain_with_delta() durchreichen, statt den
                # dortigen Default (pd.Timestamp.now() beim Funktionsaufruf)
                # implizit zu nutzen - so bleibt der Referenzzeitpunkt fuer
                # 'quote_age_minutes' konsistent mit dem, was wir dem Nutzer
                # als 'Kette abgerufen um...' anzeigen, auch wenn der
                # Streamlit-Rerun-Zyklus zwischen Fetch und Anzeige etwas
                # Zeit vergehen laesst.
                fetch_ts = pd.Timestamp.now(tz="UTC")
                spot = fetch_today_spot()
                chain = fetch_today_chain(min_dte=float(min_dte), max_dte=float(max_dte))
                enriched = enrich_chain_with_delta(
                    chain, spot,
                    ChainDeltaConfig(min_abs_delta=float(min_abs_delta),
                                      r_free=float(r_free), dividend_yield=float(dividend_yield)),
                    as_of=fetch_ts,
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
                st.session_state["livechain_fetch_ts"] = fetch_ts

    if "livechain_enriched" not in st.session_state:
        st.info("Kette konnte noch nicht geladen werden - siehe Fehlermeldung oben.")
        return

    spot = st.session_state["livechain_spot"]
    enriched = st.session_state["livechain_enriched"]
    r_free = st.session_state["livechain_r_free"]
    dividend_yield = st.session_state["livechain_dividend_yield"]
    rank_lookback = st.session_state["livechain_rank_lookback"]
    fetch_ts = st.session_state.get("livechain_fetch_ts")
    st.metric("Aktueller Spot (SPX)", f"{spot:,.2f}")
    if fetch_ts is not None:
        st.caption(
            f"Kette abgerufen um {fetch_ts.tz_convert('America/New_York').strftime('%Y-%m-%d %H:%M:%S')} ET "
            "- Referenzzeitpunkt fuer die 'Quote-Alter'-Spalten weiter unten."
        )

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

    # NEU (siehe Moduldokstring Punkt 10): Quote-Alter-Warnung - unabhaengig
    # vom Randbereich-Hinweis oben, da Staleness auch mitten in der Kette
    # (nicht nur am Rand) auftreten kann, wie der Put/Call-Vergleich gegen
    # externe Quellen gezeigt hat.
    stale_mask = (
        table["quote_age_put_min"].fillna(0) > _QUOTE_AGE_WARNING_MINUTES
    ) | (
        table["quote_age_call_min"].fillna(0) > _QUOTE_AGE_WARNING_MINUTES
    )
    if stale_mask.any():
        stale_rows = table.loc[stale_mask, "distance_pct"].tolist()
        preview = ", ".join(f"{d:g}%" for d in stale_rows[:10])
        more = f" (+{len(stale_rows) - 10} weitere)" if len(stale_rows) > 10 else ""
        st.warning(
            f"⚠️ Veraltete Einzelquote(n) vermutet: bei {len(stale_rows)} Zeile(n) "
            f"({preview}{more}) liegt der letzte tatsaechliche Trade von Put oder "
            f"Call laenger als {_QUOTE_AGE_WARNING_MINUTES:g} Minuten zurueck (siehe "
            "Spalten 'Quote-Alter Put/Call (Min.)' unten). Das ist KEIN Beweis fuer "
            "eine falsche Quote (bei ruhigem Markt bleibt eine Quote laenger gueltig), "
            "aber ein Hinweis, IV/Preis fuer diese Zeilen nicht ungeprueft zu "
            "uebernehmen - insbesondere wenn Put und Call fuer denselben Abstand "
            "deutlich unterschiedliche IV zeigen (siehe Moduldokstring Punkt 10)."
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
        "quote_age_put_min": "Quote-Alter Put (Min.)", "quote_age_call_min": "Quote-Alter Call (Min.)",
    })[[
        "Ziel-Abstand % v. Spot",
        "Δ Put % (Modell, flache HV)", "Δ Call % (Modell, flache HV)",
        "IV Kombi % (Markt, Paket-Solve)", "HV-Rang % (dieser Kombi)",
        "Strike Put (real)", "Strike Call (real)",
        "Mid Put", "Mid Call",
        "IV Put % (echt)", "IV Call % (echt)",
        "Δ Put % (echt)", "Δ Call % (echt)",
        "Quote-Alter Put (Min.)", "Quote-Alter Call (Min.)",
    ]]
    st.caption(
        "'Ziel-Δ' ist die Suchvorgabe (Zeilenlabel). 'Δ Put/Call (echt)' sind die "
        "TATSAECHLICHEN, aus dem realen Mid-Preis per Black-76 zurueckgerechneten "
        "Deltas der jeweils gewaehlten Option - koennen leicht vom Ziel abweichen, "
        "da nur reale (diskrete) Strikes existieren. 'IV Kombi' = die EINE Vola, "
        "die - in Black-76 fuer Put+Call GEMEINSAM eingesetzt - exakt die Summe "
        "beider echter Mid-Preise reproduziert (kein Mittelwert der Einzel-IVs). "
        "'HV-Rang' = wo diese Kombi-IV im HV-Min/Max-Fenster oben liegt. "
        "'Quote-Alter' = Minuten seit dem letzten TATSAECHLICHEN Trade dieses "
        "Strikes, bezogen auf den Fetch-Zeitpunkt oben (Naeherungswert fuer "
        "Staleness, siehe Warnung oben und Moduldokstring Punkt 10 - kein "
        "direkter Bid/Ask-Zeitstempel verfuegbar)."
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
            "Quote-Alter Put (Min.)": "{:.0f}", "Quote-Alter Call (Min.)": "{:.0f}",
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
                ["strike", "option_type", "bid", "ask", "mid", "implied_vol_mkt", "delta_mkt",
                 "last_trade_date", "quote_age_minutes"]
            ].rename(columns={
                "last_trade_date": "Letzter Trade (yfinance)",
                "quote_age_minutes": "Quote-Alter (Min.)",
            }).sort_values(["option_type", "strike"]),
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
        "**Kontinuierliche Gewichtung statt harter Distanz-Schwelle** (siehe "
        "Moduldokstring Punkt 12): jeder Analog-Tag geht mit einem stetigen "
        "Kernel-Gewicht ein, das mit seiner Ähnlichkeitsdistanz sinkt - statt ganz "
        "drin oder ganz draussen zu sein. In Marktphasen ohne nahe historische "
        "Entsprechung (z.B. scharfe Krisen) werden trotzdem immer ein paar der "
        "naechstgelegenen Tage herangezogen, damit die Tabelle laedt - die Spalte "
        "'Effektive n (Kish)' zeigt dann transparent an, wie wenig echte "
        "Vergleichsinformation tatsaechlich dahintersteckt (deutlich kleiner als "
        "'Analog-Trades (n)' = wenig belastbar)."
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
    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        analog_max_distance = st.number_input(
            "Max. Distanz (Performance-Grenze)", min_value=0.01, max_value=100.0, value=4.0, step=0.5,
            help="Kandidaten weiter als dieser Wert entfernt werden nur noch ueber die "
                 "Mindest-Kandidaten-Garantie (siehe Bandbreite) beruecksichtigt. Deutlich "
                 "groesser als die Kernel-Bandbreite waehlen, damit diese Grenze das "
                 "Ergebnis normalerweise nicht beeinflusst.",
        )
    with ac2:
        analog_bandwidth = st.number_input(
            "Kernel-Bandbreite σ (Gaussian)", min_value=0.01, max_value=50.0, value=1.0, step=0.1,
            help="Ersetzt den frueheren harten Cutoff: jeder Analog-Tag geht mit Gewicht "
                 "exp(-(Distanz/σ)²) ein, statt ganz drin oder ganz draussen zu sein. Kleiner σ "
                 "= staerkere Fokussierung auf die aehnlichsten Tage. Mindestens "
                 f"{_ANALOG_MIN_CANDIDATES} Tage werden IMMER aufgenommen, auch in Krisenphasen "
                 "ohne nahe historische Entsprechung (siehe 'Effektive n (Kish)' unten).",
        )
    with ac3:
        analog_min_gap = st.number_input(
            "Mindestabstand zu heute (Handelstage)", min_value=0, max_value=60, value=5,
        )

    run_bulk = st.button("Analog-Läufe für alle Zeilen starten")
    if run_bulk:
        analog_results = _run_bulk_analog(
            table, daily_df, chosen_dte_exact, r_free, dividend_yield,
            hv_length, int(rank_lookback), float(analog_max_distance), float(analog_bandwidth),
            int(analog_min_gap), spot,
        )
        if analog_results is not None:
            st.session_state["livechain_analog_results"] = analog_results
            st.session_state["livechain_analog_key"] = (chosen_expiry, tuple(distance_targets_pct))

    stored = st.session_state.get("livechain_analog_results")
    stored_key = st.session_state.get("livechain_analog_key")
    if stored is not None and stored_key == (chosen_expiry, tuple(distance_targets_pct)):
        merged = table.merge(stored, on="distance_pct", how="left")

        # NEU (siehe Moduldokstring Punkt 12): Warnung, falls ein grosser Teil
        # der Zeilen trotz vorhandener Analog-Trades kaum echte (gewichtete)
        # Vergleichsinformation traegt - das Pendant zur bestehenden
        # "< 30 Analog-Tage"-Warnung in dashboard_analog.py, hier auf n_eff
        # statt der rohen Trade-Zahl bezogen, da die min_candidates-Garantie
        # die rohe Zahl kuenstlich hochhalten kann.
        if "analog_n_eff" in merged.columns:
            low_neff = merged["analog_n_eff"].dropna()
            low_neff = low_neff[low_neff < _LOW_N_EFF_WARNING]
            if len(low_neff) > 0:
                st.warning(
                    f"⚠️ {len(low_neff)} Zeile(n) haben eine effektive Stichprobengroesse "
                    f"(Kish) unter {_LOW_N_EFF_WARNING:g} - das bedeutet, dass fuer diese "
                    "Zeilen kaum nahe historische Vergleichstage existieren (typisch in "
                    "ungewoehnlichen Marktphasen). Die Tabelle zeigt diese Zeilen trotzdem "
                    "(statt sie komplett auszulassen), aber ihre Kennzahlen sind auf einer "
                    "sehr duennen Datenbasis berechnet - siehe Spalte 'Effektive n (Kish)'."
                )

        # NEU (siehe Moduldokstring Punkt 13): Zeilenzustand (gruen/rot/'')
        # VOR dem Umbenennen/Auswaehlen der Spalten bestimmen - die Funktion
        # arbeitet auf den internen Spaltennamen von 'merged'. Der Index von
        # row_state ist derselbe wie der von show2 (rename/Spaltenauswahl
        # aendern den Index nicht), daher kann das Styling ueber r.name
        # nachschlagen.
        row_state = _robustness_row_state(merged)

        show2 = merged.rename(columns={
            "distance_pct": "Ziel-Abstand % v. Spot",
            "strike_put": "Strike Put (real)",
            "strike_call": "Strike Call (real)",
            "analog_n": "Analog-Trades (n)",
            "analog_n_eff": "Effektive n (Kish)",
            "win_rate_pct": "Gewinnrate %",
            "profit_factor": "Profit Factor",
            "avg_pnl_norm_pct": "Ø norm. P&L %",
            "entry_premium_today": "Entry-Prämie heute (Pkt)",
            "expected_profit_short": "Erw. Gewinn Short (Pkt)",
            "required_premium_pf3": "Preis für PF=3 (Pkt)",
            "tstat_full": "t-Stat (gesamt)",
            "ci_excludes_zero_full": "CI ≠ 0 (gesamt)",
            "subperiod_matching": "Perioden-Match",
            "subperiod_nonempty": "Perioden gesamt",
            "tstat_val": "t-Stat (Val)",
            "robust_train_val": "Robust (Train+Val)",
        })[[
            "Ziel-Abstand % v. Spot", "Strike Put (real)", "Strike Call (real)",
            "Analog-Trades (n)", "Effektive n (Kish)",
            "Gewinnrate %", "Profit Factor",
            "Ø norm. P&L %", "Entry-Prämie heute (Pkt)",
            "Erw. Gewinn Short (Pkt)", "Preis für PF=3 (Pkt)",
            "t-Stat (gesamt)", "CI ≠ 0 (gesamt)",
            "Perioden-Match", "Perioden gesamt", "t-Stat (Val)", "Robust (Train+Val)",
        ]]
        st.caption(
            f"T_days fuer die Analog-Bewertung: "
            f"{int(stored['T_days_used'].iloc[0]) if len(stored) else 'n/a'} "
            f"(gerundet von {chosen_dte_exact:.2f} Tagen Restlaufzeit) · "
            f"{int(stored['n_analogs_total'].iloc[0]) if len(stored) else 0} Analog-Tage gefunden "
            "(bereits inkl. Mindest-Kandidaten-Garantie, siehe oben). "
            "'Effektive n (Kish)' = gewichtete effektive Stichprobengroesse "
            "((Σw)²/Σw²) - deutlich kleiner als 'Analog-Trades (n)' bedeutet, dass "
            "wenige, sehr aehnliche Tage die Kennzahlen dominieren bzw. dass die "
            "Zeile ueberwiegend aus der Mindest-Kandidaten-Garantie (weit entfernte "
            "Tage mit winzigem Gewicht) besteht. "
            "'Robust (Train+Val)' = Konfidenzintervall schliesst 0 aus UND gleiches "
            "positives Vorzeichen in BEIDEN chronologischen Haelften des Analog-Sets. "
            "'Preis für PF=3' = welcher (einheitliche) Entry-Preis auf dem KOMPLETTEN "
            "Analog-Set (Short-Seite, gewichtet gegen die dort tatsaechlich angefallenen "
            "historischen Payouts) einen Profit Factor von genau 3 ergeben haette - "
            "liegt 'Entry-Prämie heute' darueber, waere der historische PF > 3 gewesen, "
            "liegt sie darunter, entsprechend < 3 (siehe Moduldokstring Punkt 11; rein "
            "historische Ruecksimulation, keine Zukunftsgarantie)."
        )
        st.caption(
            "**Zeilenfarben:** 🟩 grün = alle Robustheitskriterien positiv erfüllt "
            f"(t-Stat gesamt > {_ROBUST_TSTAT_THRESHOLD:g}, CI ≠ 0, Perioden-Match ≥ "
            f"{_ROBUST_MIN_PERIODS_MATCH}/6, t-Stat Val > {_ROBUST_TSTAT_THRESHOLD:g}, "
            "Robust Train+Val). 🟥 rot = spiegelbildlich negativ (t-Stat gesamt und Val "
            f"< -{_ROBUST_TSTAT_THRESHOLD:g}, CI ≠ 0, Perioden-Match ≥ "
            f"{_ROBUST_MIN_PERIODS_MATCH}/6, beide Hälften signifikant mit negativem "
            "Vorzeichen). Alle anderen Zeilen ungefärbt."
        )
        st.dataframe(
            show2.style.apply(
                lambda r: [_ROW_STYLE.get(row_state.loc[r.name], "")] * len(r), axis=1,
            ).format({
                "Ziel-Abstand % v. Spot": "{:.2f}",
                "Strike Put (real)": "{:.0f}",
                "Strike Call (real)": "{:.0f}",
                "Analog-Trades (n)": "{:.0f}",
                "Effektive n (Kish)": "{:.1f}",
                "Gewinnrate %": "{:.1f}",
                "Profit Factor": "{:.2f}",
                "Ø norm. P&L %": "{:+.2f}",
                "Entry-Prämie heute (Pkt)": "{:.2f}",
                "Erw. Gewinn Short (Pkt)": "{:+.2f}",
                "Preis für PF=3 (Pkt)": "{:.2f}",
                "t-Stat (gesamt)": "{:.2f}",
                "Perioden-Match": "{:.0f}",
                "Perioden gesamt": "{:.0f}",
                "t-Stat (Val)": "{:.2f}",
            }, na_rep="n/a"),
            use_container_width=True, hide_index=True,
        )
    elif stored is not None:
        st.info("Tabelle/Expiry hat sich seit dem letzten Analog-Lauf geaendert - bitte erneut starten.")