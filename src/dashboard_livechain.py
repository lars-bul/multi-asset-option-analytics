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

  1. VERALTET (bis zur Vereinheitlichung, siehe Punkt 11) - Ziel-HV-Rang je
     Zeile war die in Teil A berechnete Spalte 'HV-Rang % (dieser Kombi)'.
     Seit der Vereinheitlichung wird in Teil B kein Ziel-Rang und keine
     Vola-Modellierung mehr gebraucht - es zaehlt nur noch die tatsaechliche
     prozentuale Kursbewegung jedes Analog-Tages (siehe Punkt 11).
  2. VERALTET (bis zur Vereinheitlichung) - Put-/Call-Delta je Zeile kam aus
     den 'Modell-Delta (flache HV)'-Spalten aus Teil A. Seit der
     Vereinheitlichung werden direkt die ECHTEN Strikes dieser Zeile
     ('Strike Put/Call (real)' aus Teil A) verwendet - kein Modell-Delta,
     kein erneuter Strike-Solve mehr noetig (siehe Punkt 11).
  3. VERALTET (bis zur Vereinheitlichung) - vormals wurde fuer den
     Analog-Lauf explizit vol_surface=VolSurfaceConfig(use_vol_surface=False)
     gesetzt, damit die Delta-Definition durchgaengig flach blieb (siehe
     Punkt 2). Seit der Vereinheitlichung entfaellt jede Vola-/Skew-
     Modellierung in Teil B komplett - es wird nur noch die tatsaechliche
     historische Kursbewegung (kein Options-Pricing) verwendet.
  4. T_days MUSS einer der in der vorab gebauten training_table verfuegbaren
     Werte sein (siehe labels.py/build_dataset.py) - die tatsaechliche
     Restlaufzeit (dte_exact) wird auf den naechstgelegenen verfuegbaren
     Wert GERUNDET (siehe available_t_days()). Wird weiterhin gebraucht,
     um die Analog-Tage in der training_table auf ihr entry_S/exit_S
     nachzuschlagen (siehe _lookup_entry_exit(), Punkt 11) - auch ohne
     Modellbewertung bleibt T_days relevant, weil exit_S vom Handelshorizont
     abhaengt.
  5. 'Heute' fuer die Analog-Suche: waehrend eines laufenden Handelstags
     enthaelt daily_df noch keinen abgeschlossenen Balken fuer heute (montags
     waere das sonst Freitag, 3 Tage alt). Es wird daher ein synthetischer
     Balken (open=high=low=close=heutiger Eroeffnungskurs) an eine KOPIE von
     daily_df angehaengt (siehe _build_daily_df_with_today()).
  6. Performance: die Aehnlichkeits-Raum-PCA und die Analog-Tage-Suche
     (find_analogs) haengen NICHT von Delta/Rang ab - werden daher NUR
     EINMAL pro Klick berechnet, nicht pro Zeile. Seit der Vereinheitlichung
     (siehe Punkt 11) gilt das jetzt auch fuer den entry_S/exit_S-Lookup
     (_lookup_entry_exit()) - der haengt nur von T_days ab, nicht von der
     einzelnen Zeile, und wird daher ebenfalls nur EINMAL pro Klick
     berechnet. Pro Zeile bleibt nur noch eine guenstige Vektor-Operation
     (Payout-Projektion + Bisektion) statt des vormals teuren
     Strike-Solve+Black-76-Loops - der Bulk-Lauf ist dadurch insgesamt
     deutlich schneller als vor der Vereinheitlichung.
  7. Je Zeile: Kennzahlen auf dem KOMPLETTEN Analog-Set (n, Gewinnrate,
     Profit Factor, Ø norm. P&L, t-Stat, CI≠0, Perioden-Konsistenz X/6)
     PLUS Train/Val-Split-Robustheit (t-Stat Val, Robust-Flag) - identisches
     Kriterium wie leg_pair_grid_search() in analog_matching.py, aber ohne
     die dreifache Neuberechnung: die Payout-Serie (siehe Punkt 11) wird
     nur EINMAL pro Zeile gebildet, Train/Val wird danach aus dem fertigen
     Ergebnis chronologisch aufgeteilt (mathematisch identisch, da jeder
     Analog-Tag unabhaengig von den anderen ist). Die zugrunde liegenden
     entry_S/exit_S-Werte (siehe _lookup_entry_exit()) werden sogar nur
     EINMAL pro GESAMTEM Lauf nachgeschlagen (nicht pro Zeile), da sie nur
     von T_days_used abhaengen, nicht von Delta/Rang der einzelnen Zeile.
  8. NEU/VEREINHEITLICHT - 'Erwarteter Gewinn Short' (in Punkten): seit der
     Vereinheitlichung (siehe Punkt 11) ist pnl_norm je Analog-Tag auf die
     HEUTIGE, reale Entry-Praemie dieser Zeile normiert (entry_premium_today
     = mid_put + mid_call aus Teil A) - NICHT mehr auf eine je Analog-Tag
     individuell modellierte, historische Entry-Praemie wie vor der
     Vereinheitlichung. Dadurch ist

         erwarteter Gewinn Short (Pkt) = avg_pnl_norm_pct/100 * entry_premium_today

     jetzt EXAKT (nicht nur naeherungsweise) identisch zum arithmetischen
     Mittel der projizierten Netto-P&L-Werte (net_pnl, siehe Punkt 11) -
     die Multiplikation wird trotzdem so dokumentiert, weil sie den
     Zusammenhang zur Ø norm. P&L%-Spalte transparent macht. Bewertet wie
     zuvor ausschliesslich das Short-Bein (die Praemien-Seite des Condors),
     NICHT die Wings; fuer die Wing-/Long-Auswahl und v.a. fuer die
     STREUUNG/Konsistenz der KOMBINIERTEN Position weiterhin den
     Iron-Condor-Ladder-Backtest verwenden (dashboard_ladder.py) - dort
     wird Short+Long GEMEINSAM pro Handelstag bewertet und liefert eine
     echte, korrelationskonsistente Equity-Kurve.
  9. NEU - Auto-Load: die Kette wird beim ERSTEN Aufruf dieses Tabs
     automatisch geladen (kein Klick auf 'Heutige Kette laden' mehr noetig,
     siehe render()). Einmal geladene Daten (Kette, Distanz-Tabelle,
     Analog-Ergebnisse) liegen in st.session_state und bleiben daher auch
     nach einem Wechsel zu einem anderen Tab/Modus erhalten - Streamlit
     raeumt st.session_state NICHT auf, nur weil ein anderer Reiter aktiv
     ist (siehe dashboard.py-Docstring: es rendert zwar nur der aktive
     Modus, aber session_state ist global fuer die gesamte Browser-Sitzung).
  10. NEU - Staleness-Diagnose je Zeile (Quote-Alter Put/Call): siehe
      chain_delta.py-Moduldokstring - dort wird bereits `quote_age_minutes`
      je Einzelserie und `quote_age_put_min`/`quote_age_call_min` je Zeile
      der Distanz-Tabelle berechnet (Minuten seit dem letzten TATSAECHLICHEN
      Trade dieses Strikes, bezogen auf den Fetch-Zeitpunkt). Bisher wurde
      dieser Wert berechnet, aber nirgends angezeigt. Hintergrund: bei einem
      Abgleich gegen externe Quellen (Barchart, Yahoo-Website) fiel auf,
      dass einzelne, duenn gehandelte Strikes (v.a. auf der Put-Seite)
      Bid/Ask-Quoten zeigten, die deutlich (~30%) von unabhaengigen,
      fast zeitgleich abgerufenen Referenzquellen abwichen, waehrend der
      Call auf unter 3% Abweichung uebereinstimmte. Der Code selbst
      (Put/Call-Zuordnung, Strike-Auswahl, Mid-Berechnung) wurde darauf hin
      geprueft und zeigte KEINEN Zuordnungsfehler - der wahrscheinlichste
      Grund ist eine schlicht veraltete Einzelquote bei Yahoo (ein Market
      Maker aktualisiert eine kaum gehandelte Serie seltener, als der
      Gesamtfeed es suggeriert - "15 Minuten verzoegert" gilt nur pauschal
      fuer den Feed, nicht pro Strike). Ein hohes Quote-Alter ist KEIN
      Beweis fuer eine falsche Quote (bei ruhigem Markt bleibt eine Quote
      laenger gueltig) und ein niedriges Quote-Alter ist KEIN Beweis fuer
      eine korrekte Quote (der Market Maker kann die Quote seit dem letzten
      Trade bewegt haben) - es ist trotzdem der beste verfuegbare
      Anhaltspunkt, um einen Verdacht wie "dieser Preis wirkt zu hoch"
      objektiv gegenzupruefen statt ihn nur aus dem Preisvergleich selbst zu
      vermuten. Ab einem Schwellenwert (_QUOTE_AGE_WARNING_MINUTES) wird
      eine explizite Warnung mit den betroffenen Zeilen angezeigt.
  11. NEU/VEREINHEITLICHT - EINE gemeinsame Payout-Grundlage fuer ALLE
      Kennzahlen in Teil B (Profit Factor, Ø norm. P&L%, Erw. Gewinn Short,
      Preis fuer PF=3, Newey-West-Reliability): fuer jeden Analog-Tag wird
      NUR die tatsaechliche PROZENTUALE Kursbewegung (exit_S/entry_S - 1,
      aus der training_table nachgeschlagen, siehe _lookup_entry_exit())
      auf den HEUTIGEN Spot projiziert, und GEGEN DIE ECHTEN, in der
      Optionskette sichtbaren Strikes dieser Zeile ausgewertet (siehe
      _payout_at_todays_scale()) - net_pnl = entry_premium_today - payout,
      pnl_norm = net_pnl / entry_premium_today.

      VORGESCHICHTE (zwei Korrekturstufen, siehe Nutzer-Diskussion):
      1) Erste Fassung von 'Preis fuer PF=3' hat die historischen, PER-ERA
         modellierten Payouts aus evaluate_analog_set() direkt verwendet
         (dort fuer JEDEN Analog-Tag ein EIGENER, an dessen eigenem
         Spot-Niveau berechneter Modell-Strike, um am Ziel-Delta zu
         bleiben). Bei einer Analog-Suche ueber die komplette SPX-Historie
         mischt das absolute Punktbetraege aus voellig unterschiedlichen
         Index-Niveaus (SPX ~5 im Jahr 1932 vs. ~7500 heute) - das
         gewichtet automatisch die betragsmaessig grossen, MODERNEN
         Analog-Tage massiv staerker. Fix: Umstellung auf die prozentuale,
         auf den heutigen Spot projizierte Bewegung (s.o.) - zunaechst NUR
         fuer die 'Preis fuer PF=3'-Spalte.
      2) Das liess 'Profit Factor'/'Ø norm. P&L%'/'Erw. Gewinn Short'
         zunaechst auf der ALTEN, per-era-modellierten evaluate_analog_set()
         -Bewertung zurueck - zwei GRUNDSAETZLICH verschiedene
         Gedankenexperimente in derselben Zeile ('gleiches Delta, jeden Tag
         neu am jeweiligen Vola-Niveau bewertet' vs. 'heutiger fester Trade,
         historische Prozentbewegungen durchgespielt'), die nicht numerisch
         uebereinstimmen MUESSEN - in der Praxis aber wie ein Widerspruch
         wirkten (z.B. 'Preis fuer PF=3' unter der heutigen Marktpraemie
         bei GLEICHZEITIG niedrigem 'Profit Factor'). Fix (diese Fassung):
         ALLE Kennzahlen der Zeile nutzen jetzt dieselbe, oben beschriebene
         Payout-Serie. Damit ist ein Widerspruch strukturell ausgeschlossen:
         liegt die heutige Marktpraemie ueber 'Preis fuer PF=3', ist der
         (aus DERSELBEN Serie berechnete) 'Profit Factor' bei dieser
         Praemie automatisch >= 3.

      Nebeneffekt: evaluate_analog_set() wird in Teil B nicht mehr
      aufgerufen - die Modellbewertung (Strike-Solve + Black-76-Pricing pro
      Analog-Tag x Zeile) entfaellt komplett, was den Bulk-Lauf spuerbar
      beschleunigt (siehe _run_bulk_analog()-Docstring). 'Profit Factor'
      nutzt wie in analog_matching.summarize_analog_trades() pnl_norm
      (normierte Rendite) statt net_pnl (absolute Punkte) fuer
      GrossProfit/GrossLoss - hier ohnehin bereits alle auf derselben
      heutigen Skala, aus Konsistenzgruenden trotzdem beibehalten.

      Bewertet weiterhin - wie zuvor - ausschliesslich die Short-Seite;
      rein historische Ruecksimulation auf dem gefundenen Analog-Set
      (siehe alle Vorbehalte zu Clustering/Newey-West-Lag in den
      bisherigen Analysen), keine Zukunftsgarantie.
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_today_chain import fetch_today_chain, fetch_today_spot, fetch_today_open
from chain_delta import ChainDeltaConfig, enrich_chain_with_delta, build_symmetric_distance_table
from analog_matching import (
    AnalogConfig, SimilaritySpaceConfig, hv_rank_bounds, fit_similarity_space,
    transform_query, find_analogs, available_t_days,
)
from features import FeatureConfig, build_feature_matrix
from dashboard_common import load_training_table
from stats import reliability_summary

# Schwellenwert (Minuten) fuer die Quote-Alter-Warnung in Teil A - bewusst
# als Modul-Konstante statt Sidebar-Eingabe, um die Sidebar nicht mit einem
# weiteren, selten geaenderten Parameter zu ueberladen. Bei Bedarf hier
# anpassen (siehe Moduldokstring Punkt 10).
_QUOTE_AGE_WARNING_MINUTES = 60.0


def _required_premium_for_profit_factor(payout: np.ndarray, target_pf: float = 3.0,
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
    gut wie nie, da PF(x)->unendlich fuer x->unendlich)."""
    payout = np.asarray(payout, dtype=float)
    payout = payout[~np.isnan(payout)]
    if len(payout) == 0:
        return None
    if not np.any(payout > 0.0):
        return 0.0

    def pf_at(x: float) -> float:
        net = x - payout
        gross_profit = net[net >= 0.0].sum()
        gross_loss = -net[net < 0.0].sum()
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


def _payout_at_todays_scale(entry_S: np.ndarray, exit_S: np.ndarray, spot_today: float,
                             strike_put: float, strike_call: float) -> tuple[np.ndarray, np.ndarray]:
    """KORREKTUR (siehe Moduldokstring Punkt 11): loest das Skalenproblem
    einer Analog-Suche, die - bei ausreichend grosser max_distance - die
    KOMPLETTE SPX-Historie seit 1927 umfassen kann. Der Index stand dort bei
    voellig unterschiedlichen absoluten Niveaus (z.B. ~5 Punkte 1932 vs.
    ~7500 Punkte heute) - eine direkte Verwendung der historischen,
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

    KORREKTUR (Alignment-Bug): gibt jetzt zusaetzlich die verwendete
    Filter-Maske zurueck. Vorher wurde intern maskiert (entry_S<=0/NaN
    verworfen), aber nur das gefilterte payout-Array zurueckgegeben - der
    Aufrufer hat 'entry_date'/'hold_days' per einfachem Slicing
    (entry_date_arr[:len(payout)]) ausgerichtet, was NUR zufaellig korrekt
    war, solange die vorgelagerte _lookup_entry_exit() bereits ALLE
    ungueltigen Zeilen entfernt hatte. Bei jeder Abweichung davon (z.B.
    andere Aufrufstelle, laxere Vorfilterung) haette das entry_date STILL
    UND OHNE FEHLERMELDUNG mit dem falschen net_pnl verknuepft - mit
    verfaelschter Newey-West-/Perioden-Konsistenz-Berechnung als Folge,
    ohne jeden Hinweis darauf im Ergebnis. Der Aufrufer MUSS die
    zurueckgegebene Maske jetzt explizit auf alle parallelen Arrays
    (entry_date, hold_days, ...) anwenden."""
    entry_S = np.asarray(entry_S, dtype=float)
    exit_S = np.asarray(exit_S, dtype=float)
    mask = (entry_S > 0.0) & ~np.isnan(entry_S) & ~np.isnan(exit_S)
    entry_S_kept, exit_S_kept = entry_S[mask], exit_S[mask]
    if len(entry_S_kept) == 0:
        return np.array([]), mask

    pct_return = exit_S_kept / entry_S_kept - 1.0
    hyp_exit_spot = spot_today * (1.0 + pct_return)
    payout = np.maximum(strike_put - hyp_exit_spot, 0.0) + np.maximum(hyp_exit_spot - strike_call, 0.0)
    return payout, mask


def _lookup_entry_exit(analogs: pd.DataFrame, training_table: pd.DataFrame, T_days: int) -> pd.DataFrame:
    """Reines Nachschlagen von entry_S/exit_S/hold_days aus der
    training_table fuer die gefundenen Analog-Tage - UNABHAENGIG von
    Delta, Ziel-Rang oder Vola (im Unterschied zum frueher hier verwendeten
    evaluate_analog_set(), das zusaetzlich eine Modellbewertung mit
    per-era-spezifischen Strikes durchgefuehrt hat). Da dieses Ergebnis fuer
    ALLE Zeilen der Bulk-Analyse in Teil B IDENTISCH ist (haengt nur von
    T_days ab, nicht von Delta/Rang einer einzelnen Zeile), wird es hier
    EINMAL pro Lauf berechnet statt - wie vor der Vereinheitlichung (siehe
    Moduldokstring Punkt 7/8/11) - implizit ueber evaluate_analog_set() PRO
    ZEILE neu (das war der teuerste Teil des Bulk-Laufs und ist jetzt nicht
    mehr noetig, da Teil B keine Modellbewertung mehr braucht - nur noch die
    tatsaechliche prozentuale Kursbewegung, siehe _payout_at_todays_scale())."""
    tt = training_table[training_table["T_days"] == T_days].set_index("entry_date")
    rows = []
    for _, arow in analogs.iterrows():
        d = arow["analog_date"]
        if d not in tt.index:
            continue
        trow = tt.loc[d]
        if isinstance(trow, pd.DataFrame):   # doppelte Zeitstempel - Sicherheitsnetz
            trow = trow.iloc[0]
        entry_S = float(trow["entry_S"])
        exit_S = float(trow["exit_S"])
        if entry_S <= 0.0 or np.isnan(entry_S) or np.isnan(exit_S):
            continue
        rows.append({
            "entry_date": d, "entry_S": entry_S, "exit_S": exit_S,
            "hold_days": (trow["exit_date"] - d).days,
        })
    return pd.DataFrame(rows)


def _summarize_unified(net_pnl: np.ndarray, pnl_norm: np.ndarray) -> dict:
    """Handelskennzahlen (Gewinnrate, Profit Factor, Erwartungswert,
    beste/schlechteste Trade) fuer die vereinheitlichte, an heutigen
    Strikes/Spot verankerte Payout-Serie (siehe Moduldokstring Punkt 11) -
    Gegenstueck zu analog_matching.summarize_analog_trades(), das auf dem
    ALTEN, per-era-modellierten evaluate_analog_set()-Ergebnis arbeitete
    und hier in Teil B nicht mehr verwendet wird. 'profit_factor' nutzt wie
    dort pnl_norm (normierte Rendite) statt net_pnl (absolute Punkte) -
    hier zwar ohnehin schon alle auf DERSELBEN heutigen Skala, aus
    Konsistenzgruenden trotzdem mit derselben Definition."""
    n_total = len(net_pnl)
    if n_total == 0:
        return {
            "n_trades": 0, "win_rate_pct": np.nan, "profit_factor": np.nan,
            "avg_pnl_norm_pct": np.nan, "avg_pnl": np.nan,
            "best_trade": np.nan, "worst_trade": np.nan,
        }
    n_wins = int((net_pnl >= 0.0).sum())
    gross_profit_norm = pnl_norm[pnl_norm >= 0.0].sum()
    gross_loss_norm = -pnl_norm[pnl_norm < 0.0].sum()
    return {
        "n_trades": n_total,
        "win_rate_pct": n_wins / n_total * 100.0,
        "profit_factor": (gross_profit_norm / gross_loss_norm) if gross_loss_norm > 0.0 else np.nan,
        "avg_pnl_norm_pct": float(np.mean(pnl_norm)) * 100.0,
        "avg_pnl": float(np.mean(net_pnl)),
        "best_trade": float(np.max(net_pnl)),
        "worst_trade": float(np.min(net_pnl)),
    }


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
                      analog_max_distance: float, analog_min_gap: int,
                      spot_today: float) -> pd.DataFrame | None:
    """Fuehrt den Analog-Tage-Lauf fuer JEDE Zeile von 'table' aus (siehe
    Moduldokstring Punkt 1-11). Gibt None zurueck (mit st.error) bei
    Abbruch, sonst ein DataFrame mit einer Zeile je distance_pct.

    VEREINHEITLICHT (siehe Moduldokstring Punkt 11): alle Kennzahlen dieser
    Funktion (Gewinnrate, Profit Factor, Ø norm. P&L%, Erwarteter Gewinn,
    Preis fuer PF=3, Newey-West-Reliability) werden aus DERSELBEN Payout-
    Serie berechnet - der Projektion der historischen PROZENTUALEN
    Kursbewegungen jedes Analog-Tages auf den HEUTIGEN Spot, ausgewertet
    mit den ECHTEN, in der Optionskette sichtbaren Strikes dieser Zeile
    (siehe _payout_at_todays_scale()). Dadurch entfaellt die teure
    Modellbewertung (Strike-Solve + Black-76-Pricing pro Analog-Tag x
    Zeile, vormals ueber evaluate_analog_set()) komplett - es wird nur noch
    EINMAL pro Lauf (nicht mehr pro Zeile) nachgeschlagen, wie sich der
    Kurs an jedem Analog-Tag tatsaechlich bewegt hat (siehe
    _lookup_entry_exit()), was den Bulk-Lauf spuerbar beschleunigt."""
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
        except ValueError as e:
            st.error(f"Analog-Suche fehlgeschlagen: {e}")
            return None

    st.caption(f"{len(analogs)} Analog-Tage gefunden (einmalig ermittelt, gilt fuer alle Zeilen unten).")

    # NEU (siehe Funktions-Docstring): entry_S/exit_S/hold_days EINMAL fuer
    # ALLE Zeilen nachschlagen, statt wie vor der Vereinheitlichung implizit
    # ueber evaluate_analog_set() PRO ZEILE neu zu modellieren - haengt nur
    # von T_days_used ab, nicht von Delta/Rang einer einzelnen Zeile.
    analog_prices = _lookup_entry_exit(analogs, training_table, T_days_used)
    if analog_prices.empty:
        st.error(
            f"Keine der gefundenen Analog-Tage hat gueltige entry_S/exit_S-Werte "
            f"fuer T_days={T_days_used} in der training_table."
        )
        return None

    entry_S_arr = analog_prices["entry_S"].to_numpy()
    exit_S_arr = analog_prices["exit_S"].to_numpy()
    entry_date_arr = analog_prices["entry_date"].to_numpy()
    hold_days_arr = analog_prices["hold_days"].to_numpy()

    progress = st.progress(0.0)
    status = st.empty()
    results = []
    n = len(table)
    for i, row in enumerate(table.itertuples()):
        status.text(f"Analog-Lauf {i + 1}/{n} (Ziel-Abstand {row.distance_pct:g}%)...")
        progress.progress((i + 1) / n)

        entry_premium_today = None
        if pd.notna(row.mid_put) and pd.notna(row.mid_call):
            entry_premium_today = float(row.mid_put) + float(row.mid_call)

        strikes_valid = pd.notna(row.strike_put) and pd.notna(row.strike_call)

        if entry_premium_today is None or entry_premium_today <= 0.0 or not strikes_valid:
            results.append({
                "distance_pct": row.distance_pct, "analog_n": None,
                "entry_premium_today": entry_premium_today,
                "expected_profit_short": None,
                "required_premium_pf3": None,
            })
            continue

        # NEU/VEREINHEITLICHT (siehe Funktions-Docstring): EINE Payout-Serie
        # - historische PROZENTUALE Kursbewegung auf den heutigen Spot
        # projiziert, gegen die ECHTEN heutigen Strikes ausgewertet - ist
        # jetzt die Grundlage FUER ALLE Kennzahlen dieser Zeile.
        payout, keep_mask = _payout_at_todays_scale(
            entry_S_arr, exit_S_arr, spot_today, float(row.strike_put), float(row.strike_call),
        )
        if len(payout) == 0:
            results.append({
                "distance_pct": row.distance_pct, "analog_n": 0,
                "entry_premium_today": entry_premium_today,
                "expected_profit_short": None,
                "required_premium_pf3": None,
            })
            continue

        net_pnl = entry_premium_today - payout
        pnl_norm = net_pnl / entry_premium_today

        trade_summary = _summarize_unified(net_pnl, pnl_norm)

        # WICHTIG: entry_date/hold_days ueber DIESELBE Maske ausrichten, die
        # _payout_at_todays_scale intern verwendet hat (siehe dortiger
        # Docstring "Alignment-Bug") - NICHT per Slicing raten, da das nur
        # zufaellig korrekt waere.
        unified = pd.DataFrame({
            "entry_date": entry_date_arr[keep_mask],
            "net_pnl": net_pnl,
            "pnl_norm": pnl_norm,
            "hold_days": hold_days_arr[keep_mask],
        })
        rel_full = reliability_summary(unified, nw_lags=None, n_subperiods=6)

        median_date = unified["entry_date"].median()
        train_eval = unified[unified["entry_date"] < median_date]
        val_eval = unified[unified["entry_date"] >= median_date]
        train_rel = reliability_summary(train_eval, nw_lags=None, n_subperiods=4)
        val_rel = reliability_summary(val_eval, nw_lags=None, n_subperiods=4)

        robust = False
        if train_rel["n_trades"] > 0 and val_rel["n_trades"] > 0:
            robust = bool(
                train_rel["ci_excludes_zero"] and val_rel["ci_excludes_zero"]
                and np.sign(train_rel["mean_pnl_norm"]) == np.sign(val_rel["mean_pnl_norm"])
                and np.sign(train_rel["mean_pnl_norm"]) > 0.0
            )

        # NEU/VEREINHEITLICHT (siehe Moduldokstring Punkt 8): jetzt EXAKT
        # (nicht mehr nur naeherungsweise) gleich dem Mittelwert von net_pnl,
        # da pnl_norm ueberall durch dasselbe entry_premium_today geteilt ist.
        expected_profit_short = trade_summary["avg_pnl"]
        required_premium_pf3 = _required_premium_for_profit_factor(payout, target_pf=3.0)

        results.append({
            "distance_pct": row.distance_pct,
            "analog_n": trade_summary["n_trades"],
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
        "Alle Kennzahlen unten (inkl. Profit Factor und Ø norm. P&L%) werden gegen "
        "die ECHTEN, oben sichtbaren Strikes UND die HEUTIGE Entry-Praemie dieser "
        "Zeile (Mid Put + Mid Call) berechnet - historische Analog-Tage fliessen nur "
        "ueber ihre PROZENTUALE Kursbewegung ein, projiziert auf den heutigen Spot "
        "(siehe Moduldokstring Punkt 11). 'Erw. Gewinn Short' (Punkte) ist dabei exakt "
        "der Mittelwert der so projizierten Netto-P&L-Werte. Bewertet NUR das "
        "Short-Bein (Praemien-Seite), NICHT die Wings. Fuer die Wing-Auswahl und vor "
        "allem fuer die echte kombinierte Equity-Kurve (Short+Long GEMEINSAM, inkl. "
        "Korrelation zwischen den Beinen) den Iron-Condor-Ladder-Backtest verwenden."
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
            spot,
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
            "required_premium_pf3": "Preis für PF=3 (Pkt)",
            "tstat_full": "t-Stat (gesamt)",
            "ci_excludes_zero_full": "CI ≠ 0 (gesamt)",
            "subperiod_matching": "Perioden-Match",
            "subperiod_nonempty": "Perioden gesamt",
            "tstat_val": "t-Stat (Val)",
            "robust_train_val": "Robust (Train+Val)",
        })[[
            "Ziel-Abstand % v. Spot", "Analog-Trades (n)", "Gewinnrate %", "Profit Factor",
            "Ø norm. P&L %", "Entry-Prämie heute (Pkt)",
            "Erw. Gewinn Short (Pkt)", "Preis für PF=3 (Pkt)",
            "t-Stat (gesamt)", "CI ≠ 0 (gesamt)",
            "Perioden-Match", "Perioden gesamt", "t-Stat (Val)", "Robust (Train+Val)",
        ]]
        st.caption(
            f"T_days fuer die Analog-Bewertung: "
            f"{int(stored['T_days_used'].iloc[0]) if len(stored) else 'n/a'} "
            f"(gerundet von {chosen_dte_exact:.2f} Tagen Restlaufzeit) · "
            f"{int(stored['n_analogs_total'].iloc[0]) if len(stored) else 0} Analog-Tage gefunden. "
            "'Robust (Train+Val)' = Konfidenzintervall schliesst 0 aus UND gleiches "
            "positives Vorzeichen in BEIDEN chronologischen Haelften des Analog-Sets. "
            "'Preis für PF=3' = welcher (einheitliche) Entry-Preis auf dem KOMPLETTEN "
            "Analog-Set (Short-Seite, ausgewertet mit den ECHTEN heutigen Strikes gegen "
            "die auf den heutigen Spot projizierten historischen Prozentbewegungen) "
            "einen Profit Factor von genau 3 ergeben haette - liegt 'Entry-Prämie heute' "
            "darueber, ist der (aus derselben Grundlage berechnete) 'Profit Factor' bei "
            "diesem Preis automatisch > 3, liegt sie darunter, entsprechend < 3 (siehe "
            "Moduldokstring Punkt 11; rein historische Ruecksimulation, keine "
            "Zukunftsgarantie)."
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