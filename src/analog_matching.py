"""
Analog-Matching fuer die IC-Leg-Auswahl.

GRUNDIDEE: statt eines global trainierten Klassifikators (train.py) wird
direkt mit historischen "Analog-Tagen" gearbeitet:

  1. Finde alle Handelstage innerhalb einer Distanz-Schwelle im PCA-Raum zu
     einem Query-Tag (Distanz im PCA-Raum der Tages-Features - siehe
     fit_similarity_space()).
  2. Rechne die Vola jedes Analog-Tags auf den HV-Rang des Query-Tags um
     (siehe target_vol_for_rank() - direkte Erweiterung von
     backtest.compute_rank_target_vol(), nur mit variablem statt festem
     Ziel-Rang), damit ein "73%-Rang-Tag" von 1935 fair mit einem von 2021
     vergleichbar ist. Das ist GENAU dasselbe Grundprinzip wie die im
     Dashboard bereits vorhandene "Fixer IV-Rang"-Simulation
     (backtest.FixedRankConfig/compute_rank_target_vol) - nur wird dort JEDER
     Handelstag der gesamten Historie auf einen festen Rang umgerechnet,
     hier NUR die Analog-Tage innerhalb der Distanz-Schwelle, und der
     Ziel-Rang ist nicht fest eingestellt, sondern der TATSAECHLICHE heutige
     Rang.
  3. Bewerte auf diesen ranggerechten Analog-Tagen einen Short- UND Long-
     Strangle (siehe reprice_leg_pair() - Long ist exakt das Spiegelbild des
     Short, da keine Transaktionskosten modelliert werden: pnl_norm_long =
     -pnl_norm_short, keine separate Simulation noetig - beide Richtungen
     werden aus EINEM Repricing-Durchgang abgeleitet).
  4. Suche per Grid-Search ueber Put-/Call-Delta-Kombinationen das robusteste
     Leg-Paar, jeweils fuer Short UND Long (siehe leg_pair_grid_search() -
     MIT Train/Validierungs-Split, um die Mehrfachvergleichs-Falle zu
     entschaerfen, siehe Docstring dort).

WICHTIG - Kausalitaet: find_analogs() beruecksichtigt fuer einen Query-Tag
NUR Kandidaten mit Datum < Query-Datum (bei Backtests des Verfahrens selbst
zusaetzlich: Analog-Tag muss VOR dem Query-Tag bereits abgeschlossen sein,
also exit_date < query_date - sonst wuerde man in die Zukunft des simulierten
'heute' schauen). Fuer den LIVE-Einsatz (Query-Tag = echter heutiger Tag)
ist das automatisch erfuellt, da alle Analog-Tage ohnehin in der
Vergangenheit liegen.

KORREKTUREN GEGENUEBER DER URSPRUENGLICHEN FASSUNG (Review vor dem Einbau ins
Dashboard - siehe Kommentare an den jeweiligen Stellen fuer Details):

  1) KRITISCH - Newey-West auf falscher Reihenfolge: find_analogs() liefert
     die Analog-Tage nach DISTANZ sortiert (naehester zuerst). Die
     urspruengliche evaluate_analog_set() hat diese Reihenfolge unveraendert
     an reliability_summary()/newey_west_tstat() weitergereicht. Newey-West
     (Bartlett-Kernel) korrigiert Autokorrelation zwischen Zeile i und i+j -
     das ist nur sinnvoll, wenn die Zeilenreihenfolge der KALENDER-Reihenfolge
     entspricht (im Ladder-Backtest automatisch der Fall, da dort taeglich
     ein Trade eroeffnet wird). Nach Distanz sortiert ist die Korrektur
     wirkungslos fuer das eigentliche Problem, das sie loesen soll: Analog-
     Tage haeufen sich in der Kalenderzeit (Vol-Regime dauern typischerweise
     mehrere Tage bis Wochen an, daher sehen benachbarte Kalendertage oft
     sehr aehnlich aus) - bei einem Test mit synthetischen Daten lagen ~40%
     der (chronologisch sortierten) Analog-Tage nur 1 Kalendertag auseinander,
     und bei einer konkreten Delta-Kombination (k=200, Put=Call=0.16) kippte
     dadurch NACHWEISLICH die Signifikanz-Aussage (ci_excludes_zero True vs.
     False) je nach Sortierreihenfolge. Fix: evaluate_analog_set() sortiert
     das Ergebnis jetzt IMMER nach entry_date, bevor es an
     reliability_summary() weitergegeben wird.

  2) min_gap_days war als "Handelstage" dokumentiert, aber ueber
     Kalendertage (np.timedelta64(..., "D")) auf dem Datum implementiert.
     Fix: find_analogs() nutzt jetzt die Position in der (aufsteigend
     sortierten) `space.dates`-Liste - das ist exakt Handelstage, wie im
     Docstring behauptet.

  3) reprice_leg_pair()/evaluate_analog_set() haben Short und Long bisher
     ueber einen 'side'-Parameter GETRENNT bewertet - obwohl Long exakt das
     Spiegelbild von Short ist (keine Transaktionskosten), wurde bei Bedarf
     an beiden Richtungen der komplette (Strike-Solve + Black-76-Bewertungs-)
     Durchlauf ZWEIMAL ausgefuehrt. Jetzt liefert reprice_leg_pair() beide
     Richtungen aus einem einzigen Durchlauf zurueck - sowohl schneller als
     auch garantiert konsistent. leg_pair_grid_search() bewertet dadurch pro
     Delta-Kombination automatisch BEIDE Richtungen und meldet zurueck, welche
     (falls ueberhaupt eine) robust ist.

  4) NEU - Clustering-Diagnose (analog_clustering_diagnostic()): meldet, wie
     stark sich die gefundenen Analog-Tage in der Kalenderzeit haeufen. Das
     ist keine Korrektur der Statistik selbst (das leistet Punkt 1), sondern
     eine explizite Warnung fuers Dashboard, wenn ein grosser Teil der
     "unabhaengigen" Analog-Tage in Wahrheit aus wenigen historischen
     Ereignissen (z.B. einer einzelnen Krise) stammt - das kann die
     Newey-West-Korrektur zwar entschaerfen, aber nicht vollstaendig
     kompensieren, wenn die effektive Anzahl unabhaengiger Beobachtungen
     sehr viel kleiner als die Anzahl gefundener Analog-Tage ist.

  5) NEU - T_days-Kompatibilitaetspruefung (available_t_days()): die
     Analog-Bewertung nutzt entry_S/exit_S aus der vorab gebauten
     training_table (siehe labels.py), die nur fuer die dort konfigurierten
     StrangleLabelConfig.T_days_list-Werte existieren (Default [2,3,5,7,10,
     14]). Ein T_days ausserhalb dieser Liste fuehrte bisher STILLSCHWEIGEND
     zu einem leeren Ergebnis (0 Zeilen ueberall, kein Fehler). Jetzt wird
     das explizit geprueft und mit einer klaren Fehlermeldung abgebrochen.

  6) WICHTIGE ERGAENZUNG (kein Bugfix, sondern eine bewusste methodische
     Ueberarbeitung nach Nutzer-Feedback) - zwei IV-Modi statt nur Rang:
     der urspruengliche Ansatz (Punkt 2 der Grundidee oben) rechnet den
     PERZENTIL-RANG von heute auf die HV-Bandbreite jedes Analog-Tages um.
     Das ist ein rein theoretisches Konstrukt (ein "65%-Rang-Tag" ist keine
     Aussage ueber ein reales Marktbewertungsniveau, sondern eine relative
     Position in einer 90+ Jahre alten Verteilung) und spiegelt NICHT wider,
     wie sich das TATSAECHLICHE heutige Bewertungsniveau (die IV, zu der
     man heute wirklich handeln wuerde) an den Analog-Tagen verhalten haette.
     evaluate_analog_set()/leg_pair_grid_search() unterstuetzen daher WEITER
     zwei explizite Modi (Parameter `vol_mode`):
       - "direct": verwendet auf JEDEM Analog-Tag exakt dieselbe absolute
         Vola `today_iv` - das heutige Bewertungsniveau. Beantwortet:
         "Haette ich HEUTE zu HEUTIGEN Konditionen gehandelt - wie waere
         die Struktur gelaufen, waeren die tatsaechlichen Kursbewegungen
         der historisch aehnlichsten Tage eingetreten?"
       - "rank": die Perzentil-Umrechnung (s.o., siehe target_vol_for_rank).
     (Siehe Punkt 8 fuer den seit dem Nutzer-Feedback ausschliesslich
     verwendeten Modus im High-Level-Einstiegspunkt recommend_for_today().)

  7) NEU - summarize_analog_trades(): dieselben Kennzahlen (Gewinnrate,
     Profit Factor, Erwartungswert/Trade, beste/schlechteste Trade) wie
     backtest.summarize_trades() bei der Ladder-Simulation, hier fuer die
     Analog-Tage EINER Delta-Kombination/Richtung - fehlten in der
     urspruenglichen Fassung komplett (dort gab es nur die
     Newey-West-Kennzahlen aus reliability_summary(), aber keine der
     "einfachen" Handelskennzahlen).

     KORREKTUR (siehe Diskussion "dashboard_livechain.py Profit Factor
     ergibt widerspruechliche Preis-Schwelle"): 'profit_factor' summierte
     urspruenglich net_pnl_{side} - ABSOLUTE Punktbetraege - ueber alle
     Analog-Tage. Bei einer Analog-Suche mit grosser max_distance (siehe
     AnalogConfig) koennen die gefundenen Analog-Tage die GESAMTE
     SPX-Historie seit 1927 umfassen - der Index stand dort bei voellig
     unterschiedlichen absoluten Niveaus (z.B. ~5 Punkte 1932 vs. ~7500
     Punkte heute). Eine Summe absoluter Punktbetraege UEBER diese Aeren
     hinweg gewichtet automatisch die betragsmaessig grossen, MODERNEN Tage
     massiv staerker als alte Tage - unabhaengig davon, ob diese relativ
     gesehen ueberhaupt schlechter liefen. 'profit_factor' wird daher jetzt
     aus pnl_norm_{side} (auf die jeweilige TAGESEIGENE Entry-Praemie
     normierte Rendite, bereits an anderer Stelle im Modul fuer genau diesen
     Zweck verwendet) statt aus net_pnl_{side} berechnet - jeder Analog-Tag
     zaehlt dadurch mit GLEICHEM Gewicht, unabhaengig vom Index-Niveau
     seiner Aera. 'avg_pnl'/'best_trade'/'worst_trade' bleiben bewusst in
     absoluten Punkten (fuer die Einordnung einzelner Trades sinnvoll),
     nur 'profit_factor' - eine aggregierte Kennzahl UEBER alle Analog-Tage
     hinweg - wechselt auf die normierte Basis.

  8) NEU (Vereinfachung nach Nutzer-Feedback) - recommend_for_today() nutzt
     jetzt AUSSCHLIESSLICH den rang-basierten Modus (vol_mode="rank") und
     erwartet dafuer direkt einen vom Nutzer eingegebenen Ziel-HV-Rang
     (`target_rank_pct`, in Prozent, 0-100 - Werte ausserhalb extrapolieren
     bewusst ueber das historische Fenster hinaus, identisch zu
     backtest.FixedRankConfig.target_rank_pct). Der fruehere `vol_mode`-/
     `today_iv_override`-Parameter dieser Funktion (Punkt 6, "direct"-Modus
     bzw. ein manuell eingegebener/aus HV berechneter absoluter IV-Wert, aus
     dem der Rang ERST rueckgerechnet wurde) entfaellt dadurch.

     Der eigentliche Rechenkern ist UNVERAENDERT: fuer JEDEN Analog-Tag wird
     weiterhin per target_vol_for_rank() rollierend zurueckgerechnet, welche
     IV an GENAU DIESEM Tag noetig gewesen waere, um exakt den eingegebenen
     Rang in seinem eigenen HV-Min/Max-Fenster (hv_length/rank_lookback_days)
     zu erreichen - z.B. bedeutet Rang 40% an einem ruhigen Markttag eine
     niedrige IV, in einer Vola-Krise (z.B. Finanzkrise 2008) bei DEMSELBEN
     Rang 40% dagegen eine deutlich hoehere IV, weil sich dort das gesamte
     HV-Min/Max-Fenster auf hoeherem Niveau befindet. Identisches Prinzip zum
     'Fixer IV-Rang'-Modus im Iron-Condor-Ladder-Backtest
     (backtest.FixedRankConfig/compute_rank_target_vol) - nur eben auf die
     Analog-Tage innerhalb der Distanz-Schwelle statt auf JEDEN Handelstag
     angewendet.

     evaluate_analog_set()/leg_pair_grid_search() bleiben UNVERAENDERT und
     unterstuetzen weiterhin beide Modi (siehe Punkt 6) fuer direkten
     Gebrauch ausserhalb von recommend_for_today().

  9) NEU (Umstellung nach Nutzer-Feedback) - Distanz-Schwelle statt fester
     Anzahl (k): find_analogs() verwendet jetzt AnalogConfig.max_distance
     statt AnalogConfig.k - es werden ALLE Kandidaten-Tage mit Distanz
     <= max_distance verwendet, unabhaengig von ihrer Anzahl (frueher: die
     k naechsten Tage, unabhaengig von ihrer absoluten Distanz). n_analogs
     kann dadurch stark schwanken, je nachdem wie 'ungewoehnlich' der
     Query-Tag im Aehnlichkeits-Raum liegt. analog_distance_distribution()
     liefert eine Kalibrierungs-Vorschau (Perzentile der tatsaechlichen
     Distanzen aller zulaessigen Kandidaten), damit sich im Dashboard eine
     sinnvolle Schwelle ablesen laesst, OHNE dafuer schon den vollen
     Grid-Search auszufuehren.

  10) NEU - Delta-Eingabe per Strike (delta_for_strike()): das Dashboard
      erlaubt jetzt, Put-/Call-Bein wahlweise per Strike statt per Delta
      einzugeben. delta_for_strike() berechnet dafuer das zu einem
      gegebenen Strike gehoerige Delta NUMERISCH per Bump-and-Reprice auf
      dem Forward-Preis (zentrale Differenz), unter Verwendung von
      forward_price()/leg_price() - also GENAU der Bewertungslogik (inkl.
      Vol-Surface/Smile ueber `vol_surface`), die auch reprice_leg_pair()/
      solve_strike_delta_consistent() verwenden. Es gibt bewusst KEINE
      zweite, eigene Delta-Formel - dadurch bleibt das Ergebnis auch dann
      konsistent, wenn sich die Bewertungslogik in black76.py spaeter
      aendert. Delta wird als POSITIVE Groesse (Betrag) zurueckgegeben,
      identisch zur sonst im Modul verwendeten Konvention (Put-Deltas
      werden ebenfalls als positive Zahlen wie 0.10, 0.16 angegeben, nicht
      mit oekonomischem Vorzeichen).

  11) NEU - entry_S/exit_S im Ergebnis von evaluate_analog_set(): fuer die
      "Preis fuer Profit Factor X"-Kennzahl in dashboard_livechain.py wird
      die TATSAECHLICHE prozentuale Kursbewegung jedes Analog-Tages
      benoetigt (nicht nur der bereits vorhandene, modellierte pnl_norm bei
      einem PER-ERA-spezifischen Strike) - siehe dort fuer die Begruendung,
      warum eine Skalen-konsistente Umrechnung auf den HEUTIGEN Spot noetig
      ist. evaluate_analog_set() reicht entry_S/exit_S daher jetzt einfach
      aus der ohnehin bereits nachgeschlagenen training_table-Zeile durch,
      statt sie zu verwerfen - keine Verhaltensaenderung fuer bestehende
      Aufrufer, die diese Spalten ignorieren.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from black76 import VolSurfaceConfig, forward_price, leg_price
from backtest import compute_hv, solve_strike_delta_consistent
from features import FeatureConfig, build_feature_matrix, feature_columns
from stats import reliability_summary


# ═══════════════════════════════════════════════════════════════════════
#  1) AEHNLICHKEITS-RAUM (PCA, whitened)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SimilaritySpaceConfig:
    variance_target: float = 0.80   # kumulierte erklaerte Varianz, die die gewaehlten Komponenten abdecken sollen
    min_components: int = 3
    max_components: int = 20


@dataclass
class SimilaritySpace:
    scaler: StandardScaler
    pca: PCA
    feature_cols: list[str]
    dates: pd.Series          # Datum je Zeile, AUFSTEIGEND sortiert, gleiche Reihenfolge wie scores
    scores: np.ndarray        # (n_days, n_components), whitened PCA-Scores


def fit_similarity_space(feature_df: pd.DataFrame, cfg: SimilaritySpaceConfig | None = None,
                          max_date: pd.Timestamp | None = None) -> SimilaritySpace:
    """Baut den Aehnlichkeits-Raum: StandardScaler + PCA(whiten=True) auf den
    Tages-Features (siehe features.build_feature_matrix - EIN Score-Vektor
    pro Handelstag, KEIN T_days darin, da Aehnlichkeit eine Eigenschaft des
    Tages ist, nicht der Handelsstruktur).

    whiten=True: jede Hauptkomponente wird auf Einheitsvarianz normiert, bevor
    die euklidische Distanz berechnet wird - sonst wuerde PC1 (die per
    Definition die groesste Varianz traegt) die Distanz dominieren, nur weil
    sie 'PC1' ist, nicht weil sie inhaltlich wichtiger fuer Aehnlichkeit waere.

    max_date: falls gesetzt, werden nur Zeilen bis (exklusive) max_date fuers
    Fitten verwendet - Voraussetzung fuer einen sauberen Walk-Forward-
    Backtest des Analog-Verfahrens selbst (siehe Modul-Docstring). Der
    Query-Tag selbst darf/soll enthalten sein (siehe recommend_for_today,
    das max_date=query_date+1 Tag uebergibt) - das ist keine Zukunftsinfo,
    sondern genau der Tag, fuer den gerade eine Empfehlung gesucht wird."""
    if cfg is None:
        cfg = SimilaritySpaceConfig()

    cols = feature_columns(feature_df)
    df = feature_df.dropna(subset=cols).reset_index(drop=True)
    if max_date is not None:
        df = df[df["date"] < max_date].reset_index(drop=True)
    # WICHTIG fuer find_analogs()/min_gap_days: df ist bereits nach 'date'
    # aufsteigend sortiert (build_feature_matrix() sortiert selbst), hier
    # nochmal explizit sichergestellt, da find_analogs() sich auf diese
    # Reihenfolge verlaesst, um den Handelstage-Abstand ueber Array-Positionen
    # statt ueber Kalendertage zu bestimmen (siehe KORREKTUR 2 oben).
    df = df.sort_values("date").reset_index(drop=True)

    if df.empty:
        hint = f" vor {pd.Timestamp(max_date).date()}" if max_date is not None else ""
        raise ValueError(
            f"Keine Handelstage{hint} mit vollstaendigen (NaN-freien) Features "
            "gefunden - vermutlich zu wenig Historie fuer die laengsten Feature-"
            "Fenster (z.B. ret_5y braucht 5 Jahre, hv_252d braucht ~1 Jahr "
            "Warmup). Ein spaeteres Datum waehlen."
        )

    X = df[cols].to_numpy(dtype=float)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # Erst mit vollem Rang fitten, um zu sehen, wie viele Komponenten fuer
    # das Varianz-Ziel noetig sind - dann mit genau dieser Anzahl (+ whiten)
    # final fitten.
    probe = PCA(n_components=min(len(cols), 40), random_state=42).fit(X_scaled)
    cum_var = np.cumsum(probe.explained_variance_ratio_)
    n_components = int(np.searchsorted(cum_var, cfg.variance_target) + 1)
    n_components = max(cfg.min_components, min(n_components, cfg.max_components, len(cols)))

    pca = PCA(n_components=n_components, whiten=True, random_state=42).fit(X_scaled)
    scores = pca.transform(X_scaled)

    return SimilaritySpace(scaler=scaler, pca=pca, feature_cols=cols, dates=df["date"], scores=scores)


def transform_query(space: SimilaritySpace, feature_row: pd.Series) -> np.ndarray:
    """Bringt EINEN neuen Feature-Vektor (z.B. den heutigen Tag) in denselben
    Aehnlichkeits-Raum wie beim Fit."""
    x = feature_row[space.feature_cols].to_numpy(dtype=float).reshape(1, -1)
    if np.isnan(x).any():
        raise ValueError("Query-Tag hat NaN in mindestens einem Feature (Warmup? "
                          "z.B. ret_5y in den ersten 5 Jahren der Historie).")
    return space.pca.transform(space.scaler.transform(x))[0]


# ═══════════════════════════════════════════════════════════════════════
#  2) ANALOG-SUCHE
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AnalogConfig:
    max_distance: float = 3.0   # Alle Kandidaten-Tage MIT DISTANZ <= max_distance werden
                                 # verwendet, unabhaengig von ihrer Anzahl (ersetzt die
                                 # fruehere feste Anzahl k der naechsten Tage - siehe Punkt 9
                                 # im Moduldokstring). Skala haengt vom gewhitenten PCA-Raum
                                 # ab (siehe SimilaritySpaceConfig) - zur Kalibrierung siehe
                                 # analog_distance_distribution().
    min_gap_days: int = 5       # Mindestabstand in HANDELSTAGEN zum Query-Tag (siehe KORREKTUR 2)


def find_analogs(query_date: pd.Timestamp, query_score: np.ndarray, space: SimilaritySpace,
                  cfg: AnalogConfig | None = None) -> pd.DataFrame:
    """Findet ALLE Handelstage mit Distanz <= cfg.max_distance zu query_score
    (euklidische Distanz im gewhitenten PCA-Raum), beschraenkt auf Kandidaten
    VOR query_date (Kausalitaet - siehe Modul-Docstring).

    GEAENDERT (siehe Punkt 9 im Moduldokstring): frueher wurden die cfg.k
    NAEHESTEN Tage verwendet (feste Anzahl, variable Distanz). Jetzt gilt es
    umgekehrt: eine feste Distanz-Schwelle, variable (und a priori unbekannte)
    Anzahl an Tagen. Das bedeutet auch, dass n_analogs stark schwanken kann -
    je nachdem, wie 'ungewoehnlich' der Query-Tag im Aehnlichkeits-Raum liegt
    (siehe analog_clustering_diagnostic() und die Warnung im Dashboard bei
    sehr wenigen gefundenen Tagen).

    KORREKTUR (siehe Punkt 2 im Moduldokstring): min_gap_days war zuvor ueber
    Kalendertage (np.timedelta64) implementiert, obwohl als "Handelstage"
    dokumentiert. space.dates ist aufsteigend sortiert (siehe
    fit_similarity_space) - der Handelstage-Abstand zum Query-Tag entspricht
    daher exakt der Differenz der Array-Positionen, hier jetzt korrekt
    darueber bestimmt."""
    if cfg is None:
        cfg = AnalogConfig()

    dates = space.dates.to_numpy()
    query_pos = int(np.searchsorted(dates, np.datetime64(query_date), side="left"))
    max_allowed_pos = query_pos - max(cfg.min_gap_days, 0)
    if max_allowed_pos <= 0:
        raise ValueError(
            f"Keine Analog-Kandidaten vor {pd.Timestamp(query_date).date()} mit "
            f"min_gap_days={cfg.min_gap_days} Handelstagen Abstand gefunden "
            "(zu wenig Historie vor dem Query-Tag)."
        )

    cand_idx = np.arange(max_allowed_pos)
    dist = np.linalg.norm(space.scores[cand_idx] - query_score[None, :], axis=1)

    within = dist <= cfg.max_distance
    if not within.any():
        raise ValueError(
            f"Keine Analog-Tage innerhalb der Distanz-Schwelle {cfg.max_distance:.4f} "
            f"gefunden (kleinste verfuegbare Distanz: {dist.min():.4f}). "
            "max_distance erhoehen."
        )

    chosen = cand_idx[within]

    return pd.DataFrame({
        "analog_date": dates[chosen],
        "distance": dist[within],
    }).sort_values("distance").reset_index(drop=True)


def analog_distance_distribution(daily_df: pd.DataFrame, query_date: pd.Timestamp,
                                  min_gap_days: int = 5,
                                  space_cfg: SimilaritySpaceConfig | None = None) -> np.ndarray:
    """NEU (siehe Punkt 9 im Moduldokstring) - reine Kalibrierungs-Hilfe fuers
    Dashboard: fittet den Aehnlichkeits-Raum wie recommend_for_today() und
    liefert die (aufsteigend sortierten) Distanzen ALLER zulaessigen
    Kandidaten-Tage zum Query-Tag - VOR Anwendung einer max_distance-
    Schwelle. Damit laesst sich z.B. ueber Perzentile ablesen, welche
    Schwelle etwa 'die aehnlichsten 5% aller Tage' bedeutet, ohne dafuer
    schon den vollen Grid-Search auszufuehren.

    Bewusst unabhaengig von find_analogs()/recommend_for_today() gehalten
    (etwas Code-Duplikation beim Fitten des Raums), damit diese leichte
    Vorschau-Funktion nicht an die schwerere Pipeline (Grid-Search etc.)
    gekoppelt ist - ein Aufruf dieser Funktion soll im Dashboard bei jeder
    Aenderung von query_date/min_gap_days guenstig moeglich sein, ohne dass
    dafuer bereits Delta-Grids o.ae. feststehen muessten."""
    feature_df = build_feature_matrix(daily_df, FeatureConfig())
    query_date = pd.Timestamp(query_date)
    if query_date not in set(feature_df["date"]):
        raise ValueError(f"{query_date.date()} ist kein Handelstag in den geladenen Kursdaten.")

    space = fit_similarity_space(feature_df, space_cfg, max_date=query_date + pd.Timedelta(days=1))
    query_row = feature_df.loc[feature_df["date"] == query_date].iloc[0]
    query_score = transform_query(space, query_row)

    dates = space.dates.to_numpy()
    query_pos = int(np.searchsorted(dates, np.datetime64(query_date), side="left"))
    max_allowed_pos = query_pos - max(min_gap_days, 0)
    if max_allowed_pos <= 0:
        raise ValueError(
            f"Keine Analog-Kandidaten vor {query_date.date()} mit "
            f"min_gap_days={min_gap_days} Handelstagen Abstand gefunden."
        )

    cand_idx = np.arange(max_allowed_pos)
    dist = np.linalg.norm(space.scores[cand_idx] - query_score[None, :], axis=1)
    return np.sort(dist)


def analog_clustering_diagnostic(analogs: pd.DataFrame, gap_days: int = 3) -> dict:
    """NEU (siehe Punkt 4 im Moduldokstring): reine Diagnose, KEINE
    Korrektur - meldet, wie stark sich die gefundenen Analog-Tage in der
    Kalenderzeit haeufen (z.B. weil viele aus derselben historischen Krise
    stammen). Ein grosser Anteil sehr kleiner Abstaende bedeutet, dass die
    effektive Anzahl UNABHAENGIGER Beobachtungen deutlich kleiner ist als
    len(analogs) - die Newey-West-Korrektur in reliability_summary() federt
    das ab (siehe KORREKTUR 1), ersetzt aber keine gesunde Skepsis bei
    extremem Clustering."""
    if len(analogs) < 2:
        return {"frac_within_gap": 0.0, "median_gap_days": np.nan, "n_analogs": len(analogs)}

    sorted_dates = analogs["analog_date"].sort_values().reset_index(drop=True)
    gaps = sorted_dates.diff().dt.days.dropna()
    return {
        "frac_within_gap": float((gaps <= gap_days).mean()) if len(gaps) else 0.0,
        "median_gap_days": float(gaps.median()) if len(gaps) else np.nan,
        "n_analogs": len(analogs),
        "gap_days_threshold": gap_days,
    }


# ═══════════════════════════════════════════════════════════════════════
#  3) RANG-SKALIERTE BEWERTUNG (Short- UND Long-Strangle, EIN Durchgang)
# ═══════════════════════════════════════════════════════════════════════

def hv_rank_bounds(daily_df: pd.DataFrame, hv_length: int, lookback_days: int) -> pd.DataFrame:
    """Rollierende hv/hv_min/hv_max je Handelstag (kausal) - dieselbe
    Grundlage wie backtest.compute_rank_target_vol(), hier aber mit
    hv_min/hv_max als EIGENSTAENDIGE Spalten, damit spaeter pro Analog-Tag
    auf einen BELIEBIGEN (variablen) Ziel-Rang umgerechnet werden kann,
    statt wie dort auf einen fest eingestellten."""
    if "resolution" in daily_df.columns:
        daily_df = daily_df[daily_df["resolution"] == "daily"]
    daily_df = daily_df.sort_values("date").reset_index(drop=True)

    hv = compute_hv(daily_df, hv_length)
    hv_min = hv.rolling(lookback_days, min_periods=lookback_days).min()
    hv_max = hv.rolling(lookback_days, min_periods=lookback_days).max()

    return pd.DataFrame({"date": daily_df["date"], "hv": hv, "hv_min": hv_min, "hv_max": hv_max})


def target_vol_for_rank(hv_min: float, hv_max: float, rank: float) -> float:
    """Vola, die den Analog-Tag auf EXAKT denselben Rang bringt, den der
    Query-Tag (='heute') in seinem eigenen rollierenden Fenster hat.
    rank als Bruchteil 0..1 (nicht Prozent)."""
    if np.isnan(hv_min) or np.isnan(hv_max) or hv_max <= hv_min:
        return np.nan
    return hv_min + rank * (hv_max - hv_min)


def available_t_days(training_table: pd.DataFrame) -> list[int]:
    """NEU (siehe Punkt 5 im Moduldokstring): welche T_days-Werte ueberhaupt
    in der training_table vorhanden sind (siehe labels.StrangleLabelConfig.
    T_days_list, mit dem die Tabelle gebaut wurde). Ein T_days ausserhalb
    dieser Liste liefert in evaluate_analog_set() STILLSCHWEIGEND 0 Zeilen -
    diese Funktion macht die gueltigen Werte explizit abfragbar (z.B. fuer
    ein Dashboard-Dropdown, statt eines freien Sliders)."""
    return sorted(training_table["T_days"].unique().tolist())


def reprice_leg_pair(entry_S: float, exit_S: float, T_days: int, target_vol: float,
                      put_delta: float, call_delta: float, r_free: float, dividend_yield: float,
                      vol_surface: VolSurfaceConfig) -> dict | None:
    """Bewertet EINEN Strangle (Put-Delta/Call-Delta duerfen unterschiedlich
    sein - 'Strangle' im weiteren Sinn) auf einem Analog-Tag, mit target_vol
    STATT der tatsaechlichen historischen HV dieses Tages (siehe
    target_vol_for_rank).

    KORREKTUR (siehe Punkt 3 im Moduldokstring): liefert jetzt BEIDE
    Richtungen (Short und Long) aus einem einzigen Strike-Solve/Black-76-
    Durchlauf zurueck, statt wie zuvor ueber einen 'side'-Parameter nur eine
    Richtung zu liefern und bei Bedarf an beiden Richtungen denselben
    (identisch teuren) Durchlauf zweimal auszufuehren. Long ist exakt das
    Spiegelbild von Short (net_pnl_long = -net_pnl_short), da keine
    Transaktionskosten modelliert werden - das ist keine Naeherung."""
    if np.isnan(target_vol) or target_vol <= 0.0:
        return None

    T_years = T_days / 365.0
    F_entry = forward_price(entry_S, r_free, dividend_yield, T_years)

    K_put, _ = solve_strike_delta_consistent(F_entry, put_delta, T_years, target_vol, is_put=True, cfg=vol_surface)
    K_call, _ = solve_strike_delta_consistent(F_entry, call_delta, T_years, target_vol, is_put=False, cfg=vol_surface)

    premium_put = leg_price(K_put, True, T_years, F_entry, target_vol, r_free, vol_surface)
    premium_call = leg_price(K_call, False, T_years, F_entry, target_vol, r_free, vol_surface)
    entry_premium = premium_put + premium_call
    if entry_premium <= 0.0:
        return None

    settle_put = max(K_put - exit_S, 0.0)
    settle_call = max(exit_S - K_call, 0.0)
    net_pnl_short = entry_premium - (settle_put + settle_call)
    pnl_norm_short = net_pnl_short / entry_premium

    return {
        "K_put": K_put, "K_call": K_call, "entry_premium": entry_premium,
        "net_pnl_short": net_pnl_short, "pnl_norm_short": pnl_norm_short,
        "net_pnl_long": -net_pnl_short, "pnl_norm_long": -pnl_norm_short,
    }


def delta_for_strike(K: float, is_put: bool, spot: float, T_days: int, vol: float,
                      r_free: float, dividend_yield: float,
                      vol_surface: VolSurfaceConfig, bump_frac: float = 1e-4) -> float:
    """NEU (siehe Punkt 10 im Moduldokstring) - numerisches (zentrales) Delta
    fuer einen GEGEBENEN Strike K, per Bump-and-Reprice auf dem
    Forward-Preis F. Bewusst OHNE eigene/zweite Black-76-Delta-Formel
    implementiert - nutzt ausschliesslich die bereits vorhandenen,
    oeffentlichen Bewertungsfunktionen (forward_price, leg_price), damit das
    Ergebnis automatisch zu JEDER Vol-Surface-/Smile-Logik konsistent ist,
    die leg_price() ueber `vol_surface` intern anwendet (dieselbe Logik, die
    auch reprice_leg_pair()/solve_strike_delta_consistent() verwenden) -
    auch falls sich diese Logik spaeter aendert, muss hier nichts angepasst
    werden.

    bump_frac: relative Bump-Groesse auf F (Default 0.01% von F) - klein
    genug fuer eine gute Naeherung der Ableitung, gross genug, um nicht in
    numerische Rundungsprobleme zu laufen.

    Rueckgabe als POSITIVE Groesse (Betrag), konsistent zur Delta-Konvention
    im Rest des Moduls, wo Put-Deltas ebenfalls als positive Zahlen (z.B.
    0.10, 0.16) angegeben werden statt mit dem oekonomischen Vorzeichen."""
    T_years = T_days / 365.0
    F = forward_price(spot, r_free, dividend_yield, T_years)
    h = max(abs(F) * bump_frac, 1e-6)

    price_up = leg_price(K, is_put, T_years, F + h, vol, r_free, vol_surface)
    price_down = leg_price(K, is_put, T_years, F - h, vol, r_free, vol_surface)

    return abs((price_up - price_down) / (2.0 * h))


def evaluate_analog_set(analogs: pd.DataFrame, training_table: pd.DataFrame, bounds: pd.DataFrame,
                         T_days: int, put_delta: float, call_delta: float,
                         r_free: float, dividend_yield: float, vol_surface: VolSurfaceConfig,
                         vol_mode: str = "direct", today_iv: float | None = None,
                         today_rank: float | None = None) -> pd.DataFrame:
    """Fuer jeden Analog-Tag: entry_S/exit_S aus der Trainings-Tabelle holen
    (dort bereits korrekt mit der Kalendertage-Suchlogik aus labels.py
    bestimmt - wird hier NICHT neu berechnet), dann per reprice_leg_pair()
    neu bewerten (Short UND Long, siehe dort) - mit welcher Vola, haengt vom
    `vol_mode` ab (siehe Punkt 6 im Moduldokstring):

      vol_mode="direct": JEDER Analog-Tag wird mit derselben absoluten Vola
        `today_iv` bewertet (das heutige, tatsaechliche Bewertungsniveau) -
        `bounds`/`today_rank` werden dabei NICHT zur Bewertung herangezogen.
      vol_mode="rank": klassische Perzentil-Umrechnung (siehe
        target_vol_for_rank) - `today_rank` ist in diesem Modus Pflicht,
        `bounds` liefert hv_min/hv_max je Analog-Tag. Das ist der Modus, den
        recommend_for_today() verwendet (siehe Punkt 8 im Moduldokstring).

    KORREKTUR (siehe Punkt 1 im Moduldokstring, KRITISCH): das Ergebnis wird
    vor der Rueckgabe explizit nach 'entry_date' (chronologisch) sortiert.
    `analogs` kommt aus find_analogs() nach DISTANZ sortiert - in dieser
    Reihenfolge waere die spaeter in reliability_summary() aufgerufene
    Newey-West-Korrektur wirkungslos (siehe Beleg im Moduldokstring: bei
    einer getesteten Konfiguration kippte dadurch nachweislich die
    Signifikanz-Aussage).

    NEU (siehe Punkt 11 im Moduldokstring): entry_S/exit_S werden jetzt mit
    ins Ergebnis durchgereicht (waren schon in `trow` vorhanden, wurden aber
    bisher nicht in die Ausgabe uebernommen) - fuer Kennzahlen, die die
    TATSAECHLICHE prozentuale Kursbewegung eines Analog-Tages brauchen,
    unabhaengig vom hier verwendeten, per-era-spezifischen Modell-Strike
    (siehe dashboard_livechain.py, 'Preis fuer Profit Factor X')."""
    if vol_mode not in ("direct", "rank"):
        raise ValueError(f"Unbekannter vol_mode={vol_mode!r} - erwartet 'direct' oder 'rank'.")
    if vol_mode == "direct" and (today_iv is None or np.isnan(today_iv) or today_iv <= 0.0):
        raise ValueError("vol_mode='direct' braucht ein gueltiges today_iv (>0).")
    if vol_mode == "rank" and (today_rank is None or np.isnan(today_rank)):
        raise ValueError("vol_mode='rank' braucht ein gueltiges today_rank.")
    if T_days not in training_table["T_days"].unique():
        raise ValueError(
            f"T_days={T_days} ist nicht in der training_table vorhanden. "
            f"Verfuegbare Werte: {available_t_days(training_table)} "
            "(siehe labels.StrangleLabelConfig.T_days_list, mit der die "
            "Trainingstabelle gebaut wurde)."
        )

    bounds_idx = bounds.set_index("date")
    tt = training_table[training_table["T_days"] == T_days].set_index("entry_date")

    rows = []
    for _, arow in analogs.iterrows():
        d = arow["analog_date"]
        if d not in tt.index:
            continue
        trow = tt.loc[d]
        if isinstance(trow, pd.DataFrame):   # falls doppelte Zeitstempel - Sicherheitsnetz
            trow = trow.iloc[0]

        if vol_mode == "direct":
            target_vol = today_iv
        else:
            if d not in bounds_idx.index:
                continue
            brow = bounds_idx.loc[d]
            target_vol = target_vol_for_rank(brow["hv_min"], brow["hv_max"], today_rank)

        result = reprice_leg_pair(
            trow["entry_S"], trow["exit_S"], T_days, target_vol,
            put_delta, call_delta, r_free, dividend_yield, vol_surface,
        )
        if result is None:
            continue

        hold_days = (trow["exit_date"] - d).days
        rows.append({
            "entry_date": d, "hold_days": hold_days, "distance": arow["distance"],
            "target_vol": target_vol,
            "entry_S": trow["entry_S"], "exit_S": trow["exit_S"],
            **result,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("entry_date").reset_index(drop=True)
    return out


def summarize_analog_trades(evaluated: pd.DataFrame, side: str) -> dict:
    """NEU (siehe Punkt 7 im Moduldokstring): dieselben "einfachen"
    Handelskennzahlen wie backtest.summarize_trades() beim Ladder-Backtest
    (Gewinnrate, Profit Factor, Erwartungswert/Trade, beste/schlechteste
    Trade), hier fuer das evaluate_analog_set()-Ergebnis EINER
    Delta-Kombination, side='short' oder 'long'.

    KORREKTUR (siehe Punkt 7 im Moduldokstring): 'profit_factor' wird jetzt
    aus pnl_norm_{side} (normierte Rendite je Analog-Tag) statt aus
    net_pnl_{side} (absolute Punkte) berechnet. Grund: bei einer Analog-Suche
    ueber die gesamte SPX-Historie hinweg (moegliche Spanne 1927 bis heute)
    stehen absolute Punktbetraege auf VOELLIG unterschiedlichen Index-
    Niveaus (SPX ~5 im Jahr 1932 vs. ~7500 heute) - eine Summe absoluter
    Punkte ueber diese Aeren gewichtet automatisch die betragsmaessig
    grossen, MODERNEN Tage massiv staerker, unabhaengig davon ob sie relativ
    gesehen ueberhaupt schlechter liefen. Mit pnl_norm (auf die jeweilige
    TAGESEIGENE Entry-Praemie normiert) zaehlt jeder Analog-Tag mit
    gleichem Gewicht - konsistent mit 'avg_pnl_norm_pct', das im Modul
    bereits genau aus diesem Grund verwendet wird.

    BEWUSST NICHT ENTHALTEN (anders als beim Ladder-Backtest): 'touch_rate'
    (kein Stop-Loss/Touch-Tracking - der Analog-Strangle wird wie das
    Referenzinstrument in labels.py immer bis Verfall gehalten),
    'n_unlimited_risk' (der ungehedgte Strangle hat IMMER unbegrenztes
    Risiko auf der Short-Seite - das waere hier keine Fallunterscheidung,
    sondern der Normalfall) und 'n_iv_fallback' (target_vol ist hier per
    Konstruktion nie NaN, siehe evaluate_analog_set - ungueltige Zeilen
    werden dort bereits ausgefiltert)."""
    if evaluated.empty:
        return {"n_trades": 0}

    net_col, norm_col = f"net_pnl_{side}", f"pnl_norm_{side}"
    n_total = len(evaluated)
    n_wins = int((evaluated[net_col] >= 0).sum())
    gross_profit_norm = evaluated.loc[evaluated[norm_col] >= 0, norm_col].sum()
    gross_loss_norm = -evaluated.loc[evaluated[norm_col] < 0, norm_col].sum()

    return {
        "n_trades": n_total,
        "win_rate_pct": n_wins / n_total * 100.0,
        "net_pnl_sum": evaluated[net_col].sum(),
        "avg_pnl": evaluated[net_col].mean(),
        "avg_pnl_norm_pct": evaluated[norm_col].mean() * 100.0,
        "profit_factor": (gross_profit_norm / gross_loss_norm) if gross_loss_norm > 0 else np.nan,
        "best_trade": evaluated[net_col].max(),
        "worst_trade": evaluated[net_col].min(),
    }


def _reliability_for_side(evaluated: pd.DataFrame, side: str, n_subperiods: int) -> dict:
    """Baut aus dem kombinierten evaluate_analog_set()-Ergebnis die
    'pnl_norm'/'net_pnl'-Spalten, die reliability_summary() erwartet, fuer
    die gewuenschte Richtung (side='short' oder 'long'). Die chronologische
    Sortierung aus evaluate_analog_set() bleibt dabei erhalten."""
    if evaluated.empty:
        return reliability_summary(evaluated)
    sub = evaluated.rename(columns={
        f"net_pnl_{side}": "net_pnl", f"pnl_norm_{side}": "pnl_norm",
    })
    return reliability_summary(sub, nw_lags=None, n_subperiods=n_subperiods)


# ═══════════════════════════════════════════════════════════════════════
#  4) LEG-PAAR-GRID-SEARCH (mit Train/Validierungs-Split gegen Mehrfachtesten)
# ═══════════════════════════════════════════════════════════════════════

def leg_pair_grid_search(analogs: pd.DataFrame, training_table: pd.DataFrame, bounds: pd.DataFrame,
                          T_days: int, put_delta_grid: list[float],
                          call_delta_grid: list[float], r_free: float, dividend_yield: float,
                          vol_surface: VolSurfaceConfig, vol_mode: str = "direct",
                          today_iv: float | None = None, today_rank: float | None = None,
                          n_subperiods: int = 4) -> pd.DataFrame:
    """Testet ALLE Kombinationen aus put_delta_grid x call_delta_grid auf dem
    Analog-Set, JEWEILS fuer Short UND Long (siehe KORREKTUR 3 - ein
    Repricing-Durchgang liefert beide Richtungen). `vol_mode`/`today_iv`/
    `today_rank` werden 1:1 an evaluate_analog_set() durchgereicht (siehe
    dort) - sie beschreiben das heutige Bewertungsniveau und sind fuer
    Train- und Validierungshaelfte identisch (nur die Analog-TAGE selbst
    unterscheiden sich zwischen den Haelften, nicht die heutige Referenz).

    WICHTIG - Mehrfachvergleichs-Falle (siehe Moduldokstring der Diskussion):
    bei z.B. 5x5=25 Kombinationen findet sich durch reinen Zufall fast immer
    eine 'signifikante' - deshalb wird das Analog-Set chronologisch in zwei
    Haelften gesplittet (aeltere/juengere Analog-Tage). Eine Kombination gilt
    nur dann als vertrauenswuerdig, wenn BEIDE Haelften UNABHAENGIG
    voneinander einen robusten Edge in DERSELBEN Richtung zeigen
    (ci_excludes_zero in Train UND Validierung, gleiches Vorzeichen) - reines
    In-Sample-Optimieren auf die komplette Menge wird bewusst NICHT als
    alleiniges Auswahlkriterium verwendet.

    Rueckgabe: eine Zeile je (put_delta, call_delta, side)-Kombination -
    'side' ist jetzt eine eigene Spalte statt eines Funktionsparameters,
    damit Short- und Long-Ergebnisse in EINER Tabelle direkt vergleichbar
    sind (siehe KORREKTUR 3)."""
    if analogs.empty:
        return pd.DataFrame()

    median_date = analogs["analog_date"].median()
    train_analogs = analogs[analogs["analog_date"] < median_date]
    val_analogs = analogs[analogs["analog_date"] >= median_date]

    rows = []
    for pd_ in put_delta_grid:
        for cd_ in call_delta_grid:
            train_eval = evaluate_analog_set(
                train_analogs, training_table, bounds, T_days, pd_, cd_,
                r_free, dividend_yield, vol_surface,
                vol_mode=vol_mode, today_iv=today_iv, today_rank=today_rank,
            )
            val_eval = evaluate_analog_set(
                val_analogs, training_table, bounds, T_days, pd_, cd_,
                r_free, dividend_yield, vol_surface,
                vol_mode=vol_mode, today_iv=today_iv, today_rank=today_rank,
            )
            if train_eval.empty or val_eval.empty:
                continue

            for side in ("short", "long"):
                train_rel = _reliability_for_side(train_eval, side, n_subperiods)
                val_rel = _reliability_for_side(val_eval, side, n_subperiods)

                rows.append({
                    "put_delta": pd_, "call_delta": cd_, "side": side,
                    "n_train": train_rel["n_trades"], "n_val": val_rel["n_trades"],
                    "train_mean_pnl_norm": train_rel["mean_pnl_norm"], "train_tstat": train_rel["tstat"],
                    "train_ci_excludes_zero": train_rel["ci_excludes_zero"],
                    "val_mean_pnl_norm": val_rel["mean_pnl_norm"], "val_tstat": val_rel["tstat"],
                    "val_ci_excludes_zero": val_rel["ci_excludes_zero"],
                    "val_ci_low": val_rel["ci_low"], "val_ci_high": val_rel["ci_high"],
                    "val_subperiod_matching": val_rel["subperiod_matching"],
                    "val_subperiod_nonempty": val_rel["subperiod_nonempty"],
                    "robust_both_halves": bool(
                        train_rel["ci_excludes_zero"] and val_rel["ci_excludes_zero"]
                        and np.sign(train_rel["mean_pnl_norm"]) == np.sign(val_rel["mean_pnl_norm"])
                        and np.sign(train_rel["mean_pnl_norm"]) > 0.0
                    ),
                })

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    # Sortierung: robuste Kombinationen (beide Haelften signifikant + gleiches
    # POSITIVES Vorzeichen fuer die jeweilige Richtung) zuerst, innerhalb
    # dessen nach Validierungs-t-Stat - die Validierungshaelfte entscheidet,
    # NICHT die (immer optimistischere) Trainingshaelfte.
    return result.sort_values(["robust_both_halves", "val_tstat"], ascending=[False, False]).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════
#  5) HOCHLEVEL-EINSTIEGSPUNKT
# ═══════════════════════════════════════════════════════════════════════

def recommend_for_today(daily_df: pd.DataFrame, training_table: pd.DataFrame,
                         query_date: pd.Timestamp | None = None,
                         T_days: int = 7,
                         put_delta_grid: list[float] | None = None,
                         call_delta_grid: list[float] | None = None,
                         analog_cfg: AnalogConfig | None = None,
                         space_cfg: SimilaritySpaceConfig | None = None,
                         hv_length: int = 30, rank_lookback_days: int = 252,
                         r_free: float = 0.0, dividend_yield: float = 0.015,
                         vol_surface: VolSurfaceConfig | None = None,
                         target_rank_pct: float = 50.0) -> dict:
    """Kompletter Ablauf: Aehnlichkeits-Raum fitten -> Query-Tag einordnen ->
    Analog-Tage finden (alle innerhalb einer Distanz-Schwelle, siehe Punkt 9
    im Moduldokstring) -> Leg-Paar-Grid-Search (Short + Long) auf dem
    Analog-Set.

    query_date=None (Default): der spaeteste Tag in feature_df wird
    verwendet ('heute'). Ein explizites frueheres Datum erlaubt, das
    Verfahren selbst rueckblickend zu pruefen (siehe Moduldokstring zur
    Kausalitaet).

    IV-Bewertung ist HIER IMMER rang-basiert (siehe Punkt 8 im
    Moduldokstring, vol_mode="rank"): `target_rank_pct` ist ein HV-Rang in
    Prozent (0-100 ueblich, Werte ausserhalb extrapolieren bewusst ueber das
    historische Fenster hinaus - identisch zu
    backtest.FixedRankConfig.target_rank_pct), KEIN absoluter IV-Wert. Fuer
    JEDEN Analog-Tag wird rollierend zurueckgerechnet, welche IV an GENAU
    DIESEM Tag noetig gewesen waere, um exakt diesen Rang in seinem eigenen
    HV-Min/Max-Fenster (hv_length/rank_lookback_days) zu erreichen (siehe
    target_vol_for_rank) - z.B. bedeutet Rang 40% an einem ruhigen Markttag
    eine niedrige IV, in einer Vola-Krise bei DEMSELBEN Rang 40% dagegen
    eine deutlich hoehere IV.

    Fuer direkten Zugriff auf den frueheren 'direct'-Modus (feste absolute
    Vola statt Rang) evaluate_analog_set()/leg_pair_grid_search() direkt
    aufrufen (siehe Punkt 6 im Moduldokstring)."""
    if put_delta_grid is None:
        put_delta_grid = [0.10, 0.16, 0.20, 0.25, 0.30]
    if call_delta_grid is None:
        call_delta_grid = [0.10, 0.16, 0.20, 0.25, 0.30]
    if vol_surface is None:
        vol_surface = VolSurfaceConfig()

    if T_days not in available_t_days(training_table):
        raise ValueError(
            f"T_days={T_days} ist nicht in der training_table vorhanden. "
            f"Verfuegbare Werte: {available_t_days(training_table)}."
        )

    feature_df = build_feature_matrix(daily_df, FeatureConfig())
    if query_date is None:
        query_date = feature_df["date"].max()
    else:
        query_date = pd.Timestamp(query_date)

    if query_date not in set(feature_df["date"]):
        raise ValueError(
            f"{query_date.date()} ist kein Handelstag in den geladenen Kursdaten."
        )

    space = fit_similarity_space(feature_df, space_cfg, max_date=query_date + pd.Timedelta(days=1))
    query_row = feature_df.loc[feature_df["date"] == query_date].iloc[0]
    query_score = transform_query(space, query_row)

    analogs = find_analogs(query_date, query_score, space, analog_cfg)

    # hv_rank_bounds() liefert das rollierende hv_min/hv_max-Fenster JEDES
    # Handelstages (kausal) - wird unten pro Analog-Tag einzeln nachgeschlagen
    # (siehe evaluate_analog_set). Der Query-Tag selbst braucht hier KEINE
    # eigene Sonderbehandlung mehr: der Ziel-Rang kommt direkt von aussen
    # (target_rank_pct), nicht mehr aus einer am Query-Tag beobachteten/
    # eingegebenen absoluten IV (siehe Punkt 8 im Moduldokstring).
    bounds = hv_rank_bounds(daily_df, hv_length, rank_lookback_days)

    today_rank = target_rank_pct / 100.0

    grid_result = leg_pair_grid_search(
        analogs, training_table, bounds, T_days,
        put_delta_grid, call_delta_grid, r_free, dividend_yield, vol_surface,
        vol_mode="rank", today_rank=today_rank,
    )

    clustering = analog_clustering_diagnostic(analogs)

    return {
        "query_date": query_date,
        "target_rank_pct": target_rank_pct,
        "today_rank": today_rank,
        "n_analogs": len(analogs),
        "analogs": analogs,
        "grid_result": grid_result,
        "n_pca_components": space.pca.n_components_,
        "clustering": clustering,
        # Fuer die Detailansicht im Dashboard (volle Kennzahlen zu EINER
        # gewaehlten Delta-Kombination, siehe summarize_analog_trades()) -
        # damit muss recommend_for_today() dafuer nicht nochmal komplett
        # neu ausgefuehrt werden (Aehnlichkeits-Raum + Analog-Suche sind der
        # teure Teil, nicht die einzelne Neubewertung).
        "bounds": bounds, "T_days": T_days, "r_free": r_free,
        "dividend_yield": dividend_yield, "vol_surface": vol_surface,
        "training_table": training_table,
    }