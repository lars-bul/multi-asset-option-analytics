"""
Analog-Matching fuer die IC-Leg-Auswahl.

GRUNDIDEE: statt eines global trainierten Klassifikators (train.py) wird
direkt mit historischen "Analog-Tagen" gearbeitet:

  1. Finde die K historisch aehnlichsten Handelstage zu einem Query-Tag
     (Distanz im PCA-Raum der Tages-Features - siehe fit_similarity_space()).
  2. Rechne die Vola jedes Analog-Tags auf den HV-Rang des Query-Tags um
     (siehe target_vol_for_rank() - direkte Erweiterung von
     backtest.compute_rank_target_vol(), nur mit variablem statt festem
     Ziel-Rang), damit ein "73%-Rang-Tag" von 1935 fair mit einem von 2021
     vergleichbar ist. Das ist GENAU dasselbe Grundprinzip wie die im
     Dashboard bereits vorhandene "Fixer IV-Rang"-Simulation
     (backtest.FixedRankConfig/compute_rank_target_vol) - nur wird dort JEDER
     Handelstag der gesamten Historie auf einen festen Rang umgerechnet,
     hier NUR die K Analog-Tage, und der Ziel-Rang ist nicht fest eingestellt,
     sondern der TATSAECHLICHE heutige Rang.
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
     sehr viel kleiner als k ist.

  5) NEU - T_days-Kompatibilitaetspruefung (available_t_days()): die
     Analog-Bewertung nutzt entry_S/exit_S aus der vorab gebauten
     training_table (siehe labels.py), die nur fuer die dort konfigurierten
     StrangleLabelConfig.T_days_list-Werte existieren (Default [2,3,5,7,10,
     14]). Ein T_days ausserhalb dieser Liste fuehrte bisher STILLSCHWEIGEND
     zu einem leeren Ergebnis (0 Zeilen ueberall, kein Fehler). Jetzt wird
     das explizit geprueft und mit einer klaren Fehlermeldung abgebrochen.
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
    k: int = 100
    min_gap_days: int = 5   # Mindestabstand in HANDELSTAGEN zum Query-Tag (siehe KORREKTUR 2)


def find_analogs(query_date: pd.Timestamp, query_score: np.ndarray, space: SimilaritySpace,
                  cfg: AnalogConfig | None = None) -> pd.DataFrame:
    """Findet die cfg.k naehesten Analog-Tage zu query_score (euklidische
    Distanz im gewhitenten PCA-Raum), beschraenkt auf Kandidaten VOR
    query_date (Kausalitaet - siehe Modul-Docstring).

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
    k = min(cfg.k, len(cand_idx))
    order = np.argsort(dist)[:k]
    chosen = cand_idx[order]

    return pd.DataFrame({
        "analog_date": dates[chosen],
        "distance": dist[order],
    }).sort_values("distance").reset_index(drop=True)


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


def evaluate_analog_set(analogs: pd.DataFrame, training_table: pd.DataFrame, bounds: pd.DataFrame,
                         today_rank: float, T_days: int, put_delta: float, call_delta: float,
                         r_free: float, dividend_yield: float, vol_surface: VolSurfaceConfig) -> pd.DataFrame:
    """Fuer jeden Analog-Tag: entry_S/exit_S aus der Trainings-Tabelle holen
    (dort bereits korrekt mit der Kalendertage-Suchlogik aus labels.py
    bestimmt - wird hier NICHT neu berechnet), hv_min/hv_max aus `bounds`,
    dann per reprice_leg_pair() ranggerecht neu bewerten (Short UND Long,
    siehe dort).

    KORREKTUR (siehe Punkt 1 im Moduldokstring, KRITISCH): das Ergebnis wird
    jetzt vor der Rueckgabe explizit nach 'entry_date' (chronologisch)
    sortiert. `analogs` kommt aus find_analogs() nach DISTANZ sortiert - in
    dieser Reihenfolge waere die spaeter in reliability_summary()
    aufgerufene Newey-West-Korrektur wirkungslos (sie korrigiert
    Autokorrelation zwischen Zeile i und i+j, was nur bei chronologischer
    Reihenfolge dem tatsaechlichen Kalender-Ueberlapp benachbarter
    Analog-Tage entspricht - siehe Beleg im Moduldokstring: bei einer
    getesteten Konfiguration kippte dadurch nachweislich die
    Signifikanz-Aussage)."""
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
        if d not in tt.index or d not in bounds_idx.index:
            continue
        trow = tt.loc[d]
        if isinstance(trow, pd.DataFrame):   # falls doppelte Zeitstempel - Sicherheitsnetz
            trow = trow.iloc[0]
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
            "target_vol": target_vol, **result,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("entry_date").reset_index(drop=True)
    return out


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
                          today_rank: float, T_days: int, put_delta_grid: list[float],
                          call_delta_grid: list[float], r_free: float, dividend_yield: float,
                          vol_surface: VolSurfaceConfig, n_subperiods: int = 4) -> pd.DataFrame:
    """Testet ALLE Kombinationen aus put_delta_grid x call_delta_grid auf dem
    Analog-Set, JEWEILS fuer Short UND Long (siehe KORREKTUR 3 - ein
    Repricing-Durchgang liefert beide Richtungen).

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
                train_analogs, training_table, bounds, today_rank, T_days, pd_, cd_,
                r_free, dividend_yield, vol_surface,
            )
            val_eval = evaluate_analog_set(
                val_analogs, training_table, bounds, today_rank, T_days, pd_, cd_,
                r_free, dividend_yield, vol_surface,
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
                         vol_surface: VolSurfaceConfig | None = None) -> dict:
    """Kompletter Ablauf: Aehnlichkeits-Raum fitten -> Query-Tag einordnen ->
    Analog-Tage finden -> Leg-Paar-Grid-Search (Short + Long) auf dem
    Analog-Set.

    query_date=None (Default): der spaeteste Tag in feature_df wird
    verwendet ('heute'). Ein explizites frueheres Datum erlaubt, das
    Verfahren selbst rueckblickend zu pruefen (siehe Moduldokstring zur
    Kausalitaet)."""
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

    bounds = hv_rank_bounds(daily_df, hv_length, rank_lookback_days)
    today_bounds_rows = bounds.loc[bounds["date"] == query_date]
    if today_bounds_rows.empty:
        raise ValueError(f"Keine HV-Rang-Daten fuer {query_date.date()} verfuegbar.")
    today_bounds = today_bounds_rows.iloc[0]
    today_rank = (today_bounds["hv"] - today_bounds["hv_min"]) / (today_bounds["hv_max"] - today_bounds["hv_min"])
    if np.isnan(today_rank):
        raise ValueError(f"HV-Rang fuer {query_date.date()} nicht bestimmbar (zu wenig Historie im Lookback-Fenster?).")

    grid_result = leg_pair_grid_search(
        analogs, training_table, bounds, today_rank, T_days,
        put_delta_grid, call_delta_grid, r_free, dividend_yield, vol_surface,
    )

    clustering = analog_clustering_diagnostic(analogs)

    return {
        "query_date": query_date,
        "today_rank": today_rank,
        "n_analogs": len(analogs),
        "analogs": analogs,
        "grid_result": grid_result,
        "n_pca_components": space.pca.n_components_,
        "clustering": clustering,
    }