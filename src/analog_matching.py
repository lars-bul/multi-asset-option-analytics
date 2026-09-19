"""
Analog-Matching fuer die IC-Leg-Auswahl.

GRUNDIDEE: statt eines global trainierten Klassifikators wird direkt mit
historischen "Analog-Tagen" gearbeitet:

  1. Finde alle Handelstage innerhalb einer Distanz-Schwelle im PCA-Raum zu
     einem Query-Tag (siehe fit_similarity_space()).
  2. Rechne die Vola jedes Analog-Tags auf den HV-Rang des Query-Tags um
     (siehe target_vol_for_rank(), analog zu backtest.compute_rank_target_vol).
  3. Bewerte auf diesen Analog-Tagen einen Short- UND Long-Strangle (siehe
     reprice_leg_pair() - Long ist exakt das Spiegelbild des Short).
  4. Grid-Search ueber Put-/Call-Delta-Kombinationen mit Train/Validierungs-
     Split (siehe leg_pair_grid_search()).

WICHTIG - Kausalitaet: find_analogs() beruecksichtigt NUR Kandidaten mit
Datum < Query-Datum. Bei langen Laufzeiten (T bis 30 Tage) kann ein Analog-
Tag jedoch einen Exit NACH dem Query-Datum haben - evaluate_analog_set() hat
dafuer den optionalen Parameter `query_date` (filtert exit_date <= query_date).

HISTORIE DER KORREKTUREN (Kurzfassung):
  1) Newey-West/HAC braucht chronologische Reihenfolge -> evaluate_analog_set()
     sortiert immer nach entry_date.
  2) min_gap_days sind echte HANDELSTAGE (Array-Positionen), nicht Kalendertage.
  3) reprice_leg_pair() liefert Short UND Long aus einem Durchlauf.
  4) analog_clustering_diagnostic(): Warnung bei stark geclusterten Analog-Tagen.
  5) available_t_days(): explizite T_days-Pruefung gegen die training_table.
  6) Zwei IV-Modi in evaluate_analog_set(): "direct" (feste absolute Vola) und
     "rank" (HV-Rang-Umrechnung pro Analog-Tag).
  7) summarize_analog_trades(): profit_factor auf pnl_norm (nicht absolute
     Punkte), damit Index-Niveaus verschiedener Aeren nicht verzerren.
  8) recommend_for_today() nutzt ausschliesslich den Rang-Modus.
  9) Distanz-Schwelle (max_distance) statt fester Anzahl k;
     analog_distance_distribution() als Kalibrierungs-Vorschau.
 10) delta_for_strike(): Delta zu gegebenem Strike per Bump-and-Reprice.
 11) evaluate_analog_set() reicht entry_S/exit_S durch.
 12) Optionale Kernel-Gewichtung (AnalogConfig.use_kernel_weighting):
     weight = exp(-(distance/bandwidth)**2) plus min_candidates-Garantie gegen
     Totalausfall in Krisenphasen. summarize_analog_trades_weighted()/
     weighted_reliability_summary() sind die gewichteten Pendants.
 13) NEU - HAC-Standardfehler (Bartlett, Kalenderzeit) in
     weighted_reliability_summary(): Analog-Tage im Abstand < Haltedauer haben
     ueberlappende Payoffs. _hac_se_weighted_mean() korrigiert das; 'n_eff_hac'
     = Kish-n_eff / Design-Effekt. Ohne diese Korrektur waren die t-Stats
     deutlich zu hoch.
 14) NEU - Skalenfreie, beschraenkte Statistikbasis 'pnl_spot' (P&L in
     Bruchteilen des Spots) in evaluate_analog_set(); weighted_reliability_
     summary(value_prefix="pnl_spot") vermeidet die Division durch winzige
     Modellpraemien weit OTM (pnl_norm konnte dort bei -10^5 % liegen).
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
    Tages-Features (EIN Score-Vektor pro Handelstag, KEIN T_days darin).

    whiten=True: jede Hauptkomponente wird auf Einheitsvarianz normiert, damit
    PC1 die euklidische Distanz nicht allein dominiert.

    max_date: falls gesetzt, werden nur Zeilen bis (exklusive) max_date fuers
    Fitten verwendet (sauberer Walk-Forward). Der Query-Tag selbst darf/soll
    enthalten sein (recommend_for_today uebergibt query_date+1 Tag)."""
    if cfg is None:
        cfg = SimilaritySpaceConfig()

    cols = feature_columns(feature_df)
    df = feature_df.dropna(subset=cols).reset_index(drop=True)
    if max_date is not None:
        df = df[df["date"] < max_date].reset_index(drop=True)
    # find_analogs() verlaesst sich auf aufsteigende Sortierung (Handelstage-
    # Abstand ueber Array-Positionen).
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
    max_distance: float = 3.0   # use_kernel_weighting=False: harte Schwelle (Gewicht 1 innerhalb,
                                 # sonst ausgeschlossen). True: nur noch ein zusaetzliches
                                 # (grosszuegiges) Aufnahmekriterium neben Kernel/min_candidates.
    min_gap_days: int = 5       # Mindestabstand in HANDELSTAGEN zum Query-Tag

    use_kernel_weighting: bool = False   # False = harter Cutoff (Modus 2); True = Gaussian-Kernel (Modus 3)
    bandwidth: float | None = None       # sigma; None -> max_distance/2
    min_weight: float = 1e-4             # Truncation des Kernel-Auslaeufers
    min_candidates: int = 20             # Garantie: mind. so viele naechste Kandidaten IMMER aufnehmen


def find_analogs(query_date: pd.Timestamp, query_score: np.ndarray, space: SimilaritySpace,
                  cfg: AnalogConfig | None = None) -> pd.DataFrame:
    """Findet Handelstage im Aehnlichkeits-Raum zu query_score (euklidische
    Distanz im gewhitenten PCA-Raum), beschraenkt auf Kandidaten VOR
    query_date (Kausalitaet).

    use_kernel_weighting=False: harter Cutoff, ValueError falls kein Kandidat
    innerhalb max_distance liegt.

    use_kernel_weighting=True: weight = exp(-(distance/bandwidth)**2);
    aufgenommen wird ein Kandidat, wenn (a) dist <= max_distance ODER
    (b) weight >= min_weight ODER (c) er zu den min_candidates naechsten
    gehoert. ValueError nur, wenn vor query_date gar keine Kandidaten
    existieren.

    In BEIDEN Modi enthaelt das Ergebnis eine 'weight'-Spalte."""
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

    if not cfg.use_kernel_weighting:
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
            "weight": 1.0,
        }).sort_values("distance").reset_index(drop=True)

    bandwidth = cfg.bandwidth if cfg.bandwidth is not None else cfg.max_distance / 2.0
    if bandwidth <= 0.0:
        raise ValueError(f"bandwidth muss > 0 sein (erhalten: {bandwidth}).")

    weights = np.exp(-(dist / bandwidth) ** 2)

    within_max_distance = dist <= cfg.max_distance
    within_min_weight = weights >= cfg.min_weight

    n_guarantee = min(max(cfg.min_candidates, 0), len(dist))
    guarantee_mask = np.zeros(len(dist), dtype=bool)
    if n_guarantee > 0:
        nearest_idx = np.argsort(dist)[:n_guarantee]
        guarantee_mask[nearest_idx] = True

    keep = within_max_distance | within_min_weight | guarantee_mask
    chosen = cand_idx[keep]
    dist_kept = dist[keep]
    w_kept = weights[keep]

    if len(chosen) == 0:
        raise ValueError(
            f"Keine Analog-Kandidaten vor {pd.Timestamp(query_date).date()} gefunden."
        )

    return pd.DataFrame({
        "analog_date": dates[chosen],
        "distance": dist_kept,
        "weight": w_kept,
    }).sort_values("distance").reset_index(drop=True)


def analog_distance_distribution(daily_df: pd.DataFrame, query_date: pd.Timestamp,
                                  min_gap_days: int = 5,
                                  space_cfg: SimilaritySpaceConfig | None = None) -> np.ndarray:
    """Kalibrierungs-Hilfe fuers Dashboard: (aufsteigend sortierte) Distanzen
    ALLER zulaessigen Kandidaten-Tage zum Query-Tag, VOR einer Schwelle."""
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
    """Reine Diagnose: wie stark haeufen sich die Analog-Tage in der
    Kalenderzeit (z.B. weil viele aus derselben Krise stammen)?"""
    if len(analogs) < 2:
        return {"frac_within_gap": 0.0, "median_gap_days": np.nan, "n_analogs": len(analogs),
                "gap_days_threshold": gap_days}

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
    """Rollierende hv/hv_min/hv_max je Handelstag (kausal)."""
    if "resolution" in daily_df.columns:
        daily_df = daily_df[daily_df["resolution"] == "daily"]
    daily_df = daily_df.sort_values("date").reset_index(drop=True)

    hv = compute_hv(daily_df, hv_length)
    hv_min = hv.rolling(lookback_days, min_periods=lookback_days).min()
    hv_max = hv.rolling(lookback_days, min_periods=lookback_days).max()

    return pd.DataFrame({"date": daily_df["date"], "hv": hv, "hv_min": hv_min, "hv_max": hv_max})


def target_vol_for_rank(hv_min: float, hv_max: float, rank: float) -> float:
    """Vola, die einen Tag auf EXAKT den gegebenen Rang (Bruchteil 0..1) in
    seinem eigenen rollierenden Fenster bringt."""
    if np.isnan(hv_min) or np.isnan(hv_max) or hv_max <= hv_min:
        return np.nan
    return hv_min + rank * (hv_max - hv_min)


def available_t_days(training_table: pd.DataFrame) -> list[int]:
    """Welche T_days-Werte in der training_table vorhanden sind."""
    return sorted(training_table["T_days"].unique().tolist())


def reprice_leg_pair(entry_S: float, exit_S: float, T_days: int, target_vol: float,
                      put_delta: float, call_delta: float, r_free: float, dividend_yield: float,
                      vol_surface: VolSurfaceConfig) -> dict | None:
    """Bewertet EINEN Strangle auf einem Analog-Tag mit target_vol STATT der
    tatsaechlichen HV. Liefert Short UND Long (Long = exaktes Spiegelbild)."""
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
    """Numerisches (zentrales) Delta fuer einen GEGEBENEN Strike K per
    Bump-and-Reprice auf dem Forward-Preis F, ausschliesslich ueber
    forward_price()/leg_price(). Rueckgabe als POSITIVE Groesse (Betrag)."""
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
                         today_rank: float | None = None,
                         query_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Fuer jeden Analog-Tag: entry_S/exit_S aus der Trainings-Tabelle holen,
    dann per reprice_leg_pair() neu bewerten (Short UND Long).

      vol_mode="direct": JEDER Analog-Tag mit derselben absoluten Vola today_iv.
      vol_mode="rank":   Perzentil-Umrechnung (today_rank Pflicht).

    query_date (optional, NEU): falls gesetzt, werden Analog-Tage verworfen,
    deren exit_date NACH query_date liegt (Kausalitaet bei langen Laufzeiten).
    Default None = bisheriges Verhalten.

    Das Ergebnis ist chronologisch nach entry_date sortiert (Voraussetzung fuer
    jede Autokorrelationskorrektur). Spalten u.a.: weight, entry_S, exit_S,
    pnl_norm_{short,long}, NEU pnl_spot_{short,long} (= net_pnl / entry_S,
    skalenfrei und ohne Praemien-Nenner)."""
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
    q_ts = pd.Timestamp(query_date) if query_date is not None else None

    rows = []
    for _, arow in analogs.iterrows():
        d = arow["analog_date"]
        if d not in tt.index:
            continue
        trow = tt.loc[d]
        if isinstance(trow, pd.DataFrame):   # falls doppelte Zeitstempel
            trow = trow.iloc[0]

        if q_ts is not None and pd.Timestamp(trow["exit_date"]) > q_ts:
            continue

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
            "weight": arow["weight"] if "weight" in arow.index else 1.0,
            "target_vol": target_vol,
            "entry_S": trow["entry_S"], "exit_S": trow["exit_S"],
            "pnl_spot_short": result["net_pnl_short"] / trow["entry_S"],
            "pnl_spot_long": -result["net_pnl_short"] / trow["entry_S"],
            **result,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("entry_date").reset_index(drop=True)
    return out


def summarize_analog_trades(evaluated: pd.DataFrame, side: str) -> dict:
    """Handelskennzahlen (Gewinnrate, Profit Factor, Erwartungswert, beste/
    schlechteste Trade) fuer EINE Richtung, UNGEWICHTET. profit_factor basiert
    auf pnl_norm (nicht absolute Punkte), damit verschiedene Index-Niveaus
    nicht verzerren."""
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


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish: n_eff = (Sum w)^2 / Sum w^2."""
    w = np.asarray(weights, dtype=float)
    w = w[~np.isnan(w)]
    if len(w) == 0 or w.sum() <= 0.0:
        return 0.0
    return float((w.sum() ** 2) / np.sum(w ** 2))


def summarize_analog_trades_weighted(evaluated: pd.DataFrame, side: str) -> dict:
    """Gewichtetes Pendant zu summarize_analog_trades() (Kernel-Gewicht aus
    find_analogs()). 'n_eff' (Kish) ist das aussagekraeftigere Mass fuer die
    statistische Basis. HINWEIS: profit_factor/win_rate haengen weiter an
    pnl_norm."""
    if evaluated.empty or "weight" not in evaluated.columns:
        return {"n_trades": len(evaluated), "n_eff": 0.0}

    w = evaluated["weight"].to_numpy(dtype=float)
    w_sum = w.sum()
    if w_sum <= 0.0:
        return {"n_trades": len(evaluated), "n_eff": 0.0}

    net_col, norm_col = f"net_pnl_{side}", f"pnl_norm_{side}"
    net = evaluated[net_col].to_numpy(dtype=float)
    norm = evaluated[norm_col].to_numpy(dtype=float)

    win_rate_pct = float(w[net >= 0.0].sum() / w_sum * 100.0)
    gross_profit_norm = float(np.sum(w[norm >= 0.0] * norm[norm >= 0.0]))
    gross_loss_norm = float(-np.sum(w[norm < 0.0] * norm[norm < 0.0]))

    return {
        "n_trades": len(evaluated),
        "n_eff": effective_sample_size(w),
        "win_rate_pct": win_rate_pct,
        "avg_pnl": float(np.sum(w * net) / w_sum),
        "avg_pnl_norm_pct": float(np.sum(w * norm) / w_sum * 100.0),
        "profit_factor": (gross_profit_norm / gross_loss_norm) if gross_loss_norm > 0.0 else np.nan,
        "best_trade": float(net.max()),
        "worst_trade": float(net.min()),
    }


def _hac_se_weighted_mean(dates, w, x, mean, bandwidth_days: int) -> tuple[float, float]:
    """HAC-Standardfehler (Bartlett-Kernel) des gewichteten Mittelwerts auf der
    KALENDER-Zeitachse. Analog-Tage im Abstand < Haltedauer haben
    ueberlappende Payoffs -> korrelierte Fehler; ihre Kovarianz wird mit
    (1 - h/L) gewichtet. Tagesarray mit Lags 0..L-1 -> O(Spanne*L).
    Rueckgabe: (se_hac, se_iid) - se_iid nur fuer den Design-Effekt."""
    day = pd.to_datetime(pd.Series(dates)).values.astype("datetime64[D]").astype("int64")
    idx = day - day.min()
    u = w * (x - mean)
    a = np.zeros(int(idx.max()) + 1)
    np.add.at(a, idx, u)                      # doppelte Tage werden summiert
    L = max(int(bandwidth_days), 1)
    s = float(a @ a)
    for h in range(1, min(L, len(a))):
        s += 2.0 * (1.0 - h / L) * float(a[:-h] @ a[h:])
    W = w.sum()
    return float(np.sqrt(max(s, 0.0)) / W), float(np.sqrt(np.sum(u ** 2)) / W)


def weighted_reliability_summary(evaluated: pd.DataFrame, side: str, n_subperiods: int = 6,
                                  value_prefix: str = "pnl_norm") -> dict:
    """t-Statistik/95%-CI des GEWICHTETEN Mittelwerts von {value_prefix}_{side}
    (Default 'pnl_norm'; skalenfreie, beschraenkte Basis: 'pnl_spot').

    Standardfehler: HAC (Bartlett) in Kalenderzeit, Bandbreite = max.
    Haltedauer + 1 (siehe _hac_se_weighted_mean) - korrigiert die Ueberlappung
    benachbarter Analog-Tage. 'n_eff_hac' = Kish-n_eff / Design-Effekt
    (Freiheitsgrad-Basis der t-Verteilung)."""
    from scipy.stats import t as t_dist

    norm_col = f"{value_prefix}_{side}"
    empty = {"n_trades": len(evaluated), "n_eff": 0.0, "n_eff_hac": 0.0, "mean_pnl_norm": np.nan,
             "tstat": np.nan, "ci_low": np.nan, "ci_high": np.nan,
             "ci_excludes_zero": False, "subperiod_matching": 0, "subperiod_nonempty": 0}
    if evaluated.empty or "weight" not in evaluated.columns or norm_col not in evaluated.columns:
        return empty

    w = evaluated["weight"].to_numpy(dtype=float)
    x = evaluated[norm_col].to_numpy(dtype=float)
    ok = ~np.isnan(w) & ~np.isnan(x)
    evaluated, w, x = evaluated[ok], w[ok], x[ok]
    w_sum = w.sum()
    n_eff = effective_sample_size(w)
    if len(w) == 0 or w_sum <= 0.0 or n_eff < 2.0:
        return {**empty, "n_eff": n_eff}

    mean = float(np.sum(w * x) / w_sum)

    hold = int(evaluated["hold_days"].max()) if "hold_days" in evaluated.columns else 1
    se, se_iid = _hac_se_weighted_mean(evaluated["entry_date"].to_numpy(), w, x, mean, hold + 1)

    deff = max((se / se_iid) ** 2, 1.0) if se_iid > 0.0 else 1.0
    n_eff_hac = n_eff / deff
    df = max(n_eff_hac - 1.0, 1.0)

    tstat = mean / se if se > 0.0 else np.nan
    tcrit = float(t_dist.ppf(0.975, df))
    ci_low, ci_high = mean - tcrit * se, mean + tcrit * se

    sub = evaluated.sort_values("entry_date")
    blocks = np.array_split(np.arange(len(sub)), max(n_subperiods, 1))
    matching, nonempty = 0, 0
    overall_sign = np.sign(mean)
    sw = sub["weight"].to_numpy(dtype=float)
    sx = sub[norm_col].to_numpy(dtype=float)
    for idx in blocks:
        if len(idx) == 0 or sw[idx].sum() <= 0.0:
            continue
        bmean = np.sum(sw[idx] * sx[idx]) / sw[idx].sum()
        nonempty += 1
        if overall_sign != 0.0 and np.sign(bmean) == overall_sign:
            matching += 1

    return {
        "n_trades": len(evaluated), "n_eff": n_eff, "n_eff_hac": n_eff_hac,
        "mean_pnl_norm": mean, "tstat": tstat, "ci_low": ci_low, "ci_high": ci_high,
        "ci_excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
        "subperiod_matching": matching, "subperiod_nonempty": nonempty,
    }


def _reliability_for_side(evaluated: pd.DataFrame, side: str, n_subperiods: int) -> dict:
    """UNGEWICHTETE Newey-West-Reliability fuer eine Richtung (Modus 2)."""
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
    Analog-Set, jeweils fuer Short UND Long. Das Analog-Set wird chronologisch
    in zwei Haelften gesplittet; eine Kombination gilt nur als robust, wenn
    BEIDE Haelften einen signifikanten Edge mit gleichem POSITIVEN Vorzeichen
    zeigen. Nutzt die UNGEWICHTETE reliability_summary() (nur Modus 2).

    Rueckgabe: eine Zeile je (put_delta, call_delta, side)."""
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
    Analog-Tage finden -> Leg-Paar-Grid-Search (Short + Long).

    IV-Bewertung ist IMMER rang-basiert (vol_mode="rank"): target_rank_pct ist
    ein HV-Rang in Prozent; fuer JEDEN Analog-Tag wird zurueckgerechnet, welche
    IV an diesem Tag den Rang erzeugt haette. analog_cfg wird 1:1 durchgereicht
    (Default: harter Cutoff, wie von dashboard_analog.py erwartet)."""
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
        "bounds": bounds, "T_days": T_days, "r_free": r_free,
        "dividend_yield": dividend_yield, "vol_surface": vol_surface,
        "training_table": training_table,
    }