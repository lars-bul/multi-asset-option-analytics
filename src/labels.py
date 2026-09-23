"""
Label-Generierung fuer das ML-Regime-Modell / die Analog-Suche.

REFERENZ-INSTRUMENT: 25-Delta Short Strangle (Short Put 25d + Short Call
25d, KEINE Wings), bewusst gehalten bis Verfall (kein Stop-Loss/Early-Exit -
das Label soll ein sauberes, reproduzierbares Mass fuer "wurde die
eingesammelte Praemie durch die tatsaechliche Bewegung gerechtfertigt" sein).

LAUFZEITEN: T_days = 1..90 (alle ganzen Tage). Exit-Tag ist der erste spaetere
Handelstag mit mindestens T_days KALENDERTAGEN Abstand (identisch zum
Ladder-Backtest). Bei T=1 ist das freitags der Montag (3 Kalendertage
Bewegung) - dieselbe Konvention wie im Backtest.

WICHTIG - Klassen-Schwellen sind NICHT global gueltig:
generate_class_thresholds()/assign_classes() sind bewusst von der
Label-Generierung getrennt. Die Schwellen duerfen fuer ein echtes
Walk-Forward-Training NUR aus den Trainingsdaten des jeweiligen Zeitfensters
berechnet werden - alles andere ist Leakage.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from black76 import VolSurfaceConfig, forward_price, leg_price
from backtest import compute_hv, solve_strike_delta_consistent


@dataclass
class StrangleLabelConfig:
    T_days_list: list[int] = field(default_factory=lambda: list(range(1, 91)))
    delta: float = 0.25            # Put UND Call, symmetrisch (klassischer 25d Strangle)
    hv_length: int = 30            # IV-Quelle fuer Strike-Wahl + Bewertung: HV(hv_length)
    r_free: float = 0.0
    dividend_yield: float = 0.015
    vol_surface: VolSurfaceConfig = field(default_factory=VolSurfaceConfig)


def generate_strangle_labels(df: pd.DataFrame, cfg: StrangleLabelConfig | None = None) -> pd.DataFrame:
    """Fuer JEDEN echten Handelstag und JEDE Laufzeit aus cfg.T_days_list wird
    ein 25-Delta Short Strangle eroeffnet und bis Verfall gehalten.

    Rueckgabe: eine Zeile je (entry_date, T_days)-Kombination, Spalten:
      entry_date, T_days, exit_date, entry_S, exit_S, entry_iv,
      K_put, K_call, entry_premium, net_pnl, pnl_norm

    pnl_norm = net_pnl / entry_premium.

    Zeilen mit ungueltiger current_iv oder ohne Exit-Tag in der Historie
    werden ausgelassen."""
    if cfg is None:
        cfg = StrangleLabelConfig()

    if "resolution" in df.columns:
        df = df[df["resolution"] == "daily"]
    df = df.sort_values("date").reset_index(drop=True)

    hv_series = compute_hv(df, cfg.hv_length)

    dates = df["date"].to_numpy()
    closes = df["close"].to_numpy()
    hv_arr = hv_series.to_numpy()
    n = len(df)

    vc = cfg.vol_surface
    rows = []

    for T_days in cfg.T_days_list:
        T_years = T_days / 365.0
        target_dates = dates + np.timedelta64(T_days, "D")
        exit_idx_arr = np.searchsorted(dates, target_dates, side="left")

        for i in range(n):
            current_iv = hv_arr[i]
            if np.isnan(current_iv) or current_iv <= 0.0:
                continue

            j = exit_idx_arr[i]
            if j >= n:
                continue

            S = float(closes[i])
            exit_S = float(closes[j])

            F_entry = forward_price(S, cfg.r_free, cfg.dividend_yield, T_years)

            K_put, _ = solve_strike_delta_consistent(
                F_entry, cfg.delta, T_years, current_iv, is_put=True, cfg=vc,
            )
            K_call, _ = solve_strike_delta_consistent(
                F_entry, cfg.delta, T_years, current_iv, is_put=False, cfg=vc,
            )

            premium_put = leg_price(K_put, True, T_years, F_entry, current_iv, cfg.r_free, vc)
            premium_call = leg_price(K_call, False, T_years, F_entry, current_iv, cfg.r_free, vc)
            entry_premium = premium_put + premium_call

            if entry_premium <= 0.0:
                continue

            settle_put = max(K_put - exit_S, 0.0)
            settle_call = max(exit_S - K_call, 0.0)
            net_pnl = entry_premium - (settle_put + settle_call)
            pnl_norm = net_pnl / entry_premium

            rows.append({
                "entry_date": pd.Timestamp(dates[i]),
                "T_days": T_days,
                "exit_date": pd.Timestamp(dates[j]),
                "entry_S": S,
                "exit_S": exit_S,
                "entry_iv": current_iv,
                "K_put": K_put,
                "K_call": K_call,
                "entry_premium": entry_premium,
                "net_pnl": net_pnl,
                "pnl_norm": pnl_norm,
            })

    return pd.DataFrame(rows)


@dataclass
class ClassThresholds:
    """Schwellen fuer die 4 Klassen, um 0 ZENTRIERT (0 ist immer eine feste
    Grenze, keine Datenquantile) - siehe generate_class_thresholds()."""
    neg_split: float   # Grenze zwischen 'stark_negativ' und 'negativ' (<0)
    pos_split: float   # Grenze zwischen 'positiv' und 'stark_positiv' (>=0)
    n_neg: int = 0
    n_pos: int = 0


def generate_class_thresholds(pnl_norm: np.ndarray, split_quantile: float = 0.5) -> ClassThresholds:
    """Berechnet die beiden inneren Klassengrenzen, symmetrisch um 0 gebaut:
    Rohwerte werden am FESTEN Punkt 0 in Gewinner/Verlierer getrennt, innerhalb
    jeder Seite wird SEPARAT das `split_quantile`-Quantil bestimmt.

    WICHTIG: `pnl_norm` darf NUR die Werte des Trainingsfensters enthalten
    (sonst Leakage)."""
    pnl_norm = np.asarray(pnl_norm, dtype=float)
    pnl_norm = pnl_norm[~np.isnan(pnl_norm)]

    neg_vals = pnl_norm[pnl_norm < 0.0]
    pos_vals = pnl_norm[pnl_norm >= 0.0]

    neg_split = float(np.quantile(neg_vals, 1.0 - split_quantile)) if len(neg_vals) > 0 else np.nan
    pos_split = float(np.quantile(pos_vals, split_quantile)) if len(pos_vals) > 0 else np.nan

    return ClassThresholds(neg_split=neg_split, pos_split=pos_split, n_neg=len(neg_vals), n_pos=len(pos_vals))


CLASS_LABELS = ["stark_negativ", "negativ", "positiv", "stark_positiv"]


def assign_classes(pnl_norm: pd.Series, thresholds: ClassThresholds) -> pd.Categorical:
    """Weist jedem pnl_norm-Wert eine der 4 Klassen zu."""
    pnl_norm = pd.Series(pnl_norm)
    conditions = [
        pnl_norm < thresholds.neg_split,
        (pnl_norm >= thresholds.neg_split) & (pnl_norm < 0.0),
        (pnl_norm >= 0.0) & (pnl_norm < thresholds.pos_split),
        pnl_norm >= thresholds.pos_split,
    ]
    codes = np.select(conditions, [0, 1, 2, 3], default=-1)
    return pd.Categorical.from_codes(codes, categories=CLASS_LABELS)


def class_distribution(classes: pd.Categorical) -> pd.Series:
    """Diagnose-Helfer: Klassenhaeufigkeiten (absolut + Prozent)."""
    counts = pd.Series(classes).value_counts().reindex(CLASS_LABELS)
    pct = counts / counts.sum() * 100.0
    return pd.DataFrame({"n": counts, "pct": pct.round(2)})


def build_training_table(feature_df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    """Verbindet Feature-Matrix (eine Zeile je Handelstag) und Label-Tabelle
    (eine Zeile je Handelstag x T_days) ueber das Entry-Datum."""
    merged = label_df.merge(
        feature_df, left_on="entry_date", right_on="date", how="left",
    )
    return merged.drop(columns=["date"])