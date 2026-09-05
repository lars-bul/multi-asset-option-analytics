"""
Reliability-Kennzahlen fuer die Trade-Historie - Portierung der Pine-Script-
Funktionen f_newey_west_tstat() und f_subperiod_consistency().

Beide arbeiten auf dem NORMIERTEN P&L je Trade (P&L / Max-Risiko der Struktur
bei Entry), damit unterschiedliche Strukturbreiten/Zeitraeume vergleichbar
bleiben - siehe backtest.py, Spalte 'pnl_norm'.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class NeweyWestResult:
    mean: float
    tstat: float
    se: float
    n: int


def newey_west_tstat(x: np.ndarray, max_lag: int) -> NeweyWestResult:
    """Newey-West(HAC)-korrigierter t-Stat des Mittelwerts (Bartlett-Kernel).
    Korrigiert den Standardfehler fuer Autokorrelation zwischen ueberlappenden
    Trades - ein normaler t-Test wuerde die Signifikanz sonst ueberschaetzen."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return NeweyWestResult(mean=np.nan, tstat=np.nan, se=np.nan, n=n)

    mean_x = x.mean()
    L = min(max_lag, n - 1)
    e = x - mean_x

    gamma0 = float(np.mean(e * e))
    lrv = gamma0
    for j in range(1, L + 1):
        gamma_j = float(np.mean(e[j:] * e[:-j]))
        weight = 1.0 - j / (L + 1.0)
        lrv += 2.0 * weight * gamma_j

    var_mean = lrv / n
    if var_mean <= 0.0:
        return NeweyWestResult(mean=mean_x, tstat=np.nan, se=np.nan, n=n)

    se = float(np.sqrt(var_mean))
    tstat = mean_x / se
    return NeweyWestResult(mean=mean_x, tstat=tstat, se=se, n=n)


@dataclass
class SubperiodResult:
    overall_mean: float
    matching: int
    nonempty: int
    detail: list  # Liste von (mean_pnl_norm, n_trades) je Teilperiode


def subperiod_consistency(pnl_norm: np.ndarray, entry_times: np.ndarray, k: int) -> SubperiodResult:
    """Zerlegt die Trade-Historie (nach Entry-Zeitpunkt) in k gleich lange,
    chronologische Zeitfenster und prueft, in wie vielen das Vorzeichen des
    durchschnittlichen normierten P&L mit dem Gesamt-Mittelwert uebereinstimmt."""
    pnl_norm = np.asarray(pnl_norm, dtype=float)
    entry_times = np.asarray(entry_times, dtype="int64")  # z.B. unix-ns von pd.Timestamp
    n = len(pnl_norm)

    if n < k or k < 2:
        return SubperiodResult(overall_mean=np.nan, matching=0, nonempty=0, detail=[])

    overall_mean = float(pnl_norm.mean())
    t_min, t_max = entry_times.min(), entry_times.max()
    span = t_max - t_min

    matching = 0
    nonempty = 0
    detail = []

    if span <= 0:
        return SubperiodResult(overall_mean=overall_mean, matching=0, nonempty=0, detail=[])

    for b in range(k):
        lo = t_min + span * b / k
        hi = t_min + span * (b + 1) / k
        if b == k - 1:
            mask = (entry_times >= lo) & (entry_times <= hi)
        else:
            mask = (entry_times >= lo) & (entry_times < hi)

        cnt = int(mask.sum())
        if cnt > 0:
            mean_b = float(pnl_norm[mask].mean())
            nonempty += 1
            same_sign = (overall_mean >= 0.0 and mean_b >= 0.0) or (overall_mean < 0.0 and mean_b < 0.0)
            if same_sign:
                matching += 1
            detail.append((mean_b, cnt))

    return SubperiodResult(overall_mean=overall_mean, matching=matching, nonempty=nonempty, detail=detail)


def reliability_summary(trades: pd.DataFrame, nw_lags: int = 20, n_subperiods: int = 6) -> dict:
    """Bequemer Wrapper: nimmt ein Trades-DataFrame (siehe backtest.py,
    Spalten 'pnl_norm' und 'entry_date') und gibt alle Reliability-Kennzahlen
    als dict zurueck - direkt fuers Dashboard oder fuer den ML-Bucket-Check."""
    if trades.empty:
        return {
            "n_trades": 0, "mean_pnl_norm": np.nan, "tstat": np.nan, "se": np.nan,
            "ci_low": np.nan, "ci_high": np.nan, "ci_excludes_zero": False,
            "subperiod_matching": 0, "subperiod_nonempty": 0, "subperiod_detail": [],
        }

    pnl_norm = trades["pnl_norm"].to_numpy()
    entry_ns = trades["entry_date"].astype("int64").to_numpy()

    nw = newey_west_tstat(pnl_norm, nw_lags)
    sp = subperiod_consistency(pnl_norm, entry_ns, n_subperiods)

    ci_low = ci_high = np.nan
    ci_excludes_zero = False
    if not np.isnan(nw.mean) and not np.isnan(nw.se):
        ci_low = nw.mean - 1.96 * nw.se
        ci_high = nw.mean + 1.96 * nw.se
        ci_excludes_zero = ci_low > 0.0 or ci_high < 0.0

    return {
        "n_trades": nw.n,
        "mean_pnl_norm": nw.mean,
        "tstat": nw.tstat,
        "se": nw.se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_excludes_zero": ci_excludes_zero,
        "subperiod_matching": sp.matching,
        "subperiod_nonempty": sp.nonempty,
        "subperiod_detail": sp.detail,
    }
