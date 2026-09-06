"""
Reliability-Kennzahlen fuer die Trade-Historie - Portierung der Pine-Script-
Funktionen f_newey_west_tstat() und f_subperiod_consistency().

Beide arbeiten auf dem NORMIERTEN P&L je Trade (P&L / Max-Risiko der Struktur
bei Entry), damit unterschiedliche Strukturbreiten/Zeitraeume vergleichbar
bleiben - siehe backtest.py, Spalte 'pnl_norm'.

KORREKTUR (Autokorrelation nur teilweise behandelt): die vorherige Version
nutzte einen fest codierten Lag von 20 fuer JEDE Konfiguration. Das ist zu
wenig fuer lang laufende Strukturen (z.B. T_days=60 -> taegliche
ueberlappende Trades sind ueber ~60 Handelstage hinweg mechanisch
autokorreliert, weil sich benachbarte Positionen im Marktrisiko ueberlappen)
und potenziell mehr als noetig fuer sehr kurze (T_days=2-3). Die Lag-Wahl ist
jetzt zweistufig:
  1) Eine uebliche statistische Faustregel (Newey-West 1994 Plug-in:
     floor(4*(n/100)^(2/9))) als Funktion der Stichprobengroesse.
  2) Ein STRUKTURELLER Mindest-Lag = Median der Haltedauer (hold_days) der
     Trades - das ist die tatsaechliche mechanische Ueberlappungslaenge der
     Ladder-Strategie und damit eine untere Schranke fuer die noetige
     Autokorrelations-Abdeckung.
Es wird das Maximum aus beiden verwendet. Wichtig: das behebt eine
UNTERSCHAETZUNG der Autokorrelation durch einen zu kurzen Lag - es loest
NICHT das grundsaetzlichere Problem, dass HAC-Schaetzer bei Regimewechseln
(Vol-Clustering, Fat Tails, strukturelle Bruecke) selbst nur eine Naeherung
an die "wahre" Varianz des Mittelwerts sind. Dafuer bleibt der
Perioden-Konsistenz-Check (subperiod_consistency) unten die zweite,
unabhaengige Absicherung.
"""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass
class NeweyWestResult:
    mean: float
    tstat: float
    se: float
    n: int
    lag_used: int = 0


def optimal_nw_lag(n: int) -> int:
    """Automatische Bandbreiten-/Lag-Wahl nach der ueblichen Newey-West(1994)-
    Plug-in-Faustregel: floor(4*(n/100)^(2/9)). Rein stichprobengroessen-
    basiert - siehe reliability_summary() fuer die zusaetzliche strukturelle
    Untergrenze anhand der tatsaechlichen Trade-Ueberlappung."""
    if n < 2:
        return 0
    lag = int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    return max(1, min(lag, n - 1))


def newey_west_tstat(x: np.ndarray, max_lag: int | None = None) -> NeweyWestResult:
    """Newey-West(HAC)-korrigierter t-Stat des Mittelwerts (Bartlett-Kernel).
    Korrigiert den Standardfehler fuer Autokorrelation zwischen ueberlappenden
    Trades - ein normaler t-Test wuerde die Signifikanz sonst ueberschaetzen.

    max_lag=None (Default): Lag wird automatisch ueber optimal_nw_lag(n)
    bestimmt (siehe dort). Fuer die Ladder-Strategie mit strukturell
    ueberlappenden Trades sollte i.d.R. reliability_summary() verwendet
    werden, die zusaetzlich die Haltedauer als Mindest-Lag beruecksichtigt."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return NeweyWestResult(mean=np.nan, tstat=np.nan, se=np.nan, n=n, lag_used=0)

    if max_lag is None:
        max_lag = optimal_nw_lag(n)

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
        return NeweyWestResult(mean=mean_x, tstat=np.nan, se=np.nan, n=n, lag_used=L)

    se = float(np.sqrt(var_mean))
    tstat = mean_x / se
    return NeweyWestResult(mean=mean_x, tstat=tstat, se=se, n=n, lag_used=L)


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


def reliability_summary(trades: pd.DataFrame, nw_lags: int | None = None, n_subperiods: int = 6) -> dict:
    """Bequemer Wrapper: nimmt ein Trades-DataFrame (siehe backtest.py,
    Spalten 'pnl_norm' und 'entry_date') und gibt alle Reliability-Kennzahlen
    als dict zurueck - direkt fuers Dashboard oder fuer den ML-Bucket-Check.

    nw_lags=None (Default, empfohlen): der Newey-West-Lag wird automatisch
    bestimmt als max(statistische Faustregel ueber n, struktureller
    Mindest-Lag = Median der Haltedauer 'hold_days'). Letzteres ist wichtig
    fuer die Ladder-Strategie: bei taeglich neu eroeffneten, sich
    ueberlappenden Positionen ist die tatsaechliche Autokorrelationslaenge
    durch die Haltedauer der Struktur nach unten begrenzt - ein rein
    stichprobengroessenbasierter Lag kann das bei langen Laufzeiten (grosses
    T_days) unterschaetzen. Ein expliziter int ueberschreibt die Automatik
    (Rueckwaertskompatibilitaet)."""
    if trades.empty:
        return {
            "n_trades": 0, "mean_pnl_norm": np.nan, "tstat": np.nan, "se": np.nan,
            "ci_low": np.nan, "ci_high": np.nan, "ci_excludes_zero": False,
            "subperiod_matching": 0, "subperiod_nonempty": 0, "subperiod_detail": [],
            "nw_lag_used": 0,
        }

    pnl_norm = trades["pnl_norm"].to_numpy()
    entry_ns = trades["entry_date"].astype("int64").to_numpy()

    if nw_lags is None:
        n = len(trades)
        base_lag = optimal_nw_lag(n)
        if "hold_days" in trades.columns and n > 0:
            structural_lag = int(np.nanmedian(trades["hold_days"].to_numpy(dtype=float)))
        else:
            structural_lag = 0
        lag = max(base_lag, structural_lag)
    else:
        lag = nw_lags

    nw = newey_west_tstat(pnl_norm, lag)
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
        "nw_lag_used": nw.lag_used,
    }