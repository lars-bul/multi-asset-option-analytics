"""
Feature-Engineering fuer das ML-Regime-Modell.

GRUNDPRINZIP - bewusst KEINE echte implizite Volatilitaet:
Alle Features werden ausschliesslich aus OHLC-Preisdaten berechnet (Returns,
historische Vola, Momentum, Tail-Risiko). Kein VIX, kein Optionsmarkt-Skew
aus echten Quotes - beides gibt es fuer SPX erst ab ca. 1990 (VIX) bzw. noch
spaeter mit brauchbarer Tiefe (volle Optionsketten). Da das Modell auch auf
die Zeit vor 1990 (inkl. Great Depression 1929-32 und Crash 1987) angewendet
werden soll, muss JEDES Feature so konstruiert sein, dass es auf der
kompletten Tagesdaten-Historie (ab 1927-12-30, siehe data_fetch.py)
berechenbar ist.

KEIN LOOKAHEAD: jedes Feature an Zeile t wird ausschliesslich aus Daten bis
UND MIT t berechnet (rollierende Fenster sind rechtsbuendig/trailing - das
ist pandas' .rolling()-Standardverhalten und wird hier durchgehend so
genutzt). Das ist Voraussetzung fuer eine spaetere Walk-Forward-Validierung
ohne Leakage.

BEWUSST NICHT ALS FEATURE AUFGENOMMEN:
- Der Kassakurs S selbst (oder irgendein anderes Preis-LEVEL) - Preisniveaus
  sind nicht stationaer (SPX bei 20 vs. 6000 ist nicht vergleichbar), nur
  Verhaeltnisse/Returns/normierte Distanzen sind es.
- Kalenderjahr/Jahrzehnt als Feature - wuerde dem Modell erlauben, Regime an
  der Kalenderzeit statt am tatsaechlichen Marktzustand festzumachen, was
  bei Walk-Forward zwangslaeufig zu Overfitting auf vergangene, nicht
  wiederkehrende Epochen fuehrt.

WARMUP-NaNs: die laengsten Fenster (5-Jahres-Return, 252er-HV etc.) erzeugen
an den ersten paar Jahren der Historie NaN-Werte. Das ist bei ueber 90 Jahren
Gesamthistorie unproblematisch (siehe build_feature_matrix - Zeilen mit NaN
werden NICHT automatisch gedroppt, das bleibt bewusst dem Aufrufer/Training
ueberlassen, siehe train.py).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest import compute_hv


@dataclass
class FeatureConfig:
    """Alle Fenstergroessen etc. an einem Ort, damit spaeter am Feature-Set
    experimentiert werden kann, ohne die Berechnungsfunktionen anzufassen."""

    # A) Volatilitaets-/Regime-Features
    hv_windows: list[int] = field(default_factory=lambda: [5, 10, 20, 30, 60, 90, 120, 252])
    hv_term_pairs: list[tuple[int, int]] = field(default_factory=lambda: [(10, 60), (20, 120), (30, 252)])
    hv_rank_base_window: int = 30      # welches HV-Fenster fuer den Rang verwendet wird
    hv_rank_lookback: int = 252        # Laenge des rollierenden Min/Max-Fensters fuer den Rang
    vol_of_vol_base_window: int = 30   # welches HV-Fenster fuer Vol-of-Vol verwendet wird
    vol_of_vol_window: int = 60        # rollierendes Fenster ueber die HV-Zeitreihe selbst
    parkinson_windows: list[int] = field(default_factory=lambda: [10, 30])
    garman_klass_windows: list[int] = field(default_factory=lambda: [10, 30])

    # B) Momentum/Trend-Features
    ma_windows: list[int] = field(default_factory=lambda: [20, 50, 100, 200])
    ma_slope_window: int = 5
    return_horizons: dict = field(default_factory=lambda: {
        "1d": 1, "1w": 5, "1m": 21, "6m": 126, "1y": 252, "5y": 1260,
    })
    autocorr_window: int = 60
    autocorr_lag: int = 1

    # C) Tail-/Crash-Risiko-Features
    drawdown_lookback: int = 252
    skew_kurt_window: int = 60
    large_move_threshold: float = 0.03   # 3% Tagesbewegung gilt als "grosser Move"
    max_move_window: int = 20

    # D) Strukturelle Features
    include_weekday: bool = True


def _daily_log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def horizon_returns(df: pd.DataFrame, horizons: dict) -> pd.DataFrame:
    """Einfache (nicht-log) Returns ueber mehrere Horizonte, z.B.
    {'1d':1,'1w':5,'1m':21,'6m':126,'1y':252,'5y':1260} (Handelstage, nicht
    Kalendertage - 1 Jahr ~ 252 Handelstage ist Standard). Kausal: Return an
    Zeile t = close[t]/close[t-h] - 1, verwendet nur Vergangenheit+t."""
    close = df["close"]
    out = {}
    for label, h in horizons.items():
        out[f"ret_{label}"] = close / close.shift(h) - 1.0
    return pd.DataFrame(out, index=df.index)


def hv_multi_window(df: pd.DataFrame, windows: list[int]) -> dict[int, pd.Series]:
    """Annualisierte HV (Close-to-Close) fuer mehrere Fensterlaengen. Nutzt
    dieselbe Formel wie backtest.compute_hv() (single source of truth), damit
    Feature-Berechnung und spaeteres Pricing/Backtesting nie auseinanderlaufen."""
    return {w: compute_hv(df, w) for w in windows}


def hv_multi_window_frame(hv_dict: dict[int, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame({f"hv_{w}d": s for w, s in hv_dict.items()})


def hv_term_structure_slopes(hv_dict: dict[int, pd.Series], pairs: list[tuple[int, int]]) -> pd.DataFrame:
    """Verhaeltnis kurzes/langes HV-Fenster - 1. Positiv = kurzes Ende hoeher
    als langes (Markt preist aktuell akuten Stress ein, Backwardation-artig
    in der REALIZED-Vola), negativ = Markt beruhigt sich gerade wieder
    relativ zum laengeren Fenster."""
    out = {}
    for short_w, long_w in pairs:
        if short_w not in hv_dict or long_w not in hv_dict:
            continue
        out[f"hv_slope_{short_w}_{long_w}"] = hv_dict[short_w] / hv_dict[long_w] - 1.0
    return pd.DataFrame(out)


def hv_rank(hv_dict: dict[int, pd.Series], base_window: int, lookback: int) -> pd.Series:
    """Rang der aktuellen HV innerhalb des rollierenden Min/Max-Fensters der
    letzten `lookback` Tage (0 = am rollierenden Minimum, 1 = am rollierenden
    Maximum). Gleiches Grundprinzip wie backtest.compute_rank_target_vol(),
    nur in die andere Richtung: dort wird aus einem Ziel-Rang eine Vola
    zurueckgerechnet, hier wird aus der TATSAECHLICHEN Vola ihr Rang bestimmt."""
    if base_window not in hv_dict:
        raise ValueError(f"hv_rank_base_window={base_window} ist nicht in hv_windows enthalten.")
    hv = hv_dict[base_window]
    hv_min = hv.rolling(lookback, min_periods=lookback).min()
    hv_max = hv.rolling(lookback, min_periods=lookback).max()
    span = hv_max - hv_min
    rank = (hv - hv_min) / span
    return rank.where(span > 0.0)


def vol_of_vol(hv_dict: dict[int, pd.Series], base_window: int, window: int) -> pd.Series:
    """Rollierende Streuung der HV-Zeitreihe selbst - misst, wie instabil
    das Volatilitaetsniveau gerade ist (unabhaengig vom Niveau selbst)."""
    if base_window not in hv_dict:
        raise ValueError(f"vol_of_vol_base_window={base_window} ist nicht in hv_windows enthalten.")
    return hv_dict[base_window].rolling(window).std()


def parkinson_vol(df: pd.DataFrame, window: int) -> pd.Series:
    """Parkinson-Vola-Schaetzer (aus High/Low statt nur Close) - nutzt mehr
    Information pro Tag als Close-to-Close, dadurch besonders bei kurzen
    Fenstern wertvoller. Nur fuer echte Tagesdaten sinnvoll (bei den
    monatlich interpolierten Zeilen ist High=Low=Close, Ergebnis waere 0)."""
    hl_ratio = np.log(df["high"] / df["low"])
    daily_var = (hl_ratio ** 2) / (4.0 * np.log(2.0))
    ann_var = daily_var.rolling(window).mean() * 252.0
    return np.sqrt(ann_var.clip(lower=0.0))


def garman_klass_vol(df: pd.DataFrame, window: int) -> pd.Series:
    """Garman-Klass-Vola-Schaetzer (High/Low + Open/Close) - effizienter als
    Parkinson, da er zusaetzlich Open-Close-Sprungueberkennung einbezieht."""
    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])
    daily_var = 0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2
    ann_var = daily_var.rolling(window).mean() * 252.0
    return np.sqrt(ann_var.clip(lower=0.0))


def ma_distance(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Prozentualer Abstand des Schlusskurses zu gleitenden Durchschnitten -
    bewusst als Verhaeltnis statt absoluter Distanz (stationaer ueber alle
    Kursniveaus der 95+ Jahre Historie hinweg vergleichbar)."""
    close = df["close"]
    out = {}
    for w in windows:
        ma = close.rolling(w).mean()
        out[f"ma_dist_{w}d"] = close / ma - 1.0
    return pd.DataFrame(out, index=df.index)


def ma_slope(df: pd.DataFrame, windows: list[int], slope_window: int) -> pd.DataFrame:
    """Steigung der gleitenden Durchschnitte (prozentuale Veraenderung des
    MA-Werts selbst ueber die letzten `slope_window` Tage) - Trend-STAERKE,
    zusaetzlich zur reinen Trend-Distanz oben."""
    close = df["close"]
    out = {}
    for w in windows:
        ma = close.rolling(w).mean()
        out[f"ma_slope_{w}d"] = ma / ma.shift(slope_window) - 1.0
    return pd.DataFrame(out, index=df.index)


def return_autocorrelation(df: pd.DataFrame, window: int, lag: int) -> pd.Series:
    """Rollierende Autokorrelation der taeglichen Log-Returns - misst, ob das
    Regime aktuell eher trendfolgend (positive Autokorrelation) oder
    mean-reverting (negative Autokorrelation) ist."""
    log_ret = _daily_log_returns(df["close"])

    def _autocorr(x: np.ndarray) -> float:
        a, b = x[lag:], x[:-lag]
        if np.std(a) == 0.0 or np.std(b) == 0.0:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    return log_ret.rolling(window + lag).apply(_autocorr, raw=True)


def drawdown_from_high(df: pd.DataFrame, lookback: int) -> pd.Series:
    """Aktueller Abstand vom rollierenden Hoch der letzten `lookback` Tage,
    als negativer Prozentwert (0 = auf Allzeithoch des Fensters)."""
    close = df["close"]
    rolling_high = close.rolling(lookback, min_periods=1).max()
    return close / rolling_high - 1.0


def rolling_skew_kurtosis(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rollierende Schiefe/Woelbung der taeglichen Returns - Fat-Tail-
    Indikatoren, die sich oft schon vor einem HV-Anstieg veraendern."""
    log_ret = _daily_log_returns(df["close"])
    return pd.DataFrame({
        f"ret_skew_{window}d": log_ret.rolling(window).skew(),
        f"ret_kurt_{window}d": log_ret.rolling(window).kurt(),
    })


def days_since_large_move(df: pd.DataFrame, threshold: float) -> pd.Series:
    """Anzahl Handelstage seit dem letzten Tag mit |Return| >= threshold
    (z.B. 3%). Crash-Recency ist historisch ein starker Praediktor fuer
    Folgevolatilitaet (Vol-Clustering)."""
    log_ret = _daily_log_returns(df["close"])
    is_large = (log_ret.abs() >= threshold).to_numpy()

    out = np.full(len(is_large), np.nan)
    last_seen = None
    for i, flag in enumerate(is_large):
        if flag:
            last_seen = i
        if last_seen is not None:
            out[i] = i - last_seen
    return pd.Series(out, index=df.index)


def max_abs_move_window(df: pd.DataFrame, window: int) -> pd.Series:
    """Groesster absoluter Tages-Return innerhalb der letzten `window` Tage."""
    log_ret = _daily_log_returns(df["close"])
    return log_ret.abs().rolling(window).max()


def weekday_feature(df: pd.DataFrame) -> pd.Series:
    """Wochentag des Entry (0=Montag..4=Freitag, bei echten Handelstagen
    kommen Sa/So nicht vor). Schwaches, aber praktisch kostenloses Feature -
    z.B. Freitags-Entries haben vor dem Wochenende andere Theta-Eigenschaften."""
    return df["date"].dt.dayofweek.astype(float)


def build_feature_matrix(df: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """Baut die komplette Feature-Matrix fuer JEDEN echten Handelstag.

    df: muss mindestens 'date','open','high','low','close' enthalten, wird
        hier (falls vorhanden) automatisch auf resolution=='daily' gefiltert
        und nach 'date' sortiert - dieselbe Konvention wie in backtest.py.

    Rueckgabe: DataFrame mit Spalte 'date' + allen Feature-Spalten, GLEICHE
    Zeilenanzahl wie der gefilterte Input (keine NaN-Zeilen werden entfernt -
    das bleibt bewusst dem Aufrufer ueberlassen, siehe Docstring oben)."""
    if cfg is None:
        cfg = FeatureConfig()

    if "resolution" in df.columns:
        df = df[df["resolution"] == "daily"]
    df = df.sort_values("date").reset_index(drop=True)

    hv_dict = hv_multi_window(df, cfg.hv_windows)

    blocks = [
        df[["date"]],
        horizon_returns(df, cfg.return_horizons),
        hv_multi_window_frame(hv_dict),
        hv_term_structure_slopes(hv_dict, cfg.hv_term_pairs),
        hv_rank(hv_dict, cfg.hv_rank_base_window, cfg.hv_rank_lookback).rename(
            f"hv_rank_{cfg.hv_rank_base_window}d_{cfg.hv_rank_lookback}d"),
        vol_of_vol(hv_dict, cfg.vol_of_vol_base_window, cfg.vol_of_vol_window).rename(
            f"vol_of_vol_{cfg.vol_of_vol_base_window}d"),
        pd.DataFrame({f"parkinson_{w}d": parkinson_vol(df, w) for w in cfg.parkinson_windows}),
        pd.DataFrame({f"gk_{w}d": garman_klass_vol(df, w) for w in cfg.garman_klass_windows}),
        ma_distance(df, cfg.ma_windows),
        ma_slope(df, cfg.ma_windows, cfg.ma_slope_window),
        return_autocorrelation(df, cfg.autocorr_window, cfg.autocorr_lag).rename(
            f"autocorr_{cfg.autocorr_window}d_lag{cfg.autocorr_lag}"),
        drawdown_from_high(df, cfg.drawdown_lookback).rename(f"drawdown_{cfg.drawdown_lookback}d"),
        rolling_skew_kurtosis(df, cfg.skew_kurt_window),
        days_since_large_move(df, cfg.large_move_threshold).rename(
            f"days_since_move_{int(cfg.large_move_threshold * 100)}pct"),
        max_abs_move_window(df, cfg.max_move_window).rename(f"max_abs_move_{cfg.max_move_window}d"),
    ]
    if cfg.include_weekday:
        blocks.append(weekday_feature(df).rename("weekday"))

    return pd.concat(blocks, axis=1)


def feature_columns(feature_df: pd.DataFrame) -> list[str]:
    """Bequemer Helfer: alle Spaltennamen ausser 'date' - das ist die Liste,
    die spaeter train.py als X-Spalten an die Modelle uebergibt."""
    return [c for c in feature_df.columns if c != "date"]