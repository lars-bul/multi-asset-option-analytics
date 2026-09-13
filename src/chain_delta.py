"""
Black-76-Bewertung DIREKT aus echten Marktpreisen (Mid) - bewusst getrennt
von eurem internen Skew-Modell (black76.skewed_iv/VolSurfaceConfig). Zweck
hier: aus dem beobachteten Bid/Ask/Mid je Serie der heutigen SPX-Kette die
ECHTE implizite Vola und das ECHTE Delta zurueckrechnen. Das ist bewusst EIN
eigener, einfacher Black-76-Solver, KEIN Aufruf von leg_price()/skewed_iv()
- die reale Marktschraege steckt ja schon im beobachteten Preis selbst,
muss also nicht ueber euer Skew-Modell nachgebildet werden. Fuer den
naechsten Schritt (Rueckrechnung auf einen mit eurem Modell VERGLEICHBAREN
base_iv/HV-Aequivalent) wird SPAETER zusaetzlich solve_strike_delta_consistent()
aus backtest.py verwendet - das ist ein separater Schritt, NICHT Teil dieses
Moduls.

Konvention (wie im Rest des Projekts): Put-Delta wird als POSITIVE Zahl
zurueckgegeben (Betrag), nicht mit oekonomischem Vorzeichen.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq


def _d1_d2(F: float, K: float, T: float, sigma: float) -> tuple[float, float]:
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black76_price(F: float, K: float, T: float, sigma: float, r: float, is_put: bool) -> float:
    """Standard Black-76-Preis (Call oder Put) auf den Terminkurs F."""
    if T <= 0.0 or sigma <= 0.0:
        intrinsic = max(K - F, 0.0) if is_put else max(F - K, 0.0)
        return intrinsic * np.exp(-r * T)
    d1, d2 = _d1_d2(F, K, T, sigma)
    disc = np.exp(-r * T)
    if is_put:
        return disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))


def black76_delta(F: float, K: float, T: float, sigma: float, r: float, is_put: bool) -> float:
    """Delta bezogen auf den TERMINKURS F (nicht Spot) - Betrag (positiv),
    konsistent zur Delta-Konvention im Rest des Projekts (siehe
    analog_matching.delta_for_strike Docstring)."""
    if T <= 0.0 or sigma <= 0.0:
        return 1.0 if ((F > K) != is_put) else 0.0
    d1, _ = _d1_d2(F, K, T, sigma)
    disc = np.exp(-r * T)
    if is_put:
        return disc * norm.cdf(-d1)
    return disc * norm.cdf(d1)


def implied_vol(price: float, F: float, K: float, T: float, r: float, is_put: bool,
                 lo: float = 1e-4, hi: float = 5.0) -> float | None:
    """Invertiert den Black-76-Preis zur impliziten Vola (Brent-Verfahren).

    Gibt None zurueck, wenn kein gueltiges Ergebnis moeglich ist (Preis
    ausserhalb des arbitragefreien Bereichs, z.B. bid=ask=0 oder Preis unter
    dem inneren Wert - kommt bei sehr illiquiden Serien der echten Kette
    durchaus vor und wird hier bewusst NICHT hart abgebrochen, sondern als
    fehlender Wert weitergereicht, damit eine einzelne kaputte Serie nicht
    die ganze Kette blockiert)."""
    if price <= 0.0 or T <= 0.0:
        return None
    intrinsic = max(K - F, 0.0) if is_put else max(F - K, 0.0)
    if price <= intrinsic * np.exp(-r * T):
        return None   # Preis unterhalb des inneren Werts - arbitragefrei nicht invertierbar

    def f(sigma: float) -> float:
        return black76_price(F, K, T, sigma, r, is_put) - price

    try:
        if f(lo) > 0.0 or f(hi) < 0.0:
            return None
        return brentq(f, lo, hi, xtol=1e-6)
    except ValueError:
        return None


@dataclass
class ChainDeltaConfig:
    r_free: float = 0.0
    dividend_yield: float = 0.015
    min_abs_delta: float = 0.01   # Filter: Serien mit kleinerem Delta werden verworfen (zu weit OTM)


def enrich_chain_with_delta(chain: pd.DataFrame, spot: float, cfg: ChainDeltaConfig | None = None) -> pd.DataFrame:
    """Nimmt das Ergebnis von fetch_today_chain.fetch_today_chain() (Spalten
    date/expiry/dte/dte_exact/strike/option_type/bid/ask/mid/...) und
    ergaenzt je Zeile implied_vol_mkt und delta_mkt (ECHTE, aus dem
    Marktpreis invertierte Werte - siehe Moduldokstring). Zeilen ohne
    gueltiges IV-Solve (z.B. mid<=0 oder Preis unter innerem Wert) werden
    verworfen.

    Nutzt dte_exact (fraktionale Resttage aus fetch_today_chain.dte_exact)
    fuer T statt ganzzahligem dte - bei kurzen Laufzeiten (0DTE) macht das
    einen spuerbaren Unterschied in Delta/IV.

    Filtert danach auf cfg.min_abs_delta <= delta_mkt <= 1.0 - "am oder im
    Geld bis Delta 0.01"."""
    from black76 import forward_price   # bereits vorhanden im Projekt

    if cfg is None:
        cfg = ChainDeltaConfig()

    rows = []
    for r in chain.itertuples(index=False):
        dte_for_T = getattr(r, "dte_exact", None)
        if dte_for_T is None:
            dte_for_T = r.dte
        T = dte_for_T / 365.0
        if T <= 0.0 or r.mid is None or r.mid <= 0.0:
            continue
        F = forward_price(spot, cfg.r_free, cfg.dividend_yield, T)
        is_put = r.option_type == "put"
        iv = implied_vol(r.mid, F, r.strike, T, cfg.r_free, is_put)
        if iv is None:
            continue
        delta = black76_delta(F, r.strike, T, iv, cfg.r_free, is_put)
        if delta < cfg.min_abs_delta or delta > 1.0:
            continue
        rows.append({**r._asdict(), "implied_vol_mkt": iv, "delta_mkt": delta})

    return pd.DataFrame(rows)


def implied_vol_pair(price_total: float, F: float, K_put: float, K_call: float, T: float, r: float,
                      lo: float = 1e-4, hi: float = 5.0) -> float | None:
    """'Paket-IV' der KOMBINATION (Put+Call zusammen) - NICHT der Mittelwert
    der beiden Einzel-IVs, sondern die EINE Vola, die - in Black-76 fuer
    BEIDE Beine gemeinsam eingesetzt - exakt die Summe der beiden echten
    Marktpreise (price_total = mid_put + mid_call) reproduziert.

    Begruendung, warum das NICHT der einfache Durchschnitt der beiden
    Einzel-IVs ist: der Preis ist eine NICHTLINEARE Funktion der Vola
    (unterschiedliches Vega je Strike/Moneyness) - ein simpler IV-Mittelwert
    wuerde bei ungleichem Vega der beiden Beine einen SYSTEMATISCH falschen
    Kombinationspreis implizieren. Der Joint-Solve hier ist der korrekte,
    Black-76-konsistente Weg, den 'einen Marktpreis, den der Markt fuer das
    ganze Paket bieten wuerde' in eine einzelne Vola-Zahl zu uebersetzen."""
    if price_total <= 0.0 or T <= 0.0:
        return None

    def f(sigma: float) -> float:
        p_put = black76_price(F, K_put, T, sigma, r, True)
        p_call = black76_price(F, K_call, T, sigma, r, False)
        return (p_put + p_call) - price_total

    try:
        if f(lo) > 0.0 or f(hi) < 0.0:
            return None
        return brentq(f, lo, hi, xtol=1e-6)
    except ValueError:
        return None


def _nearest_strike_row(df_side: pd.DataFrame, target_strike: float):
    """Zeile mit dem naechstgelegenen tatsaechlich VORHANDENEN Strike zu
    einem (kontinuierlichen) Ziel-Strike - reale Ketten haben nur diskrete
    Strikes (meist 5- oder 25-Punkte-Raster)."""
    if df_side.empty:
        return None
    idx = (df_side["strike"] - target_strike).abs().idxmin()
    return df_side.loc[idx]


def build_symmetric_distance_table(enriched_chain: pd.DataFrame, expiry, spot: float,
                                    flat_vol: float, T_years: float,
                                    r_free: float, dividend_yield: float,
                                    distance_targets_pct: list[float] | None = None,
                                    hv_min: float | None = None, hv_max: float | None = None) -> pd.DataFrame:
    """Ersetzt build_delta_pair_table(): statt Put/Call UNABHAENGIG nach
    ihrem jeweiligen ECHTEN Marktdelta zu suchen (fuehrt bei realer
    Marktschraege zu asymmetrischen Strikes um den Spot), wird der Strike
    zuerst MODELLSEITIG bestimmt - genau wie euer Backtest das tun wuerde
    (strike_from_delta() mit EINER flachen/symmetrischen Vola fuer beide
    Beine, siehe backtest.solve_strike_delta_consistent) - und danach auf
    den naechstgelegenen TATSAECHLICH VORHANDENEN Strike gerundet. Dort wird
    der echte Marktpreis/echtes IV/echtes Delta nachgeschlagen.

    NEU: iv_pair_pct - die 'Paket-IV' der Kombination (siehe implied_vol_pair,
    NICHT der Mittelwert der Einzel-IVs). NEU: hv_rank_pct - der HV-Rang,
    den diese Paket-IV im HV-Min/Max-Fenster (hv_min/hv_max, siehe
    analog_matching.hv_rank_bounds - Fenster-Laenge MUSS zur Restlaufzeit
    passen, siehe dashboard_livechain.py) einnehmen wuerde - direkte
    Umkehrung von analog_matching.target_vol_for_rank(). Werte ausserhalb
    0-100% bedeuten Extrapolation ueber das historische Fenster hinaus
    (identisch zur Konvention von backtest.FixedRankConfig.target_rank_pct)."""
    from black76 import forward_price

    if distance_targets_pct is None:
        distance_targets_pct = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40]

    F = forward_price(spot, r_free, dividend_yield, T_years)
    sub = enriched_chain[enriched_chain["expiry"] == expiry]
    calls = sub[sub["option_type"] == "call"]
    puts = sub[sub["option_type"] == "put"]

    rows = []
    for dist_pct in distance_targets_pct:
        target_put = spot * (1.0 - dist_pct / 100.0)
        target_call = spot * (1.0 + dist_pct / 100.0)

        put_row = _nearest_strike_row(puts, target_put)
        call_row = _nearest_strike_row(calls, target_call)

        model_delta_put = (black76_delta(F, put_row["strike"], T_years, flat_vol, r_free, True)
                            if put_row is not None else None)
        model_delta_call = (black76_delta(F, call_row["strike"], T_years, flat_vol, r_free, False)
                             if call_row is not None else None)

        iv_pair, hv_rank_pct = None, None
        if put_row is not None and call_row is not None:
            price_total = put_row["mid"] + call_row["mid"]
            iv_pair = implied_vol_pair(price_total, F, put_row["strike"], call_row["strike"], T_years, r_free)
            if iv_pair is not None and hv_min is not None and hv_max is not None and hv_max > hv_min:
                hv_rank_pct = (iv_pair - hv_min) / (hv_max - hv_min) * 100.0

        rows.append({
            "distance_pct": dist_pct,
            "strike_put": put_row["strike"] if put_row is not None else None,
            "strike_call": call_row["strike"] if call_row is not None else None,
            "mid_put": put_row["mid"] if put_row is not None else None,
            "mid_call": call_row["mid"] if call_row is not None else None,
            "iv_put_pct": put_row["implied_vol_mkt"] * 100.0 if put_row is not None else None,
            "iv_call_pct": call_row["implied_vol_mkt"] * 100.0 if call_row is not None else None,
            "delta_put_pct": put_row["delta_mkt"] * 100.0 if put_row is not None else None,
            "delta_call_pct": call_row["delta_mkt"] * 100.0 if call_row is not None else None,
            "model_delta_put_pct": model_delta_put * 100.0 if model_delta_put is not None else None,
            "model_delta_call_pct": model_delta_call * 100.0 if model_delta_call is not None else None,
            "iv_pair_pct": iv_pair * 100.0 if iv_pair is not None else None,
            "hv_rank_pct": hv_rank_pct,
        })

    return pd.DataFrame(rows)