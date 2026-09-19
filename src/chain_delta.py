"""
Black-76-Bewertung DIREKT aus echten Marktpreisen (Mid) - bewusst getrennt
von eurem internen Skew-Modell (black76.skewed_iv/VolSurfaceConfig). Zweck
hier: aus dem beobachteten Bid/Ask/Mid je Serie der heutigen SPX-Kette die
ECHTE implizite Vola und das ECHTE Delta zurueckrechnen. Das ist bewusst EIN
eigener, einfacher Black-76-Solver, KEIN Aufruf von leg_price()/skewed_iv().

Konvention (wie im Rest des Projekts): Put-Delta wird als POSITIVE Zahl
zurueckgegeben (Betrag), nicht mit oekonomischem Vorzeichen.

Staleness-Diagnose: enrich_chain_with_delta() reicht `last_trade_date` durch
und ergaenzt `quote_age_minutes` (Minuten seit dem letzten tatsaechlichen
Trade, bezogen auf `as_of`). build_symmetric_distance_table() uebernimmt sie
pro gewaehltem Strike.

NEU: build_symmetric_distance_table() liefert zusaetzlich bid_put/ask_put/
bid_call/ask_call je Zeile - Grundlage der modellfreien Iron-Condor-Suche
(condor_search.py), die mit echten Bid/Ask-Preisen rechnet."""
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
    """Delta bezogen auf den TERMINKURS F (nicht Spot) - Betrag (positiv)."""
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
    ausserhalb des arbitragefreien Bereichs) - eine einzelne kaputte Serie
    blockiert nicht die ganze Kette."""
    if price <= 0.0 or T <= 0.0:
        return None
    intrinsic = max(K - F, 0.0) if is_put else max(F - K, 0.0)
    if price <= intrinsic * np.exp(-r * T):
        return None

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
    min_abs_delta: float = 0.01   # Serien mit kleinerem Delta werden verworfen (zu weit OTM)


def enrich_chain_with_delta(chain: pd.DataFrame, spot: float, cfg: ChainDeltaConfig | None = None,
                             as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Nimmt das Ergebnis von fetch_today_chain.fetch_today_chain() und
    ergaenzt je Zeile implied_vol_mkt, delta_mkt und quote_age_minutes.
    Zeilen ohne gueltiges IV-Solve werden verworfen. Nutzt dte_exact
    (fraktionale Resttage) fuer T, falls vorhanden. Filtert auf
    cfg.min_abs_delta <= delta_mkt <= 1.0."""
    from black76 import forward_price

    if cfg is None:
        cfg = ChainDeltaConfig()
    if as_of is None:
        as_of = pd.Timestamp.now(tz="UTC")

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

        last_trade_date = getattr(r, "last_trade_date", pd.NaT)
        quote_age_minutes = np.nan
        if pd.notna(last_trade_date):
            ltd = pd.Timestamp(last_trade_date)
            if ltd.tzinfo is None:
                ltd = ltd.tz_localize("UTC")
            else:
                ltd = ltd.tz_convert("UTC")
            quote_age_minutes = (as_of - ltd).total_seconds() / 60.0

        rows.append({
            **r._asdict(), "implied_vol_mkt": iv, "delta_mkt": delta,
            "quote_age_minutes": quote_age_minutes,
        })

    return pd.DataFrame(rows)


def implied_vol_pair(price_total: float, F: float, K_put: float, K_call: float, T: float, r: float,
                      lo: float = 1e-4, hi: float = 5.0) -> float | None:
    """'Paket-IV' der Kombination (Put+Call): die EINE Vola, die - in Black-76
    fuer BEIDE Beine gemeinsam eingesetzt - exakt die Summe der beiden echten
    Marktpreise reproduziert (NICHT der Mittelwert der Einzel-IVs, da der
    Preis nichtlinear in der Vola ist)."""
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
    """Zeile mit dem naechstgelegenen tatsaechlich VORHANDENEN Strike."""
    if df_side.empty:
        return None
    idx = (df_side["strike"] - target_strike).abs().idxmin()
    return df_side.loc[idx]


def build_symmetric_distance_table(enriched_chain: pd.DataFrame, expiry, spot: float,
                                    flat_vol: float, T_years: float,
                                    r_free: float, dividend_yield: float,
                                    distance_targets_pct: list[float] | None = None,
                                    hv_min: float | None = None, hv_max: float | None = None) -> pd.DataFrame:
    """Je Ziel-ABSTAND vom Spot (in %) eine Zeile; Put-/Call-Strike werden auf
    den naechstgelegenen real vorhandenen Strike gerundet, dort werden echter
    Marktpreis/IV/Delta nachgeschlagen.

    Spalten u.a.: iv_pair_pct (Paket-IV), hv_rank_pct (Rang der Paket-IV im
    HV-Min/Max-Fenster; ausserhalb 0-100 = Extrapolation),
    quote_age_put_min/quote_age_call_min (Staleness-Diagnose) und NEU
    bid_put/ask_put/bid_call/ask_call (fuer die Condor-Suche)."""
    from black76 import forward_price

    if distance_targets_pct is None:
        distance_targets_pct = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40]

    F = forward_price(spot, r_free, dividend_yield, T_years)
    sub = enriched_chain[enriched_chain["expiry"] == expiry]
    calls = sub[sub["option_type"] == "call"]
    puts = sub[sub["option_type"] == "put"]

    def _age(row) -> float | None:
        if row is None or "quote_age_minutes" not in row.index:
            return None
        val = row["quote_age_minutes"]
        return float(val) if pd.notna(val) else None

    def _g(row, col):
        if row is None or col not in row.index or pd.isna(row[col]):
            return None
        return float(row[col])

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
            "quote_age_put_min": _age(put_row),
            "quote_age_call_min": _age(call_row),
            "bid_put": _g(put_row, "bid"), "ask_put": _g(put_row, "ask"),
            "bid_call": _g(call_row, "bid"), "ask_call": _g(call_row, "ask"),
        })

    return pd.DataFrame(rows)