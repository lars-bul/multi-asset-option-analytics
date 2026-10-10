"""
Anbieterunabhaengige Plausibilitaetspruefung von Options-Quotes.

Arbeitet NUR auf Standardspalten der angereicherten Kette (egal ob Yahoo, IB oder ein anderer
Provider sie geliefert hat):
    expiry, option_type ('call'/'put'), strike, bid, ask, mid, dte_exact (oder dte)
Optional: implied_vol_mkt (Smile-Pruefung), quote_age_minutes (Staleness).
Fehlende optionale Spalten ueberspringen die jeweilige Pruefung - es wird nichts angenommen,
was nur ein bestimmter Anbieter liefert.

Pruefungen (erste zutreffende gewinnt, Grund steht in 'reason'):
  1. Basis:      kein Bid/Ask, Bid < 0, Ask <= 0, Bid > Ask (gekreuzt), kein Mid.
  2. Spread:     (Ask-Bid) > max(max_rel_spread * Mid, spread_floor_bp vom Spot).
  3. Alter:      optional, quote_age_minutes > max_quote_age_min.
  4. Parity:     Put-Call-Parity je (Verfall, Strike): der implizite Terminkurs
                 F_i = K + (C-P)/disc muss zum robusten Median-Terminkurs des Verfalls passen
                 (Toleranz = parity_mult * Halb-Spreads + parity_floor_bp). Der Median kommt aus den
                 Daten selbst (kein Dividenden-/Zinsmodell noetig). Verletzung -> beide Beine raus.
  5. Smile:      Preis weicht um mehr als smile_mult * Halb-Spread (+ Floor) von dem Preis ab, den die
                 lokal geglaettete IV der Nachbarstrikes (gleicher Verfall/Typ) ergibt.

Rueckgabe von filter_chain_quotes(): (bereinigte Kette, entfernte Zeilen mit Spalte 'reason').
Toleranzen sind in Spread-Vielfachen bzw. Basispunkten vom Spot angegeben, nicht in absoluten
Punkten - damit sie fuer jedes Instrument (SPX, Gold, Oel ...) und jeden Provider gelten.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QuoteFilterConfig:
    enabled: bool = True
    require_bid_ask: bool = True       # Condor-Suche rechnet mit Bid/Ask -> ohne beides unbrauchbar
    max_rel_spread: float = 1.0        # (Ask-Bid)/Mid
    spread_floor_bp: float = 1.0       # Spreads unterhalb (Basispunkte vom Spot) werden nie geflaggt
    max_quote_age_min: float | None = None
    parity_mult: float = 1.5           # erlaubte Parity-Abweichung in Vielfachen der Halb-Spreads
    parity_floor_bp: float = 1.0
    parity_min_strikes: int = 5        # weniger gemeinsame Strikes je Verfall -> keine Parity-Pruefung
    smile_mult: float = 3.0            # erlaubte Preisabweichung zur Smile-Glaettung in Halb-Spreads
    smile_floor_bp: float = 1.0
    smile_window: int = 7              # Fenster (Strikes) fuer den rollenden IV-Median


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def filter_chain_quotes(chain: pd.DataFrame, spot: float, r_free: float = 0.0,
                        dividend_yield: float = 0.0,
                        cfg: QuoteFilterConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Siehe Moduldokstring. Gibt (clean, removed) zurueck; removed hat zusaetzlich 'reason'."""
    cfg = cfg or QuoteFilterConfig()
    empty_removed = chain.iloc[0:0].copy()
    empty_removed["reason"] = pd.Series(dtype=object)
    if chain.empty or not cfg.enabled:
        return chain, empty_removed

    df = chain
    bid, ask = _col(df, "bid"), _col(df, "ask")
    mid = _col(df, "mid")
    mid = mid.where(mid.notna(), (bid + ask) / 2.0)
    K = _col(df, "strike")
    dte = _col(df, "dte_exact") if "dte_exact" in df.columns else _col(df, "dte")
    T = dte / 365.0
    typ = df["option_type"].astype(str).str.lower()
    spread = ask - bid
    spread_floor = cfg.spread_floor_bp / 1e4 * float(spot)

    reason = pd.Series("", index=df.index, dtype=object)

    def flag(mask: pd.Series, text: str) -> None:
        m = np.asarray(mask.fillna(False), dtype=bool) & (reason.to_numpy() == "")
        reason.loc[m] = text

    # 1) Basis
    has_ba = bid.notna() & ask.notna()
    if cfg.require_bid_ask:
        flag(~has_ba, "kein Bid/Ask")
        flag(has_ba & (bid < 0), "Bid < 0")
        flag(has_ba & (ask <= 0), "Ask <= 0")
        flag(has_ba & (ask < bid), "Bid > Ask (gekreuzt)")
    flag(mid.isna() | (mid <= 0), "kein gueltiges Mid")

    # 2) Spread
    flag(has_ba & (spread > np.maximum(cfg.max_rel_spread * mid, spread_floor)), "Spread zu weit")

    # 3) Alter (nur falls Spalte vorhanden und Schwelle gesetzt)
    if cfg.max_quote_age_min is not None and "quote_age_minutes" in df.columns:
        flag(_col(df, "quote_age_minutes") > float(cfg.max_quote_age_min),
             f"Quote aelter als {cfg.max_quote_age_min:g} Min.")

    half_spread = (spread / 2.0).fillna(0.0).clip(lower=0.0)

    # 4) Put-Call-Parity
    ok = reason.to_numpy() == ""
    base = pd.DataFrame({"expiry": df["expiry"], "strike": K, "T": T, "mid": mid,
                         "hs": half_spread, "typ": typ}).loc[ok]
    calls = base[base["typ"] == "call"].drop_duplicates(["expiry", "strike"]).set_index(["expiry", "strike"])
    puts = base[base["typ"] == "put"].drop_duplicates(["expiry", "strike"]).set_index(["expiry", "strike"])
    common = calls.index.intersection(puts.index)
    if len(common):
        j = pd.DataFrame({
            "cm": calls.loc[common, "mid"], "pm": puts.loc[common, "mid"],
            "hs": calls.loc[common, "hs"] + puts.loc[common, "hs"], "T": calls.loc[common, "T"],
        }).reset_index()
        disc = np.exp(-float(r_free) * j["T"])
        j["F_i"] = j["strike"] + (j["cm"] - j["pm"]) / disc
        j["tol"] = cfg.parity_mult * j["hs"] / disc + cfg.parity_floor_bp / 1e4 * float(spot)
        bad_keys = []
        for _exp, g in j.groupby("expiry"):
            if len(g) < cfg.parity_min_strikes:
                continue
            best = g.nsmallest(max(3, int(0.4 * len(g))), "tol")      # die praezisesten Strikes
            f_med = float(best["F_i"].median())
            bad = g[(g["F_i"] - f_med).abs() > g["tol"]]
            bad_keys.extend(zip(bad["expiry"], bad["strike"]))
        if bad_keys:
            idx = pd.MultiIndex.from_arrays([df["expiry"], K])
            mask = pd.Series(idx.isin(pd.MultiIndex.from_tuples(bad_keys)), index=df.index)
            flag(mask, "Put-Call-Parity verletzt")

    # 5) Smile-Ausreisser (Preisraum, skaliert mit dem Spread)
    if "implied_vol_mkt" in df.columns:
        from chain_delta import black76_price
        from black76 import forward_price

        ok = reason.to_numpy() == ""
        work = pd.DataFrame({"expiry": df["expiry"], "typ": typ, "strike": K,
                             "iv": _col(df, "implied_vol_mkt"), "mid": mid,
                             "hs": half_spread, "tt": T}).loc[ok]
        work = work.sort_values(["expiry", "typ", "strike"])
        work["iv_med"] = work.groupby(["expiry", "typ"])["iv"].transform(
            lambda s: s.rolling(int(cfg.smile_window), center=True, min_periods=3).median()
        )
        out_idx = []
        for row in work.itertuples():
            if not (np.isfinite(row.iv_med) and row.iv_med > 0 and row.tt > 0):
                continue
            F = forward_price(float(spot), float(r_free), float(dividend_yield), float(row.tt))
            model = black76_price(F, float(row.strike), float(row.tt), float(row.iv_med),
                                  float(r_free), row.typ == "put")
            tol = cfg.smile_mult * row.hs + cfg.smile_floor_bp / 1e4 * float(spot)
            if abs(float(row.mid) - model) > tol:
                out_idx.append(row.Index)
        if out_idx:
            flag(pd.Series(df.index.isin(out_idx), index=df.index),
                 "Smile-Ausreisser (Preis passt nicht zu Nachbarstrikes)")

    bad_mask = reason.to_numpy() != ""
    clean = chain.loc[~bad_mask]
    removed = chain.loc[bad_mask].copy()
    removed["reason"] = reason.to_numpy()[bad_mask]
    return clean, removed