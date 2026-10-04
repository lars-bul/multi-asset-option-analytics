"""
Modellfreie Iron-Condor-Suche auf Analog-Tagen (Teil C in dashboard_livechain.py).

Praemien = HEUTIGE Marktpreise der realen Strikes (Bid/Ask), Payouts = historische
PROZENTUALE Kursbewegungen der Analog-Tage, auf den heutigen Spot projiziert
(S_d = Spot_heute * exit_S/entry_S). Short-Strangle und Fluegel werden pro
Analog-Tag gemeinsam bewertet (korrelationsrichtig). Kein Vola-Modell, keine
Division durch Modellpraemien.

Condor(i, j) = Short-Zeile i, Fluegel-Zeile j (groesserer Abstand):
  credit   = (Bid Put_i + Bid Call_i) - (Ask Put_j + Ask Call_j)
  max_loss = max(Kp_i-Kp_j, Kc_j-Kc_i) - credit
  pnl_d    = credit - payout_d          (additiv: Put-Spread + Call-Spread)

Voraussetzung: analog_matching._hac_se_weighted_mean (kommt mit apply_patches.py).

KORREKTUR (Regression behoben): der Umbau zu weighted_pnl_stats() hatte
versehentlich criteria_flags() und analog_robustness() entfernt - beide
werden noch gebraucht (criteria_flags() jetzt zusaetzlich von
pair_trade_search.py, analog_robustness() von dashboard_livechain.py's
_run_bulk_analog()). Wieder ergaenzt, unveraendert gegenueber der
urspruenglichen Fassung.

NEU: evaluate_reverse_condor_grid() - Reverse Iron Condor (Long-Condor) auf
denselben Analog-Tagen/Gewichten. Alle bisherigen Funktionen sind unveraendert.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analog_matching import _hac_se_weighted_mean, weighted_reliability_summary


@dataclass
class CondorConfig:
    price_basis: str = "bidask"        # "bidask" (echt) | "mid" (Naeherung)
    t_min: float = 3.0                 # min. HAC-t-Stat (Gesamtset) - nur SCHWELLE, kein Ranking
    max_full_loss_share: float = 0.03  # max. gewichteter Anteil Tage mit vollem Max-Loss
    tail_quantile: float = 0.05
    tail_floor: float = -0.75          # min. gewichtetes 5%-Quantil von pnl/max_loss
    require_plateau: bool = True
    sort_by: str = "ror"              # Ranking der Kandidaten: "ror" (Rendite/Risiko, Standard) | "mean" (E[P&L] in Punkten) | "tstat"


def analog_moves(analogs: pd.DataFrame, training_table: pd.DataFrame, T_days: int,
                 query_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Prozentuale Kursbewegung je Analog-Tag (aus der training_table) + Gewicht."""
    tt = training_table.loc[
        training_table["T_days"] == T_days, ["entry_date", "entry_S", "exit_S", "exit_date"]
    ]
    m = (analogs.rename(columns={"analog_date": "entry_date"})[["entry_date", "distance", "weight"]]
         .merge(tt, on="entry_date", how="inner"))
    if query_date is not None:   # Kausalitaet: Exit muss vor/an 'heute' liegen
        m = m[m["exit_date"] <= pd.Timestamp(query_date)]
    m["ret"] = m["exit_S"] / m["entry_S"] - 1.0
    m["hold_days"] = (m["exit_date"] - m["entry_date"]).dt.days
    m = m[(m["entry_S"] > 0) & np.isfinite(m["ret"]) & (m["weight"] > 0)]
    return m.sort_values("entry_date").reset_index(drop=True)


def _weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    o = np.argsort(x)
    x, w = x[o], w[o]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, cw, x))


def weighted_pnl_stats(x: np.ndarray, w: np.ndarray, dates, hold_days: int,
                        train_mask: np.ndarray, max_loss: float,
                        tail_quantile: float = 0.05) -> dict:
    """Gewichtete P&L-Kennzahlen fuer EINE Zeile/Kombination auf den Analog-Tagen:
    E[P&L] ('mean'), Rendite/Risiko ('ror' = mean/max_loss), HAC-t-Stat (Bartlett-
    Kernel, Kalenderzeit, ueber _hac_se_weighted_mean), Train-/Val-Split-Mittelwerte
    ('e_train'/'e_val'), Anteil/Anzahl Tage mit vollem Max-Loss ('full_loss_share'/
    'n_full_loss'), gewichtetes tail_quantile-Quantil von x/max_loss ('q_tail') und
    gewichtete Gewinnrate ('win_rate_pct').

    x: P&L je Analog-Tag (gleiche Laenge wie w/dates), bereits auf der gewuenschten
       Basis (z.B. Condor-P&L oder Depot-Restgewinn).
    max_loss: Bezugsgroesse fuer 'ror'/'full_loss_share'/'n_full_loss'/'q_tail'.
       Bei unbegrenztem Risiko float('inf') (oder <= 0) uebergeben - diese vier
       Kennzahlen werden dann NaN/None, der Rest bleibt reguraer berechnet."""
    W = w.sum()
    mean = float(w @ x / W)
    se, _ = _hac_se_weighted_mean(dates, w, x, mean, hold_days + 1)

    def wm(mask):
        s = w[mask].sum()
        return float(w[mask] @ x[mask] / s) if s > 0 else np.nan

    bounded = np.isfinite(max_loss) and max_loss > 0.0
    return {
        "mean": mean,
        "ror": (mean / max_loss) if bounded else np.nan,
        "tstat": mean / se if se > 0 else np.nan,
        "e_train": wm(train_mask), "e_val": wm(~train_mask),
        "full_loss_share": float(w[x <= -max_loss + 1e-7].sum() / W) if bounded else np.nan,
        "n_full_loss": int((x <= -max_loss + 1e-7).sum()) if bounded else None,
        "q_tail": _weighted_quantile(x / max_loss, w, tail_quantile) if bounded else np.nan,
        "win_rate_pct": float(w[x >= 0].sum() / W * 100.0),
    }


def criteria_flags(stats: dict, cfg: "CondorConfig") -> dict:
    """Die Basis-Kriterien (Train>0, Val>0, t>Schwelle, Tail) fuer EINE Kennzahlen-Zeile - dieselbe
    Definition wie in der Iron-Condor-Suche. Nicht bestimmbare Werte (NaN, z.B. Tail bei
    unbegrenztem Risiko) gelten als NICHT bestanden."""
    def gt(a, b):
        return bool(np.isfinite(a) and a > b)
    c_train = gt(stats["e_train"], 0.0)
    c_val = gt(stats["e_val"], 0.0)
    c_t = gt(stats["tstat"], cfg.t_min)
    fl, q = stats["full_loss_share"], stats["q_tail"]
    c_tail = bool(np.isfinite(fl) and np.isfinite(q) and fl <= cfg.max_full_loss_share and q >= cfg.tail_floor)
    return {"c_train": c_train, "c_val": c_val, "c_t": c_t, "c_tail": c_tail,
            "c_base": bool(c_train and c_val and c_t and c_tail)}


def analog_robustness(moves: pd.DataFrame, pnl: np.ndarray) -> dict:
    """GEMEINSAME Robustheits-Kennzahlen der Analog-Tabelle (Teil B) UND der Depot-Analyse:
    t-Stat (gesamt), CI != 0, Perioden-Match, t-Stat (Val), Robust (Train+Val) und dessen
    spiegelbildlich negatives Gegenstueck, n_eff (HAC). pnl = P&L je Analog-Tag (in Zeilen von
    'moves'), Train/Val = Haelften um das mittlere Datum der Analog-Tage."""
    ev = moves[["entry_date", "hold_days", "weight"]].copy()
    ev["pnl_today_short"] = pnl
    median_date = ev["entry_date"].median()
    rel_full = weighted_reliability_summary(ev, "short", 6, "pnl_today")
    train_rel = weighted_reliability_summary(ev[ev["entry_date"] < median_date], "short", 4, "pnl_today")
    val_rel = weighted_reliability_summary(ev[ev["entry_date"] >= median_date], "short", 4, "pnl_today")

    robust, robust_neg = False, False
    if train_rel["n_eff_hac"] >= 2.0 and val_rel["n_eff_hac"] >= 2.0:
        same_sign = np.sign(train_rel["mean_pnl_norm"]) == np.sign(val_rel["mean_pnl_norm"])
        both_sig = train_rel["ci_excludes_zero"] and val_rel["ci_excludes_zero"]
        robust = bool(both_sig and same_sign and np.sign(train_rel["mean_pnl_norm"]) > 0.0)
        robust_neg = bool(both_sig and same_sign and np.sign(train_rel["mean_pnl_norm"]) < 0.0)
    return {
        "n_eff_hac": rel_full["n_eff_hac"], "tstat_full": rel_full["tstat"],
        "ci_excludes_zero_full": rel_full["ci_excludes_zero"],
        "subperiod_matching": rel_full["subperiod_matching"], "subperiod_nonempty": rel_full["subperiod_nonempty"],
        "tstat_val": val_rel["tstat"], "robust_train_val": robust, "robust_train_val_neg": robust_neg,
    }


def evaluate_condor_grid(table: pd.DataFrame, moves: pd.DataFrame, spot_today: float,
                         cfg: CondorConfig | None = None, short_valid: dict | None = None):
    """Bewertet ALLE Paare (Short-Zeile i, Fluegel-Zeile j>i) der Distanz-Tabelle.
    short_valid: optional {distance_pct: bool} - Gueltigkeit der Short-Zeilen
    laut Analog-Tabelle (Short-Strangle, siehe dashboard_livechain). Ist es
    gesetzt, ersetzt es die eigene Short-Pruefung, damit Short-Kriterium und
    Plateau exakt mit den sichtbaren gruenen Zeilen dieser Tabelle uebereinstimmen.
    Rueckgabe: (condor_df, side_df, meta)
      condor_df: symmetrische Condors, mit Kriterien-Flags und 'passes'
      side_df:   Put- und Call-Spreads einzeln (zeigt, welche Seite den Edge traegt)"""
    cfg = cfg or CondorConfig()
    if moves.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    t = table.sort_values("distance_pct").reset_index(drop=True)
    n = len(t)

    def num(c):
        return pd.to_numeric(t[c], errors="coerce").to_numpy(float)

    Kp, Kc, dist = num("strike_put"), num("strike_call"), num("distance_pct")
    if cfg.price_basis == "mid":
        sell_p, buy_p, sell_c, buy_c = num("mid_put"), num("mid_put"), num("mid_call"), num("mid_call")
    else:
        sell_p, buy_p = num("bid_put"), num("ask_put")
        sell_c, buy_c = num("bid_call"), num("ask_call")

    w = moves["weight"].to_numpy(float)
    S_d = spot_today * (1.0 + moves["ret"].to_numpy(float))
    ctx = {
        "dates": moves["entry_date"].to_numpy(),
        "hold": int(moves["hold_days"].max()),
        "train": (moves["entry_date"] < moves["entry_date"].median()).to_numpy(),
    }
    pay_p = np.maximum(Kp[:, None] - S_d[None, :], 0.0)   # (n_rows, n_days)
    pay_c = np.maximum(S_d[None, :] - Kc[:, None], 0.0)

    # Short-Strangle je Zeile (nur die Short-Seite, KEINE Fluegel) - Grundlage
    # von Short-Kriterium und Plateau: nur die Short-Optionen muessen "gueltig"
    # sein, die Long-Fluegel muessen keine Kriterien erfuellen.
    short_ok, short_rows = {}, []
    W_all = w.sum()
    for i in range(n):
        credit_s = sell_p[i] + sell_c[i]
        ok_i, mean_s, t_s, e_tr, e_va = False, np.nan, np.nan, np.nan, np.nan
        if np.isfinite(credit_s) and credit_s > 0:
            x = credit_s - (pay_p[i] + pay_c[i])
            mean_s = float(w @ x / W_all)
            se_s, _ = _hac_se_weighted_mean(ctx["dates"], w, x, mean_s, ctx["hold"] + 1)
            t_s = mean_s / se_s if se_s > 0 else np.nan
            trm = ctx["train"]
            e_tr = float(w[trm] @ x[trm] / w[trm].sum()) if w[trm].sum() > 0 else np.nan
            e_va = float(w[~trm] @ x[~trm] / w[~trm].sum()) if w[~trm].sum() > 0 else np.nan
            ok_i = bool(mean_s > 0 and e_tr > 0 and e_va > 0 and t_s > cfg.t_min)
        if short_valid is not None:
            ok_i = bool(short_valid.get(float(dist[i]), False))
        short_ok[i] = ok_i
        short_rows.append({"i": i, "distance": dist[i], "strike_put": Kp[i], "strike_call": Kc[i],
                           "credit": credit_s, "mean": mean_s, "tstat": t_s,
                           "e_train": e_tr, "e_val": e_va, "ok": ok_i})

    def spread(side, i, j):
        if side == "put":
            width, credit, pay = Kp[i] - Kp[j], sell_p[i] - buy_p[j], pay_p[i] - pay_p[j]
        else:
            width, credit, pay = Kc[j] - Kc[i], sell_c[i] - buy_c[j], pay_c[i] - pay_c[j]
        if not np.isfinite([width, credit]).all() or width <= 0 or credit <= 0 or credit >= width:
            return None
        return credit, width, credit - pay

    side_rows, cond_rows = [], []
    for i in range(n - 1):
        for j in range(i + 1, n):
            legs = {}
            for side in ("put", "call"):
                r = spread(side, i, j)
                if r is None:
                    continue
                credit, width, pnl = r
                st_ = weighted_pnl_stats(pnl, w, ctx["dates"], ctx["hold"], ctx["train"],
                                          width - credit, cfg.tail_quantile)
                legs[side] = (credit, width, pnl, st_)
                K = Kp if side == "put" else Kc
                side_rows.append({
                    "side": side, "i": i, "j": j,
                    "distance_short": dist[i], "distance_wing": dist[j],
                    "strike_short": K[i], "strike_wing": K[j],
                    "credit": credit, "max_loss": width - credit, **st_,
                })
            if len(legs) < 2:
                continue
            cp, wp, pp, sp_stats = legs["put"]
            cc, wc, pc, sc_stats = legs["call"]
            credit = cp + cc
            max_loss = max(wp, wc) - credit
            if max_loss <= 0:
                continue
            cond_rows.append({
                "i": i, "j": j, "distance_short": dist[i], "distance_wing": dist[j],
                "strike_put_wing": Kp[j], "strike_put_short": Kp[i],
                "strike_call_short": Kc[i], "strike_call_wing": Kc[j],
                "credit": credit, "max_loss": max_loss,
                "e_put_spread": sp_stats["mean"], "e_call_spread": sc_stats["mean"],
                **weighted_pnl_stats(pp + pc, w, ctx["dates"], ctx["hold"], ctx["train"],
                                      max_loss, cfg.tail_quantile),
            })

    cdf, sdf = pd.DataFrame(cond_rows), pd.DataFrame(side_rows)
    meta = {
        "n_days": len(moves), "hold_max": ctx["hold"],
        "n_eff": float(w.sum() ** 2 / np.sum(w ** 2)),
        "n_pairs": len(cdf),
    }
    if cdf.empty:
        return cdf, sdf, {**meta, "n_pass": 0}

    cdf["c_train"] = cdf["e_train"] > 0
    cdf["c_val"] = cdf["e_val"] > 0
    cdf["c_t"] = cdf["tstat"] > cfg.t_min
    cdf["c_tail"] = (cdf["full_loss_share"] <= cfg.max_full_loss_share) & (cdf["q_tail"] >= cfg.tail_floor)
    # Basis-Kriterien des Condors selbst (Train, Val, t, Tail)
    cdf["c_base"] = cdf[["c_train", "c_val", "c_t", "c_tail"]].all(axis=1)
    # Short-Kriterium: die Short-Zeile i (nur Short-Strangle) muss selbst gueltig sein
    cdf["c_short"] = [short_ok[int(i)] for i in cdf["i"]]

    def plateau(i):
        """Plateau in der SHORT-Achse: die Short-Zeilen direkt UEBER und UNTER
        (i-1 / i+1) muessen als Short-Strangle ebenfalls gueltig sein
        (E>0, Train>0, Val>0, t>Schwelle). Die Fluegel muessen NICHTS
        erfuellen. Fehlender Nachbar (Rand der Kette) = nicht erfuellt."""
        return (i - 1 >= 0 and short_ok.get(i - 1, False)) and (i + 1 < n and short_ok.get(i + 1, False))

    cdf["c_plateau"] = ([plateau(int(i)) for i in cdf["i"]] if cfg.require_plateau else True)
    cdf["passes"] = cdf[["c_base", "c_short", "c_plateau"]].all(axis=1)

    # Ranking der Kandidaten nach cfg.sort_by (Standard: Rendite/Risiko %).
    if cfg.sort_by not in ("ror", "mean", "tstat"):
        raise ValueError(f"sort_by muss 'ror', 'mean' oder 'tstat' sein (erhalten: {cfg.sort_by!r}).")
    cdf = cdf.sort_values(["passes", cfg.sort_by], ascending=[False, False]).reset_index(drop=True)
    meta["n_pass"] = int(cdf["passes"].sum())
    meta["n_base"] = int(cdf["c_base"].sum())
    meta["short_df"] = pd.DataFrame(short_rows)
    return cdf, sdf, meta


def evaluate_reverse_condor_grid(table: pd.DataFrame, moves: pd.DataFrame, spot_today: float,
                                 cfg: CondorConfig | None = None):
    """Reverse Iron Condor (Long-Condor): inneren Strangle KAUFEN (Ask), aeussere Fluegel
    VERKAUFEN (Bid). Gleiche Analog-Tage/Gewichte wie evaluate_condor_grid().

      debit    = (Ask Put_i + Ask Call_i) - (Bid Put_j + Bid Call_j)
      max_loss = debit
      max_gain = max(Breite_put, Breite_call) - debit
      pnl_d    = payout_d - debit

    Kriterien (spiegelbildlich zum Short-Condor, OHNE Tail - beim Long-Condor ist der
    Totalverlust der Normalfall): Train>0, Val>0, t>t_min, innerer Long-Strangle
    selbst gueltig (E>0, Train>0, Val>0, t>t_min, Ask-Preise) und Plateau (Zeile direkt
    ueber/unter ebenfalls gueltig).
    Rueckgabe: (reverse_df, meta)."""
    cfg = cfg or CondorConfig()
    if moves.empty:
        return pd.DataFrame(), {}

    t = table.sort_values("distance_pct").reset_index(drop=True)
    n = len(t)

    def num(c):
        return pd.to_numeric(t[c], errors="coerce").to_numpy(float)

    Kp, Kc, dist = num("strike_put"), num("strike_call"), num("distance_pct")
    if cfg.price_basis == "mid":
        buy_p = sell_p = num("mid_put")
        buy_c = sell_c = num("mid_call")
    else:
        buy_p, sell_p = num("ask_put"), num("bid_put")     # innen kaufen / Fluegel verkaufen
        buy_c, sell_c = num("ask_call"), num("bid_call")

    w = moves["weight"].to_numpy(float)
    W_all = w.sum()
    S_d = spot_today * (1.0 + moves["ret"].to_numpy(float))
    dates = moves["entry_date"].to_numpy()
    hold = int(moves["hold_days"].max())
    trm = (moves["entry_date"] < moves["entry_date"].median()).to_numpy()
    pay_p = np.maximum(Kp[:, None] - S_d[None, :], 0.0)
    pay_c = np.maximum(S_d[None, :] - Kc[:, None], 0.0)

    # Long-Strangle je Zeile (nur innere Optionen, Ask-Preise) - Basis von Gueltigkeit und Plateau
    long_ok, long_rows = {}, []
    for i in range(n):
        cost = buy_p[i] + buy_c[i]
        ok_i, mean_s, t_s, e_tr, e_va = False, np.nan, np.nan, np.nan, np.nan
        if np.isfinite(cost) and cost > 0:
            x = (pay_p[i] + pay_c[i]) - cost
            mean_s = float(w @ x / W_all)
            se_s, _ = _hac_se_weighted_mean(dates, w, x, mean_s, hold + 1)
            t_s = mean_s / se_s if se_s > 0 else np.nan
            e_tr = float(w[trm] @ x[trm] / w[trm].sum()) if w[trm].sum() > 0 else np.nan
            e_va = float(w[~trm] @ x[~trm] / w[~trm].sum()) if w[~trm].sum() > 0 else np.nan
            ok_i = bool(mean_s > 0 and e_tr > 0 and e_va > 0 and t_s > cfg.t_min)
        long_ok[i] = ok_i
        long_rows.append({"i": i, "distance": dist[i], "strike_put": Kp[i], "strike_call": Kc[i],
                          "debit": cost, "mean": mean_s, "tstat": t_s,
                          "e_train": e_tr, "e_val": e_va, "ok": ok_i})

    def spread(side, i, j):
        if side == "put":
            width, debit, payoff = Kp[i] - Kp[j], buy_p[i] - sell_p[j], pay_p[i] - pay_p[j]
        else:
            width, debit, payoff = Kc[j] - Kc[i], buy_c[i] - sell_c[j], pay_c[i] - pay_c[j]
        if not np.isfinite([width, debit]).all() or width <= 0 or debit <= 0 or debit >= width:
            return None
        return debit, width, payoff

    rows = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            rp, rc = spread("put", i, j), spread("call", i, j)
            if rp is None or rc is None:
                continue
            dp, wp, payoff_p = rp
            dc, wc, payoff_c = rc
            debit = dp + dc
            max_gain = max(wp, wc) - debit
            if debit <= 0 or max_gain <= 0:
                continue
            x = payoff_p + payoff_c - debit
            st_ = weighted_pnl_stats(x, w, dates, hold, trm, debit, cfg.tail_quantile)
            st_.pop("q_tail", None)   # Tail-Kennzahl beim Long-Condor nicht aussagekraeftig
            rows.append({
                "i": i, "j": j, "distance_inner": dist[i], "distance_wing": dist[j],
                "strike_put_wing": Kp[j], "strike_put_inner": Kp[i],
                "strike_call_inner": Kc[i], "strike_call_wing": Kc[j],
                "debit": debit, "max_loss": debit, "max_gain": max_gain,
                "rr_max": max_gain / debit,
                "e_put_spread": float(w @ (payoff_p - dp) / W_all),
                "e_call_spread": float(w @ (payoff_c - dc) / W_all),
                **st_,
            })

    rdf = pd.DataFrame(rows)
    meta = {"n_days": len(moves), "hold_max": hold,
            "n_eff": float(W_all ** 2 / np.sum(w ** 2)), "n_pairs": len(rdf),
            "long_df": pd.DataFrame(long_rows)}
    if rdf.empty:
        return rdf, {**meta, "n_pass": 0, "n_base": 0}

    rdf["c_train"] = rdf["e_train"] > 0
    rdf["c_val"] = rdf["e_val"] > 0
    rdf["c_t"] = rdf["tstat"] > cfg.t_min
    rdf["c_base"] = rdf[["c_train", "c_val", "c_t"]].all(axis=1)
    rdf["c_long"] = [long_ok[int(i)] for i in rdf["i"]]

    def plateau(i):
        return (i - 1 >= 0 and long_ok.get(i - 1, False)) and (i + 1 < n and long_ok.get(i + 1, False))

    rdf["c_plateau"] = ([plateau(int(i)) for i in rdf["i"]] if cfg.require_plateau else True)
    rdf["passes"] = rdf[["c_base", "c_long", "c_plateau"]].all(axis=1)

    if cfg.sort_by not in ("ror", "mean", "tstat"):
        raise ValueError(f"sort_by muss 'ror', 'mean' oder 'tstat' sein (erhalten: {cfg.sort_by!r}).")
    rdf = rdf.sort_values(["passes", cfg.sort_by], ascending=[False, False]).reset_index(drop=True)
    meta["n_pass"] = int(rdf["passes"].sum())
    meta["n_base"] = int(rdf["c_base"].sum())
    return rdf, meta