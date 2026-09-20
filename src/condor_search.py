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
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analog_matching import _hac_se_weighted_mean


@dataclass
class CondorConfig:
    price_basis: str = "bidask"        # "bidask" (echt) | "mid" (Naeherung)
    t_min: float = 3.0                 # min. HAC-t-Stat (Gesamtset) - nur SCHWELLE, kein Ranking
    max_full_loss_share: float = 0.03  # max. gewichteter Anteil Tage mit vollem Max-Loss
    tail_quantile: float = 0.05
    tail_floor: float = -0.75          # min. gewichtetes 5%-Quantil von pnl/max_loss
    require_plateau: bool = True


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


def _stats(x: np.ndarray, w: np.ndarray, ctx: dict, max_loss: float, cfg: CondorConfig) -> dict:
    W = w.sum()
    mean = float(w @ x / W)
    se, _ = _hac_se_weighted_mean(ctx["dates"], w, x, mean, ctx["hold"] + 1)
    tr = ctx["train"]

    def wm(mask):
        s = w[mask].sum()
        return float(w[mask] @ x[mask] / s) if s > 0 else np.nan

    return {
        "mean": mean, "ror": mean / max_loss,
        "tstat": mean / se if se > 0 else np.nan,
        "e_train": wm(tr), "e_val": wm(~tr),
        "full_loss_share": float(w[x <= -max_loss + 1e-7].sum() / W),
        "n_full_loss": int((x <= -max_loss + 1e-7).sum()),   # ABSOLUTE Anzahl Analog-Tage; 0 = nur "nicht beobachtet"
        "q_tail": _weighted_quantile(x / max_loss, w, cfg.tail_quantile),
        "win_rate_pct": float(w[x >= 0].sum() / W * 100.0),
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
                st_ = _stats(pnl, w, ctx, width - credit, cfg)
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
                **_stats(pp + pc, w, ctx, max_loss, cfg),
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

    # Ranking: HOECHSTER Erwartungswert in Punkten unter allen Kandidaten.
    cdf = cdf.sort_values(["passes", "mean"], ascending=[False, False]).reset_index(drop=True)
    meta["n_pass"] = int(cdf["passes"].sum())
    meta["n_base"] = int(cdf["c_base"].sum())
    meta["short_df"] = pd.DataFrame(short_rows)
    return cdf, sdf, meta