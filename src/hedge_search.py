"""
Hedge-Finder (v1) - reine Funktionen, KEINE Streamlit-Abhaengigkeit.

Ziel: zu einem gegebenen Iron Condor (Verfall = Verfall der Kette) systematisch
Absicherungen im SELBEN Instrument suchen, die das Payoff-Profil ueber das
Marktszenario moeglichst FLACH machen (Delta-, Gamma-, Tail-Exposure runter),
ohne den Erwartungswert unnoetig zu opfern.

Bausteine (Reihenfolge = Ablauf in run_hedge_search):
  1) build_scenarios()        Analog-Szenarien (Kernel-Gewichte, aus
                              condor_search.analog_moves) + deterministisches
                              Stress-Raster (OHNE Gewicht, nie im E[P&L]).
  2) generate_hedge_candidates()  Tail-Put/-Call, Put-/Call-Spreads, kleiner
                              Long-Strangle, Underlying (Delta-Hedge).
                              Preise real: Kauf = Ask, Verkauf = Bid.
  3) optimal_quantity()       je Kandidat beste Menge q (Raster ueber Lots).
  4) rank_single_hedges()     Einzelhedges, Vorher/Nachher-Report, Pareto-Flag.
  5) greedy_combo()           bis zu max_legs Hedges nacheinander dazunehmen.
  6) payoff_curve()           Payoff ueber das Stress-Raster (fuers Diagramm).

Alle Payoffs sind Funktionen der Endkurs-Bewegung (Bewertung nur am Verfall,
wie in condor_search.py). Pfad-, Vega- und Assignment-Risiko sind NICHT
modelliert.

Wiederverwendet: condor_search.weighted_pnl_stats (HAC-t, Train/Val, Tail) und
analog_matching._hac_se_weighted_mean.

Konventionen:
  - Alle P&L in Indexpunkten je 1 Condor (Menge 1), Hedge-Menge q in
    Vielfachen davon. Kosten/P&L je Einheit = Payoff - Nettodebit.
  - Delta/Gamma-Exposure: gewichtete Regression des Gesamt-P&L auf
    [1, r, r^2] mit r = Bewegung in PROZENT. beta_ret = Punkte je +1 %,
    beta_gamma = Punkte je (1 %)^2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analog_matching import _hac_se_weighted_mean
from condor_search import weighted_pnl_stats


# ═══════════════════════════════════════════════════════════════════════
#  KONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class HedgeConfig:
    # --- Kandidaten ---
    classes: tuple = ("tail_put", "tail_call", "put_spread", "call_spread",
                      "long_strangle", "underlying")
    price_basis: str = "bidask"          # "bidask" (echt) | "mid"
    max_distance_pct: float = 30.0       # Strike-Fenster um den Spot (+-%)
    spread_steps: tuple = (1, 2, 3)      # Spread-Breite in Strike-SCHRITTEN der Kette
    strangle_dists_pct: tuple = (1.0, 2.0, 3.0, 5.0)
    min_ask: float = 0.05                # kein Kauf unterhalb dieses Ask (Punkte)
    max_spread_pct: float = 0.40         # max. (Ask-Bid)/Mid je Leg
    min_open_interest: int = 0           # 0 = aus; greift nur, falls Spalte vorhanden

    # --- Mengen ---
    q_max: float = 5.0
    q_step: float = 0.25
    delta_max: float = 1.0               # Underlying: Menge in [-delta_max, +delta_max]
    delta_step: float = 0.05

    # --- Risiko-Zielfunktion (kleiner = besser) ---
    # score = CVaR - lam_mean*E + lam_delta*|beta_ret|*sig + lam_gamma*|beta_gamma|*sig^2
    #         + lam_stress*Stress-Verlust
    alpha: float = 0.05
    lam_mean: float = 0.5                # Preis des Erwartungswert-Verlusts
    lam_delta: float = 1.0
    lam_gamma: float = 1.0
    lam_stress: float = 0.25

    # --- Stress-Raster (Bewegung in %, KEIN Gewicht) ---
    stress_lo_pct: float = -25.0
    stress_hi_pct: float = 15.0
    stress_step_pct: float = 1.0

    # --- Filter / Kombination ---
    min_e_keep: float = 0.50             # HARTE Grenze: E nach Hedge >= 50 % von E des Condors
                                         # (nur wirksam, wenn E des Condors > 0). Ohne sie
                                         # "hebt" der Optimierer den Condor einfach auf.
    min_cvar_improvement: float = 0.10   # Pflicht: CVaR um >= 10 % besser
    top_n_eval: int = 60                 # nur die besten N nach Score voll auswerten
    greedy_pool: int = 80
    max_legs: int = 3
    min_step_gain: float = 0.02          # Greedy stoppt, wenn Score-Gewinn < 2 %
    t_hedge: float = 2.0                 # Schwelle fuer 'Hedge hat selbst +EV'
    prune_weight_frac: float = 1e-6      # Szenarien mit Gewicht < frac*max entfallen


# ═══════════════════════════════════════════════════════════════════════
#  DATENKLASSEN
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Condor:
    k_put_wing: float
    k_put_short: float
    k_call_short: float
    k_call_wing: float
    credit: float

    def pnl(self, S) -> np.ndarray:
        S = np.asarray(S, dtype=float)
        put = np.maximum(self.k_put_short - S, 0.0) - np.maximum(self.k_put_wing - S, 0.0)
        call = np.maximum(S - self.k_call_short, 0.0) - np.maximum(S - self.k_call_wing, 0.0)
        return self.credit - put - call

    @property
    def max_loss(self) -> float:
        return max(self.k_put_short - self.k_put_wing,
                   self.k_call_wing - self.k_call_short) - self.credit


def condor_from_row(row) -> Condor:
    """Condor aus einer Zeile von evaluate_condor_grid() (cdf)."""
    return Condor(float(row["strike_put_wing"]), float(row["strike_put_short"]),
                  float(row["strike_call_short"]), float(row["strike_call_wing"]),
                  float(row["credit"]))


@dataclass
class Leg:
    option_type: str   # "put" | "call" | "underlying"
    side: str          # "long" | "short"
    strike: float      # bei "underlying": Referenzkurs (heutiger Spot)
    price: float       # gezahlter (long) bzw. erhaltener (short) Preis je Einheit
    mid: float

    @property
    def sign(self) -> int:
        return 1 if self.side == "long" else -1

    def payoff(self, S: np.ndarray) -> np.ndarray:
        if self.option_type == "put":
            p = np.maximum(self.strike - S, 0.0)
        elif self.option_type == "call":
            p = np.maximum(S - self.strike, 0.0)
        else:
            p = S - self.strike
        return self.sign * p


@dataclass
class HedgeCandidate:
    name: str
    klass: str
    legs: list
    allow_short: bool = False            # nur Underlying darf negative Menge haben

    @property
    def cost(self) -> float:
        """Nettodebit je Einheit (positiv = wir zahlen)."""
        return float(sum(l.sign * l.price for l in self.legs))

    @property
    def spread_cost(self) -> float:
        """Handelskosten ggue. Mid je Einheit (immer >= 0)."""
        return float(sum(abs(l.price - l.mid) for l in self.legs))

    def unit_pnl(self, S) -> np.ndarray:
        S = np.asarray(S, dtype=float)
        return sum(l.payoff(S) for l in self.legs) - self.cost


@dataclass
class Scenarios:
    spot: float
    ret: np.ndarray
    S: np.ndarray
    S_adj: np.ndarray            # drift-bereinigt: ret - gewichteter Mittelwert(ret)
    w: np.ndarray
    dates: np.ndarray
    hold: int
    train: np.ndarray
    g_stress: np.ndarray         # Bewegungen in Dezimal
    S_stress: np.ndarray
    B: np.ndarray                # Regressionsbasis [1, r, r^2], r in %
    proj: np.ndarray             # (3, n): coef = proj @ x
    sig: float                   # gewichtete Std von r (in %)
    mean_ret: float


# ═══════════════════════════════════════════════════════════════════════
#  1) SZENARIEN
# ═══════════════════════════════════════════════════════════════════════

def _projection(B: np.ndarray, w: np.ndarray) -> np.ndarray:
    """(B' W B)^-1 B' W  -> coef = proj @ x  (gewichtete kleinste Quadrate)."""
    A = B.T @ (w[:, None] * B) + 1e-12 * np.eye(B.shape[1])
    return np.linalg.solve(A, B.T * w[None, :])


def build_scenarios(moves: pd.DataFrame, spot: float, cfg: HedgeConfig | None = None) -> Scenarios:
    """moves = Ergebnis von condor_search.analog_moves() (Spalten entry_date,
    ret, weight, hold_days). Train/Val-Split am Median der Analog-Daten wie in
    condor_search.evaluate_condor_grid()."""
    cfg = cfg or HedgeConfig()
    if moves.empty:
        raise ValueError("Keine Analog-Szenarien (moves leer).")

    train_all = (moves["entry_date"] < moves["entry_date"].median()).to_numpy()
    w_all = moves["weight"].to_numpy(float)
    keep = w_all >= cfg.prune_weight_frac * w_all.max()

    m = moves[keep]
    w = w_all[keep]
    ret = m["ret"].to_numpy(float)
    mean_ret = float(w @ ret / w.sum())

    g = np.arange(cfg.stress_lo_pct, cfg.stress_hi_pct + 1e-9, cfg.stress_step_pct) / 100.0

    r = ret * 100.0
    B = np.column_stack([np.ones_like(r), r, r ** 2])
    mr = float(w @ r / w.sum())
    sig = float(np.sqrt(w @ (r - mr) ** 2 / w.sum()))

    return Scenarios(
        spot=float(spot), ret=ret, S=spot * (1.0 + ret), S_adj=spot * (1.0 + ret - mean_ret),
        w=w, dates=m["entry_date"].to_numpy(), hold=int(m["hold_days"].max()),
        train=train_all[keep], g_stress=g, S_stress=spot * (1.0 + g),
        B=B, proj=_projection(B, w), sig=max(sig, 1e-9), mean_ret=mean_ret,
    )


# ═══════════════════════════════════════════════════════════════════════
#  2) KANDIDATEN
# ═══════════════════════════════════════════════════════════════════════

def _prep_side(chain: pd.DataFrame, expiry, option_type: str, spot: float, cfg: HedgeConfig) -> pd.DataFrame:
    """Eine Kettenseite (Put oder Call) mit Preisen/Liquiditaets-Flags."""
    d = chain[(chain["expiry"] == expiry) & (chain["option_type"] == option_type)].copy()
    if d.empty:
        return d
    bid = pd.to_numeric(d["bid"], errors="coerce")
    ask = pd.to_numeric(d["ask"], errors="coerce")
    mid = pd.to_numeric(d["mid"], errors="coerce") if "mid" in d.columns else (bid + ask) / 2.0
    mid = mid.where(mid.notna(), (bid + ask) / 2.0)

    lo, hi = spot * (1 - cfg.max_distance_pct / 100.0), spot * (1 + cfg.max_distance_pct / 100.0)
    out = pd.DataFrame({"strike": pd.to_numeric(d["strike"], errors="coerce"),
                        "mid": mid, "bid": bid, "ask": ask})
    for oi in ("open_interest", "openInterest"):
        if oi in d.columns:
            out["oi"] = pd.to_numeric(d[oi], errors="coerce")
            break

    if cfg.price_basis == "mid":
        out["buy"] = out["sell"] = out["mid"]
        out["ok_buy"] = out["ok_sell"] = out["mid"] > 0
    else:
        out["buy"], out["sell"] = out["ask"], out["bid"]
        spread_pct = (out["ask"] - out["bid"]) / out["mid"].where(out["mid"] > 0)
        liquid = (spread_pct <= cfg.max_spread_pct).fillna(False)
        out["ok_sell"] = (out["bid"] > 0) & liquid
        out["ok_buy"] = (out["ask"] >= cfg.min_ask) & out["ok_sell"]

    out = out[(out["strike"] >= lo) & (out["strike"] <= hi)]
    if cfg.min_open_interest > 0 and "oi" in out.columns:
        out = out[out["oi"].fillna(0) >= cfg.min_open_interest]
    return out.sort_values("strike").reset_index(drop=True)


def _nearest(df: pd.DataFrame, target: float, ok_col: str):
    d = df[df[ok_col]]
    if d.empty:
        return None
    return d.iloc[(d["strike"] - target).abs().argmin()]


def generate_hedge_candidates(chain: pd.DataFrame, expiry, spot: float,
                              cfg: HedgeConfig | None = None) -> list[HedgeCandidate]:
    """Kandidaten gemaess cfg.classes. Alle Optionen OTM (Puts <= Spot, Calls >= Spot)
    und nur mit ausreichender Liquiditaet (siehe _prep_side)."""
    cfg = cfg or HedgeConfig()
    puts = _prep_side(chain, expiry, "put", spot, cfg)
    calls = _prep_side(chain, expiry, "call", spot, cfg)
    cands: dict[str, HedgeCandidate] = {}

    def add(c: HedgeCandidate):
        cands.setdefault(c.name, c)

    def L(row, typ, side):
        px = float(row["buy"] if side == "long" else row["sell"])
        return Leg(typ, side, float(row["strike"]), px, float(row["mid"]))

    if "tail_put" in cfg.classes and not puts.empty:
        for _, r in puts[(puts["strike"] <= spot) & puts["ok_buy"]].iterrows():
            add(HedgeCandidate(f"Long Put {r.strike:g}", "tail_put", [L(r, "put", "long")]))

    if "tail_call" in cfg.classes and not calls.empty:
        for _, r in calls[(calls["strike"] >= spot) & calls["ok_buy"]].iterrows():
            add(HedgeCandidate(f"Long Call {r.strike:g}", "tail_call", [L(r, "call", "long")]))

    if "put_spread" in cfg.classes and not puts.empty:
        for p in range(len(puts)):
            rl = puts.iloc[p]
            if rl["strike"] > spot or not rl["ok_buy"]:
                continue
            for s in cfg.spread_steps:
                if p - s < 0:
                    continue
                rs = puts.iloc[p - s]
                if rs["ok_sell"]:
                    add(HedgeCandidate(f"Put-Spread {rl.strike:g}/{rs.strike:g}", "put_spread",
                                       [L(rl, "put", "long"), L(rs, "put", "short")]))

    if "call_spread" in cfg.classes and not calls.empty:
        for p in range(len(calls)):
            rl = calls.iloc[p]
            if rl["strike"] < spot or not rl["ok_buy"]:
                continue
            for s in cfg.spread_steps:
                if p + s >= len(calls):
                    continue
                rs = calls.iloc[p + s]
                if rs["ok_sell"]:
                    add(HedgeCandidate(f"Call-Spread {rl.strike:g}/{rs.strike:g}", "call_spread",
                                       [L(rl, "call", "long"), L(rs, "call", "short")]))

    if "long_strangle" in cfg.classes and not puts.empty and not calls.empty:
        for d in cfg.strangle_dists_pct:
            rp = _nearest(puts[puts["strike"] <= spot], spot * (1 - d / 100.0), "ok_buy")
            rc = _nearest(calls[calls["strike"] >= spot], spot * (1 + d / 100.0), "ok_buy")
            if rp is not None and rc is not None:
                add(HedgeCandidate(f"Strangle +-{d:g}% ({rp.strike:g}P/{rc.strike:g}C)", "long_strangle",
                                   [L(rp, "put", "long"), L(rc, "call", "long")]))

    if "underlying" in cfg.classes:
        add(HedgeCandidate("Underlying (Delta-Hedge)", "underlying",
                           [Leg("underlying", "long", float(spot), 0.0, 0.0)], allow_short=True))

    return list(cands.values())


# ═══════════════════════════════════════════════════════════════════════
#  3) KENNZAHLEN + MENGENOPTIMIERUNG
# ═══════════════════════════════════════════════════════════════════════

def _cvar_loss(X: np.ndarray, w: np.ndarray, alpha: float) -> np.ndarray:
    """Gewichteter CVaR als positiver VERLUST, zeilenweise (X: (m,n) oder (n,))."""
    X = np.atleast_2d(X)
    order = np.argsort(X, axis=1)
    Xs = np.take_along_axis(X, order, axis=1)
    Ws = w[order] / w.sum()
    cum_prev = np.cumsum(Ws, axis=1) - Ws
    take = np.clip(alpha - cum_prev, 0.0, Ws)
    return -(take * Xs).sum(axis=1) / alpha


def _score_matrix(X: np.ndarray, XS: np.ndarray, w: np.ndarray, proj: np.ndarray,
                  sc: Scenarios, cfg: HedgeConfig) -> dict:
    """Zielfunktion fuer m Positionen gleichzeitig. X: (m,n) Analog-P&L, XS: (m,k) Stress-P&L."""
    W = w.sum()
    mean = X @ w / W
    cvar = _cvar_loss(X, w, cfg.alpha)
    coef = X @ proj.T                                  # (m,3)
    b1, b2 = coef[:, 1], coef[:, 2]
    worst = XS.min(axis=1)
    score = (cvar - cfg.lam_mean * mean
             + cfg.lam_delta * np.abs(b1) * sc.sig
             + cfg.lam_gamma * np.abs(b2) * sc.sig ** 2
             + cfg.lam_stress * np.maximum(-worst, 0.0))
    return {"score": score, "mean": mean, "cvar": cvar, "b1": b1, "b2": b2, "worst": worst}


def _q_grid(cand: HedgeCandidate, cfg: HedgeConfig) -> np.ndarray:
    if cand.allow_short:
        n = int(round(cfg.delta_max / cfg.delta_step))
        return cfg.delta_step * np.arange(-n, n + 1)
    n = int(round(cfg.q_max / cfg.q_step))
    return cfg.q_step * np.arange(0, n + 1)


def e_floor_for(condor: Condor, sc: Scenarios, cfg: HedgeConfig) -> float:
    """Untergrenze fuer E[P&L] nach Hedge = min_e_keep * E des Condors allein
    (nur falls dessen E > 0, sonst keine Grenze)."""
    e0 = float(sc.w @ condor.pnl(sc.S) / sc.w.sum())
    return cfg.min_e_keep * e0 if e0 > 0 else -np.inf


def optimal_quantity(x_an: np.ndarray, x_st: np.ndarray, h_an: np.ndarray, h_st: np.ndarray,
                     cand: HedgeCandidate, sc: Scenarios, cfg: HedgeConfig,
                     w: np.ndarray | None = None, e_floor: float = -np.inf) -> dict:
    """Beste Menge q auf dem Raster (Lots) fuer EINEN Kandidaten, gegeben die
    aktuelle Position (x_an/x_st) und die Einheits-P&L des Hedges (h_an/h_st).
    w: optional alternative Gewichte (z.B. nur Train oder nur Val)."""
    w_use = sc.w if w is None else w
    proj = sc.proj if w is None else _projection(sc.B, w_use)
    q = _q_grid(cand, cfg)
    X = x_an[None, :] + q[:, None] * h_an[None, :]
    XS = x_st[None, :] + q[:, None] * h_st[None, :]
    m = _score_matrix(X, XS, w_use, proj, sc, cfg)
    k0 = int(np.argmin(np.abs(q)))
    score = np.where(m["mean"] >= e_floor, m["score"], np.inf)
    score[k0] = m["score"][k0]                          # q=0 bleibt immer zulaessig
    k = int(np.argmin(score))
    return {"q": float(q[k]), "score": float(score[k]), "score0": float(m["score"][k0])}


def _total(condor: Condor, hedges: list, S: np.ndarray) -> np.ndarray:
    x = condor.pnl(S)
    for cand, q in hedges:
        x = x + q * cand.unit_pnl(S)
    return x


def evaluate_position(condor: Condor, hedges: list, sc: Scenarios, cfg: HedgeConfig | None = None) -> dict:
    """Voller Report fuer Condor + Hedges [(cand, q), ...]. Leere Liste = Condor allein."""
    cfg = cfg or HedgeConfig()
    w = sc.w
    x = _total(condor, hedges, sc.S)
    xs = _total(condor, hedges, sc.S_stress)
    x_adj = _total(condor, hedges, sc.S_adj)

    worst_stress, worst_analog = float(xs.min()), float(x.min())
    ml = max(-worst_stress, -worst_analog)
    ml = ml if ml > 1e-9 else float("inf")
    st = weighted_pnl_stats(x, w, sc.dates, sc.hold, sc.train, ml, cfg.alpha)

    W = w.sum()
    mean = float(w @ x / W)
    coef = sc.proj @ x
    resid = x - sc.B @ coef
    sse = float(w @ resid ** 2)
    sst = float(w @ (x - mean) ** 2)

    return {
        "mean": mean, "tstat": st["tstat"], "e_train": st["e_train"], "e_val": st["e_val"],
        "win_rate_pct": st["win_rate_pct"], "full_loss_share": st["full_loss_share"],
        "q_tail": st["q_tail"],
        "mean_drift_adj": float(w @ x_adj / W),
        "cvar": float(_cvar_loss(x, w, cfg.alpha)[0]),
        "worst_stress": worst_stress, "worst_analog": worst_analog,
        "beta_ret": float(coef[1]), "beta_gamma": float(coef[2]),
        "delta_exp": float(abs(coef[1]) * sc.sig), "gamma_exp": float(abs(coef[2]) * sc.sig ** 2),
        "r2_market": (1.0 - sse / sst) if sst > 0 else float("nan"),
        "resid_sd": float(np.sqrt(sse / W)),
    }


def _hedge_leg_stats(cand: HedgeCandidate, q: float, sc: Scenarios, cfg: HedgeConfig) -> dict:
    """Der Hedge ISOLIERT: E, HAC-t, Train/Val, drift-bereinigt."""
    w = sc.w
    x = q * cand.unit_pnl(sc.S)
    W = w.sum()
    mean = float(w @ x / W)
    se, _ = _hac_se_weighted_mean(sc.dates, w, x, mean, sc.hold + 1)
    tr, va = sc.train, ~sc.train
    e_tr = float(w[tr] @ x[tr] / w[tr].sum()) if w[tr].sum() > 0 else np.nan
    e_va = float(w[va] @ x[va] / w[va].sum()) if w[va].sum() > 0 else np.nan
    adj = float(w @ (q * cand.unit_pnl(sc.S_adj)) / W)
    t = mean / se if se > 0 else np.nan
    ok = bool(np.isfinite(t) and mean > 0 and t > cfg.t_hedge and e_tr > 0 and e_va > 0 and adj > 0)
    return {"hedge_E": mean, "hedge_t": t, "hedge_e_train": e_tr, "hedge_e_val": e_va,
            "hedge_E_drift_adj": adj, "hedge_ev_ok": ok}


# ═══════════════════════════════════════════════════════════════════════
#  4) EINZELHEDGES RANKEN
# ═══════════════════════════════════════════════════════════════════════

def _unit_cache(cands: list, sc: Scenarios) -> dict:
    return {c.name: (c.unit_pnl(sc.S), c.unit_pnl(sc.S_stress)) for c in cands}


def _pareto(df: pd.DataFrame, cost_col: str, risk_col: str) -> pd.Series:
    """True = nicht dominiert (kleiner ist bei beiden Spalten besser)."""
    c, r = df[cost_col].to_numpy(float), df[risk_col].to_numpy(float)
    out = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        dom = (c <= c[i]) & (r <= r[i]) & ((c < c[i]) | (r < r[i]))
        if dom.any():
            out[i] = False
    return pd.Series(out, index=df.index)


def rank_single_hedges(condor: Condor, cands: list, sc: Scenarios, cfg: HedgeConfig | None = None,
                       cache: dict | None = None) -> pd.DataFrame:
    """Je Kandidat: optimale Menge, Vorher/Nachher-Report, Pareto-Flag.
    Sortierung: risk_ok (CVaR- und Stress-Verbesserung) zuerst, dann Score.
    'hedge_ev_ok' (Hedge hat selbst +EV, t, Train/Val, drift-bereinigt) ist nur
    ein ZUSATZ-Flag/Tiebreaker, keine Selektionsbedingung."""
    cfg = cfg or HedgeConfig()
    cache = cache or _unit_cache(cands, sc)
    x_an, x_st = condor.pnl(sc.S), condor.pnl(sc.S_stress)
    base = evaluate_position(condor, [], sc, cfg)
    ef = e_floor_for(condor, sc, cfg)

    w_tr, w_va = sc.w * sc.train, sc.w * (~sc.train)
    pre = []
    for c in cands:
        h_an, h_st = cache[c.name]
        o = optimal_quantity(x_an, x_st, h_an, h_st, c, sc, cfg, e_floor=ef)
        if o["q"] == 0.0 or o["score"] >= o["score0"] - 1e-12:
            continue
        pre.append((o["score"], c, o))
    pre.sort(key=lambda t: t[0])

    rows = []
    for _, c, o in pre[:cfg.top_n_eval]:
        h_an, h_st = cache[c.name]
        q = o["q"]
        rep = evaluate_position(condor, [(c, q)], sc, cfg)
        leg = _hedge_leg_stats(c, q, sc, cfg)
        q_tr = optimal_quantity(x_an, x_st, h_an, h_st, c, sc, cfg, w=w_tr, e_floor=ef)["q"] if w_tr.sum() > 0 else np.nan
        q_va = optimal_quantity(x_an, x_st, h_an, h_st, c, sc, cfg, w=w_va, e_floor=ef)["q"] if w_va.sum() > 0 else np.nan
        step = cfg.delta_step if c.allow_short else cfg.q_step
        q_stable = bool(np.isfinite(q_tr) and np.isfinite(q_va)
                        and abs(q_tr - q_va) <= max(0.5 * abs(q), 2 * step))
        impr = (base["cvar"] - rep["cvar"]) / abs(base["cvar"]) if base["cvar"] else np.nan
        rows.append({
            "name": c.name, "klass": c.klass, "q": q, "q_train": q_tr, "q_val": q_va, "q_stabil": q_stable,
            "debit_total": q * c.cost, "spread_kosten": abs(q) * c.spread_cost,
            "E_vorher": base["mean"], "E_nachher": rep["mean"], "dE": rep["mean"] - base["mean"],
            "cvar_vorher": base["cvar"], "cvar_nachher": rep["cvar"], "cvar_besser_pct": impr * 100.0,
            "stress_vorher": base["worst_stress"], "stress_nachher": rep["worst_stress"],
            "beta_ret": rep["beta_ret"], "beta_gamma": rep["beta_gamma"],
            "r2_market": rep["r2_market"], "resid_sd": rep["resid_sd"],
            "t_hac": rep["tstat"], "e_train": rep["e_train"], "e_val": rep["e_val"],
            "E_drift_adj": rep["mean_drift_adj"],
            **leg, "score": o["score"],
            "risk_ok": bool(np.isfinite(impr) and impr >= cfg.min_cvar_improvement
                            and rep["worst_stress"] > base["worst_stress"]),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["kosten_E"] = -df["dE"]                      # Erwartungswert, den der Hedge kostet
    df["pareto"] = False
    ok = df["risk_ok"]
    if ok.any():
        df.loc[ok, "pareto"] = _pareto(df[ok], "kosten_E", "cvar_nachher")
    return df.sort_values(["risk_ok", "score"], ascending=[False, True]).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════
#  5) KOMBINATION (GREEDY)
# ═══════════════════════════════════════════════════════════════════════

def greedy_combo(condor: Condor, cands: list, sc: Scenarios, cfg: HedgeConfig | None = None,
                 cache: dict | None = None) -> tuple[list, pd.DataFrame]:
    """Vorwaertsauswahl: pro Runde den Kandidaten (samt Menge) nehmen, der den
    Score am staerksten senkt. Stopp bei max_legs oder wenn der Gewinn unter
    cfg.min_step_gain (relativ) liegt. Rueckgabe: ([(cand, q), ...], Schritt-Tabelle)."""
    cfg = cfg or HedgeConfig()
    cache = cache or _unit_cache(cands, sc)
    x_an, x_st = condor.pnl(sc.S), condor.pnl(sc.S_stress)
    ef = e_floor_for(condor, sc, cfg)
    cur = float(_score_matrix(x_an[None, :], x_st[None, :], sc.w, sc.proj, sc, cfg)["score"][0])

    chosen: list = []
    steps = []
    for k in range(1, cfg.max_legs + 1):
        best = None
        for c in cands:
            if any(c.name == ch.name for ch, _ in chosen):
                continue
            h_an, h_st = cache[c.name]
            o = optimal_quantity(x_an, x_st, h_an, h_st, c, sc, cfg, e_floor=ef)
            if o["q"] == 0.0:
                continue
            if best is None or o["score"] < best[1]["score"]:
                best = (c, o)
        if best is None:
            break
        c, o = best
        gain = cur - o["score"]
        if gain <= 0.0 or gain < cfg.min_step_gain * abs(cur):
            break
        chosen.append((c, o["q"]))
        h_an, h_st = cache[c.name]
        x_an, x_st = x_an + o["q"] * h_an, x_st + o["q"] * h_st
        cur = o["score"]
        rep = evaluate_position(condor, chosen, sc, cfg)
        steps.append({"schritt": k, "hinzu": c.name, "q": o["q"], "score": cur, "score_gewinn": gain,
                      "E": rep["mean"], "t_hac": rep["tstat"], "e_train": rep["e_train"],
                      "e_val": rep["e_val"], "cvar": rep["cvar"], "stress_worst": rep["worst_stress"],
                      "beta_ret": rep["beta_ret"], "beta_gamma": rep["beta_gamma"],
                      "r2_market": rep["r2_market"], "E_drift_adj": rep["mean_drift_adj"]})
    return chosen, pd.DataFrame(steps)


# ═══════════════════════════════════════════════════════════════════════
#  6) DIAGRAMM + DEPOT-UEBERGABE
# ═══════════════════════════════════════════════════════════════════════

def payoff_curve(condor: Condor, hedges: list, sc: Scenarios) -> pd.DataFrame:
    """Payoff ueber das Stress-Raster: Condor allein vs. Condor+Hedges."""
    S = sc.S_stress
    return pd.DataFrame({
        "move_pct": sc.g_stress * 100.0, "S": S,
        "condor": condor.pnl(S),
        "hedge_only": sum((q * c.unit_pnl(S) for c, q in hedges), np.zeros_like(S)),
        "gesamt": _total(condor, hedges, S),
    })


def describe(cand: HedgeCandidate, q: float) -> str:
    parts = []
    for l in cand.legs:
        qty = abs(q)
        side = l.side if q >= 0 else ("short" if l.side == "long" else "long")
        if l.option_type == "underlying":
            parts.append(f"{side} {qty:g}x Underlying")
        else:
            parts.append(f"{side} {qty:g}x {l.option_type.capitalize()} {l.strike:g}")
    return " + ".join(parts)


def legs_for_depot(condor: Condor, hedges: list) -> list[dict]:
    """Beine im Format von dashboard_depot.add_combo_to_depot (Menge je Condor = 1,
    Hedge-Mengen skaliert). Underlying-Hedges werden NICHT uebergeben (Depot kennt
    nur Optionsbeine)."""
    legs = [
        {"option_type": "put", "side": "long", "strike": condor.k_put_wing, "qty": 1.0},
        {"option_type": "put", "side": "short", "strike": condor.k_put_short, "qty": 1.0},
        {"option_type": "call", "side": "short", "strike": condor.k_call_short, "qty": 1.0},
        {"option_type": "call", "side": "long", "strike": condor.k_call_wing, "qty": 1.0},
    ]
    for cand, q in hedges:
        if q <= 0:
            continue
        for l in cand.legs:
            if l.option_type == "underlying":
                continue
            legs.append({"option_type": l.option_type, "side": l.side,
                         "strike": float(l.strike), "qty": float(q)})
    return legs


# ═══════════════════════════════════════════════════════════════════════
#  HOCHLEVEL-EINSTIEGSPUNKT
# ═══════════════════════════════════════════════════════════════════════

def run_hedge_search(chain: pd.DataFrame, expiry, condor: Condor, moves: pd.DataFrame,
                     spot: float, cfg: HedgeConfig | None = None) -> dict:
    """Kompletter Ablauf: Szenarien -> Kandidaten -> Einzelranking -> Greedy-Kombi.
    chain = enriched Kette (market_data-Snapshot['enriched']), moves = analog_moves()."""
    cfg = cfg or HedgeConfig()
    sc = build_scenarios(moves, spot, cfg)
    cands = generate_hedge_candidates(chain, expiry, spot, cfg)
    if not cands:
        return {"scenarios": sc, "candidates": [], "singles": pd.DataFrame(),
                "combo": [], "combo_steps": pd.DataFrame(),
                "base": evaluate_position(condor, [], sc, cfg)}

    cache = _unit_cache(cands, sc)
    singles = rank_single_hedges(condor, cands, sc, cfg, cache)

    pool_names = set(singles["name"].head(cfg.greedy_pool)) if not singles.empty else set()
    pool = [c for c in cands if c.name in pool_names]
    combo, steps = greedy_combo(condor, pool, sc, cfg, cache) if pool else ([], pd.DataFrame())

    return {
        "scenarios": sc, "candidates": cands, "singles": singles,
        "combo": combo, "combo_steps": steps,
        "base": evaluate_position(condor, [], sc, cfg),
        "hedged": evaluate_position(condor, combo, sc, cfg) if combo else None,
        "curve": payoff_curve(condor, combo, sc),
    }