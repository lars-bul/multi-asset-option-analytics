"""
Pair-Trade-Kern: verbindet die Analog-Tage EINES fuehrenden Instruments (Lead,
auf dem geshortet wird - bestimmt den Aehnlichkeitsraum, siehe
analog_matching.find_analogs) mit der tatsaechlichen historischen Bewegung
EINES ZWEITEN Instruments (Follower) an GENAU DENSELBEN Kalendertagen.

BEIN-ROLLEN (die Spaltennamen folgen der echten Rolle):
  Beide Tabellen sind nach distance_pct aufsteigend sortiert; Index i = INNERES
  Bein (naeher am Spot), Index j = AEUSSERES Bein (weiter OTM, "Fluegel").
  Lead     (Credit-Spread): inneres Bein = SHORT, aeusseres Bein = LONG.
  Follower (Debit-Spread) : inneres Bein = LONG,  aeusseres Bein = SHORT.
  Ergebnis-Spalten:
    lead_short_distance     = Distanz des Lead-SHORT-Beins   (innen)
    lead_long_distance      = Distanz des Lead-LONG-Beins    (aussen, Fluegel)
    follower_long_distance  = Distanz des Follower-LONG-Beins  (innen)
    follower_short_distance = Distanz des Follower-SHORT-Beins (aussen, Fluegel)

ABLAUF von evaluate_pair_grid():
  1. Kreuzmenge: alle Lead-Spreads x alle Follower-Spreads x Mengenfaktoren x
     3 Strukturen (put_spread / call_spread / iron_condor).
  2. JEDE Kombination bekommt EINE Zeile. EINZIGER Ausschlussgrund:
       'ungueltige_bepreisung' - ein Bein hat keinen gueltigen Bid/Ask-Spread
     Alles andere wird bewertet.
  3. ROBUSTHEIT NUR AUF DEM GESAMTSPREAD (Lead + Follower zusammen):
     Train>0, Val>0, HAC-t>Schwelle, Tail. KEINE Einzelbein-Tests.
  4. Ausgabe: alle Zeilen; sortiert erst nach 'tested', dann 'passes', dann
     nach Verlustwahrscheinlichkeit (aufsteigend).

RANDWERTE OBERSEITE / UNTERSEITE (VORZEICHEN = ERGEBNIS: positiv = GEWINN,
negativ = VERLUST):
  'max_result_up'   = Ergebnis (P&L) der Gesamtposition, wenn BEIDE Basiswerte
                      stark steigen (S -> unendlich, jedes Bein im Extremwert).
                      Beispiel Put-Credit-Spread: Kredit = Max-Gewinn -> positiv;
                      Put-Debit: Debit = Max-Verlust -> negativ.
  'max_result_down' = Ergebnis (P&L) der Gesamtposition, wenn BEIDE Basiswerte
                      auf 0 fallen.
  (Frueher 'max_loss_up' / 'max_gain_down' mit unterschiedlichen Vorzeichen,
  dann kurz beide als Verlust; jetzt einheitlich als ERGEBNIS.)
  Beide exakt und korrelationsfrei: oben ist jeder Put wertlos und der
  Call-Anteil konstant (-Summe sign*K, da jedes Paket aus Long/Short-Paaren
  besteht); unten ist jeder Call wertlos und der Put-Anteil Summe sign*K.
  WICHTIG: Das sind Werte an den Raendern (S=0 bzw. S->inf), NICHT der
  Max-Verlust der Position. Der Max-Verlust ('max_loss_struct') ist das
  Minimum ueber alle Endkurse; hat die Position konvexe Fluegel (z.B. ein
  12 % breiter Follower-Debit-Condor), koennen die Randwerte sehr gross im
  GEWINN liegen (stark positive Werte), waehrend der Max-Verlust klein bleibt.

MAX-VERLUST (korrigiert) - warum die alte Version falsch war:
  Alt (a): Iron Condor = Summe von Put- UND Call-Max-Loss. Ein Condor kann aber
           nie beide Seiten gleichzeitig verlieren -> ueberschaetzt.
  Alt (b): '-min(pnl_d)' ueber die Analog-Tage. Das ist der schlechteste
           BEOBACHTETE Tag, kein Max-Verlust: bei gutartiger Historie winzig
           -> 'ror' (= mean/max_loss) und die Tail-Kennzahlen ('q_tail',
           'full_loss_share') werden aufgeblaeht bzw. bedeutungslos.
  Neu ('structural', Standard): der EXAKTE Max-Verlust der Gesamtposition.
           Der Gesamt-Payoff ist stueckweise linear und SEPARABEL in
           (S_lead, S_follower): pnl = f(S_lead) + g(S_follower). Das Minimum
           einer separablen Summe ist die Summe der Minima, und jedes
           Minimum liegt an einem Knickpunkt (S=0 oder einem der Strikes).
           Daher wird jedes Bein-Paket an {0} + seinen Strikes ausgewertet
           (Condor: korrekt max(Put-, Call-Verlust), nicht die Summe).
           Das ist die garantierte Obergrenze OHNE Annahme ueber die
           Korrelation. Zusaetzlich wird 'max_loss_hist' (schlechtester
           simulierter Tag) als Info-Spalte ausgegeben; mit
           cfg.max_loss_basis='empirical' kann er als Bezugsgroesse gewaehlt
           werden.
           Hinweis: Hat ein Bein-Paket zwischen zwei Strikes ein Plateau auf
           Minimum-Niveau (z.B. Put- und Call-Strike des 'Distanz 0'-Beins liegen
           auf verschiedenen Listenstrikes), kann ein einzelner Analog-Tag das
           strukturelle Minimum EXAKT erreichen (max_loss_hist == max_loss_struct).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from condor_search import weighted_pnl_stats, criteria_flags

_NAN_STATS = {
    "mean": np.nan, "ror": np.nan, "tstat": np.nan, "e_train": np.nan, "e_val": np.nan,
    "full_loss_share": np.nan, "n_full_loss": None, "q_tail": np.nan, "win_rate_pct": np.nan,
}
_NAN_FLAGS = {"c_train": False, "c_val": False, "c_t": False, "c_tail": False, "c_base": False}


# ═══════════════════════════════════════════════════════════════════════
#  Analog-Bewegungen (Lead + Follower vom SELBEN Kalendertag)
# ═══════════════════════════════════════════════════════════════════════

def pair_analog_moves(analogs: pd.DataFrame, lead_training_table: pd.DataFrame,
                       follower_daily_df: pd.DataFrame, T_days: int,
                       query_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Je Lead-Analog-Tag: Lead-Rendite aus der training_table, Follower-Rendite
    aus dessen Tageskursen (Entry = Schlusskurs am selben Kalendertag, Exit =
    erster Handelstag mit >= T_days Kalendertagen Abstand). query_date: Exit
    nach query_date wird verworfen (Kausalitaet). Rueckgabe: entry_date,
    hold_days, weight, distance, lead_ret, follower_ret (+ Entry/Exit-Kurse);
    out.attrs['n_no_follower_match'] = Anzahl verworfener Tage."""
    follower_daily_df = follower_daily_df.sort_values("date").reset_index(drop=True)
    follower_dates = follower_daily_df["date"].to_numpy()
    follower_closes = follower_daily_df["close"].to_numpy()

    tt = lead_training_table[lead_training_table["T_days"] == T_days].set_index("entry_date")
    q_ts = pd.Timestamp(query_date) if query_date is not None else None

    rows, n_no_follower_match = [], 0
    for _, arow in analogs.iterrows():
        d = arow["analog_date"]
        if d not in tt.index:
            continue
        trow = tt.loc[d]
        if isinstance(trow, pd.DataFrame):
            trow = trow.iloc[0]
        if q_ts is not None and pd.Timestamp(trow["exit_date"]) > q_ts:
            continue

        lead_ret = trow["exit_S"] / trow["entry_S"] - 1.0

        entry_idx = np.searchsorted(follower_dates, np.datetime64(d), side="left")
        if entry_idx >= len(follower_dates) or follower_dates[entry_idx] != np.datetime64(d):
            n_no_follower_match += 1
            continue
        target_exit = np.datetime64(d) + np.timedelta64(T_days, "D")
        exit_idx = np.searchsorted(follower_dates, target_exit, side="left")
        if exit_idx >= len(follower_dates):
            n_no_follower_match += 1
            continue

        f_entry, f_exit = float(follower_closes[entry_idx]), float(follower_closes[exit_idx])
        if f_entry <= 0.0 or np.isnan(f_entry) or np.isnan(f_exit):
            n_no_follower_match += 1
            continue

        rows.append({
            "entry_date": d,
            "hold_days": (trow["exit_date"] - d).days,
            "weight": arow["weight"] if "weight" in arow.index else 1.0,
            "distance": arow["distance"],
            "lead_entry_S": trow["entry_S"], "lead_exit_S": trow["exit_S"], "lead_ret": lead_ret,
            "follower_entry_S": f_entry, "follower_exit_S": f_exit,
            "follower_ret": f_exit / f_entry - 1.0,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("entry_date").reset_index(drop=True)
    out.attrs["n_no_follower_match"] = n_no_follower_match
    return out


def project_pair_payout(lead_spot_today: float, follower_spot_today: float, moves: pd.DataFrame,
                         lead_payoff_fn, follower_payoff_fn) -> pd.DataFrame:
    """Projiziert jeden Analog-Tag auf die HEUTIGEN Spots und wertet die
    Payoff-Funktionen (S -> P&L, ohne Entry-Kosten) aus."""
    out = moves.copy()
    out["S_lead_hyp"] = lead_spot_today * (1.0 + out["lead_ret"])
    out["S_follower_hyp"] = follower_spot_today * (1.0 + out["follower_ret"])
    out["lead_payoff"] = lead_payoff_fn(out["S_lead_hyp"].to_numpy())
    out["follower_payoff"] = follower_payoff_fn(out["S_follower_hyp"].to_numpy())
    out["combined_payoff"] = out["lead_payoff"] + out["follower_payoff"]
    return out


# ═══════════════════════════════════════════════════════════════════════
#  Konfiguration
# ═══════════════════════════════════════════════════════════════════════

_SORT_ASCENDING = {"ror": False, "mean": False, "tstat": False, "loss_prob_pct": True}


@dataclass
class PairTradeConfig:
    """Feldnamen t_min/max_full_loss_share/tail_quantile/tail_floor kompatibel zu
    condor_search.CondorConfig (criteria_flags() per Duck-Typing)."""
    price_basis: str = "bidask"        # "bidask" | "mid"
    t_min: float = 3.0
    max_full_loss_share: float = 0.03
    tail_quantile: float = 0.05
    tail_floor: float = -0.75
    sort_by: str = "loss_prob_pct"     # "loss_prob_pct" (aufsteigend) | "ror" | "mean" | "tstat"
    max_loss_basis: str = "structural"  # "structural" (exakt, Standard) | "empirical" (schlechtester Analog-Tag)
    max_grid_cells: int = 5_000        # Zellen = Lead-Paare x Follower-Paare x Mengenfaktoren (Zeilen = 3x)
    lead_multiplier: float = 100.0
    follower_multiplier: float = 100.0
    qty_factors: tuple = (1.0,)        # Follower-Menge = notional_ratio * Faktor


# ═══════════════════════════════════════════════════════════════════════
#  Bausteine
# ═══════════════════════════════════════════════════════════════════════

def _price_columns(table: pd.DataFrame, price_basis: str):
    if price_basis == "mid":
        mp = pd.to_numeric(table["mid_put"], errors="coerce").to_numpy(float)
        mc = pd.to_numeric(table["mid_call"], errors="coerce").to_numpy(float)
        return mp, mp, mc, mc
    return tuple(pd.to_numeric(table[c], errors="coerce").to_numpy(float)
                 for c in ("bid_put", "ask_put", "bid_call", "ask_call"))


def _row_payoffs(table: pd.DataFrame, S_hyp: np.ndarray):
    """Innerer Wert je Tabellenzeile bei Verfall gegen die Analog-Spots: (pay_p, pay_c)."""
    Kp = pd.to_numeric(table["strike_put"], errors="coerce").to_numpy(float)
    Kc = pd.to_numeric(table["strike_call"], errors="coerce").to_numpy(float)
    return np.maximum(Kp[:, None] - S_hyp[None, :], 0.0), np.maximum(S_hyp[None, :] - Kc[:, None], 0.0)


def _vertical_spread(K_inner, K_wing, bid_inner, ask_inner, bid_wing, ask_wing,
                      pay_inner, pay_wing, is_put: bool, is_short: bool):
    """Ein Vertical-Spread (eine Seite, ein Instrument). 'inner' = naeher am
    Spot (kleinere distance_pct), 'wing' = weiter OTM.
    is_short=True (Lead): Short inner / Long Wing -> Credit.
    is_short=False (Follower): Long inner / Short Wing -> Debit.
    Rueckgabe (net_cf, pay_diff, legs) oder None. legs = [(K, is_put, sign)],
    sign +1 long / -1 short. pnl(S) = net_cf + sum(sign*intrinsic(S)); fuer den
    Lead entspricht das net_cf - (pay_inner - pay_wing), fuer den Follower
    net_cf + (pay_inner - pay_wing)."""
    width = abs(K_inner - K_wing)
    if not np.isfinite(width) or width <= 0.0:
        return None
    if is_short:
        net_cf = bid_inner - ask_wing
        if not (np.isfinite(net_cf) and 0.0 < net_cf < width):
            return None
        legs = [(K_inner, is_put, -1), (K_wing, is_put, +1)]
    else:
        net_cf = -(ask_inner - bid_wing)
        debit = -net_cf
        if not (np.isfinite(debit) and 0.0 < debit < width):
            return None
        legs = [(K_inner, is_put, +1), (K_wing, is_put, -1)]
    return net_cf, pay_inner - pay_wing, legs


def _combine(a, b):
    """Zwei Spread-Pakete (Put + Call) zu einem Condor-Paket addieren."""
    if a is None or b is None:
        return None
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def _min_pnl_points(net_cf: float, legs) -> float:
    """EXAKTES Minimum der P&L (in Punkten) eines Bein-Pakets ueber alle
    Endkurse S >= 0. Payoff ist stueckweise linear mit Knicken nur an den
    Strikes -> Auswertung an {0} + Strikes genuegt (oberhalb des hoechsten
    Strikes ist der Payoff konstant, da jedes Paket Long/Short-Paare enthaelt)."""
    cand = np.array([0.0] + [k for k, _, _ in legs], dtype=float)
    pnl = np.full(cand.shape, float(net_cf))
    for K, is_put, sign in legs:
        pnl += sign * (np.maximum(K - cand, 0.0) if is_put else np.maximum(cand - K, 0.0))
    return float(pnl.min())


def _extreme_pnl_points(net_cf: float, legs, side: str) -> float:
    """P&L (in Punkten) eines Bein-Pakets im Extremwert der Oberseite
    (side='up', S -> unendlich) bzw. Unterseite (side='down', S = 0).
    Oben: Puts wertlos; Calls liefern sign*(S-K), die S-Anteile heben sich auf
    (Long/Short-Paare) -> Rest = -Summe(sign*K ueber Calls).
    Unten: Calls wertlos; Puts liefern sign*K.
    Rueckgabe ist P&L: positiv = Gewinn, negativ = Verlust (wird unveraendert
    als 'max_result_up' / 'max_result_down' ausgegeben)."""
    if side == "up":
        return float(net_cf - sum(sign * K for K, is_put, sign in legs if not is_put))
    return float(net_cf + sum(sign * K for K, is_put, sign in legs if is_put))


def dedupe_strike_rows(table: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    key = table[["strike_put", "strike_call"]].apply(pd.to_numeric, errors="coerce").round(6)
    dup = key.duplicated(keep="first").to_numpy()
    return table.loc[~dup].reset_index(drop=True), int(dup.sum())


def _spread_vol_pct(iv: np.ndarray, i: int, j: int) -> float:
    vals = [v for v in (iv[i], iv[j]) if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


# ═══════════════════════════════════════════════════════════════════════
#  Grid-Search
# ═══════════════════════════════════════════════════════════════════════

def evaluate_pair_grid(lead_table: pd.DataFrame, follower_table: pd.DataFrame, pair_moves: pd.DataFrame,
                        lead_spot_today: float, follower_spot_today: float,
                        cfg: PairTradeConfig | None = None) -> pd.DataFrame:
    """Siehe Moduldokstring. Geldeinheiten: GELD je 1 Lead-Kontrakt (Punkte x
    Multiplikator); Follower-Menge 'follower_qty' = Kontrakte je Lead-Kontrakt
    (notional_ratio * Faktor). Heutige echte Strikes/Preise fest, nur die
    %-Bewegungen der Analog-Tage werden auf den heutigen Spot projiziert.

    Rueckgabe: EINE Zeile je (Lead-Paar, Follower-Paar, Struktur, Mengenfaktor),
    inkl. der Zeilen mit ungueltiger Bepreisung ('tested'=False)."""
    if cfg is None:
        cfg = PairTradeConfig()
    if cfg.sort_by not in _SORT_ASCENDING:
        raise ValueError(f"sort_by muss eines von {sorted(_SORT_ASCENDING)} sein (erhalten: {cfg.sort_by!r}).")
    if cfg.max_loss_basis not in ("structural", "empirical"):
        raise ValueError("max_loss_basis muss 'structural' oder 'empirical' sein.")
    if pair_moves.empty:
        return pd.DataFrame()

    lead_table = lead_table.sort_values("distance_pct").reset_index(drop=True)
    follower_table = follower_table.sort_values("distance_pct").reset_index(drop=True)
    lead_table, n_lead_dup = dedupe_strike_rows(lead_table)
    follower_table, n_foll_dup = dedupe_strike_rows(follower_table)
    n_lead, n_follower = len(lead_table), len(follower_table)

    qty_factors = tuple(cfg.qty_factors) or (1.0,)
    lm, fm = float(cfg.lead_multiplier), float(cfg.follower_multiplier)
    notional_ratio = (lead_spot_today * lm) / (follower_spot_today * fm)

    def _finish(out: pd.DataFrame) -> pd.DataFrame:
        out.attrs.update({"dedupe_dropped": (n_lead_dup, n_foll_dup), "notional_ratio": notional_ratio,
                          "n_lead_rows": n_lead, "n_follower_rows": n_follower})
        return out

    n_lead_pairs = n_lead * (n_lead - 1) // 2
    n_follower_pairs = n_follower * (n_follower - 1) // 2
    n_cells = n_lead_pairs * n_follower_pairs * len(qty_factors)
    if n_cells > cfg.max_grid_cells:
        raise ValueError(
            f"Grid zu gross: {n_lead_pairs} Lead-Paare x {n_follower_pairs} Follower-Paare x "
            f"{len(qty_factors)} Mengenfaktoren = {n_cells:,} Zellen (Grenze: {cfg.max_grid_cells:,}). "
            "Distanz-Raster oder Mengenfaktoren verkleinern."
        )
    if n_lead_pairs == 0 or n_follower_pairs == 0:
        return _finish(pd.DataFrame())

    w = pair_moves["weight"].to_numpy(float)
    dates = pair_moves["entry_date"].to_numpy()
    hold = int(pair_moves["hold_days"].max())
    train_mask = (pair_moves["entry_date"] < pair_moves["entry_date"].median()).to_numpy()

    S_lead_hyp = lead_spot_today * (1.0 + pair_moves["lead_ret"].to_numpy(float))
    S_foll_hyp = follower_spot_today * (1.0 + pair_moves["follower_ret"].to_numpy(float))
    pay_p_lead, pay_c_lead = _row_payoffs(lead_table, S_lead_hyp)
    pay_p_foll, pay_c_foll = _row_payoffs(follower_table, S_foll_hyp)

    def num(t, c):
        return pd.to_numeric(t[c], errors="coerce").to_numpy(float)

    lbp, lap, lbc, lac = _price_columns(lead_table, cfg.price_basis)
    fbp, fap, fbc, fac = _price_columns(follower_table, cfg.price_basis)
    lKp, lKc, ldist = num(lead_table, "strike_put"), num(lead_table, "strike_call"), num(lead_table, "distance_pct")
    fKp, fKc, fdist = num(follower_table, "strike_put"), num(follower_table, "strike_call"), num(follower_table, "distance_pct")
    liv, fiv = num(lead_table, "iv_pair_pct"), num(follower_table, "iv_pair_pct")

    follower_cache = []
    for fi in range(n_follower - 1):
        for fj in range(fi + 1, n_follower):
            put = _vertical_spread(fKp[fi], fKp[fj], fbp[fi], fap[fi], fbp[fj], fap[fj],
                                   pay_p_foll[fi], pay_p_foll[fj], True, False)
            call = _vertical_spread(fKc[fi], fKc[fj], fbc[fi], fac[fi], fbc[fj], fac[fj],
                                    pay_c_foll[fi], pay_c_foll[fj], False, False)
            follower_cache.append((fi, fj, put, call))

    rows = []
    for li in range(n_lead - 1):
        for lj in range(li + 1, n_lead):
            lput = _vertical_spread(lKp[li], lKp[lj], lbp[li], lap[li], lbp[lj], lap[lj],
                                    pay_p_lead[li], pay_p_lead[lj], True, True)
            lcall = _vertical_spread(lKc[li], lKc[lj], lbc[li], lac[li], lbc[lj], lac[lj],
                                     pay_c_lead[li], pay_c_lead[lj], False, True)
            vol_lead = _spread_vol_pct(liv, li, lj)

            for fi, fj, fput, fcall in follower_cache:
                vol_foll = _spread_vol_pct(fiv, fi, fj)

                structures = [
                    ("put_spread", lput, fput),
                    ("call_spread", lcall, fcall),
                    ("iron_condor", _combine(lput, lcall), _combine(fput, fcall)),
                ]
                for structure, lead_leg, foll_leg in structures:
                    for factor in qty_factors:
                        q = notional_ratio * factor
                        base = {
                            "structure": structure,
                            "lead_i": li, "lead_j": lj, "follower_i": fi, "follower_j": fj,
                            # Rollenbasierte Distanzen (innen = i, aussen = j):
                            #   Lead: innen SHORT, aussen LONG | Follower: innen LONG, aussen SHORT
                            "lead_short_distance": float(ldist[li]),
                            "lead_long_distance": float(ldist[lj]),
                            "follower_long_distance": float(fdist[fi]),
                            "follower_short_distance": float(fdist[fj]),
                            "vol_lead_pct": vol_lead, "vol_follower_pct": vol_foll,
                            "follower_qty": q, "qty_factor": factor,
                            "net_cashflow": np.nan, "max_loss": np.nan,
                            "max_loss_struct": np.nan, "max_loss_hist": np.nan,
                            "max_result_up": np.nan, "max_result_down": np.nan,
                            "loss_prob_pct": np.nan, "full_loss_pct": np.nan, "lead_full_loss_pct": np.nan,
                            "lead_net_cashflow": np.nan, "follower_net_cashflow": np.nan,
                            "tested": False, "exclude_reason": "", "passes": False,
                            **_NAN_STATS, **_NAN_FLAGS,
                        }
                        if lead_leg is None or foll_leg is None:
                            base["exclude_reason"] = "ungueltige_bepreisung"
                            rows.append(base)
                            continue

                        l_net, l_pay, l_legs = lead_leg
                        f_net, f_pay, f_legs = foll_leg
                        f_scale = q * fm
                        combined_net = lm * l_net + f_scale * f_net
                        # EXAKTER struktureller Max-Verlust (separable Summe der Minima)
                        lead_min = lm * _min_pnl_points(l_net, l_legs)
                        struct_ml = max(0.0, -(lead_min + f_scale * _min_pnl_points(f_net, f_legs)))
                        base["net_cashflow"], base["max_loss_struct"] = combined_net, struct_ml
                        base["lead_net_cashflow"], base["follower_net_cashflow"] = lm * l_net, f_scale * f_net

                        # Randwerte: beide Basiswerte oben (S -> inf) bzw. unten (S = 0).
                        # Vorzeichen = ERGEBNIS: positiv = Gewinn, negativ = Verlust.
                        base["max_result_up"] = (lm * _extreme_pnl_points(l_net, l_legs, "up")
                                                 + f_scale * _extreme_pnl_points(f_net, f_legs, "up"))
                        base["max_result_down"] = (lm * _extreme_pnl_points(l_net, l_legs, "down")
                                                   + f_scale * _extreme_pnl_points(f_net, f_legs, "down"))

                        pnl_d = lm * (l_net - l_pay) + f_scale * (f_net + f_pay)
                        hist_ml = max(0.0, -float(np.min(pnl_d)))
                        max_loss = struct_ml if cfg.max_loss_basis == "structural" else hist_ml
                        if max_loss <= 0.0 < struct_ml:
                            # Kein einziger Verlusttag in den Analogdaten: Paar NICHT verwerfen (das waeren
                            # genau die besten Paare), sondern auf den strukturellen Max-Verlust zurueckfallen.
                            max_loss = struct_ml
                        base["max_loss_hist"], base["max_loss"] = hist_ml, max_loss

                        stats = weighted_pnl_stats(pnl_d, w, dates, hold, train_mask, max_loss, cfg.tail_quantile)
                        flags = criteria_flags(stats, cfg)
                        # Verlustwahrscheinlichkeit = gewichteter Anteil Analog-Tage mit P&L < 0
                        loss_prob = float(w[pnl_d < 0.0].sum() / w.sum() * 100.0)
                        lead_pnl_d = lm * (l_net - l_pay)
                        base["lead_full_loss_pct"] = float(w[lead_pnl_d <= lead_min + 1e-7].sum() / w.sum() * 100.0)
                        base["full_loss_pct"] = (float(stats["full_loss_share"]) * 100.0
                                                 if np.isfinite(stats["full_loss_share"]) else np.nan)
                        base.update(stats)
                        base.update(flags)
                        base.update({"loss_prob_pct": loss_prob, "tested": True, "passes": bool(flags["c_base"])})
                        rows.append(base)

    result = pd.DataFrame(rows)
    if result.empty:
        return _finish(result)

    result = result.sort_values(
        ["tested", "passes", cfg.sort_by], ascending=[False, False, _SORT_ASCENDING[cfg.sort_by]],
        na_position="last",
    ).reset_index(drop=True)
    return _finish(result)