"""
Pair-Trade-Kern: verbindet die Analog-Tage EINES fuehrenden Instruments
(auf dem geshortet wird - bestimmt den Aehnlichkeits-Raum, siehe
analog_matching.find_analogs) mit der tatsaechlichen historischen Bewegung
EINES ZWEITEN (Follower-)Instruments an GENAU DENSELBEN Kalendertagen.

WARUM ein Instrument fuehrend sein muss: der PCA-Aehnlichkeitsraum ist eine
Eigenschaft EINES Kursverlaufs (siehe analog_matching.fit_similarity_space -
Tages-Features wie Returns/HV ueber verschiedene Fenster). "Aehnliche Tage"
kann man nur relativ zu EINEM Referenz-Instrument definieren. Fuer den
Pair-Trade ist das face-valide: die Analog-Tage sollen beschreiben, wie sich
eine SETUP-AEHNLICHE Situation auf der Instrument entwickelt hat, auf der
der SHORT (das eigentliche Risiko) liegt - das Follower-Instrument wird an
GENAU DIESEN Tagen mitbetrachtet, nicht ueber einen eigenen Aehnlichkeits-
Raum gesucht (das waere eine andere, hier NICHT gewollte Frage: "welche
Tage aehneln dem Follower selbst am meisten").

Dieses Modul haengt bewusst NICHT von condor_search.py ab (dessen
Bewertungs-/Scoring-Funktionen wie evaluate_condor_grid/analog_robustness/
weighted_pnl_stats werden fuer den fertigen Pair-Trade-Modus noch benoetigt,
liegen mir aber noch nicht vor - siehe TODOs unten). Was hier steht, ist der
instrumentenuebergreifende Teil: Lead-Analogtage + Follower-Kurshistorie ->
eine gemeinsame Tabelle mit Lead- UND Follower-Rendite pro Analog-Tag.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analog_matching import _hac_se_weighted_mean
from condor_search import weighted_pnl_stats, criteria_flags


def pair_analog_moves(analogs: pd.DataFrame, lead_training_table: pd.DataFrame,
                       follower_daily_df: pd.DataFrame, T_days: int,
                       query_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Fuer jeden Lead-Analog-Tag (aus analog_matching.find_analogs(),
    Spalten 'analog_date', 'distance', 'weight'): die Lead-Rendite kommt aus
    lead_training_table (wie in condor_search.analog_moves() - entry_S/exit_S
    fuer dasselbe T_days), die Follower-Rendite wird HIER direkt aus der
    taeglichen Kurshistorie des Follower-Instruments (follower_daily_df,
    Spalten 'date','close') ermittelt: Follower-Entry = Schlusskurs an
    exakt demselben Kalendertag wie der Lead-Analog-Tag, Follower-Exit = der
    erste Handelstag mit mindestens T_days Kalendertagen Abstand (identische
    Konvention zu labels.generate_strangle_labels/backtest.run_daily_ladder).

    query_date (optional): wie in analog_matching.evaluate_analog_set -
    Zeilen, deren Exit NACH query_date liegt, werden verworfen (Kausalitaet
    bei langen Laufzeiten).

    Rueckgabe: entry_date, hold_days, weight, distance,
    lead_ret (= exit_S/entry_S - 1 des FUEHRENDEN Instruments),
    follower_ret (dieselbe Definition fuer das FOLGE-Instrument),
    UND ein 'follower_data_ok'-Flag (False, falls am Lead-Analog-Tag kein
    Follower-Kurs vorlag - z.B. unterschiedliche Handelskalender/Feiertage
    oder eine kuerzere Follower-Historie). Zeilen mit follower_data_ok=False
    werden bereits herausgefiltert, das Flag dient nur der Diagnose/Anzeige.
    """
    follower_daily_df = follower_daily_df.sort_values("date").reset_index(drop=True)
    follower_dates = follower_daily_df["date"].to_numpy()
    follower_closes = follower_daily_df["close"].to_numpy()

    tt = lead_training_table[lead_training_table["T_days"] == T_days].set_index("entry_date")
    q_ts = pd.Timestamp(query_date) if query_date is not None else None

    rows = []
    n_no_follower_match = 0
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

        # Follower-Entry: Schlusskurs an GENAU diesem Kalendertag.
        follower_entry_idx = np.searchsorted(follower_dates, np.datetime64(d), side="left")
        if follower_entry_idx >= len(follower_dates) or follower_dates[follower_entry_idx] != np.datetime64(d):
            n_no_follower_match += 1
            continue
        # Follower-Exit: erster Handelstag mit >= T_days Kalendertagen Abstand
        # (identische Konvention zur Lead-Seite, siehe labels.py/backtest.py).
        target_exit = np.datetime64(d) + np.timedelta64(T_days, "D")
        follower_exit_idx = np.searchsorted(follower_dates, target_exit, side="left")
        if follower_exit_idx >= len(follower_dates):
            n_no_follower_match += 1
            continue

        follower_entry_S = float(follower_closes[follower_entry_idx])
        follower_exit_S = float(follower_closes[follower_exit_idx])
        if follower_entry_S <= 0.0 or np.isnan(follower_entry_S) or np.isnan(follower_exit_S):
            n_no_follower_match += 1
            continue
        follower_ret = follower_exit_S / follower_entry_S - 1.0

        rows.append({
            "entry_date": d,
            "hold_days": (trow["exit_date"] - d).days,
            "weight": arow["weight"] if "weight" in arow.index else 1.0,
            "distance": arow["distance"],
            "lead_entry_S": trow["entry_S"], "lead_exit_S": trow["exit_S"], "lead_ret": lead_ret,
            "follower_entry_S": follower_entry_S, "follower_exit_S": follower_exit_S,
            "follower_ret": follower_ret,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("entry_date").reset_index(drop=True)
    out.attrs["n_no_follower_match"] = n_no_follower_match
    return out


def project_pair_payout(lead_spot_today: float, follower_spot_today: float,
                         moves: pd.DataFrame,
                         lead_payoff_fn, follower_payoff_fn) -> pd.DataFrame:
    """Projiziert JEDEN Analog-Tag (Lead- UND Follower-Rendite aus
    pair_analog_moves()) auf die HEUTIGEN Spot-Preise beider Instrumente und
    wertet die dort uebergebenen Payoff-Funktionen aus - identisches Prinzip
    zu dashboard_livechain._payout_at_todays_scale()/dashboard_depot.py, nur
    fuer ZWEI Instrumente gleichzeitig statt fuer eines.

    lead_payoff_fn/follower_payoff_fn: Funktionen S (np.ndarray) -> P&L
    (np.ndarray) OHNE Entry-Kosten (reiner Verfallswert je Bein-Kombination,
    z.B. -max(K_put - S, 0) fuer einen Long Put) - die Entry-Praemie
    (Kredit/Debit) wird AUSSERHALB dieser Funktion addiert (siehe
    Modul-TODO: Gesamt-Kredit/Netto-Kosten der Kombo kommt aus der echten
    Bid/Ask-Kette beider Instrumente, analog zu condor_search.py).

    Rueckgabe: moves + Spalten 'S_lead_hyp', 'S_follower_hyp',
    'lead_payoff', 'follower_payoff', 'combined_payoff'."""
    out = moves.copy()
    out["S_lead_hyp"] = lead_spot_today * (1.0 + out["lead_ret"])
    out["S_follower_hyp"] = follower_spot_today * (1.0 + out["follower_ret"])
    out["lead_payoff"] = lead_payoff_fn(out["S_lead_hyp"].to_numpy())
    out["follower_payoff"] = follower_payoff_fn(out["S_follower_hyp"].to_numpy())
    out["combined_payoff"] = out["lead_payoff"] + out["follower_payoff"]
    return out


# ═══════════════════════════════════════════════════════════════════════
#  GRID-SEARCH: Lead-Credit-Spread/Condor x Follower-Debit-Spread/Reverse-
#  Condor, ueber die VOLLE Kreuzmenge beider Strike-Tabellen.
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PairTradeConfig:
    """Konfiguration fuer evaluate_pair_grid(). Feldnamen (t_min,
    max_full_loss_share, tail_quantile, tail_floor, sort_by) bewusst
    identisch zu condor_search.CondorConfig, damit condor_search.
    criteria_flags() unveraendert (Duck-Typing) wiederverwendet werden kann.

    require_lead_plateau (statt require_plateau wie bei CondorConfig): das
    Plateau-Kriterium (Nachbar-Strikes ebenfalls robust) wird bewusst NUR
    auf der LEAD-Seite verlangt - die Lead-Seite soll wie beim bestehenden
    Iron-Condor-Search eine fuer sich genommen robuste Einkommensquelle
    sein. Die Follower-Seite ist ein HEDGE und darf fuer sich genommen
    erwartungswert-negativ sein (wie eine Versicherungspraemie) - ein
    Plateau-Zwang dort wuerde systematisch gute Hedges aussortieren, siehe
    Erklaerung im Chat vor dieser Implementierung.

    max_grid_cells: harte Sicherheitsgrenze (siehe evaluate_pair_grid()-
    Docstring, Abschnitt PERFORMANCE-WARNUNG) - wirft ValueError statt
    minutenlang zu rechnen, wenn das Distanz-Raster zu fein aufgeloest ist."""
    price_basis: str = "bidask"        # "bidask" (echt) | "mid" (Naeherung, z.B. bei duennem NDX-Orderbuch)
    t_min: float = 3.0
    max_full_loss_share: float = 0.03
    tail_quantile: float = 0.05
    tail_floor: float = -0.75
    require_lead_plateau: bool = True
    sort_by: str = "ror"
    max_grid_cells: int = 20_000


def _price_columns(table: pd.DataFrame, price_basis: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(bid_put, ask_put, bid_call, ask_call) - bei price_basis='mid' werden
    Bid=Ask=Mid gesetzt (Naeherung fuer ein duennes Orderbuch, z.B. bei
    NDX-Optionen ausserhalb der Handelszeit - identisches Prinzip zu
    condor_search.CondorConfig.price_basis)."""
    if price_basis == "mid":
        mid_p = pd.to_numeric(table["mid_put"], errors="coerce").to_numpy(float)
        mid_c = pd.to_numeric(table["mid_call"], errors="coerce").to_numpy(float)
        return mid_p, mid_p, mid_c, mid_c
    return (
        pd.to_numeric(table["bid_put"], errors="coerce").to_numpy(float),
        pd.to_numeric(table["ask_put"], errors="coerce").to_numpy(float),
        pd.to_numeric(table["bid_call"], errors="coerce").to_numpy(float),
        pd.to_numeric(table["ask_call"], errors="coerce").to_numpy(float),
    )


def _row_payoffs(table: pd.DataFrame, S_d_hyp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Innerer Wert JEDER Zeile der Distanz-Tabelle bei Verfall, gegen die
    per Analog-Tag hypothetisch projizierten Spot-Preise S_d_hyp -
    identisches Prinzip zu condor_search.evaluate_condor_grid (dort inline
    als pay_p/pay_c), hier als eigenstaendige Funktion, damit sie fuer Lead
    UND Follower identisch wiederverwendet werden kann.

    Rueckgabe: (pay_p, pay_c), je Shape (n_rows, n_days)."""
    Kp = pd.to_numeric(table["strike_put"], errors="coerce").to_numpy(float)
    Kc = pd.to_numeric(table["strike_call"], errors="coerce").to_numpy(float)
    pay_p = np.maximum(Kp[:, None] - S_d_hyp[None, :], 0.0)
    pay_c = np.maximum(S_d_hyp[None, :] - Kc[:, None], 0.0)
    return pay_p, pay_c


def _vertical_spread(K_inner: float, K_wing: float,
                      bid_inner: float, ask_inner: float, bid_wing: float, ask_wing: float,
                      pay_inner: np.ndarray, pay_wing: np.ndarray,
                      is_short: bool) -> tuple[float, float, np.ndarray] | None:
    """Wirtschaftliche Kennzahlen EINES Vertical-Spreads (Put- ODER
    Call-Seite, EIN Instrument). K_inner/pay_inner = die Zeile mit dem
    KLEINEREN Abstand vom Spot (also naeher dran), K_wing/pay_wing = die
    Zeile mit dem GROESSEREN Abstand - unabhaengig davon, ob der Strike bei
    Puts dadurch kleiner oder bei Calls groesser wird (siehe Aufrufer:
    K_inner/K_wing werden dort per Zeilen-INDEX, nicht per Strike-Betrag
    zugewiesen).

    is_short=True (LEAD-Rolle, 'teure' Seite): SHORT der innere Strike,
    LONG der Wing -> Credit-Spread. Das ist die minimal noetige Struktur
    (Anforderung: 'mindestens Credit-Spread muss nutzbar sein') - ein
    vollstaendiger Iron Condor entsteht erst spaeter durch Kombination von
    Put- UND Call-Vertical auf derselben Lead-Seite (siehe
    evaluate_pair_grid()).

    is_short=False (FOLLOWER-Rolle, 'billige'/Hedge-Seite): LONG der innere
    Strike, SHORT der Wing -> Debit-Spread, das Spiegelbild ('Reverse').
    Verdient bei grosser Bewegung des Follower-Instruments, die Kosten
    werden durch den verkauften Wing reduziert.

    Rueckgabe (net_cashflow, max_loss, pay_diff) oder None bei ungueltiger
    Konfiguration (Breite <=0, oder - je Rolle - kein echter Kredit bzw.
    kein sinnvoller, kleiner-als-die-Breite Debit):
      net_cashflow: positiv = Kredit (Lead), negativ = Debit (Follower)
      max_loss: immer positiv
      pay_diff: Array (n_days,) = pay_inner - pay_wing, OHNE Ein-/Ausstiegskosten."""
    width = abs(K_inner - K_wing)
    if not np.isfinite(width) or width <= 0.0:
        return None

    pay_diff = pay_inner - pay_wing

    if is_short:
        net_cf = bid_inner - ask_wing
        if not (np.isfinite(net_cf) and 0.0 < net_cf < width):
            return None
        max_loss = width - net_cf
    else:
        net_cf = -(ask_inner - bid_wing)
        debit = -net_cf
        if not (np.isfinite(debit) and 0.0 < debit < width):
            return None
        max_loss = debit

    return net_cf, max_loss, pay_diff


def _lead_short_validity(bid_p: np.ndarray, bid_c: np.ndarray, pay_p: np.ndarray, pay_c: np.ndarray,
                          w: np.ndarray, dates: np.ndarray, hold: int, train_mask: np.ndarray,
                          t_min: float) -> np.ndarray:
    """Fuer JEDE Zeile der Lead-Distanz-Tabelle: ist der NACKTE Short-
    Strangle an genau diesem Abstand (nur Short Put+Call, KEINE Fluegel)
    selbst robust profitabel (mean>0, Train>0, Val>0, t>t_min)?

    Identische Definition wie das 'short_ok' in
    condor_search.evaluate_condor_grid() - dort inline berechnet, nicht als
    eigenstaendige Funktion exportiert, deshalb hier bewusst dupliziert
    statt condor_search.py zu aendern. Grundlage fuer das Plateau-Kriterium
    der Lead-Seite (siehe PairTradeConfig.require_lead_plateau)."""
    n = len(bid_p)
    ok = np.zeros(n, dtype=bool)
    W_train, W_val = w[train_mask].sum(), w[~train_mask].sum()
    for i in range(n):
        credit_s = bid_p[i] + bid_c[i]
        if not (np.isfinite(credit_s) and credit_s > 0.0):
            continue
        x = credit_s - (pay_p[i] + pay_c[i])
        mean_s = float(w @ x / w.sum())
        se_s, _ = _hac_se_weighted_mean(dates, w, x, mean_s, hold + 1)
        t_s = mean_s / se_s if se_s > 0.0 else np.nan
        e_tr = float(w[train_mask] @ x[train_mask] / W_train) if W_train > 0.0 else np.nan
        e_va = float(w[~train_mask] @ x[~train_mask] / W_val) if W_val > 0.0 else np.nan
        ok[i] = bool(
            mean_s > 0.0 and np.isfinite(e_tr) and e_tr > 0.0
            and np.isfinite(e_va) and e_va > 0.0 and np.isfinite(t_s) and t_s > t_min
        )
    return ok


def _plateau_ok(ok: np.ndarray, i: int) -> bool:
    """Nachbar-Zeilen (i-1 UND i+1) muessen ebenfalls 'ok' sein - identisches
    Plateau-Prinzip wie in condor_search.evaluate_condor_grid(), hier nur
    auf die Lead-Seite angewendet. Fehlender Nachbar (Rand der Tabelle) =
    nicht erfuellt."""
    n = len(ok)
    return bool(i - 1 >= 0 and ok[i - 1] and i + 1 < n and ok[i + 1])


def evaluate_pair_grid(lead_table: pd.DataFrame, follower_table: pd.DataFrame,
                        pair_moves: pd.DataFrame,
                        lead_spot_today: float, follower_spot_today: float,
                        cfg: PairTradeConfig | None = None) -> pd.DataFrame:
    """Durchsucht das VOLLE kartesische Produkt aus Lead-Strike-Paaren
    (i<j auf lead_table) x Follower-Strike-Paaren (i'<j' auf
    follower_table) und bewertet DREI moegliche Strukturen je Kombination:
      'put_spread'  - NUR die Put-Seite (Lead Short Put Spread / Follower
                       Long Put Spread) - die MINIMALE nutzbare Struktur.
      'call_spread' - NUR die Call-Seite, analog.
      'iron_condor' - Put- UND Call-Seite zusammen (Lead = klassischer Iron
                       Condor, Follower = 'Reverse Iron Condor').
    Alle drei werden zurueckgegeben (Spalte 'structure') - das Ranking
    (cfg.sort_by, absteigend nach 'passes' dann nach sort_by) entscheidet,
    welche Struktur+Strikes am Ende vorne liegt. Es wird also NICHT vorher
    auf eine Struktur festgelegt.

    SIMULATIONSTECHNIK (wichtig, siehe Klarstellung vor dieser Implementierung):
    wie bei condor_search.py und ANDERS als beim reinen Analog-Tage-Matching
    (analog_matching.evaluate_analog_set, das Strikes PRO Analog-Tag neu
    ueber ein Ziel-Delta herleitet) werden hier die HEUTIGEN, ECHTEN Strikes/
    Bid-Ask-Preise aus lead_table/follower_table fest verwendet - NUR die
    prozentuale Bewegung der Analog-Tage (siehe pair_analog_moves()) wird
    auf den heutigen Spot projiziert. Es gibt HIER keinerlei Delta-Solve/
    Black-76-Neubewertung je Analog-Tag.

    Kombinierter Max-Loss = Lead-Max-Loss + Follower-Max-Loss (konservative
    ADDITIVE Schranke - der wahre gemeinsame Worst-Case waere ein Optimum
    ueber den gemeinsamen (S_lead, S_follower)-Raum unter Beruecksichtigung
    der Korrelation, was hier NICHT berechnet wird. Die additive Schranke
    ist pessimistischer als der reale korrelierte Worst-Case, also
    konservativ - eine bewusste Modellentscheidung, kein Automatismus aus
    condor_search.py, wo es dieses Problem bei nur einem Instrument nicht gibt).

    PERFORMANCE-WARNUNG: die Kosten skalieren mit
    (n_lead_pairs * n_follower_pairs), n_pairs = n_rows*(n_rows-1)/2 JE
    Tabelle. Bei den ueblichen ~24 Distanz-Zeilen aus dashboard_livechain.py
    ergäbe das ~276 Paare je Seite, also ~76.000 Lead-x-Follower-
    Kombinationen x 3 Strukturen = weit ueber 200.000 HAC-Berechnungen JE
    getesteter Laufzeit-Paarung - nicht praktikabel. cfg.max_grid_cells
    bricht deshalb hart ab (ValueError) statt minutenlang zu rechnen. Im
    Dashboard sollte das Distanz-Raster fuer diesen Modus DEUTLICH groeber
    sein als bei der Einzel-Instrument-Suche (z.B. 8-12 statt 24 Zeilen je
    Tabelle - das ergibt bereits ~28-66 Paare je Seite, also ~800-4.400
    Kombinationen, praktikabel)."""
    if cfg is None:
        cfg = PairTradeConfig()
    if pair_moves.empty:
        return pd.DataFrame()

    lead_table = lead_table.sort_values("distance_pct").reset_index(drop=True)
    follower_table = follower_table.sort_values("distance_pct").reset_index(drop=True)
    n_lead, n_follower = len(lead_table), len(follower_table)

    n_lead_pairs = n_lead * (n_lead - 1) // 2
    n_follower_pairs = n_follower * (n_follower - 1) // 2
    n_cells = n_lead_pairs * n_follower_pairs
    if n_cells > cfg.max_grid_cells:
        raise ValueError(
            f"Grid zu gross: {n_lead_pairs} Lead-Paare x {n_follower_pairs} "
            f"Follower-Paare = {n_cells:,} Kombinationen (Grenze: "
            f"{cfg.max_grid_cells:,}). Distanz-Raster fuer Lead und/oder "
            "Follower verkleinern (weniger Ziel-Abstaende in der Kette-Tabelle)."
        )

    w = pair_moves["weight"].to_numpy(float)
    dates = pair_moves["entry_date"].to_numpy()
    hold = int(pair_moves["hold_days"].max())
    train_mask = (pair_moves["entry_date"] < pair_moves["entry_date"].median()).to_numpy()

    S_lead_hyp = lead_spot_today * (1.0 + pair_moves["lead_ret"].to_numpy(float))
    S_follower_hyp = follower_spot_today * (1.0 + pair_moves["follower_ret"].to_numpy(float))

    pay_p_lead, pay_c_lead = _row_payoffs(lead_table, S_lead_hyp)
    pay_p_foll, pay_c_foll = _row_payoffs(follower_table, S_follower_hyp)

    lead_bid_p, lead_ask_p, lead_bid_c, lead_ask_c = _price_columns(lead_table, cfg.price_basis)
    lead_Kp = pd.to_numeric(lead_table["strike_put"], errors="coerce").to_numpy(float)
    lead_Kc = pd.to_numeric(lead_table["strike_call"], errors="coerce").to_numpy(float)

    foll_bid_p, foll_ask_p, foll_bid_c, foll_ask_c = _price_columns(follower_table, cfg.price_basis)
    foll_Kp = pd.to_numeric(follower_table["strike_put"], errors="coerce").to_numpy(float)
    foll_Kc = pd.to_numeric(follower_table["strike_call"], errors="coerce").to_numpy(float)

    lead_ok = _lead_short_validity(lead_bid_p, lead_bid_c, pay_p_lead, pay_c_lead, w, dates, hold, train_mask, cfg.t_min)

    def _lead_pairs():
        # i = innere/nahe Zeile (wird geshortet), j = Fluegel-Zeile (long) -
        # bei BEIDEN Seiten (Put UND Call) ist i>0 immer die Zeile mit dem
        # KLEINEREN distance_pct, unabhaengig vom Strike-Vorzeichen (siehe
        # _vertical_spread-Docstring).
        for i in range(n_lead - 1):
            for j in range(i + 1, n_lead):
                put = _vertical_spread(lead_Kp[i], lead_Kp[j], lead_bid_p[i], lead_ask_p[i],
                                       lead_bid_p[j], lead_ask_p[j], pay_p_lead[i], pay_p_lead[j], is_short=True)
                call = _vertical_spread(lead_Kc[i], lead_Kc[j], lead_bid_c[i], lead_ask_c[i],
                                        lead_bid_c[j], lead_ask_c[j], pay_c_lead[i], pay_c_lead[j], is_short=True)
                yield i, j, put, call

    def _follower_pairs():
        for i in range(n_follower - 1):
            for j in range(i + 1, n_follower):
                put = _vertical_spread(foll_Kp[i], foll_Kp[j], foll_bid_p[i], foll_ask_p[i],
                                       foll_bid_p[j], foll_ask_p[j], pay_p_foll[i], pay_p_foll[j], is_short=False)
                call = _vertical_spread(foll_Kc[i], foll_Kc[j], foll_bid_c[i], foll_ask_c[i],
                                        foll_bid_c[j], foll_ask_c[j], pay_c_foll[i], pay_c_foll[j], is_short=False)
                yield i, j, put, call

    follower_cache = list(_follower_pairs())

    rows = []
    for li, lj, lput, lcall in _lead_pairs():
        lead_plateau_ok = _plateau_ok(lead_ok, li)
        if cfg.require_lead_plateau and not lead_plateau_ok:
            continue
        for fi, fj, fput, fcall in follower_cache:
            structures = []
            if lput is not None and fput is not None:
                structures.append(("put_spread", lput, fput))
            if lcall is not None and fcall is not None:
                structures.append(("call_spread", lcall, fcall))
            if lput is not None and lcall is not None and fput is not None and fcall is not None:
                structures.append((
                    "iron_condor",
                    (lput[0] + lcall[0], lput[1] + lcall[1], lput[2] + lcall[2]),
                    (fput[0] + fcall[0], fput[1] + fcall[1], fput[2] + fcall[2]),
                ))

            for structure, lead_leg, foll_leg in structures:
                l_net, l_max, l_pay = lead_leg
                f_net, f_max, f_pay = foll_leg
                pnl_d = (l_net - l_pay) + (f_net + f_pay)
                combined_max_loss = l_max + f_max
                if combined_max_loss <= 0.0:
                    continue

                stats = weighted_pnl_stats(pnl_d, w, dates, hold, train_mask, combined_max_loss, cfg.tail_quantile)
                flags = criteria_flags(stats, cfg)

                rows.append({
                    "structure": structure,
                    "lead_i": li, "lead_j": lj, "follower_i": fi, "follower_j": fj,
                    "lead_short_distance": float(lead_table["distance_pct"].iloc[li]),
                    "lead_wing_distance": float(lead_table["distance_pct"].iloc[lj]),
                    "follower_short_distance": float(follower_table["distance_pct"].iloc[fi]),
                    "follower_wing_distance": float(follower_table["distance_pct"].iloc[fj]),
                    "net_cashflow": l_net + f_net, "max_loss": combined_max_loss,
                    "lead_net_cashflow": l_net, "follower_net_cashflow": f_net,
                    **stats, **flags,
                    "c_lead_plateau": lead_plateau_ok,
                })

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result["passes"] = result["c_base"] & result["c_lead_plateau"]
    if cfg.sort_by not in ("ror", "mean", "tstat"):
        raise ValueError(f"sort_by muss 'ror', 'mean' oder 'tstat' sein (erhalten: {cfg.sort_by!r}).")
    return result.sort_values(["passes", cfg.sort_by], ascending=[False, False]).reset_index(drop=True)