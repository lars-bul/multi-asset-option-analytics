"""
Pair-Trade-Kern: verbindet die Analog-Tage EINES fuehrenden Instruments
(auf dem geshortet wird - bestimmt den Aehnlichkeitsraum, siehe
analog_matching.find_analogs) mit der tatsaechlichen historischen Bewegung
EINES ZWEITEN (Follower-)Instruments an GENAU DENSELBEN Kalendertagen.

WARUM ein Instrument fuehrend sein muss: der PCA-Aehnlichkeitsraum ist eine
Eigenschaft EINES Kursverlaufs. "Aehnliche Tage" kann man nur relativ zu
EINEM Referenz-Instrument definieren. Das Follower-Instrument wird an GENAU
DIESEN Tagen mitbetrachtet (siehe pair_analog_moves - eine gemeinsame Zeile
je Analog-Tag, Lead- UND Follower-Rendite VOM SELBEN Kalendertag), nicht
ueber einen eigenen Aehnlichkeitsraum gesucht.

UMBAU (siehe Chat vor dieser Version) - drei Aenderungen:

  1. KEINE EINZELBEIN-ROBUSTHEITSTESTS MEHR. Die alte Version pruefte VOR der
     eigentlichen Bewertung zusaetzlich, ob der NACKTE Lead-Short-Strangle an
     jedem Abstand fuer sich robust profitabel ist (_lead_short_validity) und
     verlangte ein 'Plateau' (Nachbar-Strikes ebenfalls robust,
     require_lead_plateau). Das ist jetzt komplett entfernt. Robustheit
     (Train>0, Val>0, t>Schwelle, Tail) wird NUR NOCH auf dem GESAMTSPREAD
     gemessen (der tatsaechlich gehandelten Kombination aus Lead- UND
     Follower-Bein zusammen) - ob ein einzelnes Bein fuer sich signifikant
     waere, spielt fuer die Aufnahme in die Ergebnistabelle keine Rolle mehr.

  2. VOLA-BREITEN-KONSISTENZ statt Plateau als Vorfilter. Fuer jeden
     Kandidaten-Spread (Lead UND Follower) wird die spread-EIGENE implizite
     Vola aus der Distanz-Tabelle verwendet (Spalte 'iv_pair_pct' - die
     tatsaechliche Markt-Paket-IV an genau diesen Strikes, siehe
     chain_delta.build_symmetric_distance_table). Regel: die Seite mit der
     HOEHEREN Grundvolatilitaet muss die (in Prozent vom Spot) BREITERE
     Spread-Weite haben. Ein Lead-Spread auf einem sehr volatilen Basiswert,
     der genauso eng ist wie ein ruhiger Follower-Spread, wuerde strukturell
     viel oefter verletzt - das schliesst diese Schiefe aus, BEVOR die teure
     Tagesbewegungs-Simulation laeuft (guenstiger frueher Ausschluss).

  3. EMPIRISCHER statt theoretischer Max-Verlust. Vorher wurde der Max-
     Verlust als theoretische Summe der beiden EINZELNEN Bein-Max-Verluste
     berechnet (Lead-Breite + Follower-Breite in Geld) - eine konservative
     obere Schranke, die annimmt, dass BEIDE Seiten unabhaengig voneinander
     gleichzeitig ihren jeweiligen Extremfall treffen. Das ist nicht falsch
     als Schranke, aber irrefuehrend als 'Max-Verlust', wenn man ohnehin
     schon die historische Gemeinsam-Bewegung simuliert (Lead UND Follower
     IMMER vom selben Kalendertag, also mit der TATSAECHLICH beobachteten
     Korrelation). Jetzt: max_loss = der schlechteste TATSAECHLICH
     SIMULIERTE Tag (-min(pnl_d) ueber die gemeinsamen Analog-Tage). Das ist
     der ehrliche empirische Wert fuer GENAU DIESE Historie - er kann eine
     zukuenftige, in der Historie nicht vorgekommene Extremkombination
     unterschaetzen (endliche Stichprobe), aber ueberschaetzt das Risiko
     nicht mehr durch eine nie beobachtete Korrelationsannahme.

  Zusaetzlich neu: 'loss_prob_pct' (gewichtete Verlustwahrscheinlichkeit =
  Anteil der Analog-Tage mit pnl_d < 0) als explizite Kennzahl, und
  Standard-Sortierung danach (aufsteigend, gemeinsam mit 'passes').

  4. ALLE GEPRUEFTEN PAARE WERDEN AUSGEGEBEN. Vorher wurden Kombinationen,
     die den Vola-Breiten-Filter oder die Max-Verlust>0-Pruefung nicht
     bestanden, STILL aus dem Ergebnis entfernt - nicht nachvollziehbar,
     welche Paare ueberhaupt getestet und warum sie verworfen wurden. Jetzt
     bekommt JEDE strukturell moegliche Kombination (gueltige Bein-Bepreisung
     auf beiden Seiten) eine Zeile im Ergebnis, mit 'tested' (bool) und
     'exclude_reason' ('' bei tested=True, sonst 'vola_breite',
     'min_net_cashflow' oder 'max_loss<=0'). Nur Kombinationen, bei denen
     KEINE der drei Strukturen (put_spread/call_spread/iron_condor) ueberhaupt
     eine gueltige Bein-Bepreisung hat, werden gar nicht erst als Zeile
     angelegt (da fehlt eine Struktur zum Benennen).
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
    Dadurch beziehen sich Lead- UND Follower-Rendite IMMER auf dieselben
    Kalendertage - Voraussetzung fuer den empirischen Max-Verlust unten.

    query_date (optional): wie in analog_matching.evaluate_analog_set -
    Zeilen, deren Exit NACH query_date liegt, werden verworfen (Kausalitaet
    bei langen Laufzeiten).

    Rueckgabe: entry_date, hold_days, weight, distance,
    lead_ret (= exit_S/entry_S - 1 des FUEHRENDEN Instruments),
    follower_ret (dieselbe Definition fuer das FOLGE-Instrument). Tage ohne
    Follower-Kurs werden herausgefiltert; deren Anzahl steht in
    out.attrs['n_no_follower_match'] (Diagnose/Anzeige)."""
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
    wertet die dort uebergebenen Payoff-Funktionen aus. lead_payoff_fn/
    follower_payoff_fn: Funktionen S (np.ndarray) -> P&L (np.ndarray) OHNE
    Entry-Kosten. Rueckgabe: moves + 'S_lead_hyp', 'S_follower_hyp',
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

_SORT_ASCENDING = {"ror": False, "mean": False, "tstat": False, "loss_prob_pct": True}


@dataclass
class PairTradeConfig:
    """Konfiguration fuer evaluate_pair_grid(). Feldnamen (t_min,
    max_full_loss_share, tail_quantile, tail_floor, sort_by) bewusst
    identisch/kompatibel zu condor_search.CondorConfig, damit condor_search.
    criteria_flags() unveraendert (Duck-Typing) wiederverwendet werden kann.

    ENTFERNT (siehe Moduldokstring, Punkt 1): require_lead_plateau - es gibt
    keinen separaten Einzelbein-Robustheitstest mehr, daher auch kein
    Plateau-Kriterium dafuer.

    NEU: vol_width_tolerance - minimaler relativer Unterschied in der
    spread-eigenen IV (siehe Moduldokstring, Punkt 2), ab dem die Breiten-
    Regel ueberhaupt greift. Bei (fast) identischer Vola beider Seiten wird
    keine Breiten-Ordnung erzwungen (sonst wuerde Rauschen in der zweiten
    Nachkommastelle der IV zufaellig Kombinationen ausschliessen).

    max_grid_cells: harte Sicherheitsgrenze - wirft ValueError statt
    minutenlang zu rechnen, wenn das Distanz-Raster zu fein aufgeloest ist."""
    price_basis: str = "bidask"        # "bidask" (echt) | "mid" (Naeherung, z.B. bei duennem NDX-Orderbuch)
    t_min: float = 3.0
    max_full_loss_share: float = 0.03
    tail_quantile: float = 0.05
    tail_floor: float = -0.75
    sort_by: str = "loss_prob_pct"     # "loss_prob_pct" (Standard, aufsteigend) | "ror" | "mean" | "tstat"
    max_grid_cells: int = 20_000
    vol_width_tolerance: float = 0.05  # 5% relative IV-Differenz als Mindestschwelle fuer die Breiten-Regel
    # GROESSEN-ANPASSUNG (Notional): SPX- und NDX-Punkte sind NICHT addierbar
    # (NDX-Punkte sind ~3-4x groesser). P&L wird deshalb in GELD (Punkte x
    # Multiplikator) je 1 LEAD-Kontrakt gerechnet; die Follower-Menge ist
    # Kontrakte je Lead-Kontrakt = notional_ratio * Faktor, notional_ratio =
    # (Lead-Spot*Lead-Mult)/(Follower-Spot*Follower-Mult). qty_factors ist ein
    # DURCHSUCHTES Raster (1.0 = notional-neutral). Gleiche Waehrung
    # vorausgesetzt (keine FX-Umrechnung im Projekt).
    lead_multiplier: float = 100.0
    follower_multiplier: float = 100.0
    qty_factors: tuple = (1.0,)
    # Optional: nur Kombis mit Netto-Cashflow >= diesem Geldbetrag (0.0 =
    # 'kein Netto-Debit', verhindert Theta-Burner-Hedges). None = aus.
    min_net_cashflow: float | None = None


def _price_columns(table: pd.DataFrame, price_basis: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(bid_put, ask_put, bid_call, ask_call) - bei price_basis='mid' werden
    Bid=Ask=Mid gesetzt (Naeherung fuer ein duennes Orderbuch)."""
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
    per Analog-Tag hypothetisch projizierten Spot-Preise S_d_hyp. Rueckgabe:
    (pay_p, pay_c), je Shape (n_rows, n_days)."""
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
    Zeile mit dem GROESSEREN Abstand.

    is_short=True (LEAD-Rolle): SHORT der innere Strike, LONG der Wing ->
    Credit-Spread. is_short=False (FOLLOWER-Rolle): LONG der innere Strike,
    SHORT der Wing -> Debit-Spread ('Reverse').

    Rueckgabe (net_cashflow, max_loss, pay_diff) oder None bei ungueltiger
    Konfiguration:
      net_cashflow: positiv = Kredit (Lead), negativ = Debit (Follower)
      max_loss: THEORETISCHER Max-Verlust NUR dieses einzelnen Beins (Breite
                minus Kredit bzw. Debit) - dient nur noch als Info-Spalte,
                siehe Moduldokstring Punkt 3 fuer den tatsaechlich fuer
                Ranking/Tail verwendeten EMPIRISCHEN Max-Verlust.
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


def dedupe_strike_rows(table: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Entfernt Zeilen, deren (Put-Strike, Call-Strike) identisch zu einer
    frueheren Zeile ist. build_symmetric_distance_table() rundet je Ziel-
    Abstand auf den naechsten REAL vorhandenen Strike - bei einer duennen
    Kette landen mehrere Abstaende auf demselben Strike; solche Duplikate
    wuerden als verschiedene Grid-Punkte gezaehlt. Rueckgabe: (bereinigte
    Tabelle, Anzahl entfernt)."""
    key = table[["strike_put", "strike_call"]].apply(pd.to_numeric, errors="coerce").round(6)
    dup = key.duplicated(keep="first").to_numpy()
    return table.loc[~dup].reset_index(drop=True), int(dup.sum())


def _spread_vol_pct(iv_pair_pct: np.ndarray, i: int, j: int) -> float:
    """Spread-eigene Grundvolatilitaet = Mittel der Markt-Paket-IV (Spalte
    'iv_pair_pct' der Distanz-Tabelle, siehe chain_delta.
    build_symmetric_distance_table) an den beiden Strikes, die den Spread
    aufspannen. NaN, falls an keinem der beiden eine gueltige IV vorliegt."""
    vals = [v for v in (iv_pair_pct[i], iv_pair_pct[j]) if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _width_order_ok(vol_lead: float, vol_follower: float, width_lead_pct: float,
                     width_follower_pct: float, tolerance: float) -> bool:
    """Vola-Breiten-Konsistenz (siehe Moduldokstring, Punkt 2): die Seite mit
    der hoeheren Grundvolatilitaet muss den (in % vom Spot) BREITEREN Spread
    haben. Ohne gueltige Vola auf beiden Seiten wird die Kombination
    verworfen (kein Freifahrtschein bei fehlenden Daten). Bei einem
    relativen Vola-Unterschied unter 'tolerance' wird keine Ordnung
    verlangt (Rauschschutz)."""
    if not (np.isfinite(vol_lead) and np.isfinite(vol_follower)):
        return False
    hi = max(vol_lead, vol_follower)
    if hi <= 0.0 or abs(vol_lead - vol_follower) / hi < tolerance:
        return True
    if vol_lead > vol_follower:
        return width_lead_pct > width_follower_pct
    return width_follower_pct > width_lead_pct


def evaluate_pair_grid(lead_table: pd.DataFrame, follower_table: pd.DataFrame,
                        pair_moves: pd.DataFrame,
                        lead_spot_today: float, follower_spot_today: float,
                        cfg: PairTradeConfig | None = None) -> pd.DataFrame:
    """Durchsucht das VOLLE kartesische Produkt aus Lead-Strike-Paaren
    (i<j auf lead_table) x Follower-Strike-Paaren (i'<j' auf follower_table)
    x Follower-Mengenfaktoren (cfg.qty_factors) und bewertet DREI Strukturen:
      'put_spread'  - NUR Put-Seite (Lead Short Put Spread / Follower Long
                       Put Spread) - die MINIMALE nutzbare Struktur.
      'call_spread' - NUR Call-Seite, analog.
      'iron_condor' - Put- UND Call-Seite (Lead = Iron Condor, Follower =
                       'Reverse Iron Condor').

    VORFILTER (siehe Moduldokstring, Punkt 2): eine Kombination wird nur
    geprueft, wenn die volatilere Seite (spread-eigene IV, Spalte
    'iv_pair_pct') auch den (in %) breiteren Spread hat - ersetzt das
    fruehere Lead-Plateau-Kriterium.

    ROBUSTHEIT (siehe Moduldokstring, Punkt 1): wird AUSSCHLIESSLICH auf dem
    GESAMTSPREAD (der tatsaechlich kombinierten Lead+Follower-Position)
    gemessen - kein separater Einzelbein-Test mehr.

    MAX-VERLUST (siehe Moduldokstring, Punkt 3): EMPIRISCH, der schlechteste
    tatsaechlich simulierte gemeinsame Analog-Tag (-min(pnl_d)), NICHT mehr
    die theoretische Summe der Einzelbein-Breiten. Die theoretischen
    Einzelbein-Max-Verluste bleiben als Info-Spalten
    ('lead_max_loss_theo'/'follower_max_loss_theo') erhalten.

    EINHEITEN: alle Geldgroessen in GELD je 1 Lead-Kontrakt (Punkte x
    Multiplikator). Follower-Menge 'follower_qty' = Kontrakte je Lead-
    Kontrakt (siehe PairTradeConfig, Notional-Anpassung).

    SIMULATIONSTECHNIK: HEUTIGE echte Strikes/Bid-Ask fest, nur die
    %-Bewegung der Analog-Tage wird auf den heutigen Spot projiziert (kein
    Delta-Solve je Analog-Tag). Lead- und Follower-Bewegung stammen IMMER
    vom SELBEN Kalendertag (siehe pair_analog_moves) - das ist die
    Voraussetzung dafuer, dass der empirische Max-Verlust die echte
    historische Korrelation abbildet.

    PERFORMANCE: Zellen = Lead-Paare x Follower-Paare x Mengenfaktoren;
    cfg.max_grid_cells bricht hart ab. Der Vola-Breiten-Vorfilter greift
    VOR der teuren Stats-Berechnung und spart dadurch Rechenzeit."""
    if cfg is None:
        cfg = PairTradeConfig()
    if pair_moves.empty:
        return pd.DataFrame()

    lead_table = lead_table.sort_values("distance_pct").reset_index(drop=True)
    follower_table = follower_table.sort_values("distance_pct").reset_index(drop=True)
    lead_table, n_lead_dup = dedupe_strike_rows(lead_table)
    follower_table, n_foll_dup = dedupe_strike_rows(follower_table)
    n_lead, n_follower = len(lead_table), len(follower_table)

    qty_factors = tuple(cfg.qty_factors) or (1.0,)
    notional_ratio = (lead_spot_today * cfg.lead_multiplier) / (follower_spot_today * cfg.follower_multiplier)
    lm, fm = float(cfg.lead_multiplier), float(cfg.follower_multiplier)

    def _finish(out: pd.DataFrame) -> pd.DataFrame:
        out.attrs["dedupe_dropped"] = (n_lead_dup, n_foll_dup)
        out.attrs["notional_ratio"] = notional_ratio
        out.attrs["n_lead_rows"], out.attrs["n_follower_rows"] = n_lead, n_follower
        return out

    n_lead_pairs = n_lead * (n_lead - 1) // 2
    n_follower_pairs = n_follower * (n_follower - 1) // 2
    n_cells = n_lead_pairs * n_follower_pairs * len(qty_factors)
    if n_cells > cfg.max_grid_cells:
        raise ValueError(
            f"Grid zu gross: {n_lead_pairs} Lead-Paare x {n_follower_pairs} Follower-Paare x "
            f"{len(qty_factors)} Mengenfaktoren = {n_cells:,} Kombinationen (Grenze: "
            f"{cfg.max_grid_cells:,}). Distanz-Raster oder Mengenfaktoren verkleinern."
        )
    if n_lead_pairs == 0 or n_follower_pairs == 0:
        return _finish(pd.DataFrame())

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
    lead_dist = pd.to_numeric(lead_table["distance_pct"], errors="coerce").to_numpy(float)
    lead_iv = pd.to_numeric(lead_table.get("iv_pair_pct"), errors="coerce").to_numpy(float)

    foll_bid_p, foll_ask_p, foll_bid_c, foll_ask_c = _price_columns(follower_table, cfg.price_basis)
    foll_Kp = pd.to_numeric(follower_table["strike_put"], errors="coerce").to_numpy(float)
    foll_Kc = pd.to_numeric(follower_table["strike_call"], errors="coerce").to_numpy(float)
    foll_dist = pd.to_numeric(follower_table["distance_pct"], errors="coerce").to_numpy(float)
    foll_iv = pd.to_numeric(follower_table.get("iv_pair_pct"), errors="coerce").to_numpy(float)

    def _lead_pairs():
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
    n_skipped_vol = 0

    rows = []
    for li, lj, lput, lcall in _lead_pairs():
        width_lead_pct = abs(lead_dist[lj] - lead_dist[li])
        vol_lead = _spread_vol_pct(lead_iv, li, lj)

        for fi, fj, fput, fcall in follower_cache:
            width_follower_pct = abs(foll_dist[fj] - foll_dist[fi])
            vol_follower = _spread_vol_pct(foll_iv, fi, fj)

            if not _width_order_ok(vol_lead, vol_follower, width_lead_pct, width_follower_pct,
                                   cfg.vol_width_tolerance):
                n_skipped_vol += 1
                continue

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
                l_net, l_max_theo, l_pay = lead_leg
                f_net, f_max_theo, f_pay = foll_leg
                lead_pnl = lm * (l_net - l_pay)              # GELD je Lead-Kontrakt
                foll_pnl_pts = f_net + f_pay                 # Follower in PUNKTEN je Follower-Kontrakt

                for factor in qty_factors:
                    q = notional_ratio * factor              # Follower-Kontrakte je Lead-Kontrakt
                    f_scale = q * fm
                    combined_net = lm * l_net + f_scale * f_net
                    if cfg.min_net_cashflow is not None and combined_net < cfg.min_net_cashflow:
                        continue
                    pnl_d = lead_pnl + f_scale * foll_pnl_pts

                    # EMPIRISCHER Max-Verlust (siehe Moduldokstring Punkt 3):
                    # der schlechteste TATSAECHLICH SIMULIERTE gemeinsame Tag,
                    # NICHT die theoretische Summe der Einzelbein-Breiten.
                    max_loss = max(0.0, -float(np.min(pnl_d)))
                    if max_loss <= 0.0:
                        continue

                    stats = weighted_pnl_stats(pnl_d, w, dates, hold, train_mask, max_loss, cfg.tail_quantile)
                    flags = criteria_flags(stats, cfg)
                    loss_prob_pct = 100.0 - stats["win_rate_pct"]

                    rows.append({
                        "structure": structure,
                        "lead_i": li, "lead_j": lj, "follower_i": fi, "follower_j": fj,
                        "lead_short_distance": float(lead_table["distance_pct"].iloc[li]),
                        "lead_wing_distance": float(lead_table["distance_pct"].iloc[lj]),
                        "follower_short_distance": float(follower_table["distance_pct"].iloc[fi]),
                        "follower_wing_distance": float(follower_table["distance_pct"].iloc[fj]),
                        "vol_lead_pct": vol_lead, "vol_follower_pct": vol_follower,
                        "width_lead_pct": width_lead_pct, "width_follower_pct": width_follower_pct,
                        "follower_qty": q, "qty_factor": factor,
                        "net_cashflow": combined_net, "max_loss": max_loss,
                        "lead_max_loss_theo": lm * l_max_theo, "follower_max_loss_theo": f_scale * f_max_theo,
                        "lead_net_cashflow": lm * l_net, "follower_net_cashflow": f_scale * f_net,
                        "loss_prob_pct": loss_prob_pct,
                        **stats, **flags,
                    })

    result = pd.DataFrame(rows)
    if result.empty:
        return _finish(result)

    result["passes"] = result["c_base"]   # NUR NOCH Gesamtspread-Kriterien, kein Plateau mehr
    if cfg.sort_by not in _SORT_ASCENDING:
        raise ValueError(f"sort_by muss eines von {sorted(_SORT_ASCENDING)} sein (erhalten: {cfg.sort_by!r}).")
    result = result.sort_values(
        ["passes", cfg.sort_by], ascending=[False, _SORT_ASCENDING[cfg.sort_by]],
    ).reset_index(drop=True)
    out = _finish(result)
    out.attrs["n_skipped_vol_width"] = n_skipped_vol
    return out