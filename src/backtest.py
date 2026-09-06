"""
Iron-Condor-/Multi-Leg-Backtest (taegliche ueberlappende Positionen /
"Ladder"-Modus) - Portierung der entsprechenden Pine-Script-Logik, aber
generalisiert: jedes der 4 Beine ist unabhaengig konfigurierbar (Put/Call,
Long/Short, Delta, Menge) statt fix Put/Put/Call/Call wie im urspruenglichen
Iron-Condor-Default.

Bewusst NICHT uebernommen (siehe auch black76.py):
- Futures-Terminstruktur / Backwardation-Contango-Regime-Umschaltung.
- Symbol-Mapping fuer verschiedene Futures-Maerkte: wir handeln immer die
  regulaeren SPX-Kassaindex.
- Trend-Filter (200-Tage-Linie): weggelassen, damit wirklich jeden Tag
  dieselbe Struktur eroeffnet wird (kein bedingtes Gating).

IV-Quelle ist ausschliesslich die historische Volatilitaet (HV) - kein VIX,
weil das Ziel eine lange, konsistente Trainingshistorie fuers ML-Modell ist.

KORREKTUREN GEGENUEBER DER VORHERIGEN VERSION (siehe Kommentare an den
jeweiligen Stellen unten fuer Details):
  1) Strike-Auswahl (strike_from_delta) und Praemien-Bewertung (leg_price)
     verwendeten bisher UNTERSCHIEDLICHE Vola (flach vs. geskewt) - dadurch
     traf der gewaehlte Strike nicht das eingestellte Ziel-Delta. Jetzt loest
     `solve_strike_delta_consistent()` Strike und Skew-Vola simultan per
     Fixpunkt-Iteration.
  2) Beim Mark-to-Market offener Positionen (Stop-Loss/Early-Exit) konnte
     `current_iv` NaN sein (z.B. Rang-Modus mit noch nicht vollem Lookback-
     Fenster) und wurde ungeprueft an leg_price() durchgereicht -> stiller
     NaN-P&L ohne Fehlermeldung. Jetzt gibt es einen expliziten Fallback auf
     die Entry-IV der Position, inkl. Kennzeichnung in der Trade-Ausgabe
     (Spalte 'iv_fallback_used').
  3) Bewertung nutzte den KASSAKURS S direkt als Terminkurs F in Black-76
     (entspricht implizit r=q=0). Jetzt wird ueber `forward_price()` explizit
     mit einer Dividendenrendite (SPX) umgerechnet - sowohl beim Entry als
     auch beim Mark-to-Market vor Verfall. Am Verfall selbst bleibt es beim
     Kassakurs S, da Index-Optionen cash-gegen-Index (nicht gegen Future)
     abgerechnet werden.
  4) NEU - Fixer Strike je Bein: bisher war ein Bein immer ueber ein
     Ziel-Delta definiert (Strike wurde daraus taeglich neu bestimmt). Jetzt
     kann pro Bein optional stattdessen ein FESTER Strike-Preis vorgegeben
     werden (LegConfig.strike_mode == "fixed"). In dem Fall entfaellt die
     Delta->Strike-Umrechnung komplett; der angegebene Strike wird an jedem
     Entry-Tag unveraendert verwendet (nur die Praemie wird taeglich neu,
     mit der jeweils aktuellen IV, bewertet).
"""

from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd

from black76 import (
    VolSurfaceConfig, leg_price, leg_intrinsic, strike_from_delta, skewed_iv,
    forward_price,
)


@dataclass
class LegConfig:
    """Ein einzelnes Options-Bein: frei waehlbar Put/Call, Long/Short.

    strike_mode == "delta" (Default): Strike wird taeglich neu aus `delta`
    hergeleitet (siehe solve_strike_delta_consistent).
    strike_mode == "fixed": `fixed_strike` wird direkt als Strike verwendet,
    `delta` wird in diesem Fall ignoriert."""
    label: str
    option_type: str   # "put" oder "call"
    side: str          # "long" oder "short"
    delta: float
    qty: float = 1.0
    strike_mode: str = "delta"       # "delta" oder "fixed"
    fixed_strike: float | None = None

    @property
    def is_put(self) -> bool:
        return self.option_type == "put"

    @property
    def is_long(self) -> bool:
        return self.side == "long"

    @property
    def uses_fixed_strike(self) -> bool:
        return self.strike_mode == "fixed" and self.fixed_strike is not None and self.fixed_strike > 0.0


def default_iron_condor_legs() -> list[LegConfig]:
    """Klassischer symmetrischer Iron Condor - identisch zu den bisherigen
    Pine-Script-Defaults (Long Put Wing / Short Put / Short Call / Long Call
    Wing, je 1 Kontrakt, 5/16/16/5 Delta)."""
    return [
        LegConfig("Bein 1", "put", "long", 0.05),
        LegConfig("Bein 2", "put", "short", 0.16),
        LegConfig("Bein 3", "call", "short", 0.16),
        LegConfig("Bein 4", "call", "long", 0.05),
    ]


@dataclass
class FixedRankConfig:
    """Backtest-Modus 'Fixer IV-Rang' (hypothetisch) - Portierung aus dem
    Pine-Skript. Beantwortet: 'Wie haette JEDER Trade performt, waere er
    IMMER zum selben (heutigen) IV-Rang eroeffnet/bewertet worden?'

    An jedem Bar wird zurueckgerechnet, welche Vola noetig gewesen waere, um
    GENAU target_rank_pct innerhalb des zu diesem Zeitpunkt gueltigen
    rollierenden HV-Min/Max-Fensters (laenge lookback_days, HV-Fenster
    hv_length) zu erreichen. Diese Vola ersetzt current_iv komplett -
    wirkt also auf Strike-Auswahl UND Black-76-Bewertung gleichermassen."""
    enabled: bool = False
    target_rank_pct: float = 50.0
    hv_length: int = 30
    lookback_days: int = 252


@dataclass
class IronCondorConfig:
    # Optionsparameter
    T_days: int = 7
    hv_length: int = 30
    r_free: float = 0.0

    # Dividendenrendite fuer die Spot->Forward-Umrechnung (siehe black76.py,
    # forward_price()). ~1.3-1.5% ist eine grobe, ueber die Zeit gemittelte
    # Annaeherung an die historische SPX-Dividendenrendite; sie schwankt real
    # spuerbar (z.B. deutlich hoeher in den 1970ern/80ern, niedriger in
    # Boomphasen). Fuer kurze Laufzeiten (wenige Wochen) ist der Effekt auf
    # den Optionspreis klein, aber systematisch (Puts etwas teurer, Calls
    # etwas guenstiger als bei einer Spot=Forward-Annahme).
    dividend_yield: float = 0.015

    legs: list[LegConfig] = field(default_factory=default_iron_condor_legs)

    # Vorzeitiger Ausstieg / Stop-Loss (bezieht sich auf beruehrte SHORT-Beine)
    close_on_touch: bool = False
    use_early_exit: bool = False
    early_exit_days: int = 30

    vol_surface: VolSurfaceConfig = field(default_factory=VolSurfaceConfig)
    fixed_rank: FixedRankConfig = field(default_factory=FixedRankConfig)


@dataclass
class _PositionLeg:
    label: str
    K: float
    entry_premium: float   # unsigniertere Optionspreis (immer positiv)
    is_put: bool
    is_long: bool
    qty: float


@dataclass
class _Position:
    entry_date: pd.Timestamp
    legs: list[_PositionLeg]
    entry_S: float
    entry_iv: float
    touched: bool = False


def compute_hv(df: pd.DataFrame, length: int) -> pd.Series:
    """Annualisierte historische Volatilitaet (Close-to-Close), wie im
    Pine-Script. Erwartet ein DataFrame mit 'close', sortiert nach 'date'."""
    log_ret = np.log(df["close"] / df["close"].shift(1))
    hv_daily = log_ret.rolling(length).std()
    return hv_daily * np.sqrt(252)


def compute_rank_target_vol(df: pd.DataFrame, hv_length: int, lookback_days: int,
                             target_rank_pct: float) -> pd.Series:
    """Rekonstruiert, an JEDEM Bar, welche Vola noetig gewesen waere, um
    exakt target_rank_pct innerhalb des zu diesem Zeitpunkt gueltigen
    rollierenden HV-Min/Max-Fensters zu erreichen (kausal, kein Lookahead -
    das rollierende Fenster verwendet nur vergangene/aktuelle Bars)."""
    hv_r = compute_hv(df, hv_length)
    hv_min = hv_r.rolling(lookback_days, min_periods=lookback_days).min()
    hv_max = hv_r.rolling(lookback_days, min_periods=lookback_days).max()
    target_vol = hv_min + (target_rank_pct / 100.0) * (hv_max - hv_min)
    return target_vol.where(hv_max > hv_min)


def solve_strike_delta_consistent(F: float, target_delta: float, T_y: float, base_iv: float,
                                   is_put: bool, cfg: VolSurfaceConfig,
                                   max_iter: int = 6, tol: float = 1e-6) -> tuple[float, float]:
    """Loest Strike UND Bewertungs-Vola SIMULTAN (Fixpunkt-Iteration).

    Vorher wurde der Strike mit der FLACHEN Basis-Vola (HV) bestimmt
    (strike_from_delta(..., current_iv, ...)), die Praemie aber mit der
    GESKEWTEN Vola an genau diesem Strike bewertet (leg_price(...) ruft
    intern skewed_iv() auf). Das ist inkonsistent: ein 16-Delta-Put im
    Skew-Modell hat unter der geskewten (hoeheren) Put-Vola ein ANDERES
    Delta als unter der flachen HV - der real resultierende Strike lag
    dadurch systematisch naeher am Geld als das eingestellte Ziel-Delta
    suggerierte, was die eingesammelten Praemien zu hoch ausweist.

    Diese Funktion iteriert stattdessen: Strike mit aktueller Vola-Schaetzung
    bestimmen -> Skew-Vola an diesem Strike neu berechnen -> Strike erneut
    bestimmen usw., bis der Strike sich nicht mehr aendert (typischerweise
    <5 Iterationen, da Skew-Effekt auf realistischen SPX-Parametern klein
    gegenueber der Grobstruktur ist). Das Ergebnis ist ein Strike, dessen
    Delta unter der TATSAECHLICH verwendeten (geskewten) Vola korrekt dem
    Ziel-Delta entspricht."""
    sigma = base_iv
    K = strike_from_delta(F, target_delta, sigma, T_y, is_put)
    for _ in range(max_iter):
        sigma_new = skewed_iv(K, F, T_y, base_iv, cfg)
        K_new = strike_from_delta(F, target_delta, sigma_new, T_y, is_put)
        if abs(K_new - K) < tol:
            K, sigma = K_new, sigma_new
            break
        K, sigma = K_new, sigma_new
    return K, sigma


def _leg_expiry_pnl(K: float, S: float, is_put: bool, is_long: bool, qty: float, entry_premium: float) -> float:
    intrinsic = leg_intrinsic(K, S, is_put)
    if is_long:
        return (intrinsic - entry_premium) * qty
    return (entry_premium - intrinsic) * qty


def compute_max_risk(legs: list[_PositionLeg], entry_net_cashflow: float) -> float:
    """
    Worst-case Verlust der Struktur bis zum Verfall (positiver Wert), oder
    math.inf falls die Struktur unbegrenztes Verlustrisiko hat (mehr Short-
    Calls als Long-Calls -> Verlust waechst unbegrenzt mit steigendem Kurs).

    Da der Auszahlungsverlauf einer Put/Call-Kombination stueckweise linear
    in S ist, liegt das Minimum (= worst case) entweder bei S=0, bei einem
    der Strikes, oder (falls das Netto-Call-Exposure netto short ist) im
    Unendlichen.
    """
    slope_inf = sum((leg.qty if (not leg.is_put and leg.is_long) else 0.0) for leg in legs) - \
                sum((leg.qty if (not leg.is_put and not leg.is_long) else 0.0) for leg in legs)
    if slope_inf < 0.0:
        return math.inf

    candidate_S = [0.0] + [leg.K for leg in legs]
    worst = min(
        sum(_leg_expiry_pnl(leg.K, S, leg.is_put, leg.is_long, leg.qty, leg.entry_premium) for leg in legs)
        for S in candidate_S
    )
    return max(-worst, 0.0)


def run_daily_ladder(df: pd.DataFrame, cfg: IronCondorConfig) -> pd.DataFrame:
    """
    Fuehrt den Ladder-Backtest aus.

    df: muss Spalten 'date','open','high','low','close' enthalten, echte
        Tagesdaten (siehe data_fetch.py - Spalte 'resolution'=='daily'; wird
        hier zur Sicherheit nochmal gefiltert, falls vorhanden).
    cfg: IronCondorConfig mit cfg.legs (Liste von LegConfig, je Put/Call,
        Long/Short, Delta oder fixer Strike, Menge frei waehlbar).

    Rueckgabe: DataFrame, eine Zeile je abgeschlossenem Trade.
    """
    if "resolution" in df.columns:
        df = df[df["resolution"] == "daily"]
    df = df.sort_values("date").reset_index(drop=True)

    hv = compute_hv(df, cfg.hv_length)
    if cfg.fixed_rank.enabled:
        iv_series = compute_rank_target_vol(
            df, cfg.fixed_rank.hv_length, cfg.fixed_rank.lookback_days, cfg.fixed_rank.target_rank_pct,
        )
    else:
        iv_series = hv

    T_years = cfg.T_days / 365.0
    vc = cfg.vol_surface

    open_positions: list[_Position] = []
    closed_trades = []

    dates = df["date"].to_numpy()
    closes = df["close"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    iv_arr = iv_series.to_numpy()

    n = len(df)
    for i in range(n):
        today = pd.Timestamp(dates[i])
        S = float(closes[i])
        high = float(highs[i])
        low = float(lows[i])
        current_iv = iv_arr[i]

        # ── neue Position eroeffnen ──
        if not np.isnan(current_iv) and current_iv > 0.0:
            # KORREKTUR (Spot vs. Forward): Black-76 erwartet den Terminkurs F,
            # nicht den Kassakurs S. F liegt bei positiver Dividendenrendite
            # unter S - siehe forward_price()/black76.py.
            F_entry = forward_price(S, cfg.r_free, cfg.dividend_yield, T_years)

            pos_legs = []
            for leg_cfg in cfg.legs:
                if leg_cfg.uses_fixed_strike:
                    # NEU (fixer Strike): kein Delta->Strike-Solve noetig -
                    # der vorgegebene Strike wird direkt uebernommen. Die
                    # Praemie wird trotzdem taeglich neu bewertet (mit der
                    # jeweils aktuellen IV inkl. Skew an genau diesem Strike).
                    K = leg_cfg.fixed_strike
                else:
                    # KORREKTUR (Delta/Skew-Konsistenz): Strike und
                    # Bewertungs-Vola werden simultan geloest statt Strike
                    # mit flacher und Praemie mit geskewter Vola zu berechnen
                    # (siehe Docstring von solve_strike_delta_consistent).
                    K, _sigma_entry = solve_strike_delta_consistent(
                        F_entry, leg_cfg.delta, T_years, current_iv, leg_cfg.is_put, vc,
                    )
                premium = leg_price(K, leg_cfg.is_put, T_years, F_entry, current_iv, cfg.r_free, vc) * leg_cfg.qty
                pos_legs.append(_PositionLeg(
                    label=leg_cfg.label, K=K, entry_premium=premium,
                    is_put=leg_cfg.is_put, is_long=leg_cfg.is_long, qty=leg_cfg.qty,
                ))
            open_positions.append(_Position(entry_date=today, legs=pos_legs, entry_S=S, entry_iv=current_iv))

        # ── alle offenen Positionen pruefen / ggf. schliessen ──
        # HINWEIS (identisch zum Pine-Original): Geprueft wird nur an echten
        # Handelstagen (Zeilen in df), 'days_since' ist aber Kalendertage.
        # Nach einem Wochenende/Feiertag kann days_since dadurch an einem
        # einzigen Bar ueber T_days hinausspringen (z.B. Freitag Tag 8 ->
        # Montag Tag 11). Faellt in diesem Bar zusaetzlich ein Touch-Ereignis
        # zusammen, gewinnt der Stop-Loss-Grund (siehe Prioritaet unten) auch
        # wenn T_days dabei leicht ueberschritten wird - das ist keine
        # Abweichung vom Original, sondern folgt exakt derselben taeglichen
        # Bar-fuer-Bar-Pruefreihenfolge.
        still_open = []
        for pos in open_positions:
            # Touch bezieht sich nur auf SHORT-Beine (das ist das offene Risiko)
            touched_now = any(
                (low <= leg.K if leg.is_put else high >= leg.K)
                for leg in pos.legs if not leg.is_long
            )
            if touched_now:
                pos.touched = True

            days_since = (today - pos.entry_date).days
            T_elapsed_years = days_since / 365.0
            T_rem = max(T_years - T_elapsed_years, 0.0)

            stop_close = cfg.close_on_touch and touched_now
            early_close = (not stop_close) and cfg.use_early_exit and days_since >= cfg.early_exit_days
            expiry_close = (not stop_close) and (not early_close) and days_since >= cfg.T_days

            if not (stop_close or early_close or expiry_close):
                still_open.append(pos)
                continue

            # KORREKTUR (NaN-Vola-Guard): Bei Stop-Loss/Early-Exit VOR Verfall
            # muss current_iv gueltig sein, um die Restlaufzeit zu bewerten
            # (leg_price braucht eine Vola). Im Rang-Modus kann current_iv an
            # manchen Bars NaN sein (Lookback-Fenster noch nicht voll) - das
            # wurde bisher ungeprueft durchgereicht und erzeugte einen
            # stillen NaN-P&L. Jetzt: Fallback auf die Entry-IV der Position
            # (beste verfuegbare Naeherung), transparent markiert per
            # 'iv_fallback_used' in der Trade-Ausgabe. Am Verfall selbst wird
            # ohnehin nur ueber den inneren Wert (Kassakurs S) abgerechnet -
            # dort wird gar keine Vola benoetigt.
            iv_fallback_used = False
            net_pnl = 0.0

            if expiry_close or T_rem <= 0.0:
                for leg in pos.legs:
                    settle = leg_intrinsic(leg.K, S, leg.is_put) * leg.qty
                    net_pnl += (settle - leg.entry_premium) if leg.is_long else (leg.entry_premium - settle)
            else:
                iv_for_mark = current_iv
                if np.isnan(iv_for_mark) or iv_for_mark <= 0.0:
                    iv_for_mark = pos.entry_iv
                    iv_fallback_used = True

                # KORREKTUR (Spot vs. Forward): auch beim Mark-to-Market vor
                # Verfall mit dem Terminkurs zur Restlaufzeit T_rem bewerten,
                # nicht mit dem Kassakurs S.
                F_rem = forward_price(S, cfg.r_free, cfg.dividend_yield, T_rem)
                for leg in pos.legs:
                    settle = leg_price(leg.K, leg.is_put, T_rem, F_rem, iv_for_mark, cfg.r_free, vc) * leg.qty
                    net_pnl += (settle - leg.entry_premium) if leg.is_long else (leg.entry_premium - settle)

            exit_reason = "expiry" if expiry_close else ("stop_loss" if stop_close else "early_exit")

            entry_net = sum(leg.entry_premium for leg in pos.legs if not leg.is_long) - \
                        sum(leg.entry_premium for leg in pos.legs if leg.is_long)
            max_risk = compute_max_risk(pos.legs, entry_net)

            closed_trades.append({
                "entry_date": pos.entry_date, "exit_date": today, "exit_reason": exit_reason,
                "entry_S": pos.entry_S, "exit_S": S, "entry_iv": pos.entry_iv,
                **{f"K_{leg.label}": leg.K for leg in pos.legs},
                "entry_net": entry_net, "net_pnl": net_pnl,
                "max_risk": max_risk,
                "pnl_norm": (net_pnl / max_risk) if (max_risk not in (0.0, math.inf) and not math.isnan(max_risk)) else np.nan,
                "unlimited_risk": max_risk == math.inf,
                "touched": pos.touched,
                "hold_days": days_since,
                "iv_fallback_used": iv_fallback_used,
            })

        open_positions = still_open

    return pd.DataFrame(closed_trades)


def summarize_trades(trades: pd.DataFrame) -> dict:
    """Kompakte Kennzahlen analog zur Pine-Script-Ergebnistabelle."""
    if trades.empty:
        return {"n_trades": 0}

    n_total = len(trades)
    n_wins = int((trades["net_pnl"] >= 0).sum())
    gross_profit = trades.loc[trades["net_pnl"] >= 0, "net_pnl"].sum()
    gross_loss = -trades.loc[trades["net_pnl"] < 0, "net_pnl"].sum()

    n_fallback = int(trades["iv_fallback_used"].sum()) if "iv_fallback_used" in trades.columns else 0

    return {
        "n_trades": n_total,
        "win_rate_pct": n_wins / n_total * 100.0,
        "net_pnl_sum": trades["net_pnl"].sum(),
        "avg_pnl": trades["net_pnl"].mean(),
        "avg_pnl_norm_pct": trades["pnl_norm"].mean(skipna=True) * 100.0,
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else np.nan,
        "touch_rate_pct": trades["touched"].mean() * 100.0,
        "best_trade": trades["net_pnl"].max(),
        "worst_trade": trades["net_pnl"].min(),
        "n_unlimited_risk": int(trades["unlimited_risk"].sum()),
        "avg_entry_iv_pct": trades["entry_iv"].mean() * 100.0 if "entry_iv" in trades.columns else np.nan,
        "n_iv_fallback": n_fallback,
    }