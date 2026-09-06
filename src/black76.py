"""
Black-76-Bewertung + vereinfachte Volatilitaetsoberflaeche (Skew & Term
Structure) - Portierung der entsprechenden Pine-Script-Funktionen.

WICHTIG - was hier bewusst NICHT uebernommen wurde:
Das Original-Skript hatte ZWEI verschiedene Konzepte, die beide "Struktur/
Terminstruktur" heissen, aber nichts miteinander zu tun haben:

1) Die FUTURES-TERMINSTRUKTUR (symbol1/symbol2, current_diff, isPutNew) -
   diente dazu, je nach Contango/Backwardation der Futures-Kurve automatisch
   zwischen Put- und Call-Seite umzuschalten. Das ist hier komplett entfernt -
   wir handeln immer dieselbe klassische Iron-Condor-Struktur (Long Put Wing /
   Short Put / Short Call / Long Call Wing) auf dem regulaeren SPX-Kassaindex.

2) Die IV-TERM-STRUCTURE (atm_iv_term, long_run_iv, iv_term_kappa) - ist ein
   ANDERES Konzept: wie die implizite Vola selbst mit der Restlaufzeit einer
   EINZELNEN Struktur variiert (kurze Laufzeiten folgen enger der aktuellen
   HV, lange Laufzeiten konvergieren zu einem langfristigen Anker-Niveau).
   Das ist Teil des Bewertungsmodells und bleibt erhalten (siehe unten).

KORREKTUR (Spot vs. Forward): Black-76 ist ein Modell auf den TERMINKURS F,
nicht auf den Kassakurs S. Fruehere Version hat S direkt als F durchgereicht
(entspricht implizit r=q=0). Das ist bei Futures-Optionen exakt richtig
(daher der Kommentar in black76_call/put unten), bei SPX als Kassaindex mit
Dividendenrendite q>0 aber leicht falsch: der faire Terminkurs liegt unter
dem Spot (F = S*exp((r-q)*T)), was Puts tendenziell verteuert und Calls
verguenstigt gegenueber einer Spot-Bewertung. Bei kurzen Laufzeiten (T_days
im Bereich weniger Wochen) ist der Effekt klein, aber systematisch und daher
nicht zu vernachlaessigen. `forward_price()` unten macht diese Umrechnung
explizit; alle Aufrufer (siehe backtest.py) uebergeben jetzt F statt S.
"""

from dataclasses import dataclass
import math

from scipy.stats import norm


def norm_cdf(x: float) -> float:
    return norm.cdf(x)


def norm_ppf(p: float) -> float:
    """Inverse Normalverteilung (exakt, via scipy - praeziser als die
    Acklam-Approximation im Pine-Script)."""
    return norm.ppf(p)


def forward_price(S: float, r: float, q: float, T: float) -> float:
    """Terminkurs aus Spot S, risikolosem Zins r, Dividendenrendite q und
    Restlaufzeit T (Jahre). Bei r=q=0 identisch zu F=S (Rueckwaertskompatibel
    zur bisherigen vereinfachten Annahme). Fuer T<=0 (z.B. Restlaufzeit nach
    Verfall) wird S direkt zurueckgegeben, da kein Termingeschaeft mehr
    aussteht."""
    if T <= 0.0:
        return S
    return S * math.exp((r - q) * T)


def black76_call(F: float, K: float, sigma: float, T: float, r: float) -> float:
    if sigma <= 0.0 or T <= 0.0 or F <= 0.0 or K <= 0.0:
        return max(F - K, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    disc = math.exp(-r * T)
    return disc * (F * norm_cdf(d1) - K * norm_cdf(d2))


def black76_put(F: float, K: float, sigma: float, T: float, r: float) -> float:
    if sigma <= 0.0 or T <= 0.0 or F <= 0.0 or K <= 0.0:
        return max(K - F, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    disc = math.exp(-r * T)
    return disc * (K * norm_cdf(-d2) - F * norm_cdf(-d1))


def leg_intrinsic(K: float, S: float, is_put: bool) -> float:
    """Verfall-Innerer-Wert - bewusst auf Basis des KASSAKURSES S, nicht des
    Terminkurses: Cash-Settlement eines Index-Optionsverfalls (SPX) erfolgt
    gegen den Indexstand, nicht gegen einen Future/Forward."""
    return max(K - S, 0.0) if is_put else max(S - K, 0.0)


def strike_from_delta(F: float, delta: float, sigma: float, T: float, is_put: bool) -> float:
    """Strike aus einem Ziel-Delta (Normal-Approximation, identisch zur
    Pine-Logik: z aus 1-delta, Strike = F * exp(drift -/+ z*sigma*sqrt(T))).
    Erwartet den TERMINKURS F (siehe forward_price)."""
    z = norm_ppf(1.0 - delta)
    drift = 0.5 * sigma * sigma * T
    sigma_sqrtT = sigma * math.sqrt(T)
    if is_put:
        return F * math.exp(drift - z * sigma_sqrtT)
    return F * math.exp(drift + z * sigma_sqrtT)


@dataclass
class VolSurfaceConfig:
    """Parameter der vereinfachten Vol-Oberflaeche (Skew & Term Structure).
    Defaults entsprechen den Pine-Script-Defaults."""
    use_vol_surface: bool = True
    long_run_iv: float = 0.19          # langfristiges IV-Anker-Niveau (ATM), als Dezimalzahl (0.19 = 19%)
    iv_term_kappa: float = 3.0         # Mean-Reversion-Geschwindigkeit
    skew_slope_put: float = 1.4        # Vol-Punkte @ Referenz-Laufzeit
    skew_slope_call: float = 0.4
    skew_ref_days: int = 30
    skew_term_exponent: float = 0.5


def atm_iv_term(T_y: float, base_iv: float, cfg: VolSurfaceConfig) -> float:
    """ATM-IV als Funktion der Restlaufzeit: kurzes Ende folgt base_iv (i.d.R.
    HV), langes Ende konvergiert exponentiell zum langfristigen Anker."""
    if not cfg.use_vol_surface:
        return base_iv
    T_eff = max(T_y, 1.0 / 365.0)
    return cfg.long_run_iv + (base_iv - cfg.long_run_iv) * math.exp(-cfg.iv_term_kappa * T_eff)


def skewed_iv(K: float, F: float, T_y: float, base_iv: float, cfg: VolSurfaceConfig) -> float:
    """Skew-adjustierte IV fuer einen konkreten Strike (Put-Seite teurer,
    Call-Seite guenstiger/flacher - typischer SPX Volatility Smirk).
    Erwartet den TERMINKURS F als Referenzpunkt fuer die Moneyness."""
    base = atm_iv_term(T_y, base_iv, cfg)
    if not (cfg.use_vol_surface and K > 0.0 and F > 0.0 and base > 0.0):
        return base

    T_eff = max(T_y, 1.0 / 365.0)
    sqrtT = math.sqrt(T_eff)
    z = math.log(K / F) / (base * sqrtT)
    T_ref = cfg.skew_ref_days / 365.0
    T_fac = (T_ref / T_eff) ** cfg.skew_term_exponent
    slope = cfg.skew_slope_put if z < 0.0 else cfg.skew_slope_call
    iv_adj = base + (slope / 100.0) * (-z) * T_fac
    return max(iv_adj, 0.02)


def leg_price(K: float, is_put: bool, T_y: float, F: float, base_iv: float,
              r_free: float, cfg: VolSurfaceConfig) -> float:
    """Vollstaendige Beinbewertung: Skew/Term-Structure-IV -> Black-76.
    Wird sowohl fuer Entry-Praemien als auch fuer jede spaetere
    Mark-to-Market-Bewertung (mit T_y = Restlaufzeit) verwendet.
    F muss der TERMINKURS sein (siehe forward_price), nicht der Kassakurs."""
    sigma = skewed_iv(K, F, T_y, base_iv, cfg)
    if is_put:
        return black76_put(F, K, sigma, T_y, r_free)
    return black76_call(F, K, sigma, T_y, r_free)