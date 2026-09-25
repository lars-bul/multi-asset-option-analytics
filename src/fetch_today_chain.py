"""
KOMPATIBILITAETSSCHICHT (Multi-Instrument-Umbau).

Dies war urspruenglich die komplette SPX-Options-Abruflogik. Die eigentliche
Implementierung lebt jetzt instrumentenunabhaengig in
chain_providers/yfinance_provider.py (YFinanceChainProvider) - diese Datei
ruft sie mit dem SPX-Instrument (instruments.INSTRUMENTS['spx']) auf, damit
bestehender Code, der

    from fetch_today_chain import fetch_today_chain, fetch_today_spot, fetch_today_open

importiert (market_data.py, dashboard_depot.py, dashboard_livechain.py),
UNVERAENDERT weiterlaeuft - Verhalten und Rueckgabeformat sind exakt wie vorher.

NEUER CODE FUER NEUE INSTRUMENTE sollte NICHT mehr diese Datei importieren,
sondern direkt ueber die Provider-Registry gehen:

    from chain_providers.base import get_provider
    from instruments import INSTRUMENTS
    instr = INSTRUMENTS["gold"]
    provider = get_provider(instr)
    provider.fetch_chain(instr, min_dte=0.0, max_dte=90.0)

market_data.py wurde bereits auf diesen Weg umgestellt (siehe dort); diese
Datei bleibt nur fuer Abwaertskompatibilitaet mit den beiden Dashboard-Modulen
erhalten, die noch direkt fetch_today_chain()/fetch_today_spot()/
fetch_today_open() importieren, bis sie im naechsten Schritt ebenfalls
generalisiert werden.
"""
import sys

import pandas as pd

from instruments import INSTRUMENTS, InstrumentConfig
from chain_providers.yfinance_provider import YFinanceChainProvider, _dte_exact, ET

TICKER = "^SPX"
MIN_DTE = 0.0
MAX_DTE = 90.0

_SPX = INSTRUMENTS["spx"]
_provider = YFinanceChainProvider()


def dte_exact(expiry: pd.Timestamp, now_et=None) -> float:
    """Re-Export der Original-Formel (jetzt in chain_providers/yfinance_provider.py),
    damit ein direkter Import `from fetch_today_chain import dte_exact` weiter
    funktioniert."""
    return _dte_exact(expiry, now_et)


def _ad_hoc_instrument(ticker: str) -> InstrumentConfig:
    """Fuer den (seltenen) Fall, dass der urspruengliche Ticker-Parameter mit
    einem anderen Wert als TICKER='^SPX' aufgerufen wird - baut spontan eine
    minimale InstrumentConfig, damit der Provider trotzdem funktioniert."""
    return InstrumentConfig(
        key="_adhoc", label=ticker, price_ticker=ticker, currency="USD",
        multiplier=100.0, dividend_yield=0.0, has_shiller_history=False,
        settlement="cash_index", chain_provider="yfinance", chain_ticker=ticker,
    )


def fetch_today_spot(ticker: str = TICKER) -> float:
    instr = _SPX if ticker == TICKER else _ad_hoc_instrument(ticker)
    return _provider.fetch_spot(instr)


def fetch_today_open(ticker: str = TICKER) -> float | None:
    instr = _SPX if ticker == TICKER else _ad_hoc_instrument(ticker)
    return _provider.fetch_today_open(instr)


def fetch_today_chain(ticker: str = TICKER, min_dte: float = MIN_DTE, max_dte: float = MAX_DTE) -> pd.DataFrame:
    instr = _SPX if ticker == TICKER else _ad_hoc_instrument(ticker)
    return _provider.fetch_chain(instr, min_dte, max_dte)


def compare_to_model(chain: pd.DataFrame, spot: float, r_free: float, dividend_yield: float,
                      current_iv: float, vol_surface) -> pd.DataFrame:
    """SYSTEMATISCHER Abgleich: fuer JEDE Serie der heutigen Kette wird der
    theoretische Preis aus eurer eigenen black76.leg_price() berechnet und
    der Differenz zum echten Markt-Mid gegenuebergestellt. Unveraendert
    gegenueber dem Original - instrumentenunabhaengig, da nur mit dem
    uebergebenen 'chain'-DataFrame und den explizit uebergebenen Parametern
    gerechnet wird."""
    from black76 import forward_price, leg_price

    rows = chain.copy()
    rows["model_price"] = rows.apply(
        lambda r: leg_price(
            r["strike"], r["option_type"] == "put", r["dte"] / 365.0,
            forward_price(spot, r_free, dividend_yield, r["dte"] / 365.0),
            current_iv, r_free, vol_surface,
        ),
        axis=1,
    )
    rows["model_minus_mid"] = rows["model_price"] - rows["mid"]
    rows["model_minus_mid_pct"] = rows["model_minus_mid"] / rows["mid"].replace(0, pd.NA)
    return rows


if __name__ == "__main__":
    chain = fetch_today_chain()
    print(f"{len(chain)} Serien in der heutigen Kette (DTE {MIN_DTE}-{MAX_DTE}).")
    print(chain.head(10))