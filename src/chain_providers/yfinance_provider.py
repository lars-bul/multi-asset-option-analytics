"""
Optionsketten-Anbieter ueber Yahoo Finance (yfinance).

Diese Datei ist die MULTI-INSTRUMENT-VERALLGEMEINERUNG der urspruenglichen
fetch_today_chain.py: Logik 1:1 uebernommen (dieselbe dte_exact()-Formel,
dieselben Spaltennamen/Umbenennungen, dieselbe Fehlerbehandlung je Verfall),
nur dass der Ticker jetzt aus instr.chain_ticker kommt statt aus der
Konstante TICKER="^SPX". fetch_today_chain.py selbst bleibt als duenne
Kompatibilitaetsschicht bestehen (siehe dort) - bestehender Code, der davon
importiert, aendert sich nicht.

WICHTIG (aus dem Original uebernommen): fuer SPX ist der Ketten-Ticker "^SPX"
(CBOE-Feed bei Yahoo), NICHT "^GSPC" (das ist nur die Kurshistorie) - siehe
instruments.py, chain_ticker="^SPX" bei "spx".

dte_exact() nimmt fuer ALLE hier bedienten Instrumente PM-Settlement (16:00
America/New_York) an - im Original SPX-spezifisch (SPXW-Wochenserien/0DTE)
begruendet, gilt hier als Naeherung ebenso fuer NDX/RUT-Proxy(IWM)/Gold(GLD)/
Oil(USO), da es sich durchweg um US-boersengehandelte Optionen handelt.

KORREKTUR (yfinance-fast_info-Bug, "KeyError: 'currentTradingPeriod'"):
yf.Ticker(...).fast_info ist ein Lazy-Objekt - Felder wie 'open'/'lastPrice'
loesen ERST bei Zugriff einen Netzwerkruf aus (u.a. _get_1y_prices()), und
Yahoo liefert fuer manche Ticker/Marktzustaende (insbesondere auslaendische
Indizes wie ^N225/^STOXX50E, aber auch sporadisch bei US-Tickern) ein
Metadaten-Objekt OHNE den Schluessel 'currentTradingPeriod'. yfinance faengt
das intern NICHT ab und wirft einen rohen KeyError - das ist ein Bug/eine
Unzuverlaessigkeit von yfinance/Yahoo, kein Fehler in der eigenen Config.
fetch_spot()/fetch_today_open() faengt das jetzt ab: fetch_spot() wirft
stattdessen eine spechende UnsupportedChainError, fetch_today_open() gibt
None zurueck (das ist laut chain_providers.base.ChainProvider ohnehin der
vorgesehene 'kein Wert verfuegbar'-Fall - der Aufrufer faellt dann auf den
letzten echten Schlusskurs zurueck, siehe dashboard_livechain.py).
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from chain_providers.base import ChainProvider, UnsupportedChainError
from instruments import InstrumentConfig

ET = ZoneInfo("America/New_York")


def _now_et() -> datetime:
    return datetime.now(ET)


def _dte_exact(expiry: pd.Timestamp, now_et: datetime | None = None) -> float:
    """Identisch zur Originalformel aus fetch_today_chain.dte_exact():
    PM-Settlement 16:00 ET angenommen."""
    if now_et is None:
        now_et = _now_et()
    settle = datetime(expiry.year, expiry.month, expiry.day, 16, 0, 0, tzinfo=ET)
    return (settle - now_et).total_seconds() / 86400.0


class YFinanceChainProvider(ChainProvider):
    name = "yfinance"

    def fetch_spot(self, instr: InstrumentConfig) -> float:
        ticker = instr.chain_ticker or instr.price_ticker
        tk = yf.Ticker(ticker)
        try:
            fast = tk.fast_info
            price = fast.get("lastPrice") or fast.get("last_price")
        except Exception as e:
            # KEIN gezieltes except KeyError: yfinance's fast_info-Interna sind lazy und
            # koennen bei fehlenden Yahoo-Metadaten (siehe Modul-Docstring) auch andere
            # Exceptions werfen (z.B. IndexError/HTTPError je nach yfinance-Version) - hier
            # zaehlt nur, dass der Spot fuer 'ticker' gerade nicht ermittelbar war.
            raise UnsupportedChainError(
                f"Konnte aktuellen Spot-Preis fuer '{ticker}' nicht ermitteln - Yahoo hat kein "
                f"nutzbares Kurs-Metadatenobjekt geliefert ({type(e).__name__}: {e}). Das ist ein "
                "bekanntes yfinance/Yahoo-Problem bei manchen Tickern (v.a. auslaendische Indizes) "
                "und tritt manchmal auch nur voruebergehend auf - erneut versuchen."
            ) from e
        if price is None:
            raise UnsupportedChainError(
                f"Konnte aktuellen Spot-Preis fuer '{ticker}' nicht ermitteln (yfinance fast_info "
                "liefert weder 'lastPrice' noch 'last_price')."
            )
        return float(price)

    def fetch_today_open(self, instr: InstrumentConfig) -> float | None:
        ticker = instr.chain_ticker or instr.price_ticker
        tk = yf.Ticker(ticker)
        try:
            fast = tk.fast_info
            open_price = fast.get("open") or fast.get("regularMarketOpen")
        except Exception:
            # Optionaler Wert (siehe chain_providers.base.ChainProvider.fetch_today_open) - der
            # Aufrufer faellt bei None auf den letzten echten Schlusskurs zurueck, siehe
            # dashboard_livechain._build_daily_df_with_today(). Kein Grund, deswegen abzubrechen.
            return None
        if open_price is None or open_price <= 0:
            return None
        return float(open_price)

    def fetch_chain(self, instr: InstrumentConfig, min_dte: float, max_dte: float) -> pd.DataFrame:
        ticker = instr.chain_ticker or instr.price_ticker
        tk = yf.Ticker(ticker)
        now_et = _now_et()
        today = pd.Timestamp(date.today())

        try:
            expiries = tk.options
        except Exception as e:
            raise UnsupportedChainError(
                f"Konnte Expiries fuer '{ticker}' nicht laden - Yahoo-Endpoint nicht "
                f"erreichbar oder Ticker falsch? Fehler: {e}"
            ) from e

        if not expiries:
            raise UnsupportedChainError(f"Keine Expiries fuer '{ticker}' gefunden.")

        rows = []
        for exp_str in expiries:
            exp = pd.Timestamp(exp_str)
            dte_precise = _dte_exact(exp, now_et)
            if dte_precise < min_dte or dte_precise > max_dte:
                continue

            try:
                chain = tk.option_chain(exp_str)
            except Exception as e:
                print(f"  Warnung: Expiry {exp_str} uebersprungen ({e})", file=sys.stderr)
                continue

            for option_type, df_leg in (("call", chain.calls), ("put", chain.puts)):
                if df_leg.empty:
                    continue
                sub = df_leg.copy()
                sub["date"] = today
                sub["expiry"] = exp
                sub["dte"] = round(dte_precise)
                sub["dte_exact"] = dte_precise
                sub["option_type"] = option_type
                sub["mid"] = (sub["bid"] + sub["ask"]) / 2.0
                if "lastTradeDate" in sub.columns:
                    sub["last_trade_date"] = sub["lastTradeDate"]
                else:
                    sub["last_trade_date"] = pd.NaT
                rows.append(sub[[
                    "date", "expiry", "dte", "dte_exact", "strike", "option_type",
                    "bid", "ask", "mid", "lastPrice", "volume", "openInterest", "impliedVolatility",
                    "last_trade_date",
                ]].rename(columns={
                    "lastPrice": "last", "openInterest": "open_interest", "impliedVolatility": "implied_vol",
                }))

        if not rows:
            raise UnsupportedChainError(
                f"Keine Serien im DTE-Fenster [{min_dte}, {max_dte}] fuer '{ticker}' gefunden - "
                "Fenster erweitern oder Handelstag pruefen (Wochenende/Feiertag?)."
            )

        return pd.concat(rows, ignore_index=True).sort_values(
            ["expiry", "option_type", "strike"]
        ).reset_index(drop=True)