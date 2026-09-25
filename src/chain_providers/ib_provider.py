"""
Optionsketten-Anbieter ueber Interactive Brokers (fuer Eurex/EuroStoxx50).

VORAUSSETZUNGEN (ausserhalb dieses Codes, kann ich hier nicht pruefen/testen):
  1. Ein IB-Konto (auch ein Paper-Trading-Konto reicht fuer die Kontaktdaten -
     ECHTE Marktdaten brauchen aber ein Live-Konto + Datenabo, siehe Punkt 3).
  2. TWS oder IB Gateway laeuft lokal (oder erreichbar) und die API ist dort
     aktiviert (TWS: Global Configuration -> API -> Settings -> "Enable ActiveX
     and Socket Clients"). Standardports: TWS 7497 (Paper) / 7496 (Live),
     IB Gateway 4002 (Paper) / 4001 (Live).
  3. Ein aktives Marktdaten-Abo "Eurex Index and Futures Options" (fuer
     Nicht-Profis, wenige Euro/Monat) - ohne das liefert IB entweder gar keine
     oder nur verzoegerte/eingeschraenkte Quotes fuer EURO STOXX 50-Optionen.
  4. Python-Paket 'ib_async' (Nachfolger/gepflegter Fork von 'ib_insync'):
         pip install ib_async

KONFIGURATION: Host/Port/ClientId ueber Umgebungsvariablen IB_HOST/IB_PORT/
IB_CLIENT_ID (Defaults: 127.0.0.1 / 7497 / 17). Bei mehreren gleichzeitig
verbundenen Clients (z.B. TWS selbst + dieses Skript + ein anderes Tool) muss
IB_CLIENT_ID je Prozess eindeutig sein, sonst lehnt IB die Verbindung ab.

WICHTIGER HINWEIS ZU ib_trading_class/ib_index_symbol (siehe instruments.py):
IB-Vertragsdetails aendern sich gelegentlich. Bevor dieser Provider produktiv
genutzt wird: in der TWS unter "Contract Search" nach "ESTX50" (Exchange EUREX,
SecType OPT) suchen und pruefen, ob Symbol/TradingClass noch mit
instruments.py uebereinstimmen. Ich kann das von hier aus nicht verifizieren,
da mein Sandbox-Netzwerk keinen Zugriff auf IB-Endpunkte hat.

SETTLEMENT-HINWEIS: anders als bei SPX (PM-Settlement, siehe dashboard_livechain.py
_dte_exact) werden EURO STOXX 50-Indexoptionen an der Eurex per AM-Settlement
auf Basis eines Eroeffnungsauktions-Werts am Verfalltag abgerechnet. Die
dte_exact-Naeherung hier nimmt daher NICHT 16:00 wie bei SPX, sondern den
Boersenbeginn (09:00 Europe/Berlin) am Verfalltag - das ist ebenfalls eine
Naeherung (die exakte Eroeffnungszeit der Auktion variiert leicht) und sollte
bei Bedarf verfeinert werden.
"""
from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from chain_providers.base import ChainProvider, UnsupportedChainError
from instruments import InstrumentConfig

_IB_HOST = os.environ.get("IB_HOST", "127.0.0.1")
_IB_PORT = int(os.environ.get("IB_PORT", "7497"))
_IB_CLIENT_ID = int(os.environ.get("IB_CLIENT_ID", "17"))
_TZ_EUREX = ZoneInfo("Europe/Berlin")
_SNAPSHOT_TIMEOUT_S = 8.0


def _require_ib_async():
    try:
        import ib_async  # noqa: F401
        return ib_async
    except ImportError as e:
        raise UnsupportedChainError(
            "Paket 'ib_async' ist nicht installiert - fuer die EuroStoxx50-Kette "
            "noetig: `pip install ib_async`. (Falls du noch das aeltere, "
            "unveraendert Feature-eingefrorene 'ib_insync' nutzt: das laeuft mit "
            "diesem Provider NICHT direkt, bitte auf 'ib_async' wechseln.)"
        ) from e


class IBChainProvider(ChainProvider):
    name = "ib"

    def _connect(self):
        ib_async = _require_ib_async()
        from ib_async import IB

        ib = IB()
        try:
            ib.connect(_IB_HOST, _IB_PORT, clientId=_IB_CLIENT_ID, timeout=10)
        except Exception as e:
            raise UnsupportedChainError(
                f"Verbindung zu Interactive Brokers fehlgeschlagen ({e}). Pruefen: "
                f"laeuft TWS/IB Gateway auf {_IB_HOST}:{_IB_PORT}? Ist die API dort "
                "aktiviert ('Enable ActiveX and Socket Clients')? Ist IB_CLIENT_ID "
                f"({_IB_CLIENT_ID}) nicht bereits durch einen anderen Client belegt?"
            ) from e
        return ib

    def _index_contract(self, instr: InstrumentConfig):
        from ib_async import Index

        if not (instr.ib_index_symbol and instr.ib_exchange):
            raise UnsupportedChainError(
                f"Instrument '{instr.key}' hat keine ib_index_symbol/ib_exchange "
                "in instruments.py hinterlegt."
            )
        return Index(instr.ib_index_symbol, instr.ib_exchange,
                     instr.ib_currency or instr.currency)

    def fetch_spot(self, instr: InstrumentConfig) -> float:
        ib = self._connect()
        try:
            contract = self._index_contract(instr)
            ib.qualifyContracts(contract)
            ticker = ib.reqMktData(contract, "", False, False)
            ib.sleep(_SNAPSHOT_TIMEOUT_S)
            price = ticker.marketPrice()
            if price is None or np.isnan(price):
                price = ticker.close
            if price is None or (isinstance(price, float) and np.isnan(price)):
                raise UnsupportedChainError(
                    f"IB hat keinen gueltigen Kurs fuer {instr.ib_index_symbol} "
                    "geliefert (Markt geschlossen + keine 'Close' verfuegbar, oder "
                    "fehlende Marktdaten-Berechtigung)."
                )
            return float(price)
        finally:
            ib.disconnect()

    def fetch_chain(self, instr: InstrumentConfig, min_dte: float, max_dte: float) -> pd.DataFrame:
        from ib_async import Option

        ib = self._connect()
        try:
            underlying = self._index_contract(instr)
            ib.qualifyContracts(underlying)

            chains = ib.reqSecDefOptParams(
                underlying.symbol, "", underlying.secType, underlying.conId,
            )
            chains = [c for c in chains if c.exchange == instr.ib_exchange]
            if not chains:
                raise UnsupportedChainError(
                    f"IB liefert keine Options-Parameter fuer {instr.ib_index_symbol} "
                    f"auf {instr.ib_exchange} - Vertragsdetails in instruments.py pruefen."
                )
            chain = chains[0]
            trading_class = instr.ib_trading_class or chain.tradingClasses[0]

            now = pd.Timestamp.now(tz=_TZ_EUREX)
            expiries = []
            for exp_str in sorted(chain.expirations):
                exp_date = pd.Timestamp(datetime.strptime(exp_str, "%Y%m%d"), tz=_TZ_EUREX)
                # Naeherung Verfall = Boersenbeginn 09:00 lokal (siehe Modul-Docstring,
                # Settlement-Hinweis) statt 16:00 wie bei SPX/PM-Settlement.
                exp_settle = exp_date + pd.Timedelta(hours=9)
                dte_exact = (exp_settle - now).total_seconds() / 86400.0
                if min_dte <= dte_exact <= max_dte:
                    expiries.append((exp_str, dte_exact))

            if not expiries:
                return pd.DataFrame(columns=[
                    "expiry", "strike", "option_type", "bid", "ask", "mid",
                    "dte", "dte_exact", "last_trade_date",
                ])

            contracts = []
            meta = []
            for exp_str, dte_exact in expiries:
                for strike in sorted(chain.strikes):
                    for right, opt_type in (("P", "put"), ("C", "call")):
                        c = Option(
                            underlying.symbol, exp_str, float(strike), right,
                            instr.ib_exchange, tradingClass=trading_class,
                            currency=instr.ib_currency or instr.currency,
                        )
                        contracts.append(c)
                        meta.append((exp_str, dte_exact, float(strike), opt_type))

            qualified = ib.qualifyContracts(*contracts)
            valid_idx = [i for i, c in enumerate(qualified) if c.conId]
            qualified = [qualified[i] for i in valid_idx]
            meta = [meta[i] for i in valid_idx]
            if not qualified:
                raise UnsupportedChainError(
                    "Keine der angefragten EURO-STOXX-50-Optionskontrakte konnte "
                    "bei IB qualifiziert werden - Strike-/TradingClass-Angaben pruefen."
                )

            tickers = ib.reqTickers(*qualified)  # Snapshot-Abruf in einem Rutsch

            rows = []
            for t, (exp_str, dte_exact, strike, opt_type) in zip(tickers, meta):
                bid = t.bid if t.bid and t.bid > 0 else np.nan
                ask = t.ask if t.ask and t.ask > 0 else np.nan
                mid = float(np.nanmean([bid, ask])) if not (np.isnan(bid) and np.isnan(ask)) else np.nan
                last_trade_date = pd.NaT
                if getattr(t, "lastTimestamp", None):
                    last_trade_date = pd.Timestamp(t.lastTimestamp)
                rows.append({
                    "expiry": pd.Timestamp(datetime.strptime(exp_str, "%Y%m%d")),
                    "strike": strike, "option_type": opt_type,
                    "bid": bid, "ask": ask, "mid": mid,
                    "dte": int(round(dte_exact)), "dte_exact": dte_exact,
                    "last_trade_date": last_trade_date,
                })
            return pd.DataFrame(rows)
        finally:
            ib.disconnect()