"""
Abstraktion fuer Optionsketten-Anbieter (siehe instruments.py, Feld
'chain_provider'). Jeder Anbieter liefert dieselbe Rueckgabeform, damit
market_data.py/chain_delta.py NICHT wissen muessen, ob die Daten von Yahoo
Finance, Interactive Brokers oder einem spaeter ergaenzten Anbieter kommen.

Erwartetes DataFrame-Format von fetch_chain() (identisch zur bisherigen
fetch_today_chain.py-Konvention, siehe chain_delta.enrich_chain_with_delta()):
  expiry (Timestamp), strike (float), option_type ('put'|'call'), bid, ask,
  mid, dte (int), dte_exact (float, Kalendertage inkl. Bruchteil bis Verfall),
  last_trade_date (Timestamp oder NaT).

ERWEITERBARKEIT (Kernanforderung: die Anwendung soll leicht auf andere
Datenquellen umgestellt/erweitert werden koennen): statt eines fest
verdrahteten if/elif nach instr.chain_provider gibt es eine REGISTRY
(Name -> Factory-Funktion). Ein neuer Anbieter wird an EINER Stelle per
register_provider() angemeldet, ohne dass dieses Modul selbst geaendert
werden muss:

    # z.B. in einem eigenen Skript oder Plugin-Modul:
    from chain_providers.base import register_provider
    from my_polygon_provider import PolygonChainProvider
    register_provider("polygon", lambda: PolygonChainProvider(api_key=...))

    # und in instruments.py bei einem Instrument:
    chain_provider="polygon"

Die beiden mitgelieferten Anbieter ("yfinance", "ib") sind lediglich die
Standardbelegung der Registry (siehe _register_builtin_providers() unten) -
sie geniessen keine Sonderrolle gegenueber spaeter ergaenzten Anbietern.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from instruments import InstrumentConfig


class UnsupportedChainError(RuntimeError):
    """Wird geworfen, wenn fuer ein Instrument chain_provider=None gesetzt ist,
    ein Modus (Heutige Kette/Depot) aber trotzdem eine Kette braucht - oder
    wenn ein registrierter Provider aus einem instrumentenspezifischen Grund
    (z.B. fehlende ib_*-Felder, fehlendes Paket, Verbindungsfehler) nicht
    nutzbar ist."""


class ChainProvider:
    """Basisklasse (kein ABC-Zwang, damit ein minimaler Duck-Typing-Provider
    ohne Vererbung moeglich ist - Pflicht sind nur fetch_spot()/fetch_chain())."""
    name: str = "base"

    def fetch_spot(self, instr: InstrumentConfig) -> float:
        raise NotImplementedError

    def fetch_chain(self, instr: InstrumentConfig, min_dte: float, max_dte: float) -> pd.DataFrame:
        raise NotImplementedError

    def fetch_today_open(self, instr: InstrumentConfig) -> float | None:
        """Optionaler heutiger Eroeffnungskurs (fuer den synthetischen 'Heute'-
        Bar der Analog-Suche). Default: None (Aufrufer faellt auf den letzten
        echten Schlusskurs zurueck) - einzelne Provider koennen das ueberschreiben."""
        return None


_PROVIDER_FACTORIES: dict[str, Callable[[], ChainProvider]] = {}
_builtins_registered = False


def register_provider(name: str, factory: Callable[[], ChainProvider]) -> None:
    """Meldet einen Chain-Provider unter 'name' an. 'name' ist genau der
    String, der in instruments.py als chain_provider=... eingetragen wird.
    Ueberschreibt eine vorhandene Registrierung desselben Namens bewusst
    kommentarlos (so kann z.B. in Tests ein Fake-Provider eingesetzt werden)."""
    _PROVIDER_FACTORIES[name] = factory


def _register_builtin_providers() -> None:
    """Registriert die mitgelieferten Anbieter beim ERSTEN get_provider()-
    Aufruf (lazy, damit z.B. 'ib_async' nicht importiert wird, solange
    niemand ein IB-gestuetztes Instrument nutzt)."""
    global _builtins_registered
    if _builtins_registered:
        return

    def _make_yfinance() -> ChainProvider:
        from chain_providers.yfinance_provider import YFinanceChainProvider
        return YFinanceChainProvider()

    def _make_ib() -> ChainProvider:
        from chain_providers.ib_provider import IBChainProvider
        return IBChainProvider()

    _PROVIDER_FACTORIES.setdefault("yfinance", _make_yfinance)
    _PROVIDER_FACTORIES.setdefault("ib", _make_ib)
    _builtins_registered = True


def available_providers() -> list[str]:
    """Namen aller aktuell registrierten Provider (Diagnose/Dashboard-Anzeige)."""
    _register_builtin_providers()
    return sorted(_PROVIDER_FACTORIES)


def get_provider(instr: InstrumentConfig) -> ChainProvider:
    """Liefert die passende ChainProvider-Instanz fuer ein Instrument.
    Wirft UnsupportedChainError, wenn instr.chain_provider None ist oder
    keinem registrierten Provider entspricht."""
    if not instr.chain_provider:
        raise UnsupportedChainError(
            f"Fuer Instrument '{instr.key}' ({instr.label}) ist keine Live-Optionskette "
            "konfiguriert (chain_provider=None) - 'Heutige Kette' und die "
            "Live-Bewertung im Depot stehen fuer dieses Instrument nicht zur "
            "Verfuegung. Backtest (Modus 1) und Analog-Matching/ML (Modus 2) "
            "funktionieren trotzdem, da sie nur die taegliche Kurshistorie brauchen."
        )
    _register_builtin_providers()
    factory = _PROVIDER_FACTORIES.get(instr.chain_provider)
    if factory is None:
        raise UnsupportedChainError(
            f"Kein Chain-Provider unter dem Namen '{instr.chain_provider}' registriert "
            f"(Instrument '{instr.key}'). Registriert sind: {available_providers()}. "
            "Neue Anbieter mit chain_providers.base.register_provider(name, factory) "
            "anmelden, bevor sie in instruments.py referenziert werden."
        )
    return factory()