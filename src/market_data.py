"""
GEMEINSAMER MARKTDATEN-BESTAND fuer ALLE Modi (Heutige Kette, Depot-Analyse, ...).

MULTI-INSTRUMENT-UMBAU: frueher gab es GENAU EINEN globalen Snapshot (immer
SPX). Jetzt wird PRO INSTRUMENT-SCHLUESSEL (siehe instruments.py) ein eigener
Snapshot gehalten - noetig, weil im gemeinsamen Depot (depot_engine.py)
gleichzeitig Kombos verschiedener Instrumente liegen koennen und jedes davon
unabhaengig geladen/aktualisiert wird. Der Ketten-Abruf selbst laeuft NICHT
mehr fest ueber fetch_today_chain.py, sondern ueber die Provider-Registry
(chain_providers.base.get_provider) - jedes Instrument kann dadurch einen
eigenen Anbieter haben (Yahoo, Interactive Brokers, spaeter ggf. weitere),
ohne dass dieses Modul die Details kennen muss.

Wird ein Instrument OHNE chain_provider angefragt (aktuell: Nikkei,
EuroStoxx50 - siehe instruments.py), werfen load()/get() eine
chain_providers.base.UnsupportedChainError. Das ist ABSICHTLICH kein stiller
leerer Snapshot: die Dashboards muessen das explizit abfangen und 'Heutige
Kette'/Depot-Live-Bewertung fuer dieses Instrument sauber deaktivieren, statt
mit leeren/falschen Daten weiterzurechnen.

BREAKING CHANGE gegenueber der Vorversion: current()/load()/get() nehmen jetzt
ZUSAETZLICH als ERSTES Argument ein Instrument (Schluessel-String ODER
InstrumentConfig) entgegen. Aufrufer, die noch `market_data.current(max_dte)`
(ohne Instrument) nutzen, muessen auf `market_data.current(instrument, max_dte)`
umgestellt werden - das betrifft aktuell noch dashboard_livechain.py und
dashboard_depot.py (naechster Umbauschritt).
"""
import threading
from datetime import date

import pandas as pd

from instruments import InstrumentConfig, get_instrument
from chain_providers.base import get_provider
from chain_delta import ChainDeltaConfig, enrich_chain_with_delta

LOAD_MIN_ABS_DELTA = 0.0001
DEFAULT_MAX_DTE = 90.0

_LOCK = threading.Lock()
_SNAPS: dict[str, dict] = {}   # instrument_key -> Snapshot


def _resolve(instrument) -> InstrumentConfig:
    return instrument if isinstance(instrument, InstrumentConfig) else get_instrument(instrument)


def current(instrument, needed_max_dte: float = 0.0) -> dict | None:
    """Der aktuelle Snapshot FUER DIESES INSTRUMENT, falls er von heute ist und
    mindestens needed_max_dte Tage abdeckt, sonst None. Laedt NICHTS (immer sofort)."""
    instr = _resolve(instrument)
    snap = _SNAPS.get(instr.key)
    if snap is None or snap["fetch_date"] != date.today():
        return None
    if snap["max_dte"] < float(needed_max_dte) - 1e-9:
        return None
    return snap


def load(instrument, r_free: float, dividend_yield: float, needed_max_dte: float) -> dict:
    """Laedt Spot + Kette fuer DIESES INSTRUMENT neu und ersetzt dessen Snapshot.
    Wirft chain_providers.base.UnsupportedChainError, falls das Instrument
    keinen chain_provider hat, sonst die jeweilige Provider-Exception bei
    Abrufproblemen; ein fehlgeschlagener Ladeversuch laesst den bisherigen
    Snapshot DIESES Instruments unveraendert (andere Instrumente sind nie
    betroffen, da jedes seinen eigenen Snapshot-Eintrag hat)."""
    instr = _resolve(instrument)
    provider = get_provider(instr)   # wirft UnsupportedChainError fruehzeitig, falls None
    max_dte = max(float(needed_max_dte), DEFAULT_MAX_DTE)
    with _LOCK:
        fetch_ts = pd.Timestamp.now(tz="UTC")
        spot = float(provider.fetch_spot(instr))
        chain = provider.fetch_chain(instr, min_dte=0.0, max_dte=max_dte)
        enriched = enrich_chain_with_delta(
            chain, spot,
            ChainDeltaConfig(min_abs_delta=LOAD_MIN_ABS_DELTA, r_free=float(r_free),
                             dividend_yield=float(dividend_yield)),
            as_of=fetch_ts,
        )
        snap = {"instrument": instr.key, "spot": spot, "enriched": enriched, "fetch_ts": fetch_ts,
                "r_free": float(r_free), "dividend_yield": float(dividend_yield),
                "max_dte": max_dte, "fetch_date": date.today()}
        _SNAPS[instr.key] = snap
        return snap


def get(instrument, r_free: float, dividend_yield: float, needed_max_dte: float, force: bool = False) -> dict:
    """Snapshot dieses Instruments aus dem Speicher, sonst (oder bei force) neu laden."""
    instr = _resolve(instrument)
    snap = None if force else current(instr, needed_max_dte)
    return snap if snap is not None else load(instr, r_free, dividend_yield, needed_max_dte)