"""
GEMEINSAMER MARKTDATEN-BESTAND fuer ALLE Modi (Heutige Kette, Depot-Analyse, ...).

Problem vorher: jeder Modus hat die Optionskette (Yahoo-Abruf + IV-Solve je Serie) selbst und
pro Browser-Sitzung geladen. Das hat beim Wechsel der Tabs zu langen Wartezeiten gefuehrt und
dazu, dass die Modi mit UNTERSCHIEDLICHEN Datenstaenden gearbeitet haben.

Jetzt gibt es EINEN Snapshot pro Programmstart (Modul-Variable, ueberlebt Reruns, Tab-Wechsel und
neu geladene Browser-Fenster):
  - spot, enriched (Kette inkl. implied_vol_mkt/delta_mkt/mid/bid/ask), fetch_ts, r_free,
    dividend_yield, max_dte, fetch_date
  - wird neu geladen, wenn (a) noch keiner existiert, (b) er von einem frueheren Tag stammt,
    (c) ein Modus einen laengeren DTE-Bereich braucht als geladen wurde, oder (d) der Nutzer
    ausdruecklich 'neu laden' klickt (force=True).
  - Die Kette wird mit sehr kleinem |Delta|-Filter (0.0001) geladen; feinere Filter
    (DTE-Bereich, |Delta|) wendet jeder Modus selbst auf der Ansicht an, ohne neu zu laden.
  - r_free/dividend_yield gelten fuer den Zeitpunkt des Ladens (wie bisher); ein Wechsel der
    Werte wirkt erst nach 'neu laden'.
"""
import threading
from datetime import date

import pandas as pd

from fetch_today_chain import fetch_today_chain, fetch_today_spot
from chain_delta import ChainDeltaConfig, enrich_chain_with_delta

LOAD_MIN_ABS_DELTA = 0.0001
DEFAULT_MAX_DTE = 90.0

_LOCK = threading.Lock()
_SNAP: dict = {"data": None}


def current(needed_max_dte: float = 0.0) -> dict | None:
    """Der aktuelle Snapshot, falls er von heute ist und mindestens needed_max_dte Tage abdeckt,
    sonst None. Laedt NICHTS (immer sofort)."""
    snap = _SNAP["data"]
    if snap is None or snap["fetch_date"] != date.today():
        return None
    if snap["max_dte"] < float(needed_max_dte) - 1e-9:
        return None
    return snap


def load(r_free: float, dividend_yield: float, needed_max_dte: float) -> dict:
    """Laedt Spot + Kette neu und ersetzt den Snapshot. Wirft bei Fehlern (Aufrufer zeigt sie an);
    ein fehlgeschlagener Ladeversuch laesst den bisherigen Snapshot unveraendert."""
    max_dte = max(float(needed_max_dte), DEFAULT_MAX_DTE)
    with _LOCK:
        fetch_ts = pd.Timestamp.now(tz="UTC")
        spot = float(fetch_today_spot())
        chain = fetch_today_chain(min_dte=0.0, max_dte=max_dte)
        enriched = enrich_chain_with_delta(
            chain, spot,
            ChainDeltaConfig(min_abs_delta=LOAD_MIN_ABS_DELTA, r_free=float(r_free),
                             dividend_yield=float(dividend_yield)),
            as_of=fetch_ts,
        )
        snap = {"spot": spot, "enriched": enriched, "fetch_ts": fetch_ts, "r_free": float(r_free),
                "dividend_yield": float(dividend_yield), "max_dte": max_dte, "fetch_date": date.today()}
        _SNAP["data"] = snap
        return snap


def get(r_free: float, dividend_yield: float, needed_max_dte: float, force: bool = False) -> dict:
    """Snapshot aus dem Speicher, sonst (oder bei force) neu laden."""
    snap = None if force else current(needed_max_dte)
    return snap if snap is not None else load(r_free, dividend_yield, needed_max_dte)