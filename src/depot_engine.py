"""
Rechenlogik + Speicherung fuer die Depot-Analyse (ohne Streamlit, damit testbar).

DATENMODELL (JSON in data/depot.json):
  {"version": 1, "saved_at": "...", "combos": [
      {"id": "a1b2c3d4e5", "name": "IC 7300/7350/7900/7950", "expiry": "2026-10-02",
       "note": "", "created": "...",
       "legs": [{"option_type": "put"|"call", "side": "long"|"short",
                 "strike": 7300.0, "qty": 1.0, "entry_price": 3.2}, ...]}]}
  Eine Kombo hat 1 bis 4 Beine mit GEMEINSAMEM Verfall. entry_price = Optionspreis
  beim Einstieg in Indexpunkten je Bein (nicht mal 100).

VORZEICHEN-KONVENTION:
  entry_net  = Summe ueber Beine: short +preis*qty (Kredit), long -preis*qty (Debit)
  Wert bei Verfall V(S) = Summe: long +intrinsic*qty, short -intrinsic*qty
  P&L bei Verfall (seit Einstieg) = entry_net + V(S)

SPEICHERN: atomar (tmp-Datei + os.replace) mit .bak der vorherigen Version, damit
das Depot auch bei einem Absturz waehrend des Schreibens nicht verloren geht. Eine
unlesbare Datei wird NICHT ueberschrieben, sondern als depot.corrupt-<zeit>.json
gesichert.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

MULTIPLIER = 100          # SPX: 1 Indexpunkt = 100 USD je Kontrakt
SCHEMA_VERSION = 1
OPTION_TYPES = ("put", "call")
SIDES = ("long", "short")


# ───────────────────────── Datenmodell ─────────────────────────

def new_id() -> str:
    return uuid.uuid4().hex[:10]


def clean_leg(leg: dict) -> dict:
    return {
        "option_type": str(leg["option_type"]).lower(),
        "side": str(leg["side"]).lower(),
        "strike": float(leg["strike"]),
        "qty": float(leg.get("qty", 1.0)),
        "entry_price": float(leg.get("entry_price", 0.0)),
    }


def make_combo(name: str, expiry, legs: list[dict], note: str = "", combo_id: str | None = None,
               created: str | None = None) -> dict:
    return {
        "id": combo_id or new_id(),
        "name": (name or "").strip() or "Kombo",
        "expiry": str(expiry),
        "note": note or "",
        "created": created or datetime.now().isoformat(timespec="seconds"),
        "legs": [clean_leg(l) for l in legs],
    }


def validate_combo(combo: dict) -> list[str]:
    errs: list[str] = []
    legs = combo.get("legs", [])
    if not 1 <= len(legs) <= 4:
        errs.append("Eine Kombo braucht 1 bis 4 Beine.")
    try:
        datetime.strptime(str(combo.get("expiry")), "%Y-%m-%d")
    except ValueError:
        errs.append("Verfallsdatum ungueltig (Format JJJJ-MM-TT).")
    for i, leg in enumerate(legs, 1):
        try:
            if str(leg["option_type"]).lower() not in OPTION_TYPES:
                errs.append(f"Bein {i}: Typ muss 'put' oder 'call' sein.")
            if str(leg["side"]).lower() not in SIDES:
                errs.append(f"Bein {i}: Richtung muss 'long' oder 'short' sein.")
            if not float(leg["strike"]) > 0:
                errs.append(f"Bein {i}: Strike muss > 0 sein.")
            if not float(leg.get("qty", 1.0)) > 0:
                errs.append(f"Bein {i}: Menge muss > 0 sein.")
            if float(leg.get("entry_price", 0.0)) < 0:
                errs.append(f"Bein {i}: Einstiegspreis darf nicht negativ sein.")
        except (KeyError, TypeError, ValueError):
            errs.append(f"Bein {i}: unvollstaendige oder ungueltige Angaben.")
    return errs


# ───────────────────────── Speicherung ─────────────────────────

def load_depot(path: Path) -> tuple[list[dict], str | None]:
    """Laedt das Depot. Rueckgabe: (combos, warnung). Bei unlesbarer Datei wird diese
    gesichert (nicht ueberschrieben) und ein leeres Depot + Warnung geliefert."""
    path = Path(path)
    if not path.exists():
        return [], None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        combos = data["combos"]
        if not isinstance(combos, list):
            raise ValueError("'combos' ist keine Liste")
        out = []
        for c in combos:
            out.append(make_combo(c.get("name", ""), c.get("expiry", ""), c.get("legs", []),
                                  c.get("note", ""), c.get("id"), c.get("created")))
        return out, None
    except Exception as e:                                   # unlesbar/beschaedigt
        stamp = time.strftime("%Y%m%d-%H%M%S")
        bad = path.with_name(f"{path.stem}.corrupt-{stamp}{path.suffix}")
        try:
            os.replace(path, bad)
            where = f"gesichert als {bad.name}"
        except OSError:
            where = "konnte nicht gesichert werden"
        return [], f"Depot-Datei war nicht lesbar ({e}) - {where}. Es wird mit leerem Depot gestartet."


def save_depot(path: Path, combos: list[dict]) -> None:
    """Schreibt das Depot atomar. Wirft OSError bei Fehlern (Aufrufer zeigt sie an)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": SCHEMA_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "combos": combos,
    }
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    if path.exists():
        try:
            shutil.copy2(path, path.with_name(path.name + ".bak"))
        except OSError:
            pass
    os.replace(tmp, path)


# ───────────────────────── Auszahlung bei Verfall ─────────────────────────

def _sign(leg: dict) -> float:
    return 1.0 if leg["side"] == "long" else -1.0


def entry_net(combo: dict) -> float:
    """Netto-Einstiegs-Cashflow in Punkten: Kredit positiv, Debit negativ."""
    return float(sum((-_sign(l)) * l["entry_price"] * l["qty"] for l in combo["legs"]))


def expiry_value(combo: dict, S) -> np.ndarray:
    """Wert der Position bei Verfall (ohne Einstiegspreis): long +intrinsic, short -intrinsic."""
    S = np.asarray(S, dtype=float)
    total = np.zeros_like(S, dtype=float)
    for leg in combo["legs"]:
        intr = np.maximum(leg["strike"] - S, 0.0) if leg["option_type"] == "put" \
            else np.maximum(S - leg["strike"], 0.0)
        total = total + _sign(leg) * leg["qty"] * intr
    return total


def pnl_at_expiry(combo: dict, S) -> np.ndarray:
    return entry_net(combo) + expiry_value(combo, S)


def payoff_profile(combo: dict) -> dict:
    """Exakte Kennzahlen des stueckweise linearen Verfalls-Payoffs (P&L seit Einstieg):
    breakevens (Liste), max_profit, max_loss (positiv; math.inf = unbegrenzt)."""
    f = lambda s: float(pnl_at_expiry(combo, np.array([s]))[0])
    bps = sorted({0.0} | {leg["strike"] for leg in combo["legs"]})
    vals = [f(b) for b in bps]
    slope_up = sum(_sign(l) * l["qty"] for l in combo["legs"] if l["option_type"] == "call")

    roots: list[float] = []
    for (a, fa), (b, fb) in zip(zip(bps, vals), zip(bps[1:], vals[1:])):
        if fa == 0.0:
            roots.append(a)
        elif fa * fb < 0.0:
            roots.append(a - fa * (b - a) / (fb - fa))
    if vals[-1] == 0.0:
        roots.append(bps[-1])
    if slope_up != 0.0:
        r = bps[-1] - vals[-1] / slope_up
        if r > bps[-1] + 1e-9:
            roots.append(r)

    uniq: list[float] = []
    for r in sorted(roots):
        if r > 0 and (not uniq or abs(r - uniq[-1]) > 1e-6):
            uniq.append(r)

    max_profit = float("inf") if slope_up > 0 else max(vals)
    max_loss = float("inf") if slope_up < 0 else max(-min(vals), 0.0)
    return {"breakevens": uniq, "max_profit": max_profit, "max_loss": max_loss}


# ───────────────────────── Levels heute (beliebige P&L-Funktion) ─────────────────────────

def find_levels(f, target: float, lo: float, hi: float, n: int = 1200) -> list[float]:
    """Alle Spot-Werte in [lo, hi], bei denen f(S) == target (Vorzeichenwechsel-Suche +
    Brent). f muss ein numpy-Array akzeptieren und ein Array gleicher Laenge liefern."""
    xs = np.linspace(lo, hi, n)
    ys = np.asarray(f(xs), dtype=float) - target
    roots: list[float] = []
    for i in range(n - 1):
        y0, y1 = ys[i], ys[i + 1]
        if not (np.isfinite(y0) and np.isfinite(y1)):
            continue
        if y0 == 0.0:
            roots.append(float(xs[i]))
        elif y0 * y1 < 0.0:
            g = lambda x: float(np.asarray(f(np.array([x])), dtype=float)[0]) - target
            roots.append(float(brentq(g, xs[i], xs[i + 1], xtol=1e-6)))
    return roots


def nearest_levels(levels: list[float], spot: float) -> tuple[float | None, float | None]:
    """Naechstes Level unter/gleich und ueber dem Spot."""
    below = [x for x in levels if x <= spot]
    above = [x for x in levels if x > spot]
    return (max(below) if below else None), (min(above) if above else None)


# ───────────────────────── Gewichtete Statistik ─────────────────────────

def weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    o = np.argsort(x)
    x, w = x[o], w[o]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, cw, x))


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * x) / np.sum(w))