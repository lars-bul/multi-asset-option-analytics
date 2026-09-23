"""
Lokales Dashboard - laeuft nur auf deinem Rechner.

Start:
    streamlit run src/dashboard.py

Struktur (bewusst in mehrere kleine Module aufgeteilt, damit man einen Modus
aendern kann, ohne den Rest lesen zu muessen):
  - dashboard_common.py    Daten laden, Candlestick-Uebersicht, kleine UI-Helfer
                            (von ALLEN Modi genutzt)
  - dashboard_ladder.py    Modus 1: Iron-Condor-/Multi-Leg-Ladder-Backtest
  - dashboard_analog.py    Modus 2: Analog-Tage-Matching
  - dashboard_livechain.py Modus 3: Heutige Kette + Analog-Tabelle + Condor-Suche
  - dashboard_depot.py     Modus 4: Depot-Analyse (eigene Positionen, dauerhaft
                            gespeichert in data/depot.json)

Diese Datei selbst tut nur drei Dinge:
  1. Daten laden + die immer sichtbare Candlestick-Uebersicht rendern
     (dashboard_common.render_overview).
  2. Eine Reiterleiste oben anzeigen, mit der zwischen den Modi gewechselt wird.
  3. Je nach gewaehltem Modus GENAU EIN render()-Modul aufrufen.

WICHTIG zur Reiterleiste: es wird bewusst KEIN st.tabs() verwendet. Bei
st.tabs() fuehrt Streamlit den Code in JEDEM Tab bei jedem Rerun aus (auch
im nicht sichtbaren Tab) - das wuerde dazu fuehren, dass ALLE Modi ihre
Sidebar-Abschnitte gleichzeitig rendern und sich dort vermischen. Ein
horizontales st.radio() sieht optisch wie eine Reiterleiste aus, aktiviert
aber pro Rerun nur den Code des gewaehlten Modus - dadurch bleibt auch die
Sidebar sauber getrennt (siehe render()-Aufruf unten).
"""
import sys
import time
from pathlib import Path

import streamlit as st
import dashboard_livechain
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dashboard_common import load_data, render_overview
import dashboard_ladder
import dashboard_analog
import dashboard_depot

st.set_page_config(page_title="SPX Options Dashboard", layout="wide")

st.title("SPX Options Dashboard")
st.caption("Lokales Analyse-Tool - laeuft nur bei dir auf localhost")

df = load_data()
if df.empty:
    st.warning(
        "Noch keine Daten gefunden. Einmal im Terminal ausfuehren:\n\n"
        "`python src/data_fetch.py`\n\n"
        "und diese Seite danach neu laden."
    )
    st.stop()

has_resolution = "resolution" in df.columns
_t0 = time.perf_counter()
daily_df = render_overview(df, has_resolution)
_t_overview = time.perf_counter() - _t0

st.divider()


MODES = {
    "📊 Iron Condor Backtest": dashboard_ladder,
    "🔍 Analog-Tage-Matching": dashboard_analog,
    "⚡ Heutige Kette": dashboard_livechain,
    "💼 Depot-Analyse": dashboard_depot,
}
# Standard-Modus beim Oeffnen der App: "Heutige Kette" - dadurch starten Kette, Analog-Tabelle und
# Condor-Suche automatisch beim Start. Anderen Standard-Tab waehlen: DEFAULT_MODE aendern.
DEFAULT_MODE = "⚡ Heutige Kette"
active_mode_label = st.radio(
    "Modus", list(MODES.keys()), horizontal=True, label_visibility="collapsed",
    index=list(MODES.keys()).index(DEFAULT_MODE),
)
st.divider()

# Nur der aktive Modus rendert - dadurch bekommt auch nur er einen eigenen
# Sidebar-Abschnitt (siehe Modulbeschreibung oben).
_t1 = time.perf_counter()
MODES[active_mode_label].render(df, daily_df, has_resolution)
_t_mode = time.perf_counter() - _t1

# Diagnose: wie lange hat dieser Seitenaufbau gedauert? (Uebersicht laeuft bei JEDEM Klick, egal welcher Modus.)
st.caption(f"⏱ Seitenaufbau: Kursuebersicht {_t_overview:.1f} s · Modus '{active_mode_label}' {_t_mode:.1f} s")