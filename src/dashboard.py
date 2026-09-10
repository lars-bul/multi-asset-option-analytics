"""
Lokales Dashboard - laeuft nur auf deinem Rechner.

Start:
    streamlit run src/dashboard.py

Struktur (bewusst in mehrere kleine Module aufgeteilt, damit man einen Modus
aendern kann, ohne den Rest lesen zu muessen):
  - dashboard_common.py  Daten laden, Candlestick-Uebersicht, kleine UI-Helfer
                          (von BEIDEN Modi genutzt)
  - dashboard_ladder.py  Modus 1: Iron-Condor-/Multi-Leg-Ladder-Backtest
  - dashboard_analog.py  Modus 2: Analog-Tage-Matching

Diese Datei selbst tut nur drei Dinge:
  1. Daten laden + die immer sichtbare Candlestick-Uebersicht rendern
     (dashboard_common.render_overview).
  2. Eine Reiterleiste oben anzeigen, mit der zwischen den zwei Modi
     gewechselt wird.
  3. Je nach gewaehltem Modus GENAU EIN render()-Modul aufrufen.

WICHTIG zur Reiterleiste: es wird bewusst KEIN st.tabs() verwendet. Bei
st.tabs() fuehrt Streamlit den Code in JEDEM Tab bei jedem Rerun aus (auch
im nicht sichtbaren Tab) - das wuerde dazu fuehren, dass BEIDE Modi ihre
Sidebar-Abschnitte gleichzeitig rendern und sich dort vermischen. Ein
horizontales st.radio() sieht optisch wie eine Reiterleiste aus, aktiviert
aber pro Rerun nur den Code des gewaehlten Modus - dadurch bleibt auch die
Sidebar sauber getrennt (siehe render()-Aufruf unten).
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dashboard_common import load_data, render_overview
import dashboard_ladder
import dashboard_analog

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
daily_df = render_overview(df, has_resolution)

st.divider()

# ── Reiterleiste: waehlt den aktiven Modus ──
MODES = {
    "📊 Iron Condor Backtest": dashboard_ladder,
    "🔍 Analog-Tage-Matching": dashboard_analog,
}
active_mode_label = st.radio(
    "Modus", list(MODES.keys()), horizontal=True, label_visibility="collapsed",
)
st.divider()

# Nur der aktive Modus rendert - dadurch bekommt auch nur er einen eigenen
# Sidebar-Abschnitt (siehe Modulbeschreibung oben).
MODES[active_mode_label].render(df, daily_df, has_resolution)