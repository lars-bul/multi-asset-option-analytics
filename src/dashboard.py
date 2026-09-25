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

Diese Datei selbst tut vier Dinge:
  1. Instrument-Auswahl oben (SIEHE MULTI-INSTRUMENT-HINWEIS UNTEN) - gilt fuer
     die gesamte Seite.
  2. Daten laden + die immer sichtbare Candlestick-Uebersicht rendern
     (dashboard_common.render_overview) - fuer das GEWAEHLTE Instrument.
  3. Eine Reiterleiste oben anzeigen, mit der zwischen den Modi gewechselt wird.
  4. Je nach gewaehltem Modus GENAU EIN render()-Modul aufrufen.

WICHTIG zur Reiterleiste: es wird bewusst KEIN st.tabs() verwendet. Bei
st.tabs() fuehrt Streamlit den Code in JEDEM Tab bei jedem Rerun aus (auch
im nicht sichtbaren Tab) - das wuerde dazu fuehren, dass ALLE Modi ihre
Sidebar-Abschnitte gleichzeitig rendern und sich dort vermischen. Ein
horizontales st.radio() sieht optisch wie eine Reiterleiste aus, aktiviert
aber pro Rerun nur den Code des gewaehlten Modus - dadurch bleibt auch die
Sidebar sauber getrennt (siehe render()-Aufruf unten).

════════════════════════════════════════════════════════════════════════════
MULTI-INSTRUMENT-UMBAU - AKTUELLER STAND UND OFFENE ABHAENGIGKEIT
════════════════════════════════════════════════════════════════════════════
Oben steht jetzt ein Instrument-Auswaehler (SPX, Russell 2000, Nasdaq-100,
Nikkei 225, EuroStoxx50, Gold, Oel - siehe instruments.py). Die Auswahl steckt
in st.session_state['active_instrument'].

'Heutige Kette' (dashboard_livechain.py) und 'Depot-Analyse' (dashboard_depot.py)
sind VOLL instrumentenabhaengig umgebaut: eigene Trainingstabelle, eigener
Chain-Provider je Instrument, Fehlermeldung statt Absturz, wenn ein Instrument
keine Live-Kette hat (aktuell: Nikkei, EuroStoxx50).

'Iron Condor Backtest' (dashboard_ladder.py) und 'Analog-Tage-Matching'
(dashboard_analog.py) sind NOCH NICHT umgebaut - sie bekommen unveraendert nur
(df, daily_df, has_resolution) OHNE Instrument. Da 'df'/'daily_df' aber bereits
die Kursdaten des GEWAEHLTEN Instruments sind (siehe load_data(instr) unten),
funktionieren Backtest und Analog-Matching automatisch MIT JEDEM Instrument -
nur mit zwei bekannten Einschraenkungen, bis diese beiden Dateien im naechsten
Schritt ebenfalls generalisiert werden:
  1. Die Dividendenrendite-Vorgabe in IronCondorConfig ist weiterhin der feste
     SPX-Default (1.5%), nicht instr.dividend_yield - fuer Gold/Oel (~0%) also
     leicht zu hoch, bis dashboard_ladder.py das uebernimmt.
  2. dashboard_analog.py laedt seine eigene Trainingstabelle - aktuell noch
     der globale SPX-Pfad, unabhaengig vom oben gewaehlten Instrument.

BENOETIGTE AENDERUNG IN dashboard_common.py (lag mir beim Schreiben nicht vor -
bitte teilen, dann uebernehme ich das praezise statt zu raten):
    load_data(instr: InstrumentConfig) -> pd.DataFrame
        # statt load_data() - muss data/{instr.key}_daily.parquet laden
        # (siehe instruments.data_file(instr, "daily", DATA_DIR)).
    render_overview(df, has_resolution, instr: InstrumentConfig) -> pd.DataFrame
        # statt render_overview(df, has_resolution) - instr wird nur fuer die
        # Anzeige (z.B. Ueberschrift/Waehrung) gebraucht, die Chart-Logik
        # selbst bleibt unveraendert.
Bis dashboard_common.py entsprechend angepasst ist, LAEUFT DIESE DATEI NICHT.
"""
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from instruments import INSTRUMENTS, DEFAULT_INSTRUMENT, get_instrument
from dashboard_common import load_data, render_overview
import dashboard_ladder
import dashboard_analog
import dashboard_livechain
import dashboard_depot

st.set_page_config(page_title="Multi-Asset Options Dashboard", layout="wide")

st.title("Multi-Asset Options Dashboard")
st.caption("Lokales Analyse-Tool - laeuft nur bei dir auf localhost")

# ── Instrument-Auswahl (gilt fuer die gesamte Seite, alle Modi) ──
_instrument_keys = list(INSTRUMENTS.keys())
_instrument_labels = [INSTRUMENTS[k].label for k in _instrument_keys]

if "active_instrument" not in st.session_state:
    st.session_state["active_instrument"] = DEFAULT_INSTRUMENT

_current_idx = _instrument_keys.index(st.session_state["active_instrument"])
_chosen_label = st.selectbox("Instrument", _instrument_labels, index=_current_idx, key="instrument_select")
_active_key = _instrument_keys[_instrument_labels.index(_chosen_label)]

if _active_key != st.session_state["active_instrument"]:
    # Instrument gewechselt - Ergebnis-Caches der Modi NICHT unter dem neuen
    # Instrument weiterleben lassen (sonst zeigt z.B. 'Heutige Kette' kurzzeitig
    # noch die Condor-Suche des VORHERIGEN Instruments an).
    for _k in (
        "ic_trades", "ic_cfg_label",
        "livechain_table", "livechain_chosen_expiry", "livechain_chosen_dte_exact",
        "livechain_analog_results", "livechain_analog_key", "livechain_analog_attempt",
        "livechain_condor", "livechain_condor_attempt",
        "depot_results",
    ):
        st.session_state.pop(_k, None)
st.session_state["active_instrument"] = _active_key
instr = get_instrument(_active_key)

df = load_data(instr)
if df.empty:
    st.warning(
        f"Noch keine Daten fuer {instr.label} gefunden. Einmal im Terminal ausfuehren:\n\n"
        f"`python src/data_fetch.py {instr.key}`\n\n"
        "und diese Seite danach neu laden."
    )
    st.stop()

has_resolution = "resolution" in df.columns
_t0 = time.perf_counter()
daily_df = render_overview(df, has_resolution, instr)
_t_overview = time.perf_counter() - _t0

st.divider()


MODES = {
    "📊 Iron Condor Backtest": dashboard_ladder,
    "🔍 Analog-Tage-Matching": dashboard_analog,
    "⚡ Heutige Kette": dashboard_livechain,
    "💼 Depot-Analyse": dashboard_depot,
}
# Modi, die bereits vollstaendig auf Multi-Instrument umgestellt sind (bekommen
# 'instr' als zusaetzliches Argument). Die anderen laufen unveraendert mit
# (df, daily_df, has_resolution) - siehe Moduldokstring oben.
_INSTRUMENT_AWARE_MODES = {"⚡ Heutige Kette", "💼 Depot-Analyse"}

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
if active_mode_label in _INSTRUMENT_AWARE_MODES:
    MODES[active_mode_label].render(df, daily_df, has_resolution, instr)
else:
    MODES[active_mode_label].render(df, daily_df, has_resolution)
_t_mode = time.perf_counter() - _t1

# Diagnose: wie lange hat dieser Seitenaufbau gedauert? (Uebersicht laeuft bei JEDEM Klick, egal welcher Modus.)
st.caption(f"⏱ Seitenaufbau: Kursuebersicht {_t_overview:.1f} s · Modus '{active_mode_label}' {_t_mode:.1f} s")