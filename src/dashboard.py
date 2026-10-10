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
  - dashboard_pair.py      Modus 5: Pair-Trade zwischen zwei Instrumenten
                            (z.B. Short SPX / Long NDX) - braucht IMMER zwei
                            Instrumente gleichzeitig, waehlt sie deshalb selbst
                            in der eigenen Sidebar statt den EINEN oben
                            gewaehlten 'active_instrument' zu verwenden.

Diese Datei selbst tut vier Dinge:
  1. STICKY KOPFZEILE (Instrument-Auswahl + 'In neuem Tab oeffnen'-Button) -
     allgemeines Layout fuer ALLE Modi, bleibt beim Scrollen oben stehen.
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

STICKY KOPFZEILE - Technik: Streamlit hat kein 'sticky'. Die Kopfzeile liegt in
einem st.container(key='sticky_header') (CSS-Klasse .st-key-sticky_header; fuer
aeltere Versionen ohne key= greift ein :has()-Selektor auf einen Marker).
Per CSS: position: sticky + Hintergrund. Offset anpassen: 'top' in _STICKY_CSS.

NEUER TAB: der Button oeffnet dieselbe App mit denselben URL-Parametern:
  instr = aktives Instrument, mode = aktiver Modus; im Pair-Modus zusaetzlich
  pt_lead / pt_follower / pt_expiry (aus den Widget-Keys des Pair-Modus).
Beim Start liest diese Datei 'instr' und 'mode' wieder ein. Der Button wird
einmal VOR dem Modus-Rendering (sofort sichtbar, auch bei langen Berechnungen)
und einmal danach (aktueller Stand) in denselben Platzhalter geschrieben.

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

'Pair-Trade' (dashboard_pair.py) ist ein SONDERFALL: es braucht IMMER
ZWEI Instrumente gleichzeitig (Lead/Short + Follower/Long) und waehlt beide
DESHALB SELBST in seiner eigenen Sidebar, unabhaengig vom oben gewaehlten
'active_instrument' (der nur als Vorauswahl-Vorschlag fuer die Lead-Seite
dient). Es bekommt deshalb WEDER (df, daily_df, has_resolution) NOCH
zusaetzlich instr, sondern NUR instr (als Vorschlag) - siehe _PAIR_MODES unten.
"""
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from instruments import INSTRUMENTS, DEFAULT_INSTRUMENT, get_instrument
from dashboard_common import load_data, render_overview
import dashboard_ladder
import dashboard_analog
import dashboard_livechain
import dashboard_depot
import dashboard_pair

st.set_page_config(page_title="Multi-Asset Options Dashboard", layout="wide")


MODES = {
    "📊 Iron Condor Backtest": dashboard_ladder,
    "🔍 Analog-Tage-Matching": dashboard_analog,
    "⚡ Heutige Kette": dashboard_livechain,
    "💼 Depot-Analyse": dashboard_depot,
    "🔀 Pair-Trade": dashboard_pair,
}
# Modi, die bereits vollstaendig auf Multi-Instrument umgestellt sind (bekommen
# 'instr' als zusaetzliches Argument). Die anderen laufen unveraendert mit
# (df, daily_df, has_resolution) - siehe Moduldokstring oben.
_INSTRUMENT_AWARE_MODES = {"⚡ Heutige Kette", "💼 Depot-Analyse"}

# Pair-Trade ist ein SONDERFALL (siehe Moduldokstring oben): braucht IMMER
# zwei Instrumente, waehlt beide selbst in der eigenen Sidebar - bekommt
# deshalb NUR 'instr' (als Vorauswahl-Vorschlag), NICHT (df, daily_df,
# has_resolution), da die Kursdaten fuer BEIDE beteiligten Instrumente ohnehin
# separat innerhalb von dashboard_pair.py geladen werden muessen.
_PAIR_MODES = {"🔀 Pair-Trade"}

# Standard-Modus beim Oeffnen der App: "Heutige Kette" - dadurch starten Kette, Analog-Tabelle und
# Condor-Suche automatisch beim Start. Anderen Standard-Tab waehlen: DEFAULT_MODE aendern.
DEFAULT_MODE = "⚡ Heutige Kette"


# ── URL-Parameter / Neuer-Tab-Button ─────────────────────────────────

def _qp(name: str):
    """Einzelnen Query-Parameter als String lesen (oder None)."""
    try:
        val = st.query_params.get(name)
    except Exception:
        return None
    if isinstance(val, list):
        val = val[0] if val else None
    return val


def _new_tab_url(instr_key: str, mode_label: str) -> str:
    """Relative URL (nur Query-String) auf dieselbe App: aktuelle Parameter +
    Instrument + Modus; im Pair-Modus zusaetzlich Lead/Follower/Verfall."""
    try:
        params = {k: (v[0] if isinstance(v, list) else v) for k, v in dict(st.query_params).items()}
    except Exception:
        params = {}
    params["instr"] = instr_key
    params["mode"] = mode_label
    if mode_label in _PAIR_MODES:
        for k in ("pt_lead", "pt_follower"):
            if st.session_state.get(k):
                params[k] = str(st.session_state[k])
        exp = st.session_state.get("pt_expiry")
        if exp is not None:
            params["pt_expiry"] = str(pd.Timestamp(exp).date())
    return "?" + urlencode(params)


def _fill_new_tab_slot(slot, instr_key: str, mode_label: str) -> None:
    url = _new_tab_url(instr_key, mode_label)
    label = "In neuem Tab oeffnen"
    with slot.container():
        link_button = getattr(st, "link_button", None)
        if link_button is not None:
            link_button(label, url, use_container_width=True)
        else:  # aeltere Streamlit-Versionen
            st.markdown(f'<a href="{url}" target="_blank" rel="noopener">{label} ↗</a>', unsafe_allow_html=True)


# ── Sticky-Kopfzeile: CSS ────────────────────────────────────────────

_STICKY_CSS = """
<style>
.st-key-sticky_header,
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .sticky-marker) {
    position: sticky;
    top: 2.875rem;            /* Abstand zur Streamlit-Toolbar - bei Bedarf anpassen */
    z-index: 999;
    background: var(--background-color, var(--st-background-color, Canvas));
    padding-top: 0.4rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(128,128,128,0.25);
}
.sticky-marker { display: none; }
</style>
"""
st.markdown(_STICKY_CSS, unsafe_allow_html=True)

st.title("Multi-Asset Options Dashboard")
st.caption("Lokales Analyse-Tool - laeuft nur bei dir auf localhost")

# ── Instrument-Vorbelegung aus URL (z.B. nach 'In neuem Tab oeffnen') ──
_instrument_keys = list(INSTRUMENTS.keys())
_instrument_labels = [INSTRUMENTS[k].label for k in _instrument_keys]

if "active_instrument" not in st.session_state:
    _q_instr = _qp("instr")
    st.session_state["active_instrument"] = _q_instr if _q_instr in INSTRUMENTS else DEFAULT_INSTRUMENT

# ── STICKY KOPFZEILE (fuer alle Modi): Instrument + 'In neuem Tab oeffnen' ──
try:
    _header = st.container(key="sticky_header")
except TypeError:  # aeltere Streamlit-Version ohne key= -> Marker + :has()-Selektor
    _header = st.container()

_current_idx = _instrument_keys.index(st.session_state["active_instrument"])
with _header:
    st.markdown('<span class="sticky-marker"></span>', unsafe_allow_html=True)
    try:
        _hdr_select_col, _hdr_slot_col = st.columns([4, 1], vertical_alignment="bottom")
    except TypeError:
        _hdr_select_col, _hdr_slot_col = st.columns([4, 1])
    _chosen_label = _hdr_select_col.selectbox(
        "Instrument", _instrument_labels, index=_current_idx, key="instrument_select",
    )
    _header_slot = _hdr_slot_col.empty()
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
        "pair_result",
    ):
        st.session_state.pop(_k, None)
st.session_state["active_instrument"] = _active_key
instr = get_instrument(_active_key)

df = load_data(instr)
if df.empty:
    _fill_new_tab_slot(_header_slot, instr.key, st.session_state.get("mode_radio", DEFAULT_MODE))
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

# Modus-Vorbelegung aus URL, nur einmal je Session (danach fuehrt der Widget-Key)
if "mode_radio" not in st.session_state:
    _q_mode = _qp("mode")
    st.session_state["mode_radio"] = _q_mode if _q_mode in MODES else DEFAULT_MODE
active_mode_label = st.radio(
    "Modus", list(MODES.keys()), horizontal=True, label_visibility="collapsed", key="mode_radio",
)
st.divider()

# URL spiegelt Instrument + Modus (Kopieren / Reload / Neuer Tab liefern dieselbe Ansicht)
try:
    st.query_params["instr"] = instr.key
    st.query_params["mode"] = active_mode_label
except Exception:
    pass

# Button sofort anzeigen (auch waehrend langer Berechnungen) ...
_fill_new_tab_slot(_header_slot, instr.key, active_mode_label)

# Nur der aktive Modus rendert - dadurch bekommt auch nur er einen eigenen
# Sidebar-Abschnitt (siehe Modulbeschreibung oben).
_t1 = time.perf_counter()
if active_mode_label in _INSTRUMENT_AWARE_MODES:
    MODES[active_mode_label].render(df, daily_df, has_resolution, instr)
elif active_mode_label in _PAIR_MODES:
    MODES[active_mode_label].render(instr)
else:
    MODES[active_mode_label].render(df, daily_df, has_resolution)
_t_mode = time.perf_counter() - _t1

# ... und danach mit dem aktuellen Stand (z.B. neu gewaehlter Pair-Verfall) ueberschreiben.
_fill_new_tab_slot(_header_slot, instr.key, active_mode_label)

# Diagnose: wie lange hat dieser Seitenaufbau gedauert? (Uebersicht laeuft bei JEDEM Klick, egal welcher Modus.)
st.caption(f"⏱ Seitenaufbau: Kursuebersicht {_t_overview:.1f} s · Modus '{active_mode_label}' {_t_mode:.1f} s")