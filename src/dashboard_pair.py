"""
MODUS 5: Pair-Trade-Suche (Backend: pair_trade_search.py).

SONDERFALL (siehe dashboard.py): braucht zwei Instrumente (Lead + Follower),
waehlt beide selbst in der eigenen Sidebar und bekommt nur 'instr' (Vorschlag
fuer die Lead-Seite) plus 'header_slot' (Platzhalter in der Kopfzeile von
dashboard.py, rechts neben dem Instrument-Auswaehler).

LAUFZEIT: fuer beide Instrumente werden die verfuegbaren Verfallstage aus der
Live-Kette gelesen; angeboten werden nur Verfallstage, die es bei BEIDEN gibt
(gleiches Datum). Beide Seiten werden mit exakt diesem Verfall gebaut.

BEIN-ROLLEN (KORRIGIERT):
  Lead     = Credit-Spread: inneres Bein SHORT, aeusseres Bein (Fluegel) LONG.
  Follower = Debit-Spread:  inneres Bein LONG,  aeusseres Bein (Fluegel) SHORT.
  Die Spalten heissen daher: 'Lead Short %' / 'Lead Long (Fluegel) %' und
  'Follower Long %' / 'Follower Short (Fluegel) %'.

NEUER TAB: 'In neuem Tab oeffnen' sitzt in der KOPFZEILE (nicht in der Sidebar)
und oeffnet dieselbe Anwendung in einem neuen Browser-Tab (eigene Streamlit-
Session, eigenes Ergebnis). Lead, Follower und Verfall werden ueber URL-
Parameter (pt_lead/pt_follower/pt_expiry) uebernommen; bestehende URL-Parameter
bleiben erhalten (falls dashboard.py den Modus per Query-Parameter steuert, wird
er also mitgenommen). Der Button wird in den von dashboard.py uebergebenen
Platzhalter 'header_slot' gerendert.

ERGEBNIS-PERSISTENZ: das Ergebnis liegt in st.session_state['pair_result']
(dashboard.py loescht den Schluessel beim Instrumentwechsel). Sortieren/
Filtern/Spaltenklick rechnet NICHTS neu und laesst die Tabelle stehen; neu
berechnet wird automatisch nur, wenn sich Lead/Follower/Laufzeit/Parameter aendern. Die Ergebnisanzeige ist ein
st.fragment (falls verfuegbar) - Sortier-Widgets loesen dann auch keinen
Neuaufbau der ganzen Seite aus.

METHODIK: Lead bestimmt den Aehnlichkeitsraum, Follower wird an denselben
Kalendertagen mitbewertet; Robustheit NUR auf dem Gesamtspread; alle
Kombinationen werden angezeigt; Max-Verlust strukturell exakt (siehe
pair_trade_search.py). Zusaetzlich: 'Max-Verlust Oberseite' (beide Basiswerte
steigen stark) und 'Max-Verlust Unterseite' (beide fallen auf 0) - BEIDE mit
demselben Vorzeichen: positiv = Verlust, negativ = Gewinn.
"""
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from instruments import INSTRUMENTS, InstrumentConfig, get_instrument
import market_data
from chain_delta import build_symmetric_distance_table
from dashboard_common import load_data
from dashboard_livechain import _find_today_analogs, _ensure_training_table, _bounds_cached
from pair_trade_search import PairTradeConfig, pair_analog_moves, evaluate_pair_grid

pd.set_option("styler.render.max_elements", 5_000_000)

_DEFAULT_DISTANCES = "0,1,2,3,5,8,12"
_R_FREE = 0.0

_REASON_LABELS = {
    "": "geprueft",
    "ungueltige_bepreisung": "ausgeschl.: ungueltiger Bid/Ask-Spread",
}

# Anzeige-Label -> (Spalte im Ergebnis, Standard: aufsteigend?)
_SORT_OPTIONS = {
    "Verlustwahrscheinlichkeit": ("loss_prob_pct", True),
    "Vollverlust %": ("full_loss_pct", True),
    "Kredit (netto)": ("net_cashflow", False),
    "Rendite/Risiko": ("ror", False),
    "E[P&L]": ("mean", False),
    "t (HAC)": ("tstat", False),
    "Max-Verlust": ("max_loss", True),
    "Max-Verlust Oberseite": ("max_loss_up", True),
    "Max-Verlust Unterseite": ("max_loss_down", True),
}

_fragment = getattr(st, "fragment", None) or (lambda f: f)


# ── Neuer Tab / URL-Parameter ────────────────────────────────────────

def _qp(name: str):
    """Einzelnen Query-Parameter als String lesen (oder None)."""
    try:
        val = st.query_params.get(name)
    except Exception:
        return None
    if isinstance(val, list):
        val = val[0] if val else None
    return val


def _new_tab_url(lead_key: str, follower_key: str, expiry) -> str:
    """Relative URL (nur Query-String) auf dieselbe App; bestehende Parameter
    bleiben erhalten, die Pair-Parameter werden gesetzt/ueberschrieben."""
    try:
        params = {k: (v[0] if isinstance(v, list) else v) for k, v in dict(st.query_params).items()}
    except Exception:
        params = {}
    params["pt_lead"] = lead_key
    params["pt_follower"] = follower_key
    if expiry is not None:
        params["pt_expiry"] = str(pd.Timestamp(expiry).date())
    return "?" + urlencode(params)


def _new_tab_button(lead_key: str, follower_key: str, expiry, header_slot=None) -> None:
    """Button 'In neuem Tab oeffnen'. Wird in den Kopfzeilen-Platzhalter von
    dashboard.py gerendert (rechts neben dem Instrument-Auswaehler); ohne
    Platzhalter (z.B. Aufruf ausserhalb von dashboard.py) direkt an der aktuellen Stelle."""
    url = _new_tab_url(lead_key, follower_key, expiry)
    label = "In neuem Tab oeffnen"
    target = header_slot.container() if header_slot is not None else st.container()
    with target:
        if header_slot is not None:
            # Abstandshalter, damit der Button auf Hoehe der Selectbox (nicht des Labels) sitzt
            st.markdown("<div style='height:1.65rem'></div>", unsafe_allow_html=True)
        link_button = getattr(st, "link_button", None)
        if link_button is not None:
            link_button(label, url, use_container_width=True)
        else:  # aeltere Streamlit-Versionen
            st.markdown(f'<a href="{url}" target="_blank" rel="noopener">{label} ↗</a>', unsafe_allow_html=True)


# ── Kette / Verfallstage ─────────────────────────────────────────────

def _get_snapshot(instr: InstrumentConfig):
    """Kette je Instrument laden und in session_state halten; automatisch
    erneuert, wenn aelter als 5 Minuten (kein Button noetig)."""
    key = f"pair_snap_{instr.key}"
    cached = st.session_state.get(key)
    if cached is None or (time.time() - cached[0]) > 300:
        cached = (time.time(), market_data.get(instr, _R_FREE, instr.dividend_yield, 90.0))
        st.session_state[key] = cached
    return cached[1]


def _expiries(snap) -> list[pd.Timestamp]:
    enriched = snap["enriched"]
    if enriched.empty:
        return []
    return sorted(pd.to_datetime(enriched["expiry"]).dt.normalize().unique().tolist())


def _dte(expiry: pd.Timestamp) -> int:
    return max((pd.Timestamp(expiry) - pd.Timestamp.now()).days, 1)


def _distance_table_for(instr: InstrumentConfig, daily_df: pd.DataFrame, snap, expiry: pd.Timestamp,
                        distances: list[float]):
    """Symmetrische Distanz-Tabelle (inkl. 'iv_pair_pct') fuer EIN Instrument
    am GEWAEHLTEN Verfall. Rueckgabe: (table, spot) oder (None, None)."""
    enriched = snap["enriched"]
    exp_col = pd.to_datetime(enriched["expiry"]).dt.normalize()
    sub = enriched[exp_col == pd.Timestamp(expiry).normalize()]
    if sub.empty:
        return None, None
    dte_days = _dte(expiry)
    bounds = _bounds_cached(daily_df, max(2, dte_days), 252, instr)
    hv_valid = bounds["hv"].dropna()
    if hv_valid.empty:
        return None, None
    flat_hv = float(hv_valid.iloc[-1])
    table = build_symmetric_distance_table(
        enriched, pd.Timestamp(sub["expiry"].iloc[0]), snap["spot"], flat_hv, dte_days / 365.0,
        _R_FREE, instr.dividend_yield, distances,
    )
    return table, snap["spot"]


# ── Ergebnis-Anzeige (bleibt stehen, rechnet nichts neu) ─────────────

def _highlight(row):
    if row["Kandidat"]:
        return ["background-color: rgba(46,160,67,0.25)"] * len(row)
    if not str(row["Status"]).startswith("geprueft"):
        return ["color: rgba(128,128,128,0.8)"] * len(row)
    return [""] * len(row)


@_fragment
def _render_result() -> None:
    store = st.session_state.get("pair_result")
    if store is None:
        return
    result: pd.DataFrame = store["result"]
    meta = store["meta"]

    st.caption(meta["header"])
    n_total, n_tested, n_pass = len(result), int(result["tested"].sum()), int(result["passes"].sum())
    reasons = result.loc[~result["tested"], "exclude_reason"].value_counts()
    reason_txt = " · ".join(f"{_REASON_LABELS.get(k, k)}: {v}" for k, v in reasons.items()) or "keine"
    st.success(
        f"{n_total} Kombinationen untersucht · {n_tested} vollstaendig geprueft · {n_pass} bestehen alle "
        f"Gesamtspread-Kriterien (Train>0, Val>0, t>{meta['t_min']:g}, Tail)."
    )
    st.caption(f"Ausgeschlossen: {reason_txt}")

    c1, c2, c3, c4, c5 = st.columns([2, 1.2, 1.4, 1.6, 1.4])
    sort_label = c1.selectbox("Sortieren nach", list(_SORT_OPTIONS), key="pt_view_sort")
    desc_default = not _SORT_OPTIONS[sort_label][1]
    descending = c2.checkbox("Absteigend", value=desc_default, key=f"pt_view_desc_{sort_label}")
    cand_first = c3.checkbox("Kandidaten zuerst", value=True, key="pt_view_candfirst")
    only_tested = c4.checkbox("Ausgeschlossene ausblenden", value=False, key="pt_view_onlytested")
    only_pass = c5.checkbox("Nur Kandidaten", value=False, key="pt_view_onlypass")

    col = _SORT_OPTIONS[sort_label][0]
    view = result
    if only_tested:
        view = view[view["tested"]]
    if only_pass:
        view = view[view["passes"]]
    if cand_first:
        view = view.sort_values(["tested", "passes", col], ascending=[False, False, not descending], na_position="last")
    else:
        view = view.sort_values(col, ascending=not descending, na_position="last")

    show = view.copy()
    show["Status"] = show["exclude_reason"].map(lambda r: _REASON_LABELS.get(r, r))
    show["ror"] = show["ror"] * 100.0
    show = show[[
        "structure", "Status", "loss_prob_pct", "full_loss_pct", "lead_full_loss_pct", "passes",
        "net_cashflow", "lead_net_cashflow", "follower_net_cashflow",
        "c_train", "c_val", "c_t", "c_tail",
        "ror", "mean", "tstat", "e_train", "e_val",
        "max_loss", "max_loss_struct", "max_loss_hist", "max_loss_up", "max_loss_down",
        "lead_short_distance", "lead_long_distance", "vol_lead_pct",
        "follower_long_distance", "follower_short_distance", "vol_follower_pct",
        "follower_qty",
    ]].rename(columns={
        "structure": "Struktur", "loss_prob_pct": "Verlust-Wahrsch. %", "full_loss_pct": "Vollverlust %",
        "lead_full_loss_pct": "Lead-Vollverlust %", "passes": "Kandidat",
        "net_cashflow": "Kredit (netto)", "lead_net_cashflow": "Lead-Kredit",
        "follower_net_cashflow": "Follower (Debit = neg.)",
        "c_train": "Train>0", "c_val": "Val>0", "c_t": "t ok", "c_tail": "Tail ok",
        "ror": "Rendite/Risiko %", "mean": "E[P&L]", "tstat": "t (HAC)", "e_train": "E Train", "e_val": "E Val",
        "max_loss": "Max-Verlust (verwendet)", "max_loss_struct": "Max-Verlust strukturell",
        "max_loss_hist": "Schlechtester Analog-Tag",
        "max_loss_up": "Max-Verlust Oberseite", "max_loss_down": "Max-Verlust Unterseite",
        # Lead: innen = SHORT, aussen (Fluegel) = LONG
        "lead_short_distance": "Lead Short %", "lead_long_distance": "Lead Long (Fluegel) %",
        "vol_lead_pct": "Lead-IV %",
        # Follower: innen = LONG, aussen (Fluegel) = SHORT
        "follower_long_distance": "Follower Long %", "follower_short_distance": "Follower Short (Fluegel) %",
        "vol_follower_pct": "Follower-IV %",
        "follower_qty": "Follower-Menge",
    })
    st.dataframe(
        show.style.apply(_highlight, axis=1).format({
            "Verlust-Wahrsch. %": "{:.1f}", "Vollverlust %": "{:.1f}", "Lead-Vollverlust %": "{:.1f}",
            "Kredit (netto)": "{:+.2f}", "Lead-Kredit": "{:+.2f}", "Follower (Debit = neg.)": "{:+.2f}",
            "Rendite/Risiko %": "{:+.1f}", "E[P&L]": "{:+.2f}", "t (HAC)": "{:.2f}",
            "E Train": "{:+.2f}", "E Val": "{:+.2f}",
            "Max-Verlust (verwendet)": "{:.2f}", "Max-Verlust strukturell": "{:.2f}",
            "Schlechtester Analog-Tag": "{:.2f}",
            "Max-Verlust Oberseite": "{:+.2f}", "Max-Verlust Unterseite": "{:+.2f}",
            "Lead-IV %": "{:.1f}",
            "Follower-IV %": "{:.1f}", "Follower-Menge": "{:.3f}",
        }, na_rep="–"),
        use_container_width=True, hide_index=True, height=650,
    )
    st.caption(
        f"{len(view)} Zeilen. Spaltenkoepfe sind direkt anklickbar (sortiert nur die Anzeige). "
        "BEIN-ROLLEN: Lead = Credit-Spread - 'Lead Short %' = verkauftes (inneres, naeher am Spot) Bein, "
        "'Lead Long (Fluegel) %' = gekauftes (aeusseres) Schutzbein. Follower = Debit-Spread - "
        "'Follower Long %' = gekauftes (inneres) Bein, 'Follower Short (Fluegel) %' = verkauftes (aeusseres) Bein. "
        "Alle Distanzen in % vom Spot. "
        "'Kredit (netto)' = Lead-Kredit + Follower-Debit (negativ), in Geld je Lead-Kontrakt. "
        "'Verlust-Wahrsch.' = gewichteter Anteil Analog-Tage mit Gesamt-P&L < 0. "
        "'Vollverlust %' = Anteil Analog-Tage, an denen der Gesamtspread den verwendeten Max-Verlust "
        "erreicht (bei strukturellem Max-Verlust praktisch immer ~0, da er Lead- und Follower-Extrem "
        "gleichzeitig annimmt). 'Lead-Vollverlust %' = Anteil Tage, an denen der Lead-Spread allein "
        "seinen vollen Max-Verlust erreicht (reine Info-Spalte, kein Kriterium). "
        "'Max-Verlust Oberseite' = Verlust der Gesamtposition, wenn BEIDE Basiswerte stark steigen "
        "(jedes Bein im Extremwert); 'Max-Verlust Unterseite' = Verlust, wenn BEIDE Basiswerte auf 0 "
        "fallen. Beide mit demselben Vorzeichen: positiv = Verlust, negativ = Gewinn. Das sind "
        "Randwerte, NICHT der Max-Verlust der Position: bei konvexen Follower-Fluegeln koennen sie "
        "stark im Gewinn liegen, waehrend der strukturelle Max-Verlust klein bleibt."
    )


# ── Hauptfunktion ────────────────────────────────────────────────────

def render(instr: InstrumentConfig, header_slot=None) -> None:
    st.header("Pair-Trade-Suche")
    st.caption(
        "Lead bestimmt die Analog-Tage, der Follower wird an denselben Kalendertagen mitbewertet. "
        "Robustheit nur auf dem Gesamtspread. "
        "Alle untersuchten Kombinationen werden angezeigt. Berechnung startet automatisch."
    )

    providers = [k for k in INSTRUMENTS if INSTRUMENTS[k].chain_provider is not None]
    if len(providers) < 2:
        st.warning("Es werden mindestens zwei Instrumente mit Live-Kette benoetigt.")
        return

    # Vorbelegung aus URL-Parametern (z.B. nach 'In neuem Tab oeffnen'), nur einmal je Session
    # und nur, solange der Widget-Key noch nicht existiert.
    if "pt_lead" not in st.session_state and _qp("pt_lead") in providers:
        st.session_state["pt_lead"] = _qp("pt_lead")
    if "pt_follower" not in st.session_state:
        seed_lead = st.session_state.get("pt_lead", instr.key if instr.key in providers else providers[0])
        seed_follower = _qp("pt_follower")
        if seed_follower in providers and seed_follower != seed_lead:
            st.session_state["pt_follower"] = seed_follower

    with st.sidebar:
        st.divider()
        st.header("Pair-Trade")
        lead_idx = providers.index(instr.key) if instr.key in providers else 0
        lead_kwargs = {} if "pt_lead" in st.session_state else {"index": lead_idx}
        lead_key = st.selectbox("Lead-Instrument", providers,
                                format_func=lambda k: INSTRUMENTS[k].label, key="pt_lead", **lead_kwargs)
        follower_opts = [k for k in providers if k != lead_key]
        # Gespeicherter Follower nicht mehr in den Optionen (z.B. Lead gewechselt) -> verwerfen
        if st.session_state.get("pt_follower") not in follower_opts:
            st.session_state.pop("pt_follower", None)
        follower_key = st.selectbox("Follower-Instrument", follower_opts,
                                    format_func=lambda k: INSTRUMENTS[k].label, key="pt_follower")

        lead_instr, follower_instr = get_instrument(lead_key), get_instrument(follower_key)
        with st.spinner("Lade Ketten..."):
            lead_snap, follower_snap = _get_snapshot(lead_instr), _get_snapshot(follower_instr)
        common = sorted(set(_expiries(lead_snap)) & set(_expiries(follower_snap)))
        expiry = None
        if common:
            if "pt_expiry" not in st.session_state and _qp("pt_expiry"):
                match = [d for d in common if str(pd.Timestamp(d).date()) == _qp("pt_expiry")]
                if match:
                    st.session_state["pt_expiry"] = match[0]
            if st.session_state.get("pt_expiry") not in common:
                st.session_state.pop("pt_expiry", None)
            expiry = st.selectbox(
                "Laufzeit (bei beiden verfuegbar)", common,
                format_func=lambda d: f"{pd.Timestamp(d).date()} (DTE {_dte(d)})", key="pt_expiry",
            )
        else:
            st.warning("Kein gemeinsamer Verfall bei Lead und Follower.")

        with st.expander("Erweiterte Einstellungen"):
            max_distance = st.number_input("Max. Distanz (Aehnlichkeit)", 0.01, 100.0, 4.0, 0.5, key="pt_maxdist")
            bandwidth = st.number_input("Kernel-Bandbreite σ", 0.01, 50.0, 1.0, 0.1, key="pt_bw")
            min_gap = st.number_input("Mindestabstand (Handelstage)", 0, 60, 5, key="pt_gap")
            dist_input = st.text_input("Distanz-Raster (%)", _DEFAULT_DISTANCES, key="pt_dist")
            qty_input = st.text_input("Follower-Mengenfaktoren", "1.0", key="pt_qty")
            t_min = st.number_input("Min. t-Stat (HAC)", 0.0, 10.0, 3.0, 0.1, key="pt_tmin")
            ml_label = st.selectbox("Max-Verlust-Basis",
                                    ["Strukturell (exakt)", "Empirisch (schlechtester Analog-Tag)"], key="pt_mlbasis")

    # 'In neuem Tab oeffnen' in die Kopfzeile (neben den Instrument-Auswaehler) rendern
    _new_tab_button(lead_key, follower_key, expiry, header_slot)

    # URL spiegelt die aktuelle Auswahl (damit Kopieren/Neuer Tab/Reload dieselbe Ansicht liefert)
    try:
        st.query_params["pt_lead"] = lead_key
        st.query_params["pt_follower"] = follower_key
        if expiry is not None:
            st.query_params["pt_expiry"] = str(pd.Timestamp(expiry).date())
    except Exception:
        pass

    if lead_snap["enriched"].empty or follower_snap["enriched"].empty:
        st.error("Eine der beiden Ketten ist leer (Markt geschlossen / Provider-Fehler?).")
        return
    if expiry is None:
        with st.expander("Verfuegbare Verfallstage"):
            st.write(f"{lead_instr.label}: {[str(d.date()) for d in _expiries(lead_snap)]}")
            st.write(f"{follower_instr.label}: {[str(d.date()) for d in _expiries(follower_snap)]}")
        return

    _ensure_training_table(lead_instr)

    params = (lead_key, follower_key, str(pd.Timestamp(expiry).date()), max_distance, bandwidth, min_gap,
              dist_input, qty_input, t_min, ml_label)

    store = st.session_state.get("pair_result")
    if store is None or store["meta"]["params"] != params:
        st.session_state.pop("pair_result", None)
        _compute(lead_instr, follower_instr, lead_snap, follower_snap, expiry, max_distance, bandwidth,
                 min_gap, dist_input, qty_input, t_min, ml_label, params)
    _render_result()


def _compute(lead_instr, follower_instr, lead_snap, follower_snap, expiry, max_distance, bandwidth, min_gap,
             dist_input, qty_input, t_min, ml_label, params) -> None:
    try:
        distances = [float(x) for x in dist_input.split(",") if x.strip()]
        qty_factors = tuple(float(x) for x in qty_input.split(",") if x.strip()) or (1.0,)
    except ValueError:
        st.error("Distanz-Raster oder Mengenfaktoren ungueltig (Format z.B. '0,1,2,5,10').")
        return

    lead_daily, follower_daily = load_data(lead_instr), load_data(follower_instr)
    if lead_daily.empty or follower_daily.empty:
        st.error("Kursdaten fuer Lead oder Follower fehlen - erst `python src/data_fetch.py` ausfuehren.")
        return
    if "resolution" in lead_daily.columns:
        lead_daily = lead_daily[lead_daily["resolution"] == "daily"]
    if "resolution" in follower_daily.columns:
        follower_daily = follower_daily[follower_daily["resolution"] == "daily"]

    dte = _dte(expiry)
    with st.spinner("Baue Distanz-Tabellen und suche Analog-Tage..."):
        lead_table, lead_spot = _distance_table_for(lead_instr, lead_daily, lead_snap, expiry, distances)
        follower_table, follower_spot = _distance_table_for(follower_instr, follower_daily, follower_snap,
                                                            expiry, distances)
        if lead_table is None or follower_table is None:
            st.error("Distanz-Tabelle/HV fuer den gewaehlten Verfall nicht verfuegbar.")
            return
        if "iv_pair_pct" not in lead_table.columns or "iv_pair_pct" not in follower_table.columns:
            st.error("Distanz-Tabelle liefert keine 'iv_pair_pct'-Spalte - IV-Anzeige nicht moeglich.")
            return
        ctx = _find_today_analogs(lead_daily, float(dte), max_distance, bandwidth, min_gap, lead_instr)
    if ctx is None:
        return

    moves = pair_analog_moves(ctx["analogs"], ctx["training_table"], follower_daily,
                              ctx["T_days_used"], ctx["query_date"])
    if moves.empty:
        st.error("Keine gemeinsamen Lead+Follower-Analog-Tage mit gueltiger Bewegung gefunden.")
        return

    cfg = PairTradeConfig(
        t_min=float(t_min), lead_multiplier=lead_instr.multiplier, follower_multiplier=follower_instr.multiplier,
        qty_factors=qty_factors,
        max_loss_basis="structural" if ml_label.startswith("Strukturell") else "empirical",
    )
    try:
        with st.spinner("Bewerte alle Lead x Follower Kombinationen..."):
            result = evaluate_pair_grid(lead_table, follower_table, moves, lead_spot, follower_spot, cfg)
    except ValueError as e:
        st.error(str(e))
        return
    if result.empty:
        st.warning("Keine Kombination erzeugt (zu wenige Strikes im Distanz-Raster?).")
        return

    header = (
        f"{lead_instr.label} (Lead) / {follower_instr.label} (Follower) · Verfall {pd.Timestamp(expiry).date()} "
        f"(DTE {dte}, Analog-T_days={ctx['T_days_used']}) · {len(moves)} gemeinsame Analog-Tage "
        f"({moves.attrs.get('n_no_follower_match', 0)} ohne Follower-Kurs verworfen)."
    )
    st.session_state["pair_result"] = {
        "result": result,
        "meta": {"params": params, "t_min": float(t_min), "header": header},
    }