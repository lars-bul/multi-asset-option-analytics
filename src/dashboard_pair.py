"""
MODUS 5: Pair-Trade-Suche (Backend: pair_trade_search.py).

SONDERFALL (siehe dashboard.py): braucht zwei Instrumente (Lead + Follower),
waehlt beide selbst in der eigenen Sidebar und bekommt nur 'instr' (Vorschlag
fuer die Lead-Seite).

LAUFZEIT: fuer beide Instrumente werden die verfuegbaren Verfallstage aus der
Live-Kette gelesen; angeboten werden nur Verfallstage, die es bei BEIDEN gibt
(gleiches Datum). Beide Seiten werden mit exakt diesem Verfall gebaut.

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
steigen stark) und 'Max-Gewinn Unterseite' (beide fallen auf 0).
"""
import sys
import time
from pathlib import Path

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
    "Max-Gewinn Unterseite": ("max_gain_down", False),
}

_fragment = getattr(st, "fragment", None) or (lambda f: f)


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
        "max_loss", "max_loss_struct", "max_loss_hist", "max_loss_up", "max_gain_down",
        "lead_short_distance", "lead_wing_distance", "vol_lead_pct",
        "follower_short_distance", "follower_wing_distance", "vol_follower_pct",
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
        "max_loss_up": "Max-Verlust Oberseite", "max_gain_down": "Max-Gewinn Unterseite",
        "lead_short_distance": "Lead Short %", "lead_wing_distance": "Lead Fluegel %",
        "vol_lead_pct": "Lead-IV %",
        "follower_short_distance": "Follower Short %", "follower_wing_distance": "Follower Fluegel %",
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
            "Max-Verlust Oberseite": "{:+.2f}", "Max-Gewinn Unterseite": "{:+.2f}",
            "Lead-IV %": "{:.1f}",
            "Follower-IV %": "{:.1f}", "Follower-Menge": "{:.3f}",
        }, na_rep="–"),
        use_container_width=True, hide_index=True, height=650,
    )
    st.caption(
        f"{len(view)} Zeilen. Spaltenkoepfe sind direkt anklickbar (sortiert nur die Anzeige). "
        "'Kredit (netto)' = Lead-Kredit + Follower-Debit (negativ), in Geld je Lead-Kontrakt. "
        "'Verlust-Wahrsch.' = gewichteter Anteil Analog-Tage mit Gesamt-P&L < 0. "
        "'Vollverlust %' = Anteil Analog-Tage, an denen der Gesamtspread den verwendeten Max-Verlust "
        "erreicht (bei strukturellem Max-Verlust praktisch immer ~0, da er Lead- und Follower-Extrem "
        "gleichzeitig annimmt). 'Lead-Vollverlust %' = Anteil Tage, an denen der Lead-Spread allein "
        "seinen vollen Max-Verlust erreicht (reine Info-Spalte, kein Kriterium). "
        "'Max-Verlust Oberseite' = Verlust der Gesamtposition, wenn BEIDE Basiswerte stark steigen "
        "(jedes Bein im Extremwert); positiv = Verlust, negativ = Gewinn (z.B. Put-Credit-Spread: "
        "Kredit = Max-Gewinn -> negativer Wert). 'Max-Gewinn Unterseite' = P&L, wenn BEIDE Basiswerte "
        "auf 0 fallen; positiv = Gewinn, negativ = Verlust."
    )


# ── Hauptfunktion ────────────────────────────────────────────────────

def render(instr: InstrumentConfig) -> None:
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

    with st.sidebar:
        st.divider()
        st.header("Pair-Trade")
        lead_idx = providers.index(instr.key) if instr.key in providers else 0
        lead_key = st.selectbox("Lead-Instrument", providers, index=lead_idx,
                                format_func=lambda k: INSTRUMENTS[k].label, key="pt_lead")
        follower_opts = [k for k in providers if k != lead_key]
        follower_key = st.selectbox("Follower-Instrument", follower_opts,
                                    format_func=lambda k: INSTRUMENTS[k].label, key="pt_follower")

        lead_instr, follower_instr = get_instrument(lead_key), get_instrument(follower_key)
        with st.spinner("Lade Ketten..."):
            lead_snap, follower_snap = _get_snapshot(lead_instr), _get_snapshot(follower_instr)
        common = sorted(set(_expiries(lead_snap)) & set(_expiries(follower_snap)))
        expiry = None
        if common:
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