"""
MODUS 5: Pair-Trade zwischen zwei Instrumenten (Lead/Short x Follower/Long).

Ein einziger Einstiegspunkt: render(active_instr). SONDERFALL gegenueber den
anderen Modi (siehe dashboard.py-Moduldokstring): braucht IMMER zwei
Instrumente gleichzeitig, waehlt Lead UND Follower deshalb SELBST in der
eigenen Sidebar - 'active_instr' (der oben global gewaehlte) dient nur als
Vorauswahl-Vorschlag fuer die Lead-Seite.

FUEHRENDES INSTRUMENT = Short-Seite: bestimmt die Analog-Tage (PCA-
Aehnlichkeitsraum wird NUR auf dem Lead-Instrument gefittet - siehe
pair_trade_search.py-Moduldokstring fuer die Begruendung). Das Follower-
Instrument wird an GENAU DENSELBEN Kalendertagen ausgewertet, nicht ueber
einen eigenen Aehnlichkeits-Raum gesucht.

WIEDERVERWENDUNG statt Neubau: die Analog-Suche selbst laeuft ueber
dashboard_livechain._find_today_analogs() (exakt dieselbe Funktion, dieselbe
Kernel-Gewichtung, derselbe Cache wie in 'Heutige Kette'/'Depot-Analyse') -
NICHT neu implementiert. Ebenso chain_delta.build_symmetric_distance_table()
fuer beide Optionsketten und market_data.get()/current() fuer den
Ketten-Snapshot je Instrument.

LAUFZEIT-MATCHING (wichtig - siehe Erklaerung im Chat vor dieser
Implementierung): ein Pair-Trade braucht fuer BEIDE Instrumente GENAU
DIESELBE Restlaufzeit fuer alle Beine - sonst verfaellt ein Bein, waehrend
das andere noch offen ist, und die Hedge-Beziehung ist fuer den Rest-
Zeitraum nicht mehr definiert (unkontrolliertes zeitliches Basisrisiko
zusaetzlich zum eigentlich gewollten SPX-vs-NDX-Risiko). Deshalb wird die
Expiry-Auswahl unten NICHT unabhaengig je Seite auf die naechstgelegene DTE
gerundet, sondern auf die SCHNITTMENGE der tatsaechlich gelisteten
Verfallstermine BEIDER Instrumente beschraenkt (siehe _common_expiries()) -
der Nutzer waehlt EINEN gemeinsamen Kalendertag, kein Naeherungswert je
Seite. Der daraus abgeleitete T_days (ueber
dashboard_livechain._find_today_analogs(), das intern auf die verfuegbaren
Laufzeiten der Lead-Trainingstabelle rundet) wird anschliessend fuer BEIDE
Seiten identisch verwendet (siehe pair_trade_search.pair_analog_moves()).
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import market_data
import dashboard_livechain
from instruments import INSTRUMENTS, InstrumentConfig, get_instrument
from chain_providers.base import UnsupportedChainError
from chain_delta import build_symmetric_distance_table
from backtest import compute_hv
from dashboard_common import load_data, metric_box
from pair_trade_search import pair_analog_moves, evaluate_pair_grid, PairTradeConfig

# Bewusst GROB (7 Zeilen statt der ~24 in dashboard_livechain.py) - siehe
# pair_trade_search.evaluate_pair_grid()-Docstring, Abschnitt PERFORMANCE-
# WARNUNG: die Grid-Groesse skaliert QUADRATISCH mit der Zeilenzahl je Seite.
_DEFAULT_DISTANCE_GRID = "0.5,1,2,3,5,7,10"
_FLAT_VOL_FALLBACK = 0.20   # nur fuer die (rein informative) Modell-Delta-Spalte in der Distanz-Tabelle


def _get_market(instr: InstrumentConfig, r_free: float, dividend_yield: float) -> dict | None:
    """Ketten-Snapshot fuer EIN Instrument - identisches Prinzip zu
    dashboard_livechain.py/dashboard_depot.py (gemeinsamer, tagesaktueller
    Cache ueber market_data.py). None (mit Fehlermeldung), falls das
    Instrument keinen chain_provider hat oder der Abruf fehlschlaegt."""
    snap = market_data.current(instr, market_data.DEFAULT_MAX_DTE)
    if snap is not None:
        return snap
    with st.spinner(f"Lade aktuelle Optionskette fuer {instr.label}..."):
        try:
            return market_data.get(instr, float(r_free), float(dividend_yield), market_data.DEFAULT_MAX_DTE)
        except UnsupportedChainError as e:
            st.error(str(e))
            return None
        except Exception as e:
            st.error(f"Marktdaten fuer {instr.label} konnten nicht geladen werden: {e}")
            return None


def _common_expiries(enriched_lead: pd.DataFrame, enriched_follower: pd.DataFrame) -> list[pd.Timestamp]:
    """Nur Verfallstermine, die es TATSAECHLICH in BEIDEN Optionsketten gibt
    (siehe Moduldokstring, Abschnitt LAUFZEIT-MATCHING) - kein unabhaengiges
    Runden je Seite."""
    lead_exp = {pd.Timestamp(x).normalize() for x in enriched_lead["expiry"].unique()}
    foll_exp = {pd.Timestamp(x).normalize() for x in enriched_follower["expiry"].unique()}
    return sorted(lead_exp & foll_exp)


def _expiry_dte(enriched: pd.DataFrame, expiry: pd.Timestamp) -> float:
    """Exakte Restlaufzeit (Tage) eines Verfalls aus der geladenen Kette -
    identische Quelle wie in dashboard_livechain.py (Spalte 'dte_exact', vom
    jeweiligen Chain-Provider berechnet; Fallback 'dte')."""
    exp = pd.Timestamp(expiry).normalize()
    sub = enriched[pd.to_datetime(enriched["expiry"]).dt.normalize() == exp]
    col = "dte_exact" if "dte_exact" in sub.columns else "dte"
    return float(sub[col].iloc[0])


def _flat_vol(daily_df: pd.DataFrame, length: int = 30) -> float:
    """Grobe HV, NUR fuer die informative 'Modell-Delta'-Spalte der
    Distanz-Tabelle (siehe chain_delta.build_symmetric_distance_table) -
    evaluate_pair_grid() selbst nutzt ausschliesslich echte Bid/Ask-Preise,
    diese Zahl hat auf die eigentliche Bewertung KEINEN Einfluss."""
    hv = compute_hv(daily_df, length).dropna()
    return float(hv.iloc[-1]) if len(hv) else _FLAT_VOL_FALLBACK


def render(active_instr: InstrumentConfig) -> None:
    """Oeffentlicher Einstiegspunkt - siehe Moduldokstring, warum dieser
    Modus (anders als die uebrigen) nur 'active_instr' als Vorschlag
    bekommt statt (df, daily_df, has_resolution, instr)."""
    st.header("Pair-Trade: Lead (Short) x Follower (Long)")
    st.caption(
        "Sucht die beste Kombination aus Credit-Spread/Iron-Condor auf dem "
        "FUEHRENDEN Instrument (Short-Seite - bestimmt die Analog-Tage) und "
        "Debit-Spread/Reverse-Iron-Condor auf dem FOLGE-Instrument (Long-"
        "Seite - Hedge), ueber ALLE Strike-Kombinationen BEIDER Seiten fuer "
        "einen GEMEINSAMEN Verfallstermin. Details/Kriterien: siehe "
        "pair_trade_search.py."
    )

    instrument_keys = list(INSTRUMENTS.keys())
    with st.sidebar:
        st.divider()
        st.header("Pair-Trade")
        lead_key = st.selectbox(
            "Lead-Instrument (Short - die 'teure' Seite)", instrument_keys,
            index=instrument_keys.index(active_instr.key),
            format_func=lambda k: INSTRUMENTS[k].label, key="pair_lead_instr",
        )
        follower_default = "ndx" if lead_key == "spx" else ("spx" if lead_key == "ndx" else instrument_keys[0])
        follower_key = st.selectbox(
            "Follower-Instrument (Long - die 'billige'/Hedge-Seite)", instrument_keys,
            index=instrument_keys.index(follower_default) if follower_default in instrument_keys else 0,
            format_func=lambda k: INSTRUMENTS[k].label, key="pair_follower_instr",
        )
        if lead_key == follower_key:
            st.error("Lead- und Follower-Instrument muessen unterschiedlich sein.")
            return
        lead_instr, follower_instr = get_instrument(lead_key), get_instrument(follower_key)
        if lead_instr.currency != follower_instr.currency:
            st.error(
                f"Waehrungen unterschiedlich ({lead_instr.currency} vs. {follower_instr.currency}) - "
                "das Projekt hat keine FX-Umrechnung, ein Pair-Trade waere in Geld nicht addierbar."
            )
            return

        r_free = st.number_input("Risikoloser Zins", 0.0, 0.20, 0.0, 0.005, format="%.3f", key="pair_r")
        dividend_yield_lead = st.number_input(
            f"Dividenden-/Convenience-Yield ({lead_instr.label})", 0.0, 0.10,
            float(lead_instr.dividend_yield), 0.001, format="%.3f", key="pair_q_lead",
        )
        dividend_yield_foll = st.number_input(
            f"Dividenden-/Convenience-Yield ({follower_instr.label})", 0.0, 0.10,
            float(follower_instr.dividend_yield), 0.001, format="%.3f", key="pair_q_foll",
        )

        distance_grid_input = st.text_input(
            "Ziel-Abstaende (%, kommagetrennt) - gilt fuer BEIDE Instrumente",
            _DEFAULT_DISTANCE_GRID,
            help="Bewusst grob: die Grid-Groesse skaliert QUADRATISCH mit der "
                 "Zeilenzahl je Seite (n*(n-1)/2 Paare je Instrument, davon dann "
                 "das Kreuzprodukt). Mit den 7 Default-Werten ergeben sich bereits "
                 "ueber 1.300 geprüfte Kombinationen (3 Strukturen).",
        )
        max_distance = st.number_input("Max. Analog-Distanz (Lead)", 0.01, 100.0, 4.0, 0.5, key="pair_maxdist")
        bandwidth = st.number_input("Kernel-Bandbreite σ (Lead)", 0.01, 50.0, 1.0, 0.1, key="pair_bw")
        min_gap = st.number_input("Mindestabstand zu heute (Handelstage)", 0, 60, 5, key="pair_gap")

        st.subheader("Bewertungs-Kriterien")
        t_min = st.number_input("Min. t-Stat (HAC, kombiniert)", 0.0, 10.0, 2.0, 0.1, key="pair_tmin")
        max_full_loss = st.number_input("Max. Anteil Volltreffer-Verlust (%)", 0.0, 100.0, 5.0, 0.5, key="pair_fullloss")
        tail_floor = st.number_input("Min. 5%-Quantil (Rendite/Max-Loss)", -1.0, 0.0, -0.9, 0.05, key="pair_tail")
        require_lead_plateau = st.checkbox(
            "Plateau-Kriterium auf Lead-Seite verlangen", value=True, key="pair_plateau",
            help="Nur auf der Lead-Seite (siehe pair_trade_search.PairTradeConfig - "
                 "die Follower-Seite ist ein Hedge und darf fuer sich genommen "
                 "erwartungswert-negativ sein).",
        )
        qty_factors = st.multiselect(
            "Follower-Menge (Faktor auf Notional-neutral)", [0.25, 0.5, 0.75, 1.0, 1.5, 2.0], default=[0.5, 1.0],
            key="pair_qty",
            help="1.0 = Follower-Kontrakte je Lead-Kontrakt so, dass das Notional (Spot x Multiplikator) "
                 "gleich ist. Jeder Faktor multipliziert die Grid-Groesse.",
        ) or [1.0]
        no_debit = st.checkbox(
            "Nur Kombis ohne Netto-Debit", value=False, key="pair_nodebit",
            help="Verwirft Kombinationen, die netto Praemie zahlen (Theta-negativ). Grober Schutz gegen "
                 "teure Hedges - kein Ersatz fuer ein echtes Hedge-Wirksamkeits-Kriterium.",
        )
        price_basis = st.radio("Preisbasis", ["Bid/Ask (echt)", "Mid (Näherung)"], key="pair_basis")
        sort_label = st.selectbox(
            "Sortierung", ["Rendite/Risiko %", "E[P&L] (Pkt)", "t (HAC)"], key="pair_sort",
        )

        run = st.button("Pair-Trade-Suche ausfuehren", type="primary")

    lead_daily = load_data(lead_instr)
    follower_daily = load_data(follower_instr)
    if lead_daily.empty or follower_daily.empty:
        missing = [i.label for i, d in ((lead_instr, lead_daily), (follower_instr, follower_daily)) if d.empty]
        st.warning(f"Kursdaten fehlen fuer: {', '.join(missing)} - erst `data_fetch.py` dafuer ausfuehren.")
        return
    if "resolution" in lead_daily.columns:
        lead_daily = lead_daily[lead_daily["resolution"] == "daily"]
    if "resolution" in follower_daily.columns:
        follower_daily = follower_daily[follower_daily["resolution"] == "daily"]

    market_lead = _get_market(lead_instr, r_free, dividend_yield_lead)
    market_follower = _get_market(follower_instr, r_free, dividend_yield_foll)
    if market_lead is None or market_follower is None:
        return

    common = _common_expiries(market_lead["enriched"], market_follower["enriched"])
    if not common:
        st.error(
            f"{lead_instr.label} und {follower_instr.label} haben AKTUELL keinen "
            "gemeinsamen Verfallstermin in den geladenen Ketten (siehe Moduldokstring, "
            "Abschnitt LAUFZEIT-MATCHING - ein Pair-Trade braucht fuer beide Seiten "
            "identische Laufzeit). Bei Monats-Expiries (3. Freitag) meist gegeben, bei "
            "Weeklies je nach Instrument unterschiedlich dicht gelistet - ggf. Max. DTE "
            "in der Sidebar von 'Heutige Kette' fuer beide Instrumente vergleichen."
        )
        return

    expiry_labels = [
        f"{e.date().isoformat()} (Restlaufzeit {_expiry_dte(market_lead['enriched'], e):.1f} Tage)" for e in common
    ]
    chosen_label = st.selectbox("Gemeinsamer Verfallstermin (beide Instrumente)", expiry_labels)
    chosen_expiry = common[expiry_labels.index(chosen_label)]
    chosen_dte_exact = _expiry_dte(market_lead["enriched"], chosen_expiry)
    st.caption(
        f"Exakte Restlaufzeit: **{chosen_dte_exact:.2f} Tage** - GLEICH fuer {lead_instr.label} "
        f"und {follower_instr.label} (siehe Moduldokstring, LAUFZEIT-MATCHING)."
    )

    if not run:
        if "pair_result" in st.session_state:
            _render_result()
        else:
            st.info("Parameter links einstellen und 'Pair-Trade-Suche ausfuehren' klicken.")
        return

    with st.spinner(f"Suche Analog-Tage auf dem Lead-Instrument ({lead_instr.label})..."):
        ctx = dashboard_livechain._find_today_analogs(
            lead_daily, chosen_dte_exact, float(max_distance), float(bandwidth), int(min_gap), lead_instr,
        )
    if ctx is None:
        return
    T_days_used = ctx["T_days_used"]
    if abs(T_days_used - chosen_dte_exact) > 1.0:
        st.info(
            f"Restlaufzeit {chosen_dte_exact:.2f} Tage wird fuer die Analog-Bewertung auf "
            f"T_days={T_days_used} gerundet (verfuegbare Laufzeiten der Lead-Trainingstabelle) "
            "- gilt danach fuer BEIDE Instrumente identisch."
        )

    with st.spinner("Verbinde Lead-Analog-Tage mit der Follower-Bewegung..."):
        moves = pair_analog_moves(ctx["analogs"], ctx["training_table"], follower_daily, T_days_used, ctx["query_date"])
    if moves.empty:
        st.error(
            f"Keine Analog-Tage mit gueltiger {follower_instr.label}-Bewegung gefunden - "
            f"Follower-Kurshistorie zu kurz, oder Analog-Tage liegen vor dem Beginn der "
            f"{follower_instr.label}-Historie."
        )
        return
    n_no_match = moves.attrs.get("n_no_follower_match", 0)
    st.caption(
        f"{len(moves)} Analog-Tage nutzbar (T_days={T_days_used}, gemeinsamer Verfall "
        f"{chosen_expiry.date()}){f', {n_no_match} ohne passenden Follower-Kurs verworfen' if n_no_match else ''}."
    )

    try:
        distance_targets_pct = [float(x.strip()) for x in distance_grid_input.split(",") if x.strip()]
    except ValueError:
        st.error("Ziel-Abstaende konnten nicht geparst werden - Format z.B. '0.5,1,2,5,10'.")
        return

    T_years = chosen_dte_exact / 365.0
    lead_table = build_symmetric_distance_table(
        market_lead["enriched"], chosen_expiry, market_lead["spot"], _flat_vol(lead_daily), T_years,
        r_free, dividend_yield_lead, distance_targets_pct,
    )
    follower_table = build_symmetric_distance_table(
        market_follower["enriched"], chosen_expiry, market_follower["spot"], _flat_vol(follower_daily), T_years,
        r_free, dividend_yield_foll, distance_targets_pct,
    )

    with st.expander("Eingabe der Suche: Distanz-Tabellen beider Instrumente (echte Strikes/Preise)"):
        cols = ["distance_pct", "strike_put", "bid_put", "ask_put", "strike_call", "bid_call", "ask_call"]
        st.caption(f"{lead_instr.label} (Lead)")
        st.dataframe(lead_table[cols], use_container_width=True, hide_index=True)
        st.caption(f"{follower_instr.label} (Follower)")
        st.dataframe(follower_table[cols], use_container_width=True, hide_index=True)

    cfg = PairTradeConfig(
        price_basis="mid" if price_basis.startswith("Mid") else "bidask",
        t_min=float(t_min), max_full_loss_share=max_full_loss / 100.0, tail_floor=float(tail_floor),
        require_lead_plateau=bool(require_lead_plateau),
        lead_multiplier=float(lead_instr.multiplier), follower_multiplier=float(follower_instr.multiplier),
        qty_factors=tuple(float(x) for x in qty_factors),
        min_net_cashflow=0.0 if no_debit else None,
        sort_by={"Rendite/Risiko %": "ror", "E[P&L] (Pkt)": "mean", "t (HAC)": "tstat"}[sort_label],
    )
    try:
        with st.spinner("Durchsuche das Strike-Grid (kann je nach Rastergroesse dauern)..."):
            result = evaluate_pair_grid(
                lead_table, follower_table, moves, market_lead["spot"], market_follower["spot"], cfg,
            )
    except ValueError as e:
        st.error(f"Grid-Search fehlgeschlagen: {e}")
        return

    st.session_state["pair_result"] = dict(
        result=result, lead_instr=lead_instr.key, follower_instr=follower_instr.key,
        T_days_used=T_days_used, chosen_expiry=chosen_expiry, n_moves=len(moves),
        dedupe_dropped=result.attrs.get("dedupe_dropped", (0, 0)),
        notional_ratio=result.attrs.get("notional_ratio"),
        n_rows=(result.attrs.get("n_lead_rows"), result.attrs.get("n_follower_rows")),
        currency=lead_instr.currency,
    )
    _render_result()


def _render_result() -> None:
    """Ergebnistabelle fuer den zuletzt gelaufenen Pair-Trade-Search (aus
    st.session_state)."""
    state = st.session_state.get("pair_result")
    if state is None:
        return
    result = state["result"]
    if result.empty:
        st.warning(
            "Kein gueltiges Paar gefunden (Kredit-/Debit-Bedingungen auf keiner "
            "Strike-Kombination erfuellbar). Distanz-Raster erweitern oder Preisbasis "
            "auf 'Mid' stellen, falls Bid/Ask ausserhalb der Handelszeit 0 liefert."
        )
        return

    n_pass = int(result["passes"].sum())
    dd_lead, dd_foll = state.get("dedupe_dropped", (0, 0))
    if dd_lead or dd_foll:
        st.warning(
            f"⚠️ Duenne Optionskette: {dd_lead} Lead-Zeile(n) und {dd_foll} Follower-Zeile(n) landeten auf "
            "demselben realen Strike wie eine andere Zeile und wurden entfernt (sonst zaehlen Duplikate "
            "als eigene Kombinationen und das Plateau-Kriterium waere wirkungslos). Nutzbar blieben "
            f"{state['n_rows'][0]} Lead- und {state['n_rows'][1]} Follower-Zeilen. Bei sehr wenigen Zeilen "
            "ist die Suche kaum aussagekraeftig - einen naeheren Verfall oder ein anderes Distanz-Raster waehlen."
        )
    ccy = state.get("currency", "")
    lead_label = INSTRUMENTS[state["lead_instr"]].label
    follower_label = INSTRUMENTS[state["follower_instr"]].label

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_box("Getestete Kombinationen", f"{len(result):,}", "neutral")
    with m2:
        metric_box("Davon Kandidaten (alle Kriterien)", f"{n_pass:,}", "good" if n_pass > 0 else "bad")
    with m3:
        metric_box("Analog-Tage genutzt", f"{state['n_moves']:,}", "neutral")
    with m4:
        metric_box("T_days (beide Seiten)", f"{state['T_days_used']}", "neutral")

    if n_pass == 0:
        st.info("Kein Paar besteht alle Kriterien - kein belastbarer Edge fuer diese Konfiguration erkennbar.")
    else:
        best = result.iloc[0]
        st.success(
            f"Optimum (sortiert nach gewaehlter Metrik): **{best['structure']}** - "
            f"{lead_label} Short {best['lead_short_distance']:g}% / Wing {best['lead_wing_distance']:g}%, "
            f"{follower_label} Long {best['follower_short_distance']:g}% / Wing {best['follower_wing_distance']:g}% · "
            f"Follower-Menge {best['follower_qty']:.3f} je Lead-Kontrakt · Netto {best['net_cashflow']:+.0f} {ccy} · Max-Loss {best['max_loss']:.0f} {ccy} · "
            f"E[P&L] {best['mean']:+.0f} {ccy} = {best['ror']*100:+.1f}% (t={best['tstat']:.2f})."
        )

    show = result[[
        "structure", "lead_short_distance", "lead_wing_distance",
        "follower_short_distance", "follower_wing_distance",
        "net_cashflow", "max_loss", "follower_qty", "mean", "ror", "tstat",
        "e_train", "e_val", "full_loss_share", "q_tail", "win_rate_pct", "passes",
    ]].rename(columns={
        "structure": "Struktur",
        "lead_short_distance": f"{lead_label} Short %", "lead_wing_distance": f"{lead_label} Wing %",
        "follower_short_distance": f"{follower_label} Long %", "follower_wing_distance": f"{follower_label} Wing %",
        "net_cashflow": "Netto (Geld)", "max_loss": "Max-Loss (Geld)", "follower_qty": "Follower-Menge",
        "mean": "E[P&L] (Geld)", "ror": "Rendite/Risiko %",
        "tstat": "t (HAC)", "e_train": "E Train", "e_val": "E Val",
        "full_loss_share": "Voll-Verlust %", "q_tail": "5%-Quantil (R/R)",
        "win_rate_pct": "Gewinnrate %", "passes": "Kandidat",
    }).copy()
    show["Rendite/Risiko %"] = show["Rendite/Risiko %"] * 100.0
    show["Voll-Verlust %"] = show["Voll-Verlust %"] * 100.0
    yn = show["Kandidat"].map({True: "✅", False: "❌"})
    show["Kandidat"] = yn

    fmt = {c: "{:.2f}" for c in show.select_dtypes("number").columns}

    passing = show[result["passes"].to_numpy()]
    st.subheader("Kandidaten (alle Kriterien erfuellt)")
    if len(passing):
        st.dataframe(passing.style.format(fmt, na_rep="n/a"), use_container_width=True, hide_index=True)
    else:
        st.caption("Keine - siehe 'Alle getesteten Kombinationen' unten fuer die besten (nicht bestehenden) Zeilen.")

    with st.expander(f"Alle getesteten Kombinationen ({len(show)})"):
        st.dataframe(show.style.format(fmt, na_rep="n/a"), use_container_width=True, hide_index=True)