"""
MODUS 3 (neu, Ausbaustufe 1 von 2): Heutige Kette - symmetrische Abstands-Ansicht.

Zeigt die AKTUELLE (heutige, ~15 Min. verzoegerte) SPX-Optionskette als EINE
Tabelle: je Ziel-ABSTAND vom Spot (in %) eine Zeile - NICHT nach Delta
gepaart (siehe chain_delta.build_symmetric_distance_table). Put-Zielstrike =
spot*(1-x%), Call-Zielstrike = spot*(1+x%), beide unabhaengig auf den
naechstgelegenen real vorhandenen Strike gerundet - dadurch IMMER
symmetrisch um den Spot, per Definition.

WICHTIG (siehe Nutzer-Vorgabe): die HV-Periode fuer die flache/symmetrische
Modell-Vola UND fuer das HV-Min/Max-Rang-Fenster wird NICHT mehr manuell
eingestellt, sondern IMMER automatisch aus der gewaehlten Restlaufzeit
(dte_exact) abgeleitet - 7.5 Tage Restlaufzeit -> HV(7.5)-Fenster. Da
compute_hv() ein GANZZAHLIGES Rolling-Fenster (Anzahl Handelstage) braucht,
wird auf die naechste ganze Zahl gerundet (min. 2) - die fraktionale
Restlaufzeit selbst fliesst weiterhin exakt in T_years (Black-76) ein, nur
das diskrete HV-Rolling-Fenster kann nicht fraktional sein.

NEU: iv_pair_pct (die tatsaechliche 'Paket-IV' der Kombination, per
Joint-Black-76-Solve - NICHT der Mittelwert der Einzel-IVs, siehe
chain_delta.implied_vol_pair) und hv_rank_pct (welcher HV-Rang diese
Paket-IV im (restlaufzeit-synchronen) HV-Min/Max-Fenster entspricht -
Umkehrung von analog_matching.target_vol_for_rank()).
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_today_chain import fetch_today_chain, fetch_today_spot
from chain_delta import ChainDeltaConfig, enrich_chain_with_delta, build_symmetric_distance_table
from analog_matching import hv_rank_bounds


def render(df, daily_df, has_resolution) -> None:
    st.header("Heutige Kette - symmetrische Abstands-Ansicht")
    st.caption(
        "Laedt die aktuelle SPX-Kette (Yahoo, ~15 Min. verzoegert). Put- und "
        "Call-Strike werden je Zeile unabhaengig auf den naechstgelegenen realen "
        "Strike im gleichen prozentualen Abstand vom Spot gerundet - dadurch "
        "IMMER symmetrisch um den Spot. Die HV-Periode fuer Modell-Delta und "
        "HV-Rang wird automatisch an die gewaehlte Restlaufzeit gekoppelt "
        "(z.B. 7.5 Tage Restlaufzeit -> HV(7.5)-Fenster)."
    )

    with st.sidebar:
        st.divider()
        st.header("Heutige Kette")
        min_dte = st.number_input("Min. DTE (fraktional)", 0.0, 365.0, 0.0, 0.5)
        max_dte = st.number_input("Max. DTE (fraktional)", 1.0, 365.0, 90.0, 1.0)
        min_abs_delta = st.number_input(
            "Minimales |Delta| (Filter, echtes Marktdelta)", 0.0001, 0.5, 0.001, 0.001, format="%.4f",
            help="ACHTUNG: dieser Filter wird VOR dem Tabellenbau angewendet und "
                 "kappt damit effektiv, wie weit die Ziel-Abstaende (%) tatsaechlich "
                 "reichen koennen. Fuer die Abstands-Ansicht eher klein wählen "
                 "(Default 0.001 = 0.1%).",
        )
        st.caption("Abstands-Raster (Ziel-% vom Spot) fuer die Zeilen:")
        distance_grid_input = st.text_input(
            "Ziel-Abstaende (%, kommagetrennt)",
            "0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,"
            "7.5,8.0,8.5,9.0,9.5,10.0,10.5,11.0,11.5,12.0,12.5,13.0,13.5,14.0,14.5,15.0,15.5,16.0,"
            "16.5,17.0,17.5,18.0,18.5,19.0,19.5,20.0,20.5,21.0,21.5,22.0,22.5,23.0,23.5,24.0,24.5,"
            "25.0,25.5,26.0,26.5,27.0,27.5,28.0,28.5,29.0,29.5,30.0,30.5,31.0,31.5,32.0,32.5,33.0,"
            "33.5,34.0,34.5,35.0,35.5,36.0,36.5,37.0,37.5,38.0,38.5,39.0,39.5,40.0",
        )
        rank_lookback = st.number_input(
            "Lookback fuer HV-Min/Max-Rangfenster (Tage)", 30, 5000, 252,
            help="Laenge des rollierenden Fensters, in dem hv_min/hv_max fuer den "
                 "HV-Rang der Kombi-IV gesucht wird - identisches Prinzip zu "
                 "backtest.FixedRankConfig.lookback_days.",
        )
        r_free = st.number_input("Risikoloser Zins", 0.0, 0.20, 0.0, 0.005, format="%.3f")
        dividend_yield = st.number_input("Dividendenrendite", 0.0, 0.10, 0.015, 0.001, format="%.3f")
        load = st.button("Heutige Kette laden", type="primary")

    if load:
        with st.spinner("Lade heutige Kette und berechne Delta/IV je Serie..."):
            try:
                spot = fetch_today_spot()
                chain = fetch_today_chain(min_dte=float(min_dte), max_dte=float(max_dte))
                enriched = enrich_chain_with_delta(
                    chain, spot,
                    ChainDeltaConfig(min_abs_delta=float(min_abs_delta),
                                      r_free=float(r_free), dividend_yield=float(dividend_yield)),
                )
            except Exception as e:
                st.error(f"Laden fehlgeschlagen: {e}")
                return
        st.session_state["livechain_spot"] = spot
        st.session_state["livechain_enriched"] = enriched
        st.session_state["livechain_r_free"] = float(r_free)
        st.session_state["livechain_dividend_yield"] = float(dividend_yield)
        st.session_state["livechain_rank_lookback"] = int(rank_lookback)

    if "livechain_enriched" not in st.session_state:
        st.info("Links 'Heutige Kette laden' klicken.")
        return

    spot = st.session_state["livechain_spot"]
    enriched = st.session_state["livechain_enriched"]
    r_free = st.session_state["livechain_r_free"]
    dividend_yield = st.session_state["livechain_dividend_yield"]
    rank_lookback = st.session_state["livechain_rank_lookback"]
    st.metric("Aktueller Spot (SPX)", f"{spot:,.2f}")

    if enriched.empty:
        st.warning(
            "Keine Serien nach dem Delta-Filter uebrig - min. |Delta| verringern "
            "oder DTE-Fenster pruefen."
        )
        return

    if "dte_exact" not in enriched.columns:
        st.warning(
            "Spalte 'dte_exact' fehlt - alte fetch_today_chain.py? App neu starten "
            "und 'Heutige Kette laden' erneut klicken. Bis dahin Fallback auf 'dte'."
        )
        enriched = enriched.copy()
        enriched["dte_exact"] = enriched["dte"].astype(float)

    # ── Expiry-Auswahl ZUERST (Datum + exakte fraktionale Resttage) - die
    # HV-Periode haengt davon ab, muss also DANACH berechnet werden. ──
    expiry_info = (
        enriched[["expiry", "dte", "dte_exact"]]
        .drop_duplicates()
        .sort_values("expiry")
    )
    expiry_labels = [
        f"{pd.Timestamp(row.expiry).date().isoformat()}  (Restlaufzeit: {row.dte_exact:.1f} Tage)"
        for row in expiry_info.itertuples()
    ]
    chosen_label = st.selectbox("Expiry", expiry_labels)
    chosen_expiry = expiry_info.iloc[expiry_labels.index(chosen_label)]["expiry"]
    chosen_dte_exact = float(expiry_info.iloc[expiry_labels.index(chosen_label)]["dte_exact"])
    T_years = chosen_dte_exact / 365.0
    st.caption(
        f"Exakte Restlaufzeit: **{chosen_dte_exact:.2f} Tage** (Annahme "
        "PM-Settlement 16:00 ET - bei klassischen monatlichen AM-settled "
        "Serien ca. 6.5h zu hoch)."
    )

    # ── HV-Periode IMMER = Restlaufzeit (auf ganze Handelstage gerundet,
    # da compute_hv()/hv_rank_bounds() ein ganzzahliges Rolling-Fenster
    # brauchen - die fraktionale Restlaufzeit selbst geht weiterhin exakt
    # in T_years fuer Black-76 ein, siehe Moduldokstring). ──
    hv_length = max(2, round(chosen_dte_exact))
    st.caption(
        f"→ HV-Periode automatisch auf **{hv_length} Handelstage** gesetzt "
        f"(gerundete Restlaufzeit von {chosen_dte_exact:.2f} Tagen)."
    )

    bounds = hv_rank_bounds(daily_df, hv_length, int(rank_lookback))
    if bounds.empty or bounds["hv"].dropna().empty:
        st.error(
            f"HV({hv_length}) konnte nicht berechnet werden - zu wenig Historie "
            "in daily_df."
        )
        return
    today_bounds = bounds.dropna(subset=["hv"]).iloc[-1]
    flat_hv = float(today_bounds["hv"])
    hv_min = float(today_bounds["hv_min"]) if pd.notna(today_bounds["hv_min"]) else None
    hv_max = float(today_bounds["hv_max"]) if pd.notna(today_bounds["hv_max"]) else None

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(f"Flache HV({hv_length}d, = Restlaufzeit)", f"{flat_hv*100:.2f}%")
    with c2:
        st.metric("HV-Min im Rangfenster", f"{hv_min*100:.2f}%" if hv_min is not None else "n/a")
    with c3:
        st.metric("HV-Max im Rangfenster", f"{hv_max*100:.2f}%" if hv_max is not None else "n/a")
    if hv_min is None or hv_max is None or hv_max <= hv_min:
        st.warning(
            "HV-Min/Max fuer das Rangfenster nicht verfuegbar (zu wenig Historie im "
            "Lookback) - hv_rank_pct wird in der Tabelle 'n/a' bleiben. Lookback "
            "verkleinern oder mehr Kurshistorie laden."
        )

    try:
        distance_targets_pct = [float(x.strip()) for x in distance_grid_input.split(",") if x.strip()]
    except ValueError:
        st.error("Ziel-Abstaende konnten nicht geparst werden - Format z.B. '0.5,1,2,5,10'.")
        return

    table = build_symmetric_distance_table(
        enriched, chosen_expiry, spot, flat_hv, T_years,
        r_free, dividend_yield, distance_targets_pct, hv_min, hv_max,
    )

    # Grenzbereich erkennen: wiederholt sich strike_put/strike_call gegenueber
    # der VORHERIGEN Zeile (Distanz-Grid ist aufsteigend sortiert), ist die
    # Kette (nach Filter) an dieser Stelle "ausgegangen".
    put_stuck = table["strike_put"].astype(float).diff().eq(0)
    call_stuck = table["strike_call"].astype(float).diff().eq(0)
    if put_stuck.any() or call_stuck.any():
        stuck_from_put = table.loc[put_stuck, "distance_pct"].min() if put_stuck.any() else None
        stuck_from_call = table.loc[call_stuck, "distance_pct"].min() if call_stuck.any() else None
        parts = []
        if stuck_from_put is not None:
            parts.append(f"Put ab Ziel-Abstand {stuck_from_put:g}%")
        if stuck_from_call is not None:
            parts.append(f"Call ab Ziel-Abstand {stuck_from_call:g}%")
        st.warning(
            "Randbereich erreicht - Zeilen wiederholen den letzten verfuegbaren "
            "Strike: " + " und ".join(parts) + ". Ursache meist der 'Minimales "
            "|Delta|'-Filter in der Sidebar - dort verkleinern, um weiter OTM zu kommen."
        )

    show = table.rename(columns={
        "distance_pct": "Ziel-Abstand % v. Spot",
        "model_delta_put_pct": "Δ Put % (Modell, flache HV)",
        "model_delta_call_pct": "Δ Call % (Modell, flache HV)",
        "iv_pair_pct": "IV Kombi % (Markt, Paket-Solve)",
        "hv_rank_pct": "HV-Rang % (dieser Kombi)",
        "strike_put": "Strike Put (real)", "strike_call": "Strike Call (real)",
        "mid_put": "Mid Put", "mid_call": "Mid Call",
        "iv_put_pct": "IV Put % (echt)", "iv_call_pct": "IV Call % (echt)",
        "delta_put_pct": "Δ Put % (echt)", "delta_call_pct": "Δ Call % (echt)",
    })[[
        "Ziel-Abstand % v. Spot",
        "Δ Put % (Modell, flache HV)", "Δ Call % (Modell, flache HV)",
        "IV Kombi % (Markt, Paket-Solve)", "HV-Rang % (dieser Kombi)",
        "Strike Put (real)", "Strike Call (real)",
        "Mid Put", "Mid Call",
        "IV Put % (echt)", "IV Call % (echt)",
        "Δ Put % (echt)", "Δ Call % (echt)",
    ]]
    st.caption(
        "'IV Kombi' = die EINE Vola, die - in Black-76 fuer Put+Call GEMEINSAM "
        "eingesetzt - exakt die Summe beider echter Mid-Preise reproduziert "
        "(kein Mittelwert der Einzel-IVs, siehe chain_delta.implied_vol_pair). "
        "'HV-Rang' = wo diese Kombi-IV im HV-Min/Max-Fenster (oben, Laenge = "
        "gerundete Restlaufzeit) liegt - Werte ausserhalb 0-100% bedeuten "
        "Extrapolation ueber das historische Fenster hinaus."
    )
    st.dataframe(
        show.style.format({
            "Ziel-Abstand % v. Spot": "{:.2f}",
            "Strike Put (real)": "{:.0f}", "Strike Call (real)": "{:.0f}",
            "Mid Put": "{:.2f}", "Mid Call": "{:.2f}",
            "IV Put % (echt)": "{:.2f}", "IV Call % (echt)": "{:.2f}",
            "Δ Put % (echt)": "{:.2f}", "Δ Call % (echt)": "{:.2f}",
            "Δ Put % (Modell, flache HV)": "{:.2f}", "Δ Call % (Modell, flache HV)": "{:.2f}",
            "IV Kombi % (Markt, Paket-Solve)": "{:.2f}", "HV-Rang % (dieser Kombi)": "{:.1f}",
        }, na_rep="n/a"),
        use_container_width=True, hide_index=True,
    )

    st.session_state["livechain_table"] = table
    st.session_state["livechain_chosen_expiry"] = chosen_expiry
    st.session_state["livechain_chosen_dte_exact"] = chosen_dte_exact
    st.session_state["livechain_flat_hv"] = flat_hv
    st.session_state["livechain_hv_min"] = hv_min
    st.session_state["livechain_hv_max"] = hv_max

    with st.expander("Rohdaten dieser Expiry (alle Serien nach Delta-Filter)"):
        st.dataframe(
            enriched[enriched["expiry"] == chosen_expiry][
                ["strike", "option_type", "bid", "ask", "mid", "implied_vol_mkt", "delta_mkt"]
            ].sort_values(["option_type", "strike"]),
            use_container_width=True, hide_index=True,
        )