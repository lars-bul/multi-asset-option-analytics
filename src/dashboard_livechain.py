"""
MODUS 3: Heutige Kette - symmetrische Abstands-Ansicht + Analog-Tage-Tabelle
+ modellfreie Iron-Condor-Suche.

TEIL A: aktuelle SPX-Optionskette als EINE Tabelle, je Ziel-ABSTAND vom Spot
(in %) eine Zeile, Put/Call-Strike unabhaengig auf den naechstgelegenen real
vorhandenen Strike gerundet (chain_delta.build_symmetric_distance_table).

TEIL B (Analog-Tage-Tabelle je Zeile) - EINE Rechnung fuer alles, identisch zur
Condor-Suche ("Rechnung 2"): heutige REALE Strikes und heutige Mid-Praemie
(Mid Put + Mid Call) der Zeile, historische PROZENTUALE Kursbewegungen der
Analog-Tage (exit_S/entry_S - 1) auf den heutigen Spot projiziert.
    pnl_d = Praemie_heute - [max(Kp - S_d, 0) + max(S_d - Kc, 0)],
    S_d = Spot_heute * (1 + Bewegung_d).
Daraus kommen ALLE Kennzahlen der Tabelle: Gewinnrate, Profit Factor,
Erw. Gewinn (= gewichteter Mittelwert von pnl_d), t-Stat (HAC, Kalenderzeit),
Konfidenzintervall, Perioden-Konsistenz, Train/Val, Robust und 'Preis fuer
PF=3'. KEIN Black-76-Modell, keine Vola/HV-Rang-Umrechnung, keine Deltas mehr
in dieser Tabelle (die Spalten aus Teil A dienen nur der Anzeige).
Nebeneffekt: schneller (kein Strike-Solve je Analog-Tag) und ohne die
Division durch winzige Modellpraemien.

ZEILENFARBE = 'SHORT GUELTIG': Gruen (alle Kriterien positiv, siehe
_robustness_row_state) ist die EINZIGE Definition einer gueltigen Short-Zeile.
Die Condor-Suche (Teil C) nutzt genau diese Gruen-Markierung fuer das
Short-Kriterium und das Plateau (Zeile darueber und darunter muessen ebenfalls
gruen sein). Die Long-Fluegel muessen nichts erfuellen. Die Zeilenfarbe nutzt
die feste Schwelle _ROBUST_TSTAT_THRESHOLD; die Condor-Schwelle (Min. t-Stat)
gilt nur fuer den Condor selbst.

Gemeinsame Bausteine: Analog-Suche mit Kernel-Gewichten (analog_matching.
find_analogs), einmalig pro Klick und in st.session_state gecacht
(_find_today_analogs); T_days = naechstgelegener Wert der training_table (1..30,
fehlende Laufzeiten -> Button in _ensure_training_table()); 'Heute' =
synthetischer Balken mit heutigem Eroeffnungskurs; Staleness-Warnung
(Quote-Alter) ab _QUOTE_AGE_WARNING_MINUTES.

TEIL C: Iron-Condor-Suche (condor_search.py) mit Bid/Ask-Preisen der Kette.
"""
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_today_chain import fetch_today_chain, fetch_today_spot, fetch_today_open
from chain_delta import ChainDeltaConfig, enrich_chain_with_delta, build_symmetric_distance_table
from analog_matching import (
    AnalogConfig, SimilaritySpaceConfig, hv_rank_bounds, fit_similarity_space,
    transform_query, find_analogs, weighted_reliability_summary,
    effective_sample_size, available_t_days,
)
from features import FeatureConfig, build_feature_matrix
from dashboard_common import load_training_table
from condor_search import CondorConfig, analog_moves, evaluate_condor_grid

_QUOTE_AGE_WARNING_MINUTES = 60.0
_ANALOG_MIN_CANDIDATES = 20
_LOW_N_EFF_WARNING = 10.0
_ROBUST_TSTAT_THRESHOLD = 2.0
_ROBUST_MIN_PERIODS_MATCH = 5   # mindestens 5 von 6 Teilperioden
_REQUIRED_T_DAYS = list(range(1, 31))

_ROW_STYLE = {
    "green": "background-color: rgba(46, 160, 67, 0.30)",
    "red": "background-color: rgba(218, 54, 51, 0.30)",
}


def _ensure_training_table() -> None:
    """Prueft, ob die training_table alle Laufzeiten 1..30 enthaelt. Falls
    nicht (alte Tabelle), wird ein Hinweis + Button angezeigt, der
    build_dataset.py in einem Subprozess ausfuehrt (dauert einige Minuten).
    Die Seite bleibt bis dahin mit den vorhandenen Laufzeiten nutzbar."""
    try:
        tt = load_training_table()
    except Exception:
        return
    if tt is None or tt.empty:
        return
    have = set(available_t_days(tt))
    missing = [t for t in _REQUIRED_T_DAYS if t not in have]
    if not missing:
        return
    st.warning(
        f"Die Trainingstabelle enthaelt nur die Laufzeiten {sorted(have)} - es fehlen "
        f"{len(missing)} von 30 (T_days 1..30). Einmal neu bauen (dauert einige Minuten)."
    )
    if st.button("Trainingstabelle jetzt neu bauen (T_days 1-30)"):
        script = Path(__file__).resolve().parent / "build_dataset.py"
        with st.spinner("Baue Trainingstabelle - bitte warten, das kann mehrere Minuten dauern..."):
            proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
        if proc.returncode == 0:
            try:
                st.cache_data.clear()
            except Exception:
                pass
            st.session_state.pop("livechain_analog_ctx", None)
            st.success("Trainingstabelle neu gebaut. Seite wird neu geladen.")
            st.rerun()
        else:
            st.error("build_dataset.py ist fehlgeschlagen:")
            st.code((proc.stdout or "")[-1500:] + "\n" + (proc.stderr or "")[-3000:])


def _robustness_row_state(df: pd.DataFrame) -> pd.Series:
    """'green' (alle Robustheitskriterien positiv), 'red' (spiegelbildlich
    negativ) oder '' (Rest). NaN -> ''."""
    T = _ROBUST_TSTAT_THRESHOLD
    tstat_full = pd.to_numeric(df["tstat_full"], errors="coerce")
    tstat_val = pd.to_numeric(df["tstat_val"], errors="coerce")
    periods_ok = pd.to_numeric(df["subperiod_matching"], errors="coerce") >= _ROBUST_MIN_PERIODS_MATCH
    ci_ok = df["ci_excludes_zero_full"].eq(True)
    robust_pos = df["robust_train_val"].eq(True)
    robust_neg = df.get("robust_train_val_neg", pd.Series(False, index=df.index)).eq(True)

    green = (tstat_full > T) & ci_ok & periods_ok & (tstat_val > T) & robust_pos
    red = (tstat_full < -T) & ci_ok & periods_ok & (tstat_val < -T) & robust_neg

    state = pd.Series("", index=df.index)
    state.loc[green] = "green"
    state.loc[red] = "red"
    return state


def _required_premium_for_profit_factor(payout: np.ndarray, weights: np.ndarray,
                                         target_pf: float = 3.0,
                                         max_iter: int = 60) -> float | None:
    """Bisektion: einheitlicher Entry-Preis x, bei dem sich (gewichtet, gegen
    die Payouts auf heutiger Skala) ein Profit Factor von target_pf ergibt.
    PF(x) ist monoton nicht-fallend in x. Sind alle Payouts 0, wird 0.0
    zurueckgegeben; None, falls nicht loesbar oder keine gueltigen Payouts."""
    payout = np.asarray(payout, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = ~np.isnan(payout) & ~np.isnan(weights) & (weights > 0.0)
    payout, weights = payout[mask], weights[mask]
    if len(payout) == 0:
        return None
    if not np.any(payout > 0.0):
        return 0.0

    def pf_at(x: float) -> float:
        net = x - payout
        win = net >= 0.0
        gross_profit = np.sum(weights[win] * net[win])
        gross_loss = -np.sum(weights[~win] * net[~win])
        if gross_loss <= 0.0:
            return np.inf
        return gross_profit / gross_loss

    lo, hi = 0.0, float(payout.max()) * 10.0 + 1.0
    if pf_at(hi) < target_pf:
        return None

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        if pf_at(mid) < target_pf:
            lo = mid
        else:
            hi = mid
    return hi


def _payout_at_todays_scale(entry_S: np.ndarray, exit_S: np.ndarray, weights: np.ndarray,
                             spot_today: float, strike_put: float, strike_call: float
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Payout je Analog-Tag auf HEUTIGER Skala: nur die PROZENTUALE
    Kursbewegung (exit_S/entry_S - 1) wird uebernommen, auf den heutigen Spot
    projiziert und gegen die ECHTEN Strikes ausgewertet. Gibt (payout,
    weights) mit gleicher Maskierung zurueck."""
    entry_S = np.asarray(entry_S, dtype=float)
    exit_S = np.asarray(exit_S, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = (entry_S > 0.0) & ~np.isnan(entry_S) & ~np.isnan(exit_S) & ~np.isnan(weights)
    entry_S, exit_S, weights = entry_S[mask], exit_S[mask], weights[mask]
    if len(entry_S) == 0:
        return np.array([]), np.array([])

    pct_return = exit_S / entry_S - 1.0
    hyp_exit_spot = spot_today * (1.0 + pct_return)
    payout = np.maximum(strike_put - hyp_exit_spot, 0.0) + np.maximum(hyp_exit_spot - strike_call, 0.0)
    return payout, weights


def _build_daily_df_with_today(daily_df: pd.DataFrame):
    """Haengt einen synthetischen 'Heute'-Tagesbalken (open=high=low=close=
    heutiger Eroeffnungskurs) an eine KOPIE von daily_df an. Faellt auf den
    letzten echten Schlusskurs zurueck (used_fallback=True), falls kein
    Eroeffnungskurs verfuegbar ist.
    Rueckgabe: (daily_df_erweitert, query_date, used_fallback)."""
    today = pd.Timestamp(date.today())
    if (daily_df["date"] == today).any():
        return daily_df.copy(), today, False

    open_price = fetch_today_open()
    if open_price is None:
        return daily_df.copy(), daily_df["date"].max(), True

    new_row = pd.DataFrame([{
        "date": today, "open": open_price, "high": open_price,
        "low": open_price, "close": open_price,
    }])
    if "resolution" in daily_df.columns:
        new_row["resolution"] = "daily"
    out = pd.concat([daily_df, new_row], ignore_index=True).sort_values("date").reset_index(drop=True)
    return out, today, False


def _find_today_analogs(daily_df, chosen_dte_exact, max_distance, bandwidth, min_gap):
    """Analog-Suche fuer 'heute' (von Teil B und Teil C gemeinsam genutzt,
    in st.session_state gecacht)."""
    training_table = load_training_table()
    if training_table.empty:
        st.error("Keine Trainingstabelle gefunden - `python src/build_dataset.py` ausfuehren.")
        return None
    valid_t_days = available_t_days(training_table)
    key = (round(chosen_dte_exact, 4), float(max_distance), float(bandwidth), int(min_gap),
           date.today(), len(training_table), tuple(valid_t_days))
    cached = st.session_state.get("livechain_analog_ctx")
    if cached is not None and cached["key"] == key:
        return cached

    T_days_used = min(valid_t_days, key=lambda t: abs(t - chosen_dte_exact))
    if abs(T_days_used - chosen_dte_exact) > 1.0:
        st.info(f"Restlaufzeit {chosen_dte_exact:.2f} Tage wird auf T_days={T_days_used} gerundet "
                f"(verfuegbar: {valid_t_days}).")

    daily_df_today, query_date, used_fallback = _build_daily_df_with_today(daily_df)
    if used_fallback:
        st.warning(f"Kein heutiger Eroeffnungskurs verfuegbar - Analog-Suche nutzt "
                   f"{pd.Timestamp(query_date).date()}.")

    with st.spinner("Baue Aehnlichkeits-Raum und suche Analog-Tage..."):
        try:
            feature_df = build_feature_matrix(daily_df_today, FeatureConfig())
            if query_date not in set(feature_df["date"]):
                st.error(f"{pd.Timestamp(query_date).date()} hat keine vollstaendigen Features "
                         "(Warmup-Fenster noch nicht voll?).")
                return None
            space = fit_similarity_space(feature_df, SimilaritySpaceConfig(),
                                         max_date=query_date + pd.Timedelta(days=1))
            query_score = transform_query(space, feature_df.loc[feature_df["date"] == query_date].iloc[0])
            analogs = find_analogs(
                query_date, query_score, space,
                AnalogConfig(max_distance=max_distance, min_gap_days=min_gap,
                             use_kernel_weighting=True, bandwidth=bandwidth,
                             min_candidates=_ANALOG_MIN_CANDIDATES),
            )
        except ValueError as e:
            st.error(f"Analog-Suche fehlgeschlagen: {e}")
            return None

    ctx = dict(key=key, analogs=analogs, T_days_used=T_days_used, training_table=training_table,
               daily_df_today=daily_df_today, query_date=query_date)
    st.session_state["livechain_analog_ctx"] = ctx
    return ctx


def _run_bulk_analog(table: pd.DataFrame, daily_df: pd.DataFrame,
                      chosen_dte_exact: float, r_free: float, dividend_yield: float,
                      hv_length: int, rank_lookback: int,
                      analog_max_distance: float, analog_bandwidth: float, analog_min_gap: int,
                      spot_today: float) -> pd.DataFrame | None:
    """Analog-Tabelle je Zeile (siehe Moduldokstring, TEIL B): heutige reale
    Strikes + heutige Mid-Praemie gegen die historischen prozentualen
    Kursbewegungen der Analog-Tage. Keine Modellbewertung. (r_free,
    dividend_yield, hv_length, rank_lookback bleiben nur aus Kompatibilitaet in
    der Signatur - sie werden hier nicht mehr gebraucht.)
    Gibt None zurueck (mit st.error) bei Abbruch, sonst ein DataFrame mit einer
    Zeile je distance_pct."""
    ctx = _find_today_analogs(daily_df, chosen_dte_exact, analog_max_distance,
                              analog_bandwidth, analog_min_gap)
    if ctx is None:
        return None
    analogs, T_days_used, training_table = ctx["analogs"], ctx["T_days_used"], ctx["training_table"]
    moves = analog_moves(analogs, training_table, T_days_used, ctx["query_date"])
    if moves.empty:
        st.error("Keine Analog-Tage mit gueltiger Kursbewegung (Exit vor heute) gefunden.")
        return None
    st.caption(f"{len(moves)} Analog-Tage nutzbar (von {len(analogs)} gefunden; gilt fuer alle Zeilen, "
               "stetige Kernel-Gewichte).")

    w = moves["weight"].to_numpy(dtype=float)
    w_sum = float(w.sum())
    S_d = spot_today * (1.0 + moves["ret"].to_numpy(dtype=float))
    n_eff = effective_sample_size(w)
    median_date = moves["entry_date"].median()

    progress = st.progress(0.0)
    status = st.empty()
    results = []
    n = len(table)
    for i, row in enumerate(table.itertuples()):
        status.text(f"Zeile {i + 1}/{n} (Ziel-Abstand {row.distance_pct:g}%)...")
        progress.progress((i + 1) / n)

        empty_row = {
            "distance_pct": row.distance_pct, "analog_n": None, "analog_n_eff": None,
            "analog_n_eff_hac": None, "win_rate_pct": None, "profit_factor": None,
            "avg_pnl_norm_pct": None, "entry_premium_today": None,
            "expected_profit_short": None, "required_premium_pf3": None,
            "tstat_full": None, "ci_excludes_zero_full": False,
            "subperiod_matching": None, "subperiod_nonempty": None,
            "tstat_val": None, "robust_train_val": False, "robust_train_val_neg": False,
        }
        if (pd.isna(row.mid_put) or pd.isna(row.mid_call)
                or pd.isna(row.strike_put) or pd.isna(row.strike_call)):
            results.append(empty_row)
            continue

        premium = float(row.mid_put) + float(row.mid_call)
        payout = (np.maximum(float(row.strike_put) - S_d, 0.0)
                  + np.maximum(S_d - float(row.strike_call), 0.0))
        pnl = premium - payout                       # Gewinn je Analog-Tag, in heutigen Punkten

        ev = moves[["entry_date", "hold_days", "weight"]].copy()
        ev["pnl_today_short"] = pnl
        rel_full = weighted_reliability_summary(ev, "short", 6, "pnl_today")
        train_rel = weighted_reliability_summary(ev[ev["entry_date"] < median_date], "short", 4, "pnl_today")
        val_rel = weighted_reliability_summary(ev[ev["entry_date"] >= median_date], "short", 4, "pnl_today")

        robust, robust_neg = False, False
        if train_rel["n_eff_hac"] >= 2.0 and val_rel["n_eff_hac"] >= 2.0:
            same_sign = np.sign(train_rel["mean_pnl_norm"]) == np.sign(val_rel["mean_pnl_norm"])
            both_sig = train_rel["ci_excludes_zero"] and val_rel["ci_excludes_zero"]
            robust = bool(both_sig and same_sign and np.sign(train_rel["mean_pnl_norm"]) > 0.0)
            robust_neg = bool(both_sig and same_sign and np.sign(train_rel["mean_pnl_norm"]) < 0.0)

        mean_pnl = float(np.sum(w * pnl) / w_sum)
        win = pnl >= 0.0
        gross_profit = float(np.sum(w[win] * pnl[win]))
        gross_loss = float(-np.sum(w[~win] * pnl[~win]))

        results.append({
            "distance_pct": row.distance_pct,
            "analog_n": len(moves),
            "analog_n_eff": n_eff,
            "analog_n_eff_hac": rel_full["n_eff_hac"],
            "win_rate_pct": float(w[win].sum() / w_sum * 100.0),
            "profit_factor": (gross_profit / gross_loss) if gross_loss > 0.0 else np.nan,
            "avg_pnl_norm_pct": (mean_pnl / premium * 100.0) if premium > 0.0 else None,
            "entry_premium_today": premium,
            "expected_profit_short": mean_pnl,
            "required_premium_pf3": _required_premium_for_profit_factor(payout, w, target_pf=3.0),
            "tstat_full": rel_full["tstat"],
            "ci_excludes_zero_full": rel_full["ci_excludes_zero"],
            "subperiod_matching": rel_full["subperiod_matching"],
            "subperiod_nonempty": rel_full["subperiod_nonempty"],
            "tstat_val": val_rel["tstat"],
            "robust_train_val": robust,
            "robust_train_val_neg": robust_neg,
        })

    status.empty()
    progress.empty()

    out = pd.DataFrame(results)
    out["T_days_used"] = T_days_used
    out["n_analogs_total"] = len(moves)
    return out


def _render_condor_section(table, daily_df, chosen_expiry, chosen_dte_exact, spot,
                           max_distance, bandwidth, min_gap, distance_key) -> None:
    """TEIL C: modellfreie Iron-Condor-Suche (siehe condor_search.py)."""
    st.divider()
    st.subheader("Iron-Condor-Suche (modellfrei, tagesgenau)")
    st.caption(
        "Jedes Paar (Short-Zeile i, Fluegel-Zeile j mit groesserem Abstand) wird auf DENSELBEN "
        "Analog-Tagen bewertet: Kredit = heutige Bid/Ask-Preise, Payout = historische %-Bewegung "
        "auf den heutigen Spot projiziert. Short und Fluegel gemeinsam pro Tag (korrelationsrichtig), "
        "kein Vola-Modell. RANKING: unter allen Kandidaten nach der gewaehlten Sortierung (Standard: Rendite/Risiko %). "
        "Kandidat = Condor mit Erwartungswert in Train UND Val > 0, HAC-t-Stat > Schwelle (nur Schwelle, "
        "kein Ranking), Tail-Kriterien erfuellt, UND die Short-Zeile ist in der Analog-Tabelle oben GRUEN "
        "(= 'Short gueltig', eine gemeinsame Definition) UND Plateau: die Zeilen DIREKT ueber und "
        "unter (Abstand -1/+1 Raster) sind dort ebenfalls gruen. "
        "Die Long-Fluegel muessen keine Kriterien erfuellen. Randzeilen der Kette koennen nie Optimum sein."
    )
    if "bid_put" not in table.columns:
        st.error("Distanz-Tabelle hat keine Bid/Ask-Spalten - Kette neu laden.")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        basis = st.radio("Preisbasis", ["Bid/Ask (echt)", "Mid (Näherung)"], key="condor_basis",
                         help="Ausserhalb der Handelszeit liefert Yahoo oft Bid/Ask=0 - dann Mid waehlen.")
    with c2:
        t_min = st.number_input("Min. t-Stat (HAC)", 0.0, 10.0, 3.0, 0.1, key="condor_tmin3")
    with c3:
        max_full = st.number_input("Max. Anteil Volltreffer-Verlust (%)", 0.0, 100.0, 3.0, 0.5,
                                   key="condor_fullloss")
    with c4:
        tail_floor = st.number_input("Min. 5%-Quantil (Rendite/Max-Loss)", -1.0, 0.0, -0.75, 0.05,
                                     key="condor_tail")
    with c5:
        sort_label = st.selectbox(
            "Sortierung", ["Rendite/Risiko %", "E[P&L] (Pkt)", "t (HAC)"], index=0, key="condor_sort",
            help="Standard: Rendite/Risiko % (= E[P&L] / Max-Loss). Achtung: bevorzugt schmale Fluegel, "
                 "weil der Max-Loss im Nenner steht - Voll-Verlust-Haeufigkeit und 5%-Quantil in der "
                 "Tabelle mitlesen.",
        )

    # Short-Gueltigkeit aus der Analog-Tabelle oben (Short-Strangle je Zeile) - identisch
    # zu dem, was dort gruen/robust angezeigt wird.
    stored_sv = st.session_state.get("livechain_analog_results")
    stored_key = st.session_state.get("livechain_analog_key")
    short_valid = None
    if stored_sv is not None and stored_key == distance_key:
        # EINE Definition: Short gueltig == Zeile ist in der Analog-Tabelle GRUEN.
        state = _robustness_row_state(stored_sv)
        short_valid = dict(zip(stored_sv["distance_pct"].astype(float).tolist(),
                               state.eq("green").astype(bool).tolist()))

    if short_valid is None:
        st.info("Bitte zuerst oben 'Analog-Läufe für alle Zeilen starten' - das Short-Kriterium "
                "und das Plateau der Condor-Suche stützen sich auf diese Tabelle.")

    if st.button("Iron-Condor-Suche starten") and short_valid is not None:
        ctx = _find_today_analogs(daily_df, chosen_dte_exact, max_distance, bandwidth, min_gap)
        if ctx is not None:
            moves = analog_moves(ctx["analogs"], ctx["training_table"], ctx["T_days_used"], ctx["query_date"])
            cfg = CondorConfig(price_basis="mid" if basis.startswith("Mid") else "bidask",
                               t_min=float(t_min), max_full_loss_share=max_full / 100.0,
                               tail_floor=float(tail_floor),
                               sort_by={"Rendite/Risiko %": "ror", "E[P&L] (Pkt)": "mean", "t (HAC)": "tstat"}[sort_label])
            meta_sort = sort_label
            cdf, sdf, meta = evaluate_condor_grid(table, moves, spot, cfg, short_valid)
            meta["T_days_used"] = ctx["T_days_used"]
            meta["sort_label"] = meta_sort
            st.session_state["livechain_condor"] = dict(key=distance_key, cdf=cdf, sdf=sdf, meta=meta)

    res = st.session_state.get("livechain_condor")
    if res is None:
        return
    if res["key"] != distance_key:
        st.info("Tabelle/Expiry hat sich geaendert - Suche erneut starten.")
        return
    cdf, sdf, meta = res["cdf"], res["sdf"], res["meta"]
    if cdf.empty:
        st.warning("Kein gueltiges Paar (Kredit<=0 oder Bid/Ask fehlt). Bei Bid/Ask=0: Preisbasis 'Mid' testen.")
        return

    st.caption(
        f"T_days={meta['T_days_used']} · {meta['n_days']} Analog-Tage (n_eff Kish {meta['n_eff']:.1f}, "
        f"max. Haltedauer {meta['hold_max']}d) · {meta['n_pairs']} Paare getestet, "
        f"{meta['n_base']} erfuellen die Basis-Kriterien, {meta['n_pass']} bestehen zusaetzlich Short-Kriterium und Plateau."
    )
    if meta["n_pass"] == 0:
        st.info("Kein Paar besteht alle Kriterien - kein belastbarer Condor-Edge fuer die heutige Lage.")
    else:
        b = cdf.iloc[0]
        st.success(
            f"Optimum (beste Zeile nach '{meta.get('sort_label', 'Rendite/Risiko %')}' unter allen Kriterien): Short {b['strike_put_short']:.0f}P / {b['strike_call_short']:.0f}C, "
            f"Fluegel {b['strike_put_wing']:.0f}P / {b['strike_call_wing']:.0f}C · Kredit {b['credit']:.2f}, "
            f"Max-Loss {b['max_loss']:.2f} · E[P&L] {b['mean']:+.2f} Pkt = {b['ror']*100:+.1f}% auf Risikokapital "
            f"(t={b['tstat']:.2f}, Val-E {b['e_val']:+.2f})."
        )

    yn = lambda s: s.map({True: "✅", False: "❌"})
    show = pd.DataFrame({
        "Short-Abstand %": cdf["distance_short"], "Flügel-Abstand %": cdf["distance_wing"],
        "Put Flügel": cdf["strike_put_wing"], "Put Short": cdf["strike_put_short"],
        "Call Short": cdf["strike_call_short"], "Call Flügel": cdf["strike_call_wing"],
        "Kredit": cdf["credit"], "Max-Loss": cdf["max_loss"],
        "E[P&L] (Pkt)": cdf["mean"], "Rendite/Risiko %": cdf["ror"] * 100,
        "E Put-Spread": cdf["e_put_spread"], "E Call-Spread": cdf["e_call_spread"],
        "t (HAC)": cdf["tstat"], "E Train": cdf["e_train"], "E Val": cdf["e_val"],
        "Voll-Verlust %": cdf["full_loss_share"] * 100, "Voll-Verlust-Tage (n)": cdf["n_full_loss"],
        "5%-Quantil (R/R)": cdf["q_tail"],
        "Gewinnrate %": cdf["win_rate_pct"],
        "Train>0": yn(cdf["c_train"]), "Val>0": yn(cdf["c_val"]), "t": yn(cdf["c_t"]),
        "Tail": yn(cdf["c_tail"]), "Short grün": yn(cdf["c_short"]),
        "Plateau (Short ±1)": yn(cdf["c_plateau"]),
        "Kandidat": yn(cdf["passes"]),
    })
    fmt = {c: "{:.2f}" for c in show.select_dtypes("number").columns}
    fmt.update({c: "{:.0f}" for c in ["Put Flügel", "Put Short", "Call Short", "Call Flügel", "Voll-Verlust-Tage (n)"]})
    passing = show[cdf["passes"].to_numpy()]
    if len(passing):
        st.dataframe(passing.style.format(fmt, na_rep="n/a"), use_container_width=True, hide_index=True)
    with st.expander(f"Alle getesteten Paare ({len(show)}) inkl. Kriterien"):
        st.dataframe(show.style.format(fmt, na_rep="n/a"), use_container_width=True, hide_index=True)
    if "short_df" in meta:
        with st.expander("Short-Strangle je Zeile (Basis für Short-Kriterium und Plateau)"):
            sh = meta["short_df"].copy()
            sh["ok"] = sh["ok"].map({True: "✅", False: "❌"})
            st.dataframe(sh.style.format({c: "{:.2f}" for c in sh.select_dtypes("number").columns},
                                         na_rep="n/a"), use_container_width=True, hide_index=True)
    with st.expander("Put- und Call-Spreads einzeln (welche Seite trägt den Edge?)"):
        sd = sdf.sort_values(["side", "ror"], ascending=[True, False])[[
            "side", "distance_short", "distance_wing", "strike_short", "strike_wing",
            "credit", "max_loss", "mean", "ror", "tstat", "e_train", "e_val",
        ]].copy()
        sd["ror"] *= 100
        st.dataframe(sd.style.format({c: "{:.2f}" for c in sd.select_dtypes("number").columns}, na_rep="n/a"),
                     use_container_width=True, hide_index=True)


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

    _ensure_training_table()

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
            "0,0.01,0.1,0.2,0.25,0.3,0.4,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.5,4,5,6,8,10",
        )
        rank_lookback = st.number_input(
            "Lookback fuer HV-Min/Max-Rangfenster (Tage)", 30, 5000, 252,
            help="Laenge des rollierenden Fensters, in dem hv_min/hv_max fuer den "
                 "HV-Rang der Kombi-IV gesucht wird - identisches Prinzip zu "
                 "backtest.FixedRankConfig.lookback_days.",
        )
        r_free = st.number_input("Risikoloser Zins", 0.0, 0.20, 0.0, 0.005, format="%.3f")
        dividend_yield = st.number_input("Dividendenrendite", 0.0, 0.10, 0.015, 0.001, format="%.3f")
        reload_chain = st.button(
            "Heutige Kette neu laden", type="primary",
            help="Beim ERSTEN Oeffnen dieses Tabs wird die Kette automatisch geladen - "
                 "dieser Button ist nur fuer ein manuelles Neuladen noetig, z.B. nach "
                 "Aenderung von DTE-Filter/Delta-Filter oder fuer einen aktuelleren Snapshot.",
        )

    needs_initial_load = "livechain_enriched" not in st.session_state
    if reload_chain or needs_initial_load:
        with st.spinner("Lade heutige Kette und berechne Delta/IV je Serie..."):
            try:
                fetch_ts = pd.Timestamp.now(tz="UTC")
                spot = fetch_today_spot()
                chain = fetch_today_chain(min_dte=float(min_dte), max_dte=float(max_dte))
                enriched = enrich_chain_with_delta(
                    chain, spot,
                    ChainDeltaConfig(min_abs_delta=float(min_abs_delta),
                                      r_free=float(r_free), dividend_yield=float(dividend_yield)),
                    as_of=fetch_ts,
                )
            except Exception as e:
                st.error(f"Laden fehlgeschlagen: {e}")
                if needs_initial_load:
                    return
            else:
                st.session_state["livechain_spot"] = spot
                st.session_state["livechain_enriched"] = enriched
                st.session_state["livechain_r_free"] = float(r_free)
                st.session_state["livechain_dividend_yield"] = float(dividend_yield)
                st.session_state["livechain_rank_lookback"] = int(rank_lookback)
                st.session_state["livechain_fetch_ts"] = fetch_ts

    if "livechain_enriched" not in st.session_state:
        st.info("Kette konnte noch nicht geladen werden - siehe Fehlermeldung oben.")
        return

    spot = st.session_state["livechain_spot"]
    enriched = st.session_state["livechain_enriched"]
    r_free = st.session_state["livechain_r_free"]
    dividend_yield = st.session_state["livechain_dividend_yield"]
    rank_lookback = st.session_state["livechain_rank_lookback"]
    fetch_ts = st.session_state.get("livechain_fetch_ts")
    st.metric("Aktueller Spot (SPX)", f"{spot:,.2f}")
    if fetch_ts is not None:
        st.caption(
            f"Kette abgerufen um {fetch_ts.tz_convert('America/New_York').strftime('%Y-%m-%d %H:%M:%S')} ET "
            "- Referenzzeitpunkt fuer die 'Quote-Alter'-Spalten weiter unten."
        )

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

    stale_mask = (
        table["quote_age_put_min"].fillna(0) > _QUOTE_AGE_WARNING_MINUTES
    ) | (
        table["quote_age_call_min"].fillna(0) > _QUOTE_AGE_WARNING_MINUTES
    )
    if stale_mask.any():
        stale_rows = table.loc[stale_mask, "distance_pct"].tolist()
        preview = ", ".join(f"{d:g}%" for d in stale_rows[:10])
        more = f" (+{len(stale_rows) - 10} weitere)" if len(stale_rows) > 10 else ""
        st.warning(
            f"⚠️ Veraltete Einzelquote(n) vermutet: bei {len(stale_rows)} Zeile(n) "
            f"({preview}{more}) liegt der letzte tatsaechliche Trade von Put oder "
            f"Call laenger als {_QUOTE_AGE_WARNING_MINUTES:g} Minuten zurueck (siehe "
            "Spalten 'Quote-Alter Put/Call (Min.)' unten). Das ist KEIN Beweis fuer "
            "eine falsche Quote (bei ruhigem Markt bleibt eine Quote laenger gueltig), "
            "aber ein Hinweis, IV/Preis fuer diese Zeilen nicht ungeprueft zu "
            "uebernehmen - insbesondere wenn Put und Call fuer denselben Abstand "
            "deutlich unterschiedliche IV zeigen."
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
        "quote_age_put_min": "Quote-Alter Put (Min.)", "quote_age_call_min": "Quote-Alter Call (Min.)",
    })[[
        "Ziel-Abstand % v. Spot",
        "Δ Put % (Modell, flache HV)", "Δ Call % (Modell, flache HV)",
        "IV Kombi % (Markt, Paket-Solve)", "HV-Rang % (dieser Kombi)",
        "Strike Put (real)", "Strike Call (real)",
        "Mid Put", "Mid Call",
        "IV Put % (echt)", "IV Call % (echt)",
        "Δ Put % (echt)", "Δ Call % (echt)",
        "Quote-Alter Put (Min.)", "Quote-Alter Call (Min.)",
    ]]
    st.caption(
        "'Δ Put/Call (echt)' sind die TATSAECHLICHEN, aus dem realen Mid-Preis per "
        "Black-76 zurueckgerechneten Deltas der jeweils gewaehlten Option - koennen "
        "leicht vom Ziel abweichen, da nur reale (diskrete) Strikes existieren. "
        "'IV Kombi' = die EINE Vola, die - in Black-76 fuer Put+Call GEMEINSAM "
        "eingesetzt - exakt die Summe beider echter Mid-Preise reproduziert (kein "
        "Mittelwert der Einzel-IVs). 'HV-Rang' = wo diese Kombi-IV im HV-Min/Max-"
        "Fenster oben liegt. 'Quote-Alter' = Minuten seit dem letzten TATSAECHLICHEN "
        "Trade dieses Strikes (Naeherungswert fuer Staleness)."
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
            "Quote-Alter Put (Min.)": "{:.0f}", "Quote-Alter Call (Min.)": "{:.0f}",
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
                ["strike", "option_type", "bid", "ask", "mid", "implied_vol_mkt", "delta_mkt",
                 "last_trade_date", "quote_age_minutes"]
            ].rename(columns={
                "last_trade_date": "Letzter Trade (yfinance)",
                "quote_age_minutes": "Quote-Alter (Min.)",
            }).sort_values(["option_type", "strike"]),
            use_container_width=True, hide_index=True,
        )

    # ── TEIL B: Analog-Tage-Simulation je Zeile ──
    st.divider()
    st.subheader("Analog-Tage-Simulation je Zeile")
    st.caption(
        "Fuer JEDE Zeile oben: heutige REALE Strikes und heutige Mid-Praemie (Mid Put + Mid Call), "
        "bewertet gegen die historischen prozentualen Kursbewegungen der Analog-Tage (auf den heutigen "
        "Spot projiziert, gewichtet, bis Verfall gehalten). Kein Vola-/Black-Modell, keine "
        "Rang-Umrechnung. Diese EINE Rechnung liefert alle Kennzahlen der Tabelle und dieselbe "
        "Grundlage wie die Iron-Condor-Suche unten."
    )
    st.caption(
        "**Kontinuierliche Gewichtung statt harter Distanz-Schwelle:** jeder Analog-Tag geht "
        "mit einem stetigen Kernel-Gewicht ein; mindestens "
        f"{_ANALOG_MIN_CANDIDATES} Tage werden immer aufgenommen. 'Effektive n (Kish)' und "
        "'Effektive n (HAC)' zeigen, wie viel echte Vergleichsinformation dahintersteckt "
        "(HAC beruecksichtigt zusaetzlich die Ueberlappung benachbarter Analog-Tage - "
        "die t-Stats sind autokorrelationskorrigiert)."
    )
    st.caption(
        "'Erw. Gewinn Short' (Punkte) = gewichteter Mittelwert von (heutige Praemie - Payout) ueber "
        "die Analog-Tage; 'Ø norm. P&L %' = dieser Wert / heutige Praemie; Gewinnrate und Profit "
        "Factor sind gewichtet auf derselben Basis. Bewertet NUR das Short-Bein, NICHT die Wings - "
        "fuer Condors siehe Teil C unten."
    )
    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        analog_max_distance = st.number_input(
            "Max. Distanz (Performance-Grenze)", min_value=0.01, max_value=100.0, value=4.0, step=0.5,
            help="Kandidaten weiter als dieser Wert entfernt werden nur noch ueber die "
                 "Mindest-Kandidaten-Garantie beruecksichtigt.",
        )
    with ac2:
        analog_bandwidth = st.number_input(
            "Kernel-Bandbreite σ (Gaussian)", min_value=0.01, max_value=50.0, value=1.0, step=0.1,
            help="Jeder Analog-Tag geht mit Gewicht exp(-(Distanz/σ)²) ein. Kleiner σ = "
                 "staerkere Fokussierung auf die aehnlichsten Tage. Mindestens "
                 f"{_ANALOG_MIN_CANDIDATES} Tage werden IMMER aufgenommen.",
        )
    with ac3:
        analog_min_gap = st.number_input(
            "Mindestabstand zu heute (Handelstage)", min_value=0, max_value=60, value=5,
        )

    run_bulk = st.button("Analog-Läufe für alle Zeilen starten")
    if run_bulk:
        analog_results = _run_bulk_analog(
            table, daily_df, chosen_dte_exact, r_free, dividend_yield,
            hv_length, int(rank_lookback), float(analog_max_distance), float(analog_bandwidth),
            int(analog_min_gap), spot,
        )
        if analog_results is not None:
            st.session_state["livechain_analog_results"] = analog_results
            st.session_state["livechain_analog_key"] = (chosen_expiry, tuple(distance_targets_pct))

    stored = st.session_state.get("livechain_analog_results")
    stored_key = st.session_state.get("livechain_analog_key")
    if stored is not None and stored_key == (chosen_expiry, tuple(distance_targets_pct)):
        merged = table.merge(stored, on="distance_pct", how="left")

        if "analog_n_eff_hac" in merged.columns:
            low_neff = merged["analog_n_eff_hac"].dropna()
            low_neff = low_neff[low_neff < _LOW_N_EFF_WARNING]
            if len(low_neff) > 0:
                st.warning(
                    f"⚠️ {len(low_neff)} Zeile(n) haben eine effektive Stichprobengroesse "
                    f"(HAC) unter {_LOW_N_EFF_WARNING:g} - fuer diese Zeilen existieren kaum "
                    "unabhaengige nahe historische Vergleichstage (oder sie stammen aus wenigen "
                    "ueberlappenden Ereignissen). Die Kennzahlen sind auf sehr duenner "
                    "Datenbasis berechnet."
                )

        row_state = _robustness_row_state(merged)

        show2 = merged.rename(columns={
            "distance_pct": "Ziel-Abstand % v. Spot",
            "strike_put": "Strike Put (real)",
            "strike_call": "Strike Call (real)",
            "analog_n": "Analog-Trades (n)",
            "analog_n_eff": "Effektive n (Kish)",
            "analog_n_eff_hac": "Effektive n (HAC)",
            "win_rate_pct": "Gewinnrate %",
            "profit_factor": "Profit Factor",
            "avg_pnl_norm_pct": "Ø norm. P&L %",
            "entry_premium_today": "Entry-Prämie heute (Pkt)",
            "expected_profit_short": "Erw. Gewinn Short (Pkt)",
            "required_premium_pf3": "Preis für PF=3 (Pkt)",
            "tstat_full": "t-Stat (gesamt)",
            "ci_excludes_zero_full": "CI ≠ 0 (gesamt)",
            "subperiod_matching": "Perioden-Match",
            "subperiod_nonempty": "Perioden gesamt",
            "tstat_val": "t-Stat (Val)",
            "robust_train_val": "Robust (Train+Val)",
        })[[
            "Ziel-Abstand % v. Spot", "Strike Put (real)", "Strike Call (real)",
            "Analog-Trades (n)", "Effektive n (Kish)", "Effektive n (HAC)",
            "Gewinnrate %", "Profit Factor",
            "Ø norm. P&L %", "Entry-Prämie heute (Pkt)",
            "Erw. Gewinn Short (Pkt)", "Preis für PF=3 (Pkt)",
            "t-Stat (gesamt)", "CI ≠ 0 (gesamt)",
            "Perioden-Match", "Perioden gesamt", "t-Stat (Val)", "Robust (Train+Val)",
        ]]
        st.caption(
            f"T_days fuer die Analog-Bewertung: "
            f"{int(stored['T_days_used'].iloc[0]) if len(stored) else 'n/a'} "
            f"(gerundet von {chosen_dte_exact:.2f} Tagen Restlaufzeit) · "
            f"{int(stored['n_analogs_total'].iloc[0]) if len(stored) else 0} Analog-Tage gefunden. "
            "'Robust (Train+Val)' = Konfidenzintervall schliesst 0 aus UND gleiches positives "
            "Vorzeichen in BEIDEN chronologischen Haelften. 'Preis für PF=3' = welcher "
            "einheitliche Entry-Preis auf dem Analog-Set (gewichtet, gegen die historischen "
            "Payouts auf heutiger Skala) einen Profit Factor von genau 3 ergeben haette - rein "
            "historische Rueckrechnung, keine Zukunftsgarantie. Alle Kennzahlen dieser Tabelle "
            "(Gewinnrate, Profit Factor, t-Stat, Robust) nutzen dieselbe Rechnung: heutige "
            "Mid-Praemie minus historischer Payout."
        )
        st.caption(
            "**Zeilenfarben:** 🟩 grün = alle Robustheitskriterien positiv erfüllt "
            f"(t-Stat gesamt > {_ROBUST_TSTAT_THRESHOLD:g}, CI ≠ 0, Perioden-Match ≥ "
            f"{_ROBUST_MIN_PERIODS_MATCH}/6, t-Stat Val > {_ROBUST_TSTAT_THRESHOLD:g}, "
            "Robust Train+Val). 🟥 rot = spiegelbildlich negativ. Alle t-Werte sind "
            "HAC-korrigiert. Grün = 'Short gültig' in der Iron-Condor-Suche (gemeinsame Definition). "
            "Alle anderen Zeilen ungefärbt."
        )
        st.dataframe(
            show2.style.apply(
                lambda r: [_ROW_STYLE.get(row_state.loc[r.name], "")] * len(r), axis=1,
            ).format({
                "Ziel-Abstand % v. Spot": "{:.2f}",
                "Strike Put (real)": "{:.0f}",
                "Strike Call (real)": "{:.0f}",
                "Analog-Trades (n)": "{:.0f}",
                "Effektive n (Kish)": "{:.1f}",
                "Effektive n (HAC)": "{:.1f}",
                "Gewinnrate %": "{:.1f}",
                "Profit Factor": "{:.2f}",
                "Ø norm. P&L %": "{:+.2f}",
                "Entry-Prämie heute (Pkt)": "{:.2f}",
                "Erw. Gewinn Short (Pkt)": "{:+.2f}",
                "Preis für PF=3 (Pkt)": "{:.2f}",
                "t-Stat (gesamt)": "{:.2f}",
                "Perioden-Match": "{:.0f}",
                "Perioden gesamt": "{:.0f}",
                "t-Stat (Val)": "{:.2f}",
            }, na_rep="n/a"),
            use_container_width=True, hide_index=True,
        )
    elif stored is not None:
        st.info("Tabelle/Expiry hat sich seit dem letzten Analog-Lauf geaendert - bitte erneut starten.")

    # ── TEIL C: Iron-Condor-Suche ──
    _render_condor_section(
        table, daily_df, chosen_expiry, chosen_dte_exact, spot,
        float(analog_max_distance), float(analog_bandwidth), int(analog_min_gap),
        (chosen_expiry, tuple(distance_targets_pct)),
    )