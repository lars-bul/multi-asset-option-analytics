"""
MODUS 3: Heutige Kette - symmetrische Abstands-Ansicht + Analog-Tage-Tabelle
+ modellfreie Iron-Condor-Suche (inkl. Reverse Iron Condor).

MULTI-INSTRUMENT-UMBAU: dieses Modul bekommt jetzt zusaetzlich das oben
gewaehlte Instrument (siehe dashboard.py) und ist vollstaendig darauf
umgestellt:
  - Optionskette kommt ueber die Provider-Registry (chain_providers.base.
    get_provider(instr)) statt fest ueber fetch_today_chain - Instrumente ohne
    Live-Kette (aktuell: Nikkei, EuroStoxx50 - siehe instruments.py) zeigen
    hier einen Hinweis statt abzustuerzen.
  - Trainingstabelle/Marktdaten-Snapshot/alle internen Caches sind pro
    Instrument getrennt (Schluessel enthaelt instr.key bzw. den
    instrumentenspezifischen Dateipfad), damit ein Wechsel des Instruments
    NICHT versehentlich Ergebnisse eines anderen Instruments anzeigt.
  - Die 'Zum Depot hinzufuegen'-Buttons schreiben das aktive Instrument mit
    in die neue Kombo (gemeinsames Depot, siehe depot_engine.py).

KORREKTUR (Trainingstabellen-Neubau jetzt fuer JEDES Instrument): build_dataset.py
ist nicht mehr SPX-spezifisch fest verdrahtet (nimmt jetzt einen Instrument-
Schluessel als Kommandozeilenargument, siehe dort) - der 'Trainingstabelle
jetzt (neu) bauen'-Button in _ensure_training_table() steht deshalb jetzt fuer
ALLE Instrumente zur Verfuegung, nicht mehr nur fuer SPX. Fehlt die Tabelle
komplett ODER fehlen nur einzelne Laufzeiten (z.B. weil eine gefragte
Restlaufzeit noch nicht in T_days 1..90 abgedeckt war), wird derselbe Button
angezeigt.

TEIL A: aktuelle Optionskette als EINE Tabelle, je Ziel-ABSTAND vom Spot
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

ZEILENFARBE = 'SHORT GUELTIG': Gruen (alle Kriterien positiv, siehe
_robustness_row_state) ist die EINZIGE Definition einer gueltigen Short-Zeile.
Die Condor-Suche (Teil C) nutzt genau diese Gruen-Markierung fuer das
Short-Kriterium und das Plateau (Zeile darueber und darunter muessen ebenfalls
gruen sein). Die Long-Fluegel muessen nichts erfuellen.

Gemeinsame Bausteine: Analog-Suche mit Kernel-Gewichten (analog_matching.
find_analogs), einmalig pro Klick und in st.session_state gecacht
(_find_today_analogs); T_days = naechstgelegener Wert der training_table
DIESES INSTRUMENTS (1..90, fehlende Laufzeiten -> Button in
_ensure_training_table(), jetzt fuer jedes Instrument, siehe oben);
'Heute' = synthetischer Balken mit heutigem Eroeffnungskurs; Staleness-Warnung
(Quote-Alter) ab _QUOTE_AGE_WARNING_MINUTES.

TEIL C: Iron-Condor-Suche (condor_search.py) mit Bid/Ask-Preisen der Kette.
NEU: direkt darunter die Reverse-Iron-Condor-Suche (Long-Condor, inneren
Strangle kaufen, Fluegel verkaufen) - gleiche Analog-Tage (moves), gleiche
Tabelle, gleicher Auto-Lauf-Block und gleicher Cache wie der Short-Condor
(kein zusaetzlicher Analog-Lauf, keine zusaetzliche Ladezeit).

AUTOMATISCHER ABLAUF: Sobald dieser Tab geoeffnet wird, laedt die Kette (Auto-Load), danach
rechnet Teil B (Analog-Tabelle) und anschliessend Teil C (Condor-Suche) von selbst. Neu gerechnet
wird nur bei geaenderter Kette/Verfall/Raster/Einstellungen/Instrument (Schluessel in
st.session_state); ein Tab-Wechsel loescht nichts. Die Buttons 'neu berechnen' erzwingen eine
Neuberechnung.
"""
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import market_data
from instruments import InstrumentConfig, INSTRUMENTS
from chain_providers.base import get_provider, UnsupportedChainError
from chain_delta import build_symmetric_distance_table
from analog_matching import (
    AnalogConfig, SimilaritySpaceConfig, hv_rank_bounds, fit_similarity_space,
    transform_query, find_analogs,
    effective_sample_size, available_t_days,
)
from features import FeatureConfig, build_feature_matrix
from dashboard_common import load_training_table
from condor_search import (CondorConfig, analog_moves, evaluate_condor_grid, analog_robustness,
                           evaluate_reverse_condor_grid)

_QUOTE_AGE_WARNING_MINUTES = 60.0
_ANALOG_MIN_CANDIDATES = 20
_LOW_N_EFF_WARNING = 10.0
_ROBUST_TSTAT_THRESHOLD = 2.0
_ROBUST_MIN_PERIODS_MATCH = 5   # mindestens 5 von 6 Teilperioden
_REQUIRED_T_DAYS = list(range(1, 91))

_ROW_STYLE = {
    "green": "background-color: rgba(46, 160, 67, 0.30)",
    "red": "background-color: rgba(218, 54, 51, 0.30)",
}


_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# MODULWEITE CACHES (gelten fuer ALLE Modi und Browser-Sitzungen, ueberleben Reruns/Tab-Wechsel):
_ANALOG_CORE: dict = {}    # Analog-Suche (PCA-Fit + Analog-Tage), Schluessel enthaelt instr.key
_TT_DAYS: dict = {}        # vorhandene T_days der Trainingstabelle je (instr.key, Dateisignatur)
_BOUNDS_CACHE: dict = {}   # HV-Min/Max-Fenster je (instr.key, Datenstand, HV-Laenge, Lookback)
_RESULT_CACHE: dict = {}   # letzte Ergebnisse von Teil B ('analog') und Teil C ('condor') samt Schluessel


def _tt_file(instr: InstrumentConfig) -> Path:
    """Instrumentenspezifischer Pfad zur Trainingstabelle, z.B.
    data/gold_training_table.parquet statt eines einzigen globalen SPX-Pfads."""
    from instruments import data_file
    return data_file(instr, "training_table", _DATA_DIR)


def _training_table_signature(instr: InstrumentConfig):
    """(mtime, Groesse) der Trainingstabelle DIESES Instruments - aendert sich
    nur beim Neubau. None, falls die Datei fehlt."""
    try:
        s = _tt_file(instr).stat()
        return (s.st_mtime_ns, s.st_size)
    except OSError:
        return None


def _training_days(instr: InstrumentConfig) -> set | None:
    """Vorhandene T_days DIESES Instruments, OHNE die ganze Tabelle
    (hunderttausende Zeilen) zu laden. None, falls nicht bestimmbar."""
    sig = _training_table_signature(instr)
    cache_key = (instr.key, sig)
    if sig is not None and cache_key in _TT_DAYS:
        return _TT_DAYS[cache_key]
    days = None
    if sig is not None:
        try:
            days = set(pd.read_parquet(_tt_file(instr), columns=["T_days"])["T_days"].unique().tolist())
        except Exception:
            days = None
    if days is None:
        try:
            tt = load_training_table(instr)
            if tt is None or tt.empty:
                return None
            days = set(available_t_days(tt))
        except Exception:
            return None
    if sig is not None:
        _TT_DAYS[cache_key] = days
    return days


def _bounds_cached(daily_df: pd.DataFrame, hv_length: int, lookback: int, instr: InstrumentConfig) -> pd.DataFrame:
    key = (instr.key, len(daily_df), daily_df["date"].max(), int(hv_length), int(lookback))
    if key not in _BOUNDS_CACHE:
        if len(_BOUNDS_CACHE) > 24:
            _BOUNDS_CACHE.clear()
        _BOUNDS_CACHE[key] = hv_rank_bounds(daily_df, hv_length, lookback)
    return _BOUNDS_CACHE[key]


def _ensure_training_table(instr: InstrumentConfig) -> None:
    """Prueft, ob die Trainingstabelle DIESES Instruments alle Laufzeiten
    1..90 enthaelt. Fehlt sie komplett ODER fehlen nur einzelne Laufzeiten
    (z.B. eine gerade gefragte Restlaufzeit), wird ein Hinweis + Button zum
    (Neu-)Bauen angezeigt.

    KORREKTUR (vorher SPX-spezifisch): build_dataset.py nimmt jetzt einen
    Instrument-Schluessel als Kommandozeilenargument entgegen (statt fest an
    data/spx_daily.parquet -> data/training_table.parquet zu haengen) - der
    Button ruft es deshalb mit `instr.key` auf und steht fuer JEDES Instrument
    zur Verfuegung, nicht mehr nur fuer SPX."""
    have = _training_days(instr)
    if not have:
        st.info(
            f"Noch keine Trainingstabelle fuer {instr.label} gefunden "
            f"(`{_tt_file(instr).name}`)."
        )
        missing = list(_REQUIRED_T_DAYS)
    else:
        missing = [t for t in _REQUIRED_T_DAYS if t not in have]
        if not missing:
            return
        st.warning(
            f"Die Trainingstabelle fuer {instr.label} enthaelt nur die Laufzeiten {sorted(have)} - "
            f"es fehlen {len(missing)} von 90 (T_days 1..90). Einmal (neu) bauen (dauert laenger, siehe unten)."
        )

    if st.button(
        f"Trainingstabelle fuer {instr.label} jetzt (neu) bauen (T_days 1-90)",
        key=f"build_tt_{instr.key}",
    ):
        script = Path(__file__).resolve().parent / "build_dataset.py"
        with st.spinner(
            f"Baue Trainingstabelle fuer {instr.label} - bitte warten, das kann mehrere Minuten dauern..."
        ):
            proc = subprocess.run(
                [sys.executable, str(script), instr.key], capture_output=True, text=True,
            )
        if proc.returncode == 0:
            try:
                st.cache_data.clear()
            except Exception:
                pass
            _ANALOG_CORE.clear()
            _TT_DAYS.clear()
            st.success(f"Trainingstabelle fuer {instr.label} (neu) gebaut. Seite wird neu geladen.")
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


def _build_daily_df_with_today(daily_df: pd.DataFrame, instr: InstrumentConfig):
    """Haengt einen synthetischen 'Heute'-Tagesbalken (open=high=low=close=
    heutiger Eroeffnungskurs) an eine KOPIE von daily_df an. Faellt auf den
    letzten echten Schlusskurs zurueck (used_fallback=True), falls kein
    Eroeffnungskurs verfuegbar ist (auch wenn das Instrument gar keinen
    chain_provider hat - dann ist ein Eroeffnungskurs prinzipiell nicht
    abrufbar, siehe UnsupportedChainError unten).

    SKALEN-KORREKTUR (Bugfix): bei Instrumenten, deren Kette ueber einen
    ANDEREN Ticker laeuft als die Kurshistorie (instr.chain_ticker !=
    instr.price_ticker - z.B. Russell 2000: price_ticker='^RUT' (Indexstand,
    ~2000-3000 Punkte) vs. chain_ticker='IWM' (ETF-Kurs, ~1/10 davon)) liefert
    provider.fetch_today_open() einen Preis auf der FALSCHEN Skala fuer die
    Index-Kurshistorie. Ein direktes Anhaengen erzeugt einen kuenstlichen
    ~90%-Kurssprung im synthetischen 'Heute'-Balken - die Analog-Suche
    (analog_matching.find_analogs) bewertet 'heute' dadurch als extrem
    unaehnlich zu JEDEM historischen Tag, die Gaussian-Kernel-Gewichte aller
    Kandidaten unterlaufen numerisch auf 0.0, und der 'weight > 0'-Filter in
    condor_search.analog_moves() wirft anschliessend ALLE Analog-Tage raus
    ('Keine Analog-Tage mit gueltiger Kursbewegung gefunden', obwohl Analog-
    Tage durchaus gefunden und in der Trainingstabelle vorhanden waren).
    Fix: statt des absoluten ETF-Preises wird die TAGESRENDITE des ETFs
    (heutiger Open / letzter ETF-Schlusskurs) auf den letzten echten
    Index-Schlusskurs der Kurshistorie projiziert - das haelt den
    synthetischen Balken auf der richtigen Skala. Schlaegt die ETF-Historie
    dafuer fehl (Netzwerk, kein Handelstag-Match etc.), wird auf den echten
    letzten Schlusskurs zurueckgefallen (used_fallback=True) statt eine
    falsche Skala zu riskieren.

    Rueckgabe: (daily_df_erweitert, query_date, used_fallback)."""
    today = pd.Timestamp(date.today())
    if (daily_df["date"] == today).any():
        return daily_df.copy(), today, False

    open_price = None
    try:
        provider = get_provider(instr)
        open_price = provider.fetch_today_open(instr)
    except UnsupportedChainError:
        open_price = None

    if open_price is None:
        return daily_df.copy(), daily_df["date"].max(), True

    if instr.chain_ticker and instr.chain_ticker != instr.price_ticker:
        try:
            import yfinance as yf
            etf_hist = yf.Ticker(instr.chain_ticker).history(period="5d")
            etf_closes = etf_hist["Close"]
            etf_idx = pd.to_datetime(etf_closes.index)
            if etf_idx.tz is not None:
                etf_idx = etf_idx.tz_convert(None)
            etf_closes.index = etf_idx.normalize()

            last_index_date = pd.Timestamp(daily_df["date"].max()).normalize()
            match = etf_closes.loc[etf_closes.index == last_index_date]
            if match.empty or float(match.iloc[-1]) <= 0:
                return daily_df.copy(), daily_df["date"].max(), True

            ratio = float(daily_df["close"].iloc[-1]) / float(match.iloc[-1])
            open_price = open_price * ratio
        except Exception:
            # Skalierung fehlgeschlagen - lieber sauberer Fallback auf den
            # letzten echten Schlusskurs als ein Wert auf falscher Skala.
            return daily_df.copy(), daily_df["date"].max(), True

    new_row = pd.DataFrame([{
        "date": today, "open": open_price, "high": open_price,
        "low": open_price, "close": open_price,
    }])
    if "resolution" in daily_df.columns:
        new_row["resolution"] = "daily"
    out = pd.concat([daily_df, new_row], ignore_index=True).sort_values("date").reset_index(drop=True)
    return out, today, False


def _find_today_analogs(daily_df, chosen_dte_exact, max_distance, bandwidth, min_gap, instr: InstrumentConfig):
    """Analog-Suche fuer 'heute' (von Teil B, Teil C und der Depot-Analyse
    gemeinsam genutzt) - jetzt FUER EIN BESTIMMTES INSTRUMENT.

    CACHE: die teure Arbeit (Feature-Matrix, PCA-Fit, Analog-Tage) haengt NICHT
    von der Restlaufzeit ab und wird deshalb einmal je (Instrument, Parameter,
    Tag, Datenstand) gerechnet und modulweit gehalten - alle Modi und
    Browser-Sitzungen arbeiten fuer DIESES Instrument auf demselben Bestand.
    Die Restlaufzeit bestimmt nur T_days_used (billig). Bei einem Cache-Treffer
    wird die grosse Trainingstabelle NICHT neu geladen."""
    sig = _training_table_signature(instr)
    core_key = (instr.key, float(max_distance), float(bandwidth), int(min_gap), date.today(), sig,
                len(daily_df), daily_df["date"].max())
    core = _ANALOG_CORE.get("core") if (sig is not None and _ANALOG_CORE.get("key") == core_key) else None

    if core is None:
        training_table = load_training_table(instr)
        if training_table.empty:
            st.error(
                f"Keine Trainingstabelle fuer {instr.label} gefunden - siehe Hinweis/Button "
                "weiter oben ('Trainingstabelle jetzt (neu) bauen')."
            )
            return None
        valid_t_days = available_t_days(training_table)

        daily_df_today, query_date, used_fallback = _build_daily_df_with_today(daily_df, instr)
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

        core = dict(analogs=analogs, training_table=training_table, daily_df_today=daily_df_today,
                    query_date=query_date, valid_t_days=valid_t_days)
        if sig is not None:
            _ANALOG_CORE.clear()
            _ANALOG_CORE.update(key=core_key, core=core)

    valid_t_days = core["valid_t_days"]
    T_days_used = min(valid_t_days, key=lambda t: abs(t - chosen_dte_exact))
    if abs(T_days_used - chosen_dte_exact) > 1.0:
        st.info(f"Restlaufzeit {chosen_dte_exact:.2f} Tage wird auf T_days={T_days_used} gerundet "
                f"(verfuegbar: {valid_t_days}).")
    return {**core, "T_days_used": T_days_used, "key": core_key}


def _run_bulk_analog(table: pd.DataFrame, daily_df: pd.DataFrame,
                      chosen_dte_exact: float, r_free: float, dividend_yield: float,
                      hv_length: int, rank_lookback: int,
                      analog_max_distance: float, analog_bandwidth: float, analog_min_gap: int,
                      spot_today: float, instr: InstrumentConfig) -> pd.DataFrame | None:
    """Analog-Tabelle je Zeile (siehe Moduldokstring, TEIL B): heutige reale
    Strikes + heutige Mid-Praemie gegen die historischen prozentualen
    Kursbewegungen der Analog-Tage. Keine Modellbewertung. (r_free,
    dividend_yield, hv_length, rank_lookback bleiben nur aus Kompatibilitaet in
    der Signatur - sie werden hier nicht mehr gebraucht.)
    Gibt None zurueck (mit st.error) bei Abbruch, sonst ein DataFrame mit einer
    Zeile je distance_pct."""
    ctx = _find_today_analogs(daily_df, chosen_dte_exact, analog_max_distance,
                              analog_bandwidth, analog_min_gap, instr)
    if ctx is None:
        return None
    analogs, T_days_used, training_table = ctx["analogs"], ctx["T_days_used"], ctx["training_table"]
    moves = analog_moves(analogs, training_table, T_days_used, ctx["query_date"])
    if moves.empty:
        # DIAGNOSE statt Blackbox-Fehlermeldung: zeigt, an welcher Stelle die
        # Analog-Tage verlorengehen - (a) kein Treffer in der Trainingstabelle
        # fuer dieses T_days (Datums-/Abdeckungs-Mismatch, z.B. weil die
        # Trainingstabelle mit einem AELTEREN Kursdaten-Stand gebaut wurde als
        # der aktuell live geladene) oder (b) Treffer vorhanden, aber Exit-Datum
        # liegt NACH dem Query-Tag (Kausalitaets-Filter).
        tt_this_T = training_table.loc[training_table["T_days"] == T_days_used]
        analog_dates = pd.to_datetime(analogs["analog_date"])
        in_tt = analog_dates.isin(set(tt_this_T["entry_date"]))
        n_matched = int(in_tt.sum())
        n_exit_ok = 0
        n_weight_ok = 0
        if n_matched > 0:
            exit_by_date = tt_this_T.set_index("entry_date")["exit_date"]
            matched_exits = exit_by_date.reindex(analog_dates[in_tt])
            exit_ok_mask = matched_exits <= pd.Timestamp(ctx["query_date"])
            n_exit_ok = int(exit_ok_mask.sum())
            matched_weights = analogs.loc[in_tt.to_numpy(), "weight"].to_numpy()
            n_weight_ok = int((matched_weights[exit_ok_mask.to_numpy()] > 0).sum()) if n_exit_ok else 0
        min_w = float(analogs["weight"].min()) if len(analogs) else float("nan")
        max_w = float(analogs["weight"].max()) if len(analogs) else float("nan")
        tt_range = (
            f"{tt_this_T['entry_date'].min().date()} bis {tt_this_T['entry_date'].max().date()}"
            if len(tt_this_T) else "leer (keine Zeilen fuer dieses T_days)"
        )
        if n_matched == 0:
            hint = (
                "→ 0 Treffer in der Trainingstabelle trotz gefundener Analog-Tage: die "
                "Trainingstabelle wurde vermutlich mit einem AELTEREN/ANDEREN Kursdaten-"
                "Stand gebaut als der, den das Dashboard gerade live laedt. Fix: "
                f"`python src/build_dataset.py {instr.key}` erneut ausfuehren (oder "
                "Button oben) und Seite neu laden."
            )
        elif n_exit_ok == 0:
            hint = (
                "→ Treffer vorhanden, aber alle Exit-Daten liegen nach dem Query-Tag: "
                "T_days zu gross fuer die verfuegbare Restlaufzeit-Rundung, oder der "
                "synthetische 'Heute'-Balken liegt zeitlich vor den passenden "
                "historischen Exit-Terminen."
            )
        elif n_weight_ok == 0:
            hint = (
                f"→ Treffer und Exit-Daten sind ok, aber ALLE Kernel-Gewichte sind 0 "
                f"(Gewichte im Analog-Set reichen von {min_w:.3g} bis {max_w:.3g}). Das "
                "deutet auf eine kaputte/falsch skalierte Feature-Basis fuer den "
                "synthetischen 'Heute'-Balken hin - z.B. wenn instr.chain_ticker und "
                "instr.price_ticker auf unterschiedlichen Preis-Skalen liegen (wie bei "
                "Russell 2000: ^RUT-Indexpunkte vs. IWM-ETF-Preis) und der heutige "
                "Eroeffnungskurs versehentlich auf der FALSCHEN Skala an die "
                "Kurshistorie angehaengt wurde (siehe _build_daily_df_with_today())."
            )
        else:
            hint = "→ Unklar - bitte Rohdaten pruefen."
        st.error(
            "Keine Analog-Tage mit gueltiger Kursbewegung (Exit vor heute) gefunden.\n\n"
            f"Diagnose: {len(analogs)} Analog-Tage gefunden "
            f"({analog_dates.min().date()} bis {analog_dates.max().date()}), davon "
            f"**{n_matched}** mit passendem Eintrag in der Trainingstabelle fuer "
            f"T_days={T_days_used} ({len(tt_this_T)} Zeilen dort, Zeitraum {tt_range}), "
            f"davon **{n_exit_ok}** mit Exit-Datum <= Query-Tag "
            f"({pd.Timestamp(ctx['query_date']).date()}), davon **{n_weight_ok}** mit "
            f"Kernel-Gewicht > 0.\n\n" + hint
        )
        return None
    st.caption(f"{len(moves)} Analog-Tage nutzbar (von {len(analogs)} gefunden; gilt fuer alle Zeilen, "
               "stetige Kernel-Gewichte).")
    # Fuer den Transparenz-Expander in render() (Teil B): zeigt, WELCHE
    # historischen Tage mit welchem Gewicht/welcher Distanz und welcher
    # tatsaechlich realisierten Bewegung in die Kennzahlen oben eingeflossen
    # sind - Nachvollziehbarkeit statt Blackbox-Mittelwert.
    st.session_state["livechain_analog_moves_debug"] = moves.copy()

    w = moves["weight"].to_numpy(dtype=float)
    w_sum = float(w.sum())
    S_d = spot_today * (1.0 + moves["ret"].to_numpy(dtype=float))
    n_eff = effective_sample_size(w)

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

        rb = analog_robustness(moves, pnl)     # gleiche Funktion wie in der Depot-Analyse

        mean_pnl = float(np.sum(w * pnl) / w_sum)
        win = pnl >= 0.0
        gross_profit = float(np.sum(w[win] * pnl[win]))
        gross_loss = float(-np.sum(w[~win] * pnl[~win]))

        results.append({
            "distance_pct": row.distance_pct,
            "analog_n": len(moves),
            "analog_n_eff": n_eff,
            "analog_n_eff_hac": rb["n_eff_hac"],
            "win_rate_pct": float(w[win].sum() / w_sum * 100.0),
            "profit_factor": (gross_profit / gross_loss) if gross_loss > 0.0 else np.nan,
            "avg_pnl_norm_pct": (mean_pnl / premium * 100.0) if premium > 0.0 else None,
            "entry_premium_today": premium,
            "expected_profit_short": mean_pnl,
            "required_premium_pf3": _required_premium_for_profit_factor(payout, w, target_pf=3.0),
            "tstat_full": rb["tstat_full"],
            "ci_excludes_zero_full": rb["ci_excludes_zero_full"],
            "subperiod_matching": rb["subperiod_matching"],
            "subperiod_nonempty": rb["subperiod_nonempty"],
            "tstat_val": rb["tstat_val"],
            "robust_train_val": rb["robust_train_val"],
            "robust_train_val_neg": rb["robust_train_val_neg"],
        })

    status.empty()
    progress.empty()

    out = pd.DataFrame(results)
    out["T_days_used"] = T_days_used
    out["n_analogs_total"] = len(moves)
    return out


def _render_reverse_condor(res: dict) -> None:
    """Reverse Iron Condor (Long-Condor): inneren Strangle kaufen, Fluegel verkaufen.
    Nutzt die in res gespeicherten Ergebnisse (gleiche Analog-Tage wie der Short-Condor)."""
    st.markdown("### Reverse Iron Condor (Long-Condor)")
    rdf, rmeta = res.get("rcdf"), res.get("rmeta")
    if rdf is None:
        st.info("Reverse-Condor-Ergebnis fehlt - 'Iron-Condor-Suche neu berechnen' klicken.")
        return
    if rdf.empty:
        st.warning("Kein gültiger Reverse Condor (Debit<=0, Debit>=Breite oder Bid/Ask fehlt). "
                   "Bei Bid/Ask=0: Preisbasis 'Mid' testen.")
        return
    st.caption(
        "Inneren Strangle KAUFEN (Ask), äußere Flügel VERKAUFEN (Bid). Debit = Max-Loss "
        "(Kurs endet zwischen den inneren Strikes). Max-Gewinn = größere Flügelbreite − Debit. "
        "P&L je Analog-Tag = Payout − Debit, auf denselben Analog-Tagen und mit denselben "
        "Gewichten wie der Short-Condor. Rendite/Risiko % = E[P&L] / Debit. "
        "KEIN Tail-Kriterium (beim Long-Condor ist der Totalverlust der Normalfall); "
        "'Max-Verlust-Tage' ist nur Information. Kandidat = Train>0 UND Val>0 UND t (HAC) > Schwelle "
        "UND der innere Long-Strangle ist selbst gültig (E>0, Train>0, Val>0, t>Schwelle, Ask-Preise) "
        "UND Plateau (Zeile direkt darüber und darunter ebenfalls gültig)."
    )
    st.caption(
        f"{rmeta['n_days']} Analog-Tage (n_eff Kish {rmeta['n_eff']:.1f}, max. Haltedauer "
        f"{rmeta['hold_max']}d) · {rmeta['n_pairs']} Paare getestet, {rmeta['n_base']} erfüllen die "
        f"Basis-Kriterien, {rmeta['n_pass']} bestehen zusätzlich Long-Strangle-Kriterium und Plateau."
    )
    if rmeta["n_pass"] == 0:
        st.info("Kein Reverse Condor besteht alle Kriterien - kein belastbarer Long-Edge für die heutige Lage.")
    else:
        b = rdf.iloc[0]
        st.success(
            f"Optimum Reverse Condor: Long {b['strike_put_inner']:.0f}P / {b['strike_call_inner']:.0f}C, "
            f"Short-Flügel {b['strike_put_wing']:.0f}P / {b['strike_call_wing']:.0f}C · "
            f"Debit {b['debit']:.2f}, Max-Gewinn {b['max_gain']:.2f} · E[P&L] {b['mean']:+.2f} Pkt = "
            f"{b['ror']*100:+.1f}% auf Risikokapital (t={b['tstat']:.2f}, Val-E {b['e_val']:+.2f})."
        )

    yn = lambda s: s.map({True: "✅", False: "❌"})
    rshow = pd.DataFrame({
        "Innen-Abstand %": rdf["distance_inner"], "Flügel-Abstand %": rdf["distance_wing"],
        "Put Flügel (short)": rdf["strike_put_wing"], "Put Innen (long)": rdf["strike_put_inner"],
        "Call Innen (long)": rdf["strike_call_inner"], "Call Flügel (short)": rdf["strike_call_wing"],
        "Debit": rdf["debit"], "Max-Loss": rdf["max_loss"], "Max-Gewinn": rdf["max_gain"],
        "Max-Gewinn/Max-Loss": rdf["rr_max"],
        "E[P&L] (Pkt)": rdf["mean"], "Rendite/Risiko %": rdf["ror"] * 100,
        "E Put-Spread": rdf["e_put_spread"], "E Call-Spread": rdf["e_call_spread"],
        "t (HAC)": rdf["tstat"], "E Train": rdf["e_train"], "E Val": rdf["e_val"],
        "Max-Verlust-Tage %": rdf["full_loss_share"] * 100, "Max-Verlust-Tage (n)": rdf["n_full_loss"],
        "Gewinnrate %": rdf["win_rate_pct"],
        "Train>0": yn(rdf["c_train"]), "Val>0": yn(rdf["c_val"]), "t": yn(rdf["c_t"]),
        "Long-Strangle gültig": yn(rdf["c_long"]), "Plateau (Innen ±1)": yn(rdf["c_plateau"]),
        "Kandidat": yn(rdf["passes"]),
    })
    rfmt = {c: "{:.2f}" for c in rshow.select_dtypes("number").columns}
    rfmt.update({c: "{:.0f}" for c in ["Put Flügel (short)", "Put Innen (long)", "Call Innen (long)",
                                       "Call Flügel (short)", "Max-Verlust-Tage (n)"]})
    rpassing = rshow[rdf["passes"].to_numpy()]
    if len(rpassing):
        st.dataframe(rpassing.style.format(rfmt, na_rep="n/a"), use_container_width=True, hide_index=True)
    with st.expander(f"Alle getesteten Paare ({len(rshow)}) inkl. Kriterien – Reverse Condor"):
        st.dataframe(rshow.style.format(rfmt, na_rep="n/a"), use_container_width=True, hide_index=True)
    if "long_df" in rmeta:
        with st.expander("Long-Strangle je Zeile (Basis für Long-Kriterium und Plateau)"):
            lg = rmeta["long_df"].copy()
            lg["ok"] = lg["ok"].map({True: "✅", False: "❌"})
            st.dataframe(lg.style.format({c: "{:.2f}" for c in lg.select_dtypes("number").columns},
                                         na_rep="n/a"), use_container_width=True, hide_index=True)


def _render_condor_section(table, daily_df, chosen_expiry, chosen_dte_exact, spot,
                           max_distance, bandwidth, min_gap, distance_key, instr: InstrumentConfig) -> None:
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
                         help="Ausserhalb der Handelszeit liefert die Kette oft Bid/Ask=0 - dann Mid waehlen.")
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
        st.info("Die Condor-Suche startet automatisch, sobald die Analog-Tabelle oben berechnet ist "
                "(sie liefert das Short-Kriterium und das Plateau). Steht oben eine Fehlermeldung, "
                "zuerst diese beheben.")

    # AUTO-LAUF: die Suche startet von selbst, sobald die Analog-Tabelle vorliegt und sich Tabelle
    # oder Einstellungen geaendert haben. Der Versuch wird MIT SCHLUESSEL gemerkt - ein Fehlschlag
    # wird deshalb nicht bei jedem Rerun wiederholt. Der Button erzwingt eine Neuberechnung.
    # Das Versionskuerzel "v2-reverse" sorgt dafuer, dass alte Cache-Eintraege ohne
    # Reverse-Condor-Ergebnis automatisch neu gerechnet werden.
    cparams = (basis, float(t_min), float(max_full), float(tail_floor), sort_label, "v2-reverse")
    st.session_state["shared_condor_criteria"] = dict(basis=basis, t_min=float(t_min), max_full=float(max_full),
                                                      tail_floor=float(tail_floor))
    if st.session_state.get("livechain_condor_attempt") != (distance_key, cparams):
        cached_c = _RESULT_CACHE.get("condor")            # z.B. neue Browser-Sitzung, gleicher Datenstand
        if cached_c is not None and cached_c[0] == (distance_key, cparams):
            st.session_state["livechain_condor"] = cached_c[1]
            st.session_state["livechain_condor_attempt"] = (distance_key, cparams)
    rerun_btn = st.button("Iron-Condor-Suche neu berechnen")
    auto_c = st.session_state.get("livechain_condor_attempt") != (distance_key, cparams)
    if (rerun_btn or auto_c) and short_valid is not None:
        st.session_state["livechain_condor_attempt"] = (distance_key, cparams)
        with st.spinner("Berechne Iron-Condor-Suche..."):
            ctx = _find_today_analogs(daily_df, chosen_dte_exact, max_distance, bandwidth, min_gap, instr)
            if ctx is not None:
                moves = analog_moves(ctx["analogs"], ctx["training_table"], ctx["T_days_used"], ctx["query_date"])
                cfg = CondorConfig(price_basis="mid" if basis.startswith("Mid") else "bidask",
                                   t_min=float(t_min), max_full_loss_share=max_full / 100.0,
                                   tail_floor=float(tail_floor),
                                   sort_by={"Rendite/Risiko %": "ror", "E[P&L] (Pkt)": "mean", "t (HAC)": "tstat"}[sort_label])
                cdf, sdf, meta = evaluate_condor_grid(table, moves, spot, cfg, short_valid)
                meta["T_days_used"] = ctx["T_days_used"]
                meta["sort_label"] = sort_label
                # Reverse Iron Condor: gleiche moves/table/cfg - kein zusaetzlicher Analog-Lauf.
                rcdf, rmeta = evaluate_reverse_condor_grid(table, moves, spot, cfg)
                st.session_state["livechain_condor"] = dict(key=distance_key, params=cparams, cdf=cdf, sdf=sdf,
                                                            meta=meta, rcdf=rcdf, rmeta=rmeta)
                _RESULT_CACHE["condor"] = ((distance_key, cparams), st.session_state["livechain_condor"])

    res = st.session_state.get("livechain_condor")
    if res is None:
        return
    if res["key"] != distance_key:
        st.info("Tabelle/Expiry/Instrument hat sich geaendert - Suche erneut starten.")
        return
    cdf, sdf, meta = res["cdf"], res["sdf"], res["meta"]
    if cdf.empty:
        st.warning("Kein gueltiges Paar (Kredit<=0 oder Bid/Ask fehlt). Bei Bid/Ask=0: Preisbasis 'Mid' testen.")
        _render_reverse_condor(res)
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

    # ── Reverse Iron Condor (Long-Condor) ──
    st.divider()
    _render_reverse_condor(res)
    st.divider()

    # ── Condor aus dieser Zeile ins Depot uebernehmen ──
    st.markdown("**Condor aus einer Zeile zum Depot hinzufuegen**")
    st.caption(
        f"Legt den 4-beinigen Iron Condor der gewaehlten Zeile (Short Put/Call + Long-Fluegel, "
        f"gemeinsamer Verfall) als neue Kombo im Depot an, fuer das Instrument **{instr.label}** - "
        "dauerhaft gespeichert, erscheint danach im Tab 'Depot-Analyse'. Auswahl aus ALLEN "
        "getesteten Paaren, nicht nur Kandidaten."
    )
    cdf_sorted = cdf.sort_values(["distance_short", "distance_wing"]).reset_index(drop=True)
    condor_opts = [
        f"{'✅' if r.passes else '❌'} Short {r.distance_short:g}% / Fluegel {r.distance_wing:g}% "
        f"({r.strike_put_wing:.0f}P/{r.strike_put_short:.0f}P .. {r.strike_call_short:.0f}C/{r.strike_call_wing:.0f}C)"
        for r in cdf_sorted.itertuples()
    ]
    cc1, cc2 = st.columns([3, 1])
    with cc1:
        condor_pick_label = st.selectbox("Paar", condor_opts, key="livechain_add_condor_row")
    with cc2:
        condor_qty = st.number_input("Menge", min_value=0.01, value=1.0, step=1.0, key="livechain_add_condor_qty")
    picked_c = cdf_sorted.iloc[condor_opts.index(condor_pick_label)]
    default_condor_name = (
        f"IC {picked_c.strike_put_wing:.0f}/{picked_c.strike_put_short:.0f}/"
        f"{picked_c.strike_call_short:.0f}/{picked_c.strike_call_wing:.0f} {pd.Timestamp(chosen_expiry).date()}"
    )
    condor_name = st.text_input("Name der Kombo", value=default_condor_name, key="livechain_add_condor_name")
    if st.button("Zum Depot hinzufuegen", key="livechain_add_condor_btn"):
        import dashboard_depot
        legs = [
            {"option_type": "put", "side": "long", "strike": float(picked_c.strike_put_wing), "qty": float(condor_qty)},
            {"option_type": "put", "side": "short", "strike": float(picked_c.strike_put_short), "qty": float(condor_qty)},
            {"option_type": "call", "side": "short", "strike": float(picked_c.strike_call_short), "qty": float(condor_qty)},
            {"option_type": "call", "side": "long", "strike": float(picked_c.strike_call_wing), "qty": float(condor_qty)},
        ]
        try:
            dashboard_depot.add_combo_to_depot(
                condor_name, pd.Timestamp(chosen_expiry).strftime("%Y-%m-%d"), legs,
                note="Aus Iron-Condor-Suche (Heutige Kette)", instrument=instr.key,
            )
        except ValueError as e:
            st.error(f"Nicht hinzugefuegt: {e}")
        else:
            st.success(f"'{condor_name}' ({instr.label}) zum Depot hinzugefuegt (siehe Tab 'Depot-Analyse').")


def render(df, daily_df, has_resolution, instr: InstrumentConfig) -> None:
    st.header("Heutige Kette - symmetrische Abstands-Ansicht")
    st.caption(
        f"Aktives Instrument: **{instr.label}**. Laedt die aktuelle Optionskette "
        "(Anbieter je Instrument, siehe instruments.py; bei Yahoo ~15 Min. verzoegert). Put- und "
        "Call-Strike werden je Zeile unabhaengig auf den naechstgelegenen realen "
        "Strike im gleichen prozentualen Abstand vom Spot gerundet - dadurch "
        "IMMER symmetrisch um den Spot. Die HV-Periode fuer Modell-Delta und "
        "HV-Rang wird automatisch an die gewaehlte Restlaufzeit gekoppelt "
        "(z.B. 7.5 Tage Restlaufzeit -> HV(7.5)-Fenster)."
    )

    if instr.chain_provider is None:
        st.info(
            f"Fuer {instr.label} ist keine Live-Optionskette konfiguriert (instruments.py, "
            "chain_provider=None) - 'Heutige Kette' steht fuer dieses Instrument nicht zur "
            "Verfuegung. Nutze 'Iron Condor Backtest' oder 'Analog-Tage-Matching' fuer die "
            "historische Auswertung dieses Instruments, oder aktiviere - falls vorhanden und "
            "erreichbar - einen Chain-Provider in instruments.py (z.B. 'ib' fuer EuroStoxx50)."
        )
        return

    _ensure_training_table(instr)

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
        dividend_yield = st.number_input(
            "Dividenden-/Convenience-Yield", 0.0, 0.10, float(instr.dividend_yield), 0.001,
            format="%.3f", key=f"livechain_q_{instr.key}",
            help="Voreingestellt aus instruments.py fuer dieses Instrument.",
        )
        reload_chain = st.button(
            "Heutige Kette neu laden", type="primary",
            help="Beim ERSTEN Oeffnen dieses Tabs (bzw. nach Instrumentenwechsel) wird die Kette "
                 "automatisch geladen - dieser Button ist nur fuer ein manuelles Neuladen noetig, "
                 "z.B. nach Aenderung von DTE-Filter/Delta-Filter oder fuer einen aktuelleren Snapshot.",
        )

    # GEMEINSAMER DATENBESTAND (siehe market_data.py): die Kette wird einmal je Programmstart/Tag UND
    # INSTRUMENT geladen und von allen Modi genutzt. DTE-/Delta-Filter wirken nur auf der Ansicht.
    snap = market_data.current(instr, float(max_dte))
    if reload_chain or snap is None:
        with st.spinner(f"Lade heutige Kette fuer {instr.label} und berechne Delta/IV je Serie..."):
            try:
                snap = market_data.get(instr, float(r_free), float(dividend_yield), float(max_dte),
                                       force=bool(reload_chain))
            except Exception as e:
                st.error(f"Laden fehlgeschlagen: {e}")
                snap = market_data.current(instr, 0.0)
                if snap is None:
                    st.info("Kette konnte noch nicht geladen werden - siehe Fehlermeldung oben.")
                    return

    spot = snap["spot"]
    r_free = snap["r_free"]
    dividend_yield = snap["dividend_yield"]
    fetch_ts = snap["fetch_ts"]
    enriched_all = snap["enriched"]
    if enriched_all.empty:
        enriched = enriched_all
    else:
        dcol = "dte_exact" if "dte_exact" in enriched_all.columns else "dte"
        dvals = enriched_all[dcol].astype(float)
        enriched = enriched_all[(dvals >= float(min_dte)) & (dvals <= float(max_dte))
                                & (enriched_all["delta_mkt"] >= float(min_abs_delta))]
    st.metric(f"Aktueller Spot ({instr.label})", f"{spot:,.2f} {instr.currency}")
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
            "Spalte 'dte_exact' fehlt - 'Heutige Kette' neu laden."
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
        f"Exakte Restlaufzeit: **{chosen_dte_exact:.2f} Tage** (Naeherung: PM-Settlement 16:00 "
        "lokal fuer US-Instrumente; bei einem spaeter aktivierten Eurex/IB-Provider gilt "
        "stattdessen die dortige AM-Settlement-Naeherung, siehe chain_providers/ib_provider.py)."
    )

    hv_length = max(2, round(chosen_dte_exact))
    st.caption(
        f"→ HV-Periode automatisch auf **{hv_length} Handelstage** gesetzt "
        f"(gerundete Restlaufzeit von {chosen_dte_exact:.2f} Tagen)."
    )

    bounds = _bounds_cached(daily_df, hv_length, int(rank_lookback), instr)
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
                "last_trade_date": "Letzter Trade",
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

    st.session_state["shared_analog_params"] = dict(
        max_distance=float(analog_max_distance), bandwidth=float(analog_bandwidth), min_gap=int(analog_min_gap))

    # SCHLUESSEL: aendert sich bei neuer Kette (fetch_ts), anderem Verfall/Raster, anderen
    # Analog-Parametern, neuem Tag ODER anderem Instrument. Nur dann wird NEU gerechnet - sonst
    # bleibt das Ergebnis aus st.session_state (auch nach einem Tab-Wechsel) stehen.
    analog_key = (instr.key, chosen_expiry, tuple(distance_targets_pct), fetch_ts, float(analog_max_distance),
                  float(analog_bandwidth), int(analog_min_gap), date.today(),
                  float(min_dte), float(max_dte), float(min_abs_delta))
    if st.session_state.get("livechain_analog_key") != analog_key:
        cached_b = _RESULT_CACHE.get("analog")            # z.B. neue Browser-Sitzung, gleicher Datenstand
        if cached_b is not None and cached_b[0] == analog_key:
            st.session_state["livechain_analog_results"] = cached_b[1]
            st.session_state["livechain_analog_key"] = analog_key
            st.session_state["livechain_analog_attempt"] = analog_key
    run_bulk = st.button("Analog-Läufe neu berechnen")
    auto_run = st.session_state.get("livechain_analog_attempt") != analog_key
    if run_bulk or auto_run:
        # Versuch VOR dem Lauf merken: schlaegt er fehl, wird er nicht bei jedem Rerun wiederholt.
        st.session_state["livechain_analog_attempt"] = analog_key
        analog_results = _run_bulk_analog(
            table, daily_df, chosen_dte_exact, r_free, dividend_yield,
            hv_length, int(rank_lookback), float(analog_max_distance), float(analog_bandwidth),
            int(analog_min_gap), spot, instr,
        )
        if analog_results is not None:
            st.session_state["livechain_analog_results"] = analog_results
            st.session_state["livechain_analog_key"] = analog_key
            _RESULT_CACHE["analog"] = (analog_key, analog_results)

    stored = st.session_state.get("livechain_analog_results")
    stored_key = st.session_state.get("livechain_analog_key")
    if stored is not None and stored_key == analog_key:
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
            "historische Rueckrechnung, keine Zukunftsgarantie."
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

        # ── Transparenz: welche historischen Tage stecken dahinter? ──
        moves_dbg = st.session_state.get("livechain_analog_moves_debug")
        if moves_dbg is not None and not moves_dbg.empty:
            with st.expander(
                f"Welche historischen Tage stecken hinter der Analog-Bewertung? ({len(moves_dbg)} Tage)"
            ):
                st.caption(
                    "Alle Tage, die in JEDE Zeile der Tabelle oben einfliessen (das Gewicht "
                    "kommt aus der Aehnlichkeit zu 'heute' im PCA-Raum, NICHT aus dieser "
                    "Zeile selbst - Put/Call-Strike sind je Zeile unterschiedlich, die "
                    "Analog-Tage und ihre Gewichte aber gleich). Sortiert nach Gewicht "
                    "absteigend - die obersten Zeilen dominieren den gewichteten Mittelwert "
                    "am staerksten. 'Krisen-Fenster' markiert grob bekannte Stress-Perioden "
                    "(1987-Crash, Finanzkrise 2008/09, Corona-Crash 2020) rein zur Orientierung "
                    "- keine harte Kategorisierung."
                )
                _CRISIS_WINDOWS = [
                    ("1987-10-01", "1987-11-15", "Crash Okt. 1987"),
                    ("2008-09-01", "2009-03-31", "Finanzkrise 2008/09"),
                    ("2020-02-15", "2020-04-15", "Corona-Crash 2020"),
                ]
                dbg = moves_dbg.sort_values("weight", ascending=False).copy()
                dbg["weight_share_pct"] = dbg["weight"] / dbg["weight"].sum() * 100.0
                dbg["ret_pct"] = dbg["ret"] * 100.0

                def _tag_crisis(d: pd.Timestamp) -> str:
                    for lo, hi, label in _CRISIS_WINDOWS:
                        if pd.Timestamp(lo) <= d <= pd.Timestamp(hi):
                            return label
                    return ""
                dbg["krisen_fenster"] = dbg["entry_date"].map(_tag_crisis)

                top_n = min(50, len(dbg))
                show_dbg = dbg.head(top_n)[[
                    "entry_date", "exit_date", "distance", "weight", "weight_share_pct",
                    "ret_pct", "hold_days", "krisen_fenster",
                ]].rename(columns={
                    "entry_date": "Entry-Datum", "exit_date": "Exit-Datum",
                    "distance": "PCA-Distanz", "weight": "Kernel-Gewicht",
                    "weight_share_pct": "Anteil am Gesamtgewicht %",
                    "ret_pct": "Realisierte Bewegung %", "hold_days": "Haltedauer (Tage)",
                    "krisen_fenster": "Krisen-Fenster",
                })
                st.dataframe(
                    show_dbg.assign(**{
                        "Entry-Datum": show_dbg["Entry-Datum"].dt.date,
                        "Exit-Datum": show_dbg["Exit-Datum"].dt.date,
                    }).style.format({
                        "PCA-Distanz": "{:.4f}", "Kernel-Gewicht": "{:.4f}",
                        "Anteil am Gesamtgewicht %": "{:.2f}", "Realisierte Bewegung %": "{:+.2f}",
                        "Haltedauer (Tage)": "{:.0f}",
                    }, na_rep=""),
                    use_container_width=True, hide_index=True,
                )

                crisis_weight_share = float(dbg.loc[dbg["krisen_fenster"] != "", "weight"].sum()
                                            / dbg["weight"].sum() * 100.0) if dbg["weight"].sum() > 0 else 0.0
                top10_weight_share = float(dbg.head(min(10, len(dbg)))["weight"].sum()
                                           / dbg["weight"].sum() * 100.0) if dbg["weight"].sum() > 0 else 0.0
                st.caption(
                    f"Anteil der oben markierten Krisen-Fenster am GESAMTGEWICHT: "
                    f"**{crisis_weight_share:.1f}%**. Anteil der Top-10-Tage (nach Gewicht) am "
                    f"Gesamtgewicht: **{top10_weight_share:.1f}%** - hoch (z.B. > 30-40%), bedeutet "
                    "das Ergebnis haengt stark an wenigen Tagen, unabhaengig davon ob diese in "
                    "einem Krisen-Fenster liegen oder nicht."
                )

        # ── Position aus dieser Zeile ins Depot uebernehmen ──
        addable = merged.dropna(subset=["strike_put", "strike_call"])
        if not addable.empty:
            st.markdown("**Position aus einer Zeile zum Depot hinzufuegen**")
            st.caption(
                f"Legt einen Short- oder Long-Strangle (Put + Call der gewaehlten Zeile, "
                f"gemeinsamer Verfall) als neue Kombo im Depot an, fuer das Instrument "
                f"**{instr.label}** - dauerhaft gespeichert, erscheint danach im Tab 'Depot-Analyse'."
            )
            opts = [
                f"{r.distance_pct:g} % ({r.strike_put:.0f}P / {r.strike_call:.0f}C)"
                for r in addable.itertuples()
            ]
            ac1, ac2, ac3 = st.columns([2, 1, 1])
            with ac1:
                pick_label = st.selectbox("Zeile", opts, key="livechain_add_analog_row")
            with ac2:
                add_side = st.selectbox("Richtung", ["Short Strangle", "Long Strangle"],
                                        key="livechain_add_analog_side")
            with ac3:
                add_qty = st.number_input("Menge", min_value=0.01, value=1.0, step=1.0,
                                          key="livechain_add_analog_qty")
            picked = addable.iloc[opts.index(pick_label)]
            side = "short" if add_side.startswith("Short") else "long"
            default_name = f"{add_side} {picked.distance_pct:g}% {pd.Timestamp(chosen_expiry).date()}"
            add_name = st.text_input("Name der Kombo", value=default_name, key="livechain_add_analog_name")
            if st.button("Zum Depot hinzufuegen", key="livechain_add_analog_btn"):
                import dashboard_depot
                legs = [
                    {"option_type": "put", "side": side, "strike": float(picked.strike_put), "qty": float(add_qty)},
                    {"option_type": "call", "side": side, "strike": float(picked.strike_call), "qty": float(add_qty)},
                ]
                try:
                    dashboard_depot.add_combo_to_depot(
                        add_name, pd.Timestamp(chosen_expiry).strftime("%Y-%m-%d"), legs,
                        note="Aus Analog-Tage-Simulation je Zeile (Heutige Kette)", instrument=instr.key,
                    )
                except ValueError as e:
                    st.error(f"Nicht hinzugefuegt: {e}")
                else:
                    st.success(f"'{add_name}' ({instr.label}) zum Depot hinzugefuegt (siehe Tab 'Depot-Analyse').")
    elif stored is not None:
        st.info("Die Analog-Tabelle wird fuer die geaenderte Tabelle/Einstellung/Instrument neu berechnet "
                "oder ist fehlgeschlagen - siehe Meldungen oben; ggf. 'Analog-Läufe neu berechnen' klicken.")

    # ── TEIL C: Iron-Condor-Suche ──
    _render_condor_section(
        table, daily_df, chosen_expiry, chosen_dte_exact, spot,
        float(analog_max_distance), float(analog_bandwidth), int(analog_min_gap),
        analog_key, instr,
    )