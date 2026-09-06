"""
Walk-Forward-Training des Ensemble-Klassifikators.

Einfach ausfuehren mit:
    python src/train.py

Voraussetzung: data/training_table.parquet muss existieren (siehe
src/build_dataset.py).

GRUNDPRINZIP - Purged & Embargoed Walk-Forward:
Die Zeitreihe wird in aufeinanderfolgende (Trainingsfenster, Testfenster)-
Paare zerlegt. Fuer jedes Fenster gilt:

  1. PURGING: Trainings-Zeilen, deren Label-Zeitraum (entry_date bis
     exit_date) in das Testfenster hineinreicht, werden entfernt - sonst
     wuerde das Modell indirekt Information aus der Zukunft sehen (die
     Ladder-Strategie eroeffnet taeglich ueberlappende Positionen, ein Trade
     der am 28. eines Monats eroeffnet und 14 Tage haelt, reicht z.B. schon
     in den naechsten Monat hinein).
  2. EMBARGO: zusaetzlich werden Trainings-Zeilen entfernt, deren Entry
     weniger als `embargo_days` VOR dem Testfenster-Start liegt - das
     schuetzt zusaetzlich gegen Leakage durch Autokorrelation des
     Marktzustands selbst (nicht nur durch ueberlappende Label-Zeitraeume).
  3. KLASSEN-SCHWELLEN NUR AUS DEM TRAININGSFENSTER: die 4 Klassen (siehe
     labels.py) werden pro Fold NEU berechnet, ausschliesslich aus den
     Trainingsdaten dieses Folds - und zwar SEPARAT JE LAUFZEIT (T_days),
     weil kurze Laufzeiten strukturell anders verteilt sind als lange (siehe
     Cap-Effekt-Diskussion: bei kurzen Laufzeiten verfaellt ein Short
     Strangle oft komplett wertlos, pnl_norm=+1.0 ist dann keine Ausnahme,
     sondern haeufig - eine gemeinsame Schwelle ueber alle Laufzeiten hinweg
     wuerde von den kurzen Laufzeiten dominiert).

ENSEMBLE: drei bewusst unterschiedliche Modellfamilien (Soft-Voting, d.h.
gemittelte Klassenwahrscheinlichkeiten):
  - LightGBM (Gradient Boosted Trees) - i.d.R. staerkster Einzelkandidat bei
    solchen tabellarischen Feature-Sets.
  - Random Forest - anderer Bias als Boosting, robuster gegen einzelne
    dominante Features.
  - Multinomiale logistische Regression (mit Skalierung) - linearer
    Gegenpol; wenn die Baummodelle sie deutlich outperformen, ist das ein
    Hinweis, dass echte Nichtlinearitaet/Interaktionen im Feature-Set
    vorhanden sind, nicht nur Rauschen.

GESPEICHERT WIRD JE FOLD ("Vintage") EIN VOLLSTAENDIGES, EIGENSTAENDIGES SET:
models/vintage_<test_start>/
    model_lgbm.joblib
    model_rf.joblib
    model_logreg.joblib
    thresholds.joblib          (dict: T_days -> ClassThresholds)
    metadata.json              (Feature-Liste, Config, Metriken, Versionen)

So ist jeder Trainingsstand rueckwirkend vollstaendig reproduzierbar - u.a.
Voraussetzung dafuer, die "Referenzverteilung muss out-of-sample gebaut
werden"-Anforderung aus der urspruenglichen Diskussion sauber umzusetzen.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm
    from lightgbm import LGBMClassifier
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False

sys.path.insert(0, str(Path(__file__).resolve().parent))

from labels import CLASS_LABELS, generate_class_thresholds, assign_classes, ClassThresholds

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
TRAINING_TABLE_FILE = DATA_DIR / "training_table.parquet"

# Spalten aus labels.build_training_table(), die NICHT als Modell-Feature
# verwendet werden duerfen (Label-Rohdaten/Identifikatoren). Alles andere in
# der Tabelle (inkl. 'T_days'!) ist ein Feature - so bleibt die Feature-Liste
# automatisch konsistent mit features.py, ohne sie hier doppelt zu pflegen.
NON_FEATURE_COLS = [
    "entry_date", "exit_date", "entry_S", "exit_S", "entry_iv",
    "K_put", "K_call", "entry_premium", "net_pnl", "pnl_norm",
]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


@dataclass
class WalkForwardConfig:
    initial_train_years: float = 20.0   # Laenge des allerersten Trainingsfensters
    test_window_years: float = 1.0      # Laenge jedes Testfensters (danach: naechster Fold)
    embargo_days: int = 21              # Puffer zwischen Trainings-Entry und Test-Start
    min_train_rows: int = 2000          # Fold wird uebersprungen, wenn weniger Trainingszeilen uebrig bleiben
    split_quantile: float = 0.5         # Quantil je Seite fuer die Klassen-Schwellen (siehe labels.py)
    random_state: int = 42


@dataclass
class Fold:
    fold_id: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_cutoff: pd.Timestamp   # Entry muss STRIKT davor liegen, um im Training zu landen


def generate_folds(entry_dates: pd.Series, cfg: WalkForwardConfig) -> list[Fold]:
    """Baut die Fold-Grenzen als reine Zeitfenster (noch ohne Bezug auf die
    eigentlichen Datenzeilen - siehe fold_masks() fuer die Anwendung inkl.
    Purging/Embargo)."""
    start = entry_dates.min()
    end = entry_dates.max()

    folds = []
    fold_id = 0
    test_start = start + pd.DateOffset(years=cfg.initial_train_years)
    while test_start < end:
        test_end = min(test_start + pd.DateOffset(years=cfg.test_window_years), end + pd.Timedelta(days=1))
        train_cutoff = test_start - pd.Timedelta(days=cfg.embargo_days)
        folds.append(Fold(fold_id=fold_id, test_start=test_start, test_end=test_end, train_cutoff=train_cutoff))
        fold_id += 1
        test_start = test_end
    return folds


def fold_masks(df: pd.DataFrame, fold: Fold) -> tuple[np.ndarray, np.ndarray]:
    """Purging + Embargo in einem Schritt:
    - Trainings-Zeile ist zulaessig, wenn ihr Entry VOR dem Embargo-Cutoff
      liegt UND ihr Exit VOR dem Test-Start liegt (kein Ueberlappen des
      Label-Zeitraums mit dem Testfenster - das ist das eigentliche Purging).
    - Test-Zeile ist einfach: Entry faellt in [test_start, test_end)."""
    train_mask = (df["entry_date"] < fold.train_cutoff) & (df["exit_date"] < fold.test_start)
    test_mask = (df["entry_date"] >= fold.test_start) & (df["entry_date"] < fold.test_end)

    # Defensive Absicherung (siehe Modul-Docstring) - sollte durch die
    # Maskenkonstruktion oben bereits ausgeschlossen sein, wird hier aber
    # zusaetzlich hart geprueft, damit ein zukuenftiger Refactoring-Fehler
    # nicht still Leakage einschleust.
    if train_mask.any() and test_mask.any():
        assert df.loc[train_mask, "exit_date"].max() < fold.test_start, \
            "LEAKAGE: eine Trainings-Zeile reicht in das Testfenster hinein."
        assert df.loc[test_mask, "entry_date"].min() >= fold.test_start, \
            "LEAKAGE: eine Test-Zeile beginnt vor dem Testfenster-Start."

    return train_mask.to_numpy(), test_mask.to_numpy()


def compute_thresholds_by_horizon(train_df: pd.DataFrame, split_quantile: float) -> dict[int, ClassThresholds]:
    """Klassen-Schwellen SEPARAT je T_days, ausschliesslich aus den
    Trainingsdaten dieses Folds (siehe Modul-Docstring, Punkt 3)."""
    out = {}
    for T_days, g in train_df.groupby("T_days"):
        out[int(T_days)] = generate_class_thresholds(g["pnl_norm"].to_numpy(), split_quantile)
    return out


def apply_classes_by_horizon(df: pd.DataFrame, thresholds_by_T: dict[int, ClassThresholds]) -> pd.Categorical:
    """Wendet die (extern uebergebenen, siehe oben) Schwellen je T_days auf
    einen beliebigen Datensatz an - identische Funktion fuer Trainings- UND
    Test-Zeilen, damit garantiert dieselben Schwellen verwendet werden."""
    codes = np.full(len(df), -1, dtype=int)
    t_days_arr = df["T_days"].to_numpy()
    pnl_arr = df["pnl_norm"].to_numpy()
    for T_days, th in thresholds_by_T.items():
        m = t_days_arr == T_days
        if not m.any():
            continue
        sub_classes = assign_classes(pd.Series(pnl_arr[m]), th)
        codes[m] = sub_classes.codes
    return pd.Categorical.from_codes(codes, categories=CLASS_LABELS)


def build_base_models(random_state: int) -> dict:
    models = {
        "rf": RandomForestClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=20,
            class_weight="balanced", random_state=random_state, n_jobs=-1,
        ),
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                solver="lbfgs", max_iter=2000, class_weight="balanced", C=1.0,
            )),
        ]),
    }
    if _HAS_LIGHTGBM:
        models["lgbm"] = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            class_weight="balanced", random_state=random_state, verbosity=-1,
        )
    else:
        print("[warnung] lightgbm nicht installiert - Ensemble laeuft nur mit "
              "Random Forest + logistischer Regression. `pip install lightgbm` "
              "fuer das vollstaendige Ensemble.")
    return models


def _align_proba(model, X: np.ndarray) -> np.ndarray:
    """Bringt die Wahrscheinlichkeits-Spalten eines Modells in die feste
    Reihenfolge CLASS_LABELS - noetig, weil sklearn/lightgbm die Klassen in
    model.classes_ alphabetisch sortieren, nicht in unserer inhaltlichen
    Reihenfolge (stark_negativ < negativ < positiv < stark_positiv), und weil
    in seltenen Faellen eine Klasse im Trainingsfold komplett fehlen kann."""
    proba = model.predict_proba(X)
    aligned = np.zeros((X.shape[0], len(CLASS_LABELS)))
    for j, cls in enumerate(model.classes_):
        aligned[:, CLASS_LABELS.index(cls)] = proba[:, j]
    return aligned


def ensemble_predict_proba(models: dict, X: np.ndarray) -> np.ndarray:
    """Soft-Voting: ungewichtetes Mittel der (spaltenausgerichteten)
    Klassenwahrscheinlichkeiten aller Basismodelle."""
    aligned = [_align_proba(m, X) for m in models.values()]
    return np.mean(aligned, axis=0)


def train_one_fold(df: pd.DataFrame, fold: Fold, feature_cols: list[str], cfg: WalkForwardConfig) -> dict | None:
    """Trainiert und evaluiert einen einzelnen Fold. Gibt None zurueck, wenn
    der Fold uebersprungen wird (zu wenig Trainingsdaten)."""
    train_mask, test_mask = fold_masks(df, fold)

    train_df = df.loc[train_mask].dropna(subset=feature_cols)
    test_df = df.loc[test_mask].dropna(subset=feature_cols)

    if len(train_df) < cfg.min_train_rows:
        print(f"[fold {fold.fold_id}] uebersprungen - nur {len(train_df)} Trainingszeilen "
              f"(< min_train_rows={cfg.min_train_rows}).")
        return None
    if test_df.empty:
        print(f"[fold {fold.fold_id}] uebersprungen - keine Test-Zeilen nach NaN-Filter.")
        return None

    thresholds_by_T = compute_thresholds_by_horizon(train_df, cfg.split_quantile)
    train_df = train_df.assign(cls=apply_classes_by_horizon(train_df, thresholds_by_T))
    test_df = test_df.assign(cls=apply_classes_by_horizon(test_df, thresholds_by_T))

    train_df = train_df.dropna(subset=["cls"])
    test_df = test_df.dropna(subset=["cls"])

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["cls"].astype(str).to_numpy()
    X_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df["cls"].astype(str).to_numpy()

    models = build_base_models(cfg.random_state)
    for name, model in models.items():
        model.fit(X_train, y_train)

    ensemble_proba = ensemble_predict_proba(models, X_test)
    ensemble_pred = np.array(CLASS_LABELS)[np.argmax(ensemble_proba, axis=1)]

    report = classification_report(
        y_test, ensemble_pred, labels=CLASS_LABELS, output_dict=True, zero_division=0,
    )

    # Oekonomischer Realitaets-Check (siehe Modul-Docstring der Diskussion):
    # nicht nur Klassifikationsguete, sondern was der TATSAECHLICHE pnl_norm
    # der als 'stark_positiv'/'stark_negativ' klassifizierten Test-Trades war.
    econ_check = {}
    for cls in CLASS_LABELS:
        sel = ensemble_pred == cls
        econ_check[cls] = {
            "n": int(sel.sum()),
            "avg_pnl_norm": float(test_df.loc[sel, "pnl_norm"].mean()) if sel.any() else None,
        }

    return {
        "fold": fold,
        "models": models,
        "thresholds_by_T": thresholds_by_T,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "classification_report": report,
        "econ_check": econ_check,
        "feature_cols": feature_cols,
    }


def save_vintage(result: dict, cfg: WalkForwardConfig) -> Path:
    fold = result["fold"]
    vintage_dir = MODELS_DIR / f"vintage_{fold.test_start.date()}"
    vintage_dir.mkdir(parents=True, exist_ok=True)

    for name, model in result["models"].items():
        joblib.dump(model, vintage_dir / f"model_{name}.joblib")
    joblib.dump(result["thresholds_by_T"], vintage_dir / "thresholds.joblib")

    metadata = {
        "test_start": str(fold.test_start.date()),
        "test_end": str(fold.test_end.date()),
        "train_cutoff": str(fold.train_cutoff.date()),
        "n_train_rows": result["n_train"],
        "n_test_rows": result["n_test"],
        "feature_columns": result["feature_cols"],
        "walk_forward_config": asdict(cfg),
        "thresholds_by_T_days": {
            str(T): asdict(th) for T, th in result["thresholds_by_T"].items()
        },
        "classification_report": result["classification_report"],
        "econ_check": result["econ_check"],
        "library_versions": {
            "sklearn": sklearn.__version__,
            "lightgbm": lightgbm.__version__ if _HAS_LIGHTGBM else None,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "models_included": list(result["models"].keys()),
    }
    with open(vintage_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return vintage_dir


def load_vintage(vintage_dir: Path) -> dict:
    """Laedt ein vollstaendiges Vintage zurueck - fuer spaeteren Einsatz im
    Backtest/Dashboard (Anwendung der Klassifikation auf neue/aktuelle Tage)."""
    models = {}
    for p in Path(vintage_dir).glob("model_*.joblib"):
        name = p.stem.replace("model_", "")
        models[name] = joblib.load(p)
    thresholds_by_T = joblib.load(Path(vintage_dir) / "thresholds.joblib")
    with open(Path(vintage_dir) / "metadata.json") as f:
        metadata = json.load(f)
    return {"models": models, "thresholds_by_T": thresholds_by_T, "metadata": metadata}


def main():
    if not TRAINING_TABLE_FILE.exists():
        print(f"[fehler] {TRAINING_TABLE_FILE} nicht gefunden.")
        print("[info] Erst ausfuehren: python src/build_dataset.py")
        sys.exit(1)

    print(f"[ok] lade {TRAINING_TABLE_FILE} ...")
    df = pd.read_parquet(TRAINING_TABLE_FILE)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df = df.sort_values("entry_date").reset_index(drop=True)

    feature_cols = get_feature_columns(df)
    print(f"[ok] {len(df)} Zeilen, {len(feature_cols)} Feature-Spalten "
          f"(inkl. T_days): {feature_cols}")

    cfg = WalkForwardConfig()
    folds = generate_folds(df["entry_date"], cfg)
    print(f"[ok] {len(folds)} Walk-Forward-Folds geplant "
          f"(initial_train_years={cfg.initial_train_years}, "
          f"test_window_years={cfg.test_window_years}, embargo_days={cfg.embargo_days}).")

    MODELS_DIR.mkdir(exist_ok=True)
    fold_summaries = []

    for fold in folds:
        result = train_one_fold(df, fold, feature_cols, cfg)
        if result is None:
            continue

        vintage_dir = save_vintage(result, cfg)
        macro_f1 = result["classification_report"]["macro avg"]["f1-score"]
        acc = result["classification_report"]["accuracy"]
        sp_econ = result["econ_check"]["stark_positiv"]
        sn_econ = result["econ_check"]["stark_negativ"]
        print(
            f"[fold {fold.fold_id}] {fold.test_start.date()} - {fold.test_end.date()} | "
            f"train={result['n_train']:,} test={result['n_test']:,} | "
            f"acc={acc:.3f} macro-F1={macro_f1:.3f} | "
            f"stark_positiv: n={sp_econ['n']} avg_pnl_norm={sp_econ['avg_pnl_norm']} | "
            f"stark_negativ: n={sn_econ['n']} avg_pnl_norm={sn_econ['avg_pnl_norm']} | "
            f"gespeichert unter {vintage_dir}"
        )

        fold_summaries.append({
            "fold_id": fold.fold_id,
            "test_start": fold.test_start,
            "test_end": fold.test_end,
            "n_train": result["n_train"],
            "n_test": result["n_test"],
            "accuracy": acc,
            "macro_f1": macro_f1,
            "stark_positiv_n": sp_econ["n"],
            "stark_positiv_avg_pnl_norm": sp_econ["avg_pnl_norm"],
            "stark_negativ_n": sn_econ["n"],
            "stark_negativ_avg_pnl_norm": sn_econ["avg_pnl_norm"],
        })

    if not fold_summaries:
        print("[warnung] kein einziger Fold konnte trainiert werden - "
              "Datensatz zu kurz fuer die gewaehlte initial_train_years/test_window_years?")
        sys.exit(1)

    report_df = pd.DataFrame(fold_summaries)
    report_path = MODELS_DIR / "walkforward_report.csv"
    report_df.to_csv(report_path, index=False)
    print(f"\n[ok] Walk-Forward-Report gespeichert unter: {report_path}")
    print(report_df[["fold_id", "test_start", "test_end", "n_train", "n_test", "accuracy", "macro_f1"]])


if __name__ == "__main__":
    main()