"""
Konsolidierte Auswertung ueber ALLE trainierten Walk-Forward-Vintages.

Einfach ausfuehren mit:
    python src/evaluate.py

Voraussetzung: mindestens ein Vintage unter models/vintage_*/ (siehe
train.py) sowie data/training_table.parquet.

WAS HIER PASSIERT (und warum das mehr ist als walkforward_report.csv):
train.py speichert pro Fold nur Aggregat-Kennzahlen (Accuracy, Macro-F1,
einfacher Durchschnitts-pnl_norm je Klasse) in metadata.json. Das sagt noch
nichts darueber aus, ob der Edge STATISTISCH ROBUST ist (Newey-West-t-Stat,
Perioden-Konsistenz - siehe stats.py) oder nur zufaellig in diesem einen
Testjahr so aussah.

Dieses Skript geht einen Schritt weiter:
  1. Fuer JEDEN Vintage werden die ORIGINALEN Testzeilen (Out-of-Sample,
     also Daten, die dieses Vintage beim Training nie gesehen hat) erneut
     durch das gespeicherte Modell geschickt.
  2. Alle Vorhersagen ueber ALLE Vintages hinweg werden zu EINER durchgehenden
     Zeitreihe zusammengefuegt (jedes Vintage deckt ein anderes,
     nicht-ueberlappendes Testfenster ab - siehe train.py, generate_folds -
     das Zusammenfuegen erzeugt also KEINE neue Ueberlappung).
  3. Auf diese durchgehende Out-of-Sample-Zeitreihe wird
     stats.reliability_summary() angewendet - GENAU dieselbe Newey-West/
     Perioden-Konsistenz-Pruefung, die schon fuer den reinen Preis-Backtest
     (siehe dashboard.py) verwendet wird, jetzt aber auf die vom ML-Modell
     als 'stark_positiv'/'stark_negativ' klassifizierten Trades beschraenkt.

Ergebnis: eine belastbare Antwort auf "wenn ich nur den Trades folge, die
das Modell als stark_positiv/stark_negativ einstuft, ist das dann ueber die
GESAMTE Historie hinweg ein robuster Edge, oder nur Rauschen in einzelnen
Jahren?"
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from labels import CLASS_LABELS
from train import (
    get_feature_columns, load_vintage, ensemble_predict_proba, apply_classes_by_horizon,
)
from stats import reliability_summary

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
TRAINING_TABLE_FILE = DATA_DIR / "training_table.parquet"
OOS_PREDICTIONS_FILE = DATA_DIR / "oos_predictions.parquet"


def collect_oos_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Baut die durchgehende Out-of-Sample-Vorhersage-Zeitreihe ueber alle
    Vintages hinweg (siehe Modul-Docstring, Schritte 1+2)."""
    vintage_dirs = sorted(MODELS_DIR.glob("vintage_*"))
    if not vintage_dirs:
        print(f"[fehler] keine vintage_*-Ordner unter {MODELS_DIR} gefunden.")
        print("[info] Erst ausfuehren: python src/train.py")
        sys.exit(1)

    all_rows = []
    for vintage_dir in vintage_dirs:
        bundle = load_vintage(vintage_dir)
        meta = bundle["metadata"]
        feature_cols = meta["feature_columns"]
        test_start = pd.Timestamp(meta["test_start"])
        test_end = pd.Timestamp(meta["test_end"])

        test_mask = (df["entry_date"] >= test_start) & (df["entry_date"] < test_end)
        test_df = df.loc[test_mask].dropna(subset=feature_cols)
        if test_df.empty:
            print(f"[warnung] {vintage_dir.name}: keine gueltigen Test-Zeilen - uebersprungen.")
            continue

        X_test = test_df[feature_cols].to_numpy(dtype=float)
        proba = ensemble_predict_proba(bundle["models"], X_test)
        pred_class = np.array(CLASS_LABELS)[np.argmax(proba, axis=1)]

        true_class_cat = apply_classes_by_horizon(test_df, bundle["thresholds_by_T"])
        true_class = pd.Series(true_class_cat, index=test_df.index).astype(str)
        # Categorical.astype(str) macht aus fehlenden Codes (-1, siehe
        # apply_classes_by_horizon) den String 'nan' statt eines echten NaN -
        # hier wieder korrekt als fehlend markieren.
        true_class = true_class.replace("nan", np.nan)

        hold_days = (test_df["exit_date"] - test_df["entry_date"]).dt.days

        all_rows.append(pd.DataFrame({
            "vintage": vintage_dir.name,
            "entry_date": test_df["entry_date"].to_numpy(),
            "T_days": test_df["T_days"].to_numpy(),
            "pnl_norm": test_df["pnl_norm"].to_numpy(),
            "hold_days": hold_days.to_numpy(),
            "true_class": true_class.to_numpy(),
            "pred_class": pred_class,
            "pred_proba_max": proba.max(axis=1),
        }))

    oos_df = pd.concat(all_rows, ignore_index=True)
    oos_df["entry_date"] = pd.to_datetime(oos_df["entry_date"])
    return oos_df.sort_values("entry_date").reset_index(drop=True)


def reliability_by_predicted_class(oos_df: pd.DataFrame, by_horizon: bool = False) -> pd.DataFrame:
    """Wendet stats.reliability_summary() auf jede vom Modell vorhergesagte
    Klasse an (optional zusaetzlich je Laufzeit aufgeschluesselt)."""
    rows = []
    group_cols = ["pred_class"] + (["T_days"] if by_horizon else [])
    for keys, g in oos_df.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rel = reliability_summary(g, nw_lags=None, n_subperiods=6)
        row = dict(zip(group_cols, keys))
        row.update({
            "n_trades": rel["n_trades"],
            "mean_pnl_norm": rel["mean_pnl_norm"],
            "tstat": rel["tstat"],
            "ci_low": rel["ci_low"],
            "ci_high": rel["ci_high"],
            "ci_excludes_zero": rel["ci_excludes_zero"],
            "subperiod_matching": rel["subperiod_matching"],
            "subperiod_nonempty": rel["subperiod_nonempty"],
            "nw_lag_used": rel["nw_lag_used"],
        })
        rows.append(row)
    return pd.DataFrame(rows)


def confusion_summary(oos_df: pd.DataFrame) -> pd.DataFrame:
    """Konfusionsmatrix (Zeilen=tatsaechliche Klasse, Spalten=Vorhersage),
    normiert je Zeile (Recall-Sicht: 'von allen echten stark_positiv-Tagen,
    wie viele hat das Modell erkannt')."""
    valid = oos_df.dropna(subset=["true_class"])
    ct = pd.crosstab(valid["true_class"], valid["pred_class"], normalize="index")
    return ct.reindex(index=CLASS_LABELS, columns=CLASS_LABELS)


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

    print("[ok] sammle Out-of-Sample-Vorhersagen ueber alle Vintages ...")
    oos_df = collect_oos_predictions(df)
    oos_df.to_parquet(OOS_PREDICTIONS_FILE, index=False)
    print(f"[ok] {len(oos_df):,} Out-of-Sample-Zeilen ueber "
          f"{oos_df['vintage'].nunique()} Vintages, "
          f"Zeitraum {oos_df['entry_date'].min().date()} bis {oos_df['entry_date'].max().date()}.")
    print(f"[ok] gespeichert unter: {OOS_PREDICTIONS_FILE}")

    print("\n=== RELIABILITY JE VORHERGESAGTER KLASSE (alle Laufzeiten zusammen) ===")
    overall = reliability_by_predicted_class(oos_df, by_horizon=False)
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(overall.to_string(index=False))

    print("\n=== RELIABILITY JE VORHERGESAGTER KLASSE x LAUFZEIT ===")
    by_t = reliability_by_predicted_class(oos_df, by_horizon=True)
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(by_t.sort_values(["pred_class", "T_days"]).to_string(index=False))

    print("\n=== KONFUSIONSMATRIX (Zeile=tatsaechliche Klasse, Spalte=Vorhersage, Recall je Zeile) ===")
    cm = confusion_summary(oos_df)
    print(cm.round(3).to_string())

    overall_path = MODELS_DIR / "evaluation_reliability_overall.csv"
    by_t_path = MODELS_DIR / "evaluation_reliability_by_horizon.csv"
    cm_path = MODELS_DIR / "evaluation_confusion_matrix.csv"
    overall.to_csv(overall_path, index=False)
    by_t.to_csv(by_t_path, index=False)
    cm.to_csv(cm_path)
    print(f"\n[ok] Reports gespeichert:\n  {overall_path}\n  {by_t_path}\n  {cm_path}")


if __name__ == "__main__":
    main()