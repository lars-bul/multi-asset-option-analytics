"""
Baut den kompletten Trainings-Datensatz fuers ML-Modell / die Analog-Suche aus
den lokalen SPX-Kursdaten.

Einfach ausfuehren mit:
    python src/build_dataset.py

Voraussetzung: data/spx_daily.parquet muss existieren (siehe src/data_fetch.py).

WAS HIER PASSIERT (siehe features.py/labels.py fuer die Details je Schritt):
  1. Kursdaten laden, auf echte Tagesdaten filtern.
  2. Feature-Matrix bauen (eine Zeile je Handelstag).
  3. Strangle-Labels generieren (eine Zeile je Handelstag x Laufzeit,
     T_days = 1..90, 25-Delta Short Strangle bis Verfall gehalten).
     Jede Laufzeit laeuft in einem eigenen Prozess (Python-Loop dominiert).
  4. Beides ueber das Entry-Datum zusammenfuehren.
  5. Feature-Spalten als float32, Ergebnis als data/training_table.parquet.

WICHTIG: 'pnl_norm' bleibt roh - die Klassen-Schwellen duerfen NUR pro
Walk-Forward-Trainingsfenster berechnet werden (siehe labels.py).
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import FeatureConfig, build_feature_matrix, feature_columns
from labels import StrangleLabelConfig, generate_strangle_labels, build_training_table

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SPX_FILE = DATA_DIR / "spx_daily.parquet"
OUT_FILE = DATA_DIR / "training_table.parquet"


def _labels_for_T(args):
    df, cfg, T = args
    return generate_strangle_labels(df, replace(cfg, T_days_list=[T]))


def main():
    if not SPX_FILE.exists():
        print(f"[fehler] {SPX_FILE} nicht gefunden.")
        print("[info] Erst einmal ausfuehren: python src/data_fetch.py")
        sys.exit(1)

    print(f"[ok] lade {SPX_FILE} ...")
    df = pd.read_parquet(SPX_FILE)
    df["date"] = pd.to_datetime(df["date"])

    n_daily = (df["resolution"] == "daily").sum() if "resolution" in df.columns else len(df)
    print(f"[ok] {n_daily} echte Handelstage gefunden.")

    print("[ok] berechne Feature-Matrix ...")
    feat_df = build_feature_matrix(df, FeatureConfig())
    print(f"[ok] {feat_df.shape[0]} Zeilen, {feat_df.shape[1] - 1} Feature-Spalten.")

    label_cfg = StrangleLabelConfig()
    workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"[ok] generiere Strangle-Labels fuer T_days={label_cfg.T_days_list[0]}..."
          f"{label_cfg.T_days_list[-1]} mit {workers} Prozessen (dauert am laengsten) ...")
    jobs = [(df, label_cfg, T) for T in label_cfg.T_days_list]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        label_df = pd.concat(list(ex.map(_labels_for_T, jobs)), ignore_index=True)
    label_df = label_df.sort_values(["entry_date", "T_days"]).reset_index(drop=True)
    print(f"[ok] {label_df.shape[0]} Label-Zeilen ueber {len(label_cfg.T_days_list)} Laufzeiten.")

    print("[ok] fuehre Features und Labels zusammen ...")
    train_df = build_training_table(feat_df, label_df)

    fcols = feature_columns(feat_df)
    train_df[fcols] = train_df[fcols].astype("float32")

    train_df.to_parquet(OUT_FILE, index=False)
    print(f"[ok] gespeichert unter: {OUT_FILE}")
    print(f"     {train_df.shape[0]} Zeilen, {train_df.shape[1]} Spalten")
    print(f"     Zeitraum: {train_df['entry_date'].min().date()} bis {train_df['entry_date'].max().date()}")


if __name__ == "__main__":
    main()