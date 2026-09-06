"""
Baut den kompletten Trainings-Datensatz fuers ML-Modell aus den lokalen
SPX-Kursdaten.

Einfach ausfuehren mit:
    python src/build_dataset.py

Voraussetzung: data/spx_daily.parquet muss existieren (siehe
src/data_fetch.py - falls die Datei fehlt, wird hier direkt abgebrochen mit
einem entsprechenden Hinweis).

WAS HIER PASSIERT (siehe features.py/labels.py fuer die Details je Schritt):
  1. Kursdaten laden, auf echte Tagesdaten filtern (ab 1927-12-30).
  2. Feature-Matrix bauen (eine Zeile je Handelstag, ~38 Spalten:
     Returns/HV/Momentum/Tail-Risiko - siehe features.build_feature_matrix).
  3. Strangle-Labels generieren (eine Zeile je Handelstag x Laufzeit,
     25-Delta Short Strangle bis Verfall gehalten - siehe
     labels.generate_strangle_labels).
  4. Beides ueber das Entry-Datum zusammenfuehren (labels.build_training_table).
  5. Ergebnis als data/training_table.parquet speichern.

WICHTIG - was hier NOCH NICHT passiert (das ist Aufgabe von train.py, das
noch fehlt): die Umwandlung von 'pnl_norm' in die 4 Klassen (stark_negativ/
negativ/positiv/stark_positiv). Diese Schwellen duerfen NUR pro
Walk-Forward-Trainingsfenster berechnet werden (siehe labels.py,
generate_class_thresholds - Docstring-Warnung zu Leakage), deshalb enthaelt
diese Datei bewusst nur den rohen 'pnl_norm'-Wert, keine fertigen Klassen.
train.py liest diese Datei ein, schneidet sie in Zeitfenster und berechnet
die Klassen dann JE FENSTER neu.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import FeatureConfig, build_feature_matrix
from labels import StrangleLabelConfig, generate_strangle_labels, build_training_table

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SPX_FILE = DATA_DIR / "spx_daily.parquet"
OUT_FILE = DATA_DIR / "training_table.parquet"


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
    feature_cfg = FeatureConfig()
    feat_df = build_feature_matrix(df, feature_cfg)
    print(f"[ok] {feat_df.shape[0]} Zeilen, {feat_df.shape[1] - 1} Feature-Spalten.")

    print("[ok] generiere Strangle-Labels (das dauert am laengsten - "
          "pro Handelstag x Laufzeit wird ein Strike-Solve + Black-76-Bewertung "
          "durchgefuehrt) ...")
    label_cfg = StrangleLabelConfig()
    label_df = generate_strangle_labels(df, label_cfg)
    print(f"[ok] {label_df.shape[0]} Label-Zeilen ueber {len(label_cfg.T_days_list)} Laufzeiten "
          f"({label_cfg.T_days_list}).")

    print("[ok] fuehre Features und Labels zusammen ...")
    train_df = build_training_table(feat_df, label_df)

    train_df.to_parquet(OUT_FILE, index=False)
    print(f"[ok] gespeichert unter: {OUT_FILE}")
    print(f"     {train_df.shape[0]} Zeilen, {train_df.shape[1]} Spalten")
    print(f"     Zeitraum: {train_df['entry_date'].min().date()} bis {train_df['entry_date'].max().date()}")
    print("\n[info] Naechster Schritt (noch zu bauen): src/train.py liest diese Datei, "
          "berechnet die Klassen-Schwellen PRO Walk-Forward-Fenster und trainiert "
          "das Ensemble.")


if __name__ == "__main__":
    main()