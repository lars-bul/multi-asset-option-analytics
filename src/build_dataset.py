"""
Baut den kompletten Trainings-Datensatz fuers ML-Modell / die Analog-Suche aus
den lokalen Tageskursdaten - fuer EIN ODER MEHRERE Instrumente aus
instruments.py (vorher SPX-spezifisch fest verdrahtet, siehe Aenderungshinweis
weiter unten).

Einfach ausfuehren mit:
    python src/build_dataset.py            # baut ALLE Instrumente aus instruments.py
    python src/build_dataset.py spx        # nur SPX
    python src/build_dataset.py gold rut   # mehrere gezielt

Voraussetzung je Instrument: data/{instr.key}_daily.parquet muss existieren
(siehe src/data_fetch.py {instr.key}).

WAS HIER PASSIERT (siehe features.py/labels.py fuer die Details je Schritt):
  1. Kursdaten des Instruments laden, auf echte Tagesdaten filtern
     (resolution == 'daily' - bei SPX schliesst das die vorwaertsgefuellten
     Shiller-Monatswerte vor 1927-12-30 aus; bei allen anderen Instrumenten
     ist 'resolution' laut data_fetch.py durchgehend 'daily', der Filter ist
     dort also ein No-Op).
  2. Feature-Matrix bauen (eine Zeile je Handelstag).
  3. Strangle-Labels generieren (eine Zeile je Handelstag x Laufzeit,
     T_days = 1..90, 25-Delta Short Strangle bis Verfall gehalten).
     Jede Laufzeit laeuft in einem eigenen Prozess (Python-Loop dominiert).
  4. Beides ueber das Entry-Datum zusammenfuehren.
  5. Feature-Spalten als float32, Ergebnis als
     data/{instr.key}_training_table.parquet (ueber instruments.data_file()).

AENDERUNG (Multi-Instrument, vorher nur SPX): frueher schrieb dieses Skript
IMMER nach dem einen globalen Pfad data/training_table.parquet, unabhaengig
vom Instrument, und las die Kurse immer aus dem festen data/spx_daily.parquet.
Das Dashboard (dashboard_livechain._tt_file() / dashboard_common.
load_training_table()) erwartet aber je Instrument einen EIGENEN,
instrumentenspezifischen Dateinamen (z.B. data/spx_training_table.parquet,
data/gold_training_table.parquet) - eine mit der alten Version erzeugte
data/training_table.parquet wird von der App seitdem NICHT MEHR gefunden
('Keine Trainingstabelle gefunden', auch nach erneutem Lauf dieses Skripts).
Falls eine solche alte Datei noch existiert, kann sie einmalig manuell nach
data/spx_training_table.parquet kopiert werden, um sich den ersten vollen
Neubau zu sparen:
    Windows (PowerShell):  Copy-Item data\\training_table.parquet data\\spx_training_table.parquet
    macOS/Linux:            cp data/training_table.parquet data/spx_training_table.parquet

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

from instruments import INSTRUMENTS, InstrumentConfig, data_file
from features import FeatureConfig, build_feature_matrix, feature_columns
from labels import StrangleLabelConfig, generate_strangle_labels, build_training_table

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _labels_for_T(args):
    df, cfg, T = args
    return generate_strangle_labels(df, replace(cfg, T_days_list=[T]))


def build_and_save(instr: InstrumentConfig, out_file: Path | None = None) -> pd.DataFrame | None:
    """Baut die Trainingstabelle fuer EIN Instrument und speichert sie unter
    data/{instr.key}_training_table.parquet (oder out_file, falls angegeben).
    Gibt None zurueck (mit Konsolen-Meldung), falls die Tageskurse fuer dieses
    Instrument noch nicht per data_fetch.py geladen wurden."""
    daily_file = data_file(instr, "daily", DATA_DIR)
    out_file = out_file or data_file(instr, "training_table", DATA_DIR)

    print(f"\n=== {instr.label} ({instr.key}) ===")
    if not daily_file.exists():
        print(f"[fehler] {daily_file} nicht gefunden.")
        print(f"[info] Erst einmal ausfuehren: python src/data_fetch.py {instr.key}")
        return None

    print(f"[ok] lade {daily_file} ...")
    df = pd.read_parquet(daily_file)
    df["date"] = pd.to_datetime(df["date"])

    if "resolution" in df.columns:
        df = df[df["resolution"] == "daily"].reset_index(drop=True)
    n_daily = len(df)
    print(f"[ok] {n_daily} echte Handelstage gefunden.")
    if n_daily < 300:
        print(f"[warnung] nur {n_daily} echte Handelstage - Feature-/Label-Fenster "
              "(z.B. ret_5y, HV-Lookbacks) brauchen deutlich mehr Historie; die "
              "Trainingstabelle kann dadurch fuer die meisten/alle Tage leer bzw. "
              "sehr kurz ausfallen.")

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

    out_file.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(out_file, index=False)
    print(f"[ok] gespeichert unter: {out_file}")
    print(f"     {train_df.shape[0]} Zeilen, {train_df.shape[1]} Spalten")
    if len(train_df):
        print(f"     Zeitraum: {train_df['entry_date'].min().date()} bis {train_df['entry_date'].max().date()}")
    return train_df


def main():
    keys = sys.argv[1:] or list(INSTRUMENTS.keys())
    unknown = [k for k in keys if k not in INSTRUMENTS]
    if unknown:
        print(f"[fehler] unbekannte Instrument-Schluessel: {unknown}. "
              f"Verfuegbar: {sorted(INSTRUMENTS)}")
        return 1

    failures = []
    for key in keys:
        result = build_and_save(INSTRUMENTS[key])
        if result is None:
            failures.append(key)

    if failures:
        print(f"\n[fehler] Fuer folgende Instrumente konnte KEINE Trainingstabelle gebaut werden "
              f"(fehlende Tageskurse): {failures}. Erst "
              f"'python src/data_fetch.py {' '.join(failures)}' ausfuehren, dann dieses Skript erneut.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())