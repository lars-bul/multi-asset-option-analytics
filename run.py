"""
Einziger Befehl, um alles zu starten:

    python run.py

Macht automatisch, in dieser Reihenfolge:
  1. Legt ./venv an, falls noch nicht vorhanden.
  2. Installiert/aktualisiert die Pakete aus requirements.txt - aber nur,
     wenn sich requirements.txt seit dem letzten Lauf geaendert hat (schneller
     Start bei wiederholtem Aufruf).
  3. Aktualisiert data/spx_daily.parquet - aber nur, wenn die Datei fehlt
     oder aelter als 20 Stunden ist (kein unnoetiger Netzwerk-Call bei jedem
     Start, aber taeglich frische Daten).
  4. Startet das Streamlit-Dashboard (oeffnet automatisch den Browser).

Laeuft mit dem System-Python (egal ob 'python' oder 'python3'), erledigt
den Rest selbststaendig ueber das lokale venv.
"""

import hashlib
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / "venv"
REQUIREMENTS = ROOT / "requirements.txt"
REQ_HASH_MARKER = VENV_DIR / ".requirements_hash"
DATA_FILE = ROOT / "data" / "spx_daily.parquet"
DATA_MAX_AGE_HOURS = 20

IS_WINDOWS = sys.platform.startswith("win")
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def log(msg: str):
    print(f"[run.py] {msg}")


def ensure_venv():
    if VENV_PYTHON.exists():
        return
    log("Kein venv gefunden - lege eins an unter ./venv ...")
    subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
    log("venv angelegt.")


def requirements_changed() -> bool:
    if not REQUIREMENTS.exists():
        return False
    current_hash = hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()
    if not REQ_HASH_MARKER.exists():
        return True
    return REQ_HASH_MARKER.read_text().strip() != current_hash


def ensure_requirements():
    if not requirements_changed():
        log("Pakete bereits aktuell (requirements.txt unveraendert) - ueberspringe Installation.")
        return
    log("Installiere/aktualisiere Pakete aus requirements.txt ...")
    subprocess.run(
        [str(VENV_PYTHON), "-m", "pip", "install", "--quiet", "--upgrade", "pip"],
        check=True,
    )
    subprocess.run(
        [str(VENV_PYTHON), "-m", "pip", "install", "--quiet", "-r", str(REQUIREMENTS)],
        check=True,
    )
    current_hash = hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()
    REQ_HASH_MARKER.write_text(current_hash)
    log("Pakete installiert.")


def data_is_stale() -> bool:
    if not DATA_FILE.exists():
        return True
    age_hours = (time.time() - DATA_FILE.stat().st_mtime) / 3600.0
    return age_hours > DATA_MAX_AGE_HOURS


def ensure_data():
    if not data_is_stale():
        age_hours = (time.time() - DATA_FILE.stat().st_mtime) / 3600.0
        log(f"Daten sind noch aktuell ({age_hours:.1f}h alt) - ueberspringe Abruf.")
        return
    log("Aktualisiere SPX-Daten (data_fetch.py) ...")
    result = subprocess.run(
        [str(VENV_PYTHON), str(ROOT / "src" / "data_fetch.py")],
    )
    if result.returncode != 0:
        log("Datenabruf hat ein Problem gemeldet - Dashboard startet trotzdem "
            "mit den zuletzt vorhandenen Daten (falls welche existieren).")


def launch_dashboard():
    log("Starte Dashboard - oeffnet gleich im Browser (Strg+C zum Beenden) ...")
    subprocess.run(
        [str(VENV_PYTHON), "-m", "streamlit", "run", str(ROOT / "src" / "dashboard.py")]
    )


def main():
    ensure_venv()
    ensure_requirements()
    ensure_data()
    launch_dashboard()


if __name__ == "__main__":
    main()
