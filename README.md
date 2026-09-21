# SPX Options Dashboard (lokal)

Rein lokales Analyse-Tool. Kein Server, kein Docker, kein Deployment -
laeuft auf `localhost` und wird nur bei Bedarf gestartet.

## Start (ein einziger Befehl)

```bash
python run.py
```

Das war's - macht automatisch alles Noetige:
1. Legt `./venv` an, falls noch nicht vorhanden.
2. Installiert/aktualisiert die Pakete aus `requirements.txt` (nur beim
   ersten Mal bzw. wenn sich die Datei geaendert hat - sonst uebersprungen,
   damit der Start beim naechsten Mal schnell geht).
3. Aktualisiert `data/spx_daily.parquet`, falls die Datei fehlt oder aelter
   als 20 Stunden ist (kein unnoetiger Netzwerk-Call bei jedem Start, aber
   taeglich frische Daten).
4. Startet das Dashboard und oeffnet `http://localhost:8501` im Browser.

Mit `Strg+C` im Terminal wieder stoppen.

**Alternativ per Doppelklick** (statt Terminal): `start.bat` (Windows) bzw.
`start.sh` (macOS/Linux, vorher einmalig `chmod +x start.sh`).

**Manuell/Schritt-fuer-Schritt geht natuerlich weiterhin**, falls gewuenscht
(z.B. fuer Debugging) - siehe Abschnitt "Manuelle Einzelschritte" unten.

## Daten laden

```bash
python src/data_fetch.py
```

(Wird von `python run.py` automatisch erledigt - manuell nur noetig, wenn
du sofort frische Daten erzwingen willst, unabhaengig vom 20h-Intervall.)

Laedt SPX-Kursdaten so frueh wie moeglich beginnend und speichert sie unter
`data/spx_daily.parquet`. Kann jederzeit erneut ausgefuehrt werden, um zu
aktualisieren.

**Zwei kombinierte Quellen, klar gekennzeichnet:**

- **ab 1927-12-30**: echte taegliche OHLC-Daten via `yfinance` (`^GSPC`) -
  das ist die harte Grenze, vor der es fuer den S&P schlicht keine echten
  Tagesdaten gibt, egal welcher Anbieter.
- **1871-01 bis 1927-12-29**: Robert Shillers monatliche S&P-Composite-Reihe
  (frei via GitHub, `datasets/s-and-p-500`), auf Handelstage vorwaertsgefuellt.
  Deckt damit 1900, die Great Depression (1929-1932 faellt schon in den
  Tagesdaten-Bereich) und alles davor ab - nur mit monatlicher statt
  taeglicher Aufloesung.

Jede Zeile hat eine Spalte `resolution` (`daily` oder
`monthly_interpolated`), damit spaeterer Code (z.B. Realized-Vol-Berechnung)
weiss, welche Tage echte Bewegungen zeigen und welche nur grob interpoliert
sind. Im Dashboard gibt es dazu eine Sidebar-Checkbox, um die interpolierten
Tage bei Bedarf auszublenden.

Falls (noch) keine Internetverbindung besteht, wird automatisch auf die
jeweils erreichbare Quelle zurueckgefallen (z.B. nur Shiller-Monatsdaten,
falls yfinance nicht erreichbar ist); nur wenn wirklich gar keine Quelle
erreichbar ist, springen synthetische Platzhalterdaten ein, damit das
Dashboard trotzdem sofort lauffaehig bleibt.

## Dashboard starten

```bash
streamlit run src/dashboard.py
```

(Wird von `python run.py` automatisch erledigt.)

Oeffnet automatisch `http://localhost:8501` im Browser. Mit `Strg+C` im
Terminal wieder stoppen.

## Manuelle Einzelschritte (nur falls gewuenscht, z.B. zum Debuggen)

```bash
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
python src/data_fetch.py
streamlit run src/dashboard.py
```

## Projektstruktur

```
spx-dashboard/
├── run.py              # Einzelner Startbefehl - regelt venv/Pakete/Daten/Start
├── start.bat           # Doppelklick-Starter fuer Windows
├── start.sh            # Doppelklick-/Terminal-Starter fuer macOS/Linux
├── data/               # Parquet-Dateien (SPX-Historie, spaeter Options-Snapshots)
├── models/             # trainierte ML-Modelle (joblib .pkl + Metadaten .json)
├── src/
│   ├── data_fetch.py   # taeglicher Datenabruf (SPX ab 1871, siehe unten)
│   ├── black76.py      # Black-76-Bewertung + Vol-Oberflaeche (Skew/Term Structure)
│   ├── backtest.py     # Iron-Condor-Ladder-Backtest (portiert aus dem Pine-Skript)
│   ├── stats.py        # Newey-West t-Stat + Perioden-Konsistenz (Reliability-Check)
│   └── dashboard.py    # Streamlit-Oberflaeche: Chart + Backtest-Panel
├── requirements.txt
└── README.md
```

## Iron Condor Backtest (im Dashboard)

Im Dashboard, unterhalb des Charts, gibt es ein Backtest-Panel. Parameter
werden in der Sidebar eingestellt (Laufzeit T, HV-Periode, Delta je Bein,
optionaler Stop-Loss bei Short-Strike-Beruehrung, optionaler vorzeitiger
Ausstieg, Vol-Oberflaeche an/aus), dann Button "Backtest ausfuehren".

**Was portiert wurde (aus dem Pine-Skript, Modus "taegliche ueberlappende
Positionen"):**
- Black-76-Bewertung mit Skew + Term-Structure-Vol-Oberflaeche (`black76.py`)
- Taeglich neue, unabhaengige Iron-Condor-Struktur (Long Put Wing / Short Put
  / Short Call / Long Call Wing), IV-Quelle = historische Volatilitaet (HV)
- Optionaler Stop-Loss bei Short-Strike-Beruehrung, optionaler laufzeit-
  basierter vorzeitiger Ausstieg
- Newey-West(HAC)-korrigierter t-Stat + Perioden-Konsistenz-Check als
  Reliability-Kennzahlen (`stats.py`)

**Was bewusst NICHT uebernommen wurde:**
- Die Futures-Terminstruktur-Logik (Backwardation/Contango-basierte Put/Call-
  Umschaltung, Symbol-Mapping fuer verschiedene Maerkte) - es wird immer
  derselbe klassische Iron Condor auf dem regulaeren SPX-Kassaindex gehandelt.
- Der VIX/RVX als alternative IV-Quelle - nur HV, damit der Backtest auf der
  gesamten verfuegbaren Kurshistorie laeuft (VIX gibt es erst ab 1990).
- Der 200-Tage-Trend-Filter - damit wirklich jeden Handelstag dieselbe
  Struktur eroeffnet wird, ohne bedingtes Gating.

**Wichtig zur Datenreichweite:** Der Backtest filtert automatisch auf echte
Tagesdaten (`resolution == "daily"`, siehe unten) - er nutzt also nur den
Zeitraum ab 1927-12-30, nicht die monatlich interpolierten Daten davor (HV-
Berechnung auf kuenstlich flach fortgeschriebenen Monatswerten waere
bedeutungslos).


  basiert um 0 zentriert), speichert Modell nach `models/`
- Options-Preis-Abruf (Polygon/Tradier) als zweite Datenquelle fuer den
  Live-Marktpreis-Vergleich
- Weitere Dashboard-Panels: IV-Rank-Verlauf, Bucket-Klassifikation je
  Handelstag, Trade-Empfehlungstabelle
