"""
Zentrale Instrument-Konfiguration fuer Multi-Underlying-Unterstuetzung.

Jedes unterstuetzte Basisinstrument wird hier EINMAL beschrieben - alle
anderen Module (data_fetch, backtest, features, labels, market_data,
depot_engine, dashboard_*) lesen ihre instrumentenspezifischen Parameter
(Ticker, Waehrung, Multiplikator, Dividenden-/Convenience-Yield,
Datenpfade, welcher Options-Ketten-Anbieter zustaendig ist) ausschliesslich
von hier - NICHT mehr hartcodiert verstreut.

CHAIN-PROVIDER (siehe chain_providers/base.py, Registry-Pattern):
  "yfinance": Optionskette via Yahoo Finance. Funktioniert nur fuer US-Indizes/
              ETFs mit an US-Boersen gelisteten, liquiden Optionen.
  "ib":       Optionskette via Interactive-Brokers-API (ib_async). Noetig fuer
              Eurex-Produkte, fuer die es KEINE kostenlose Retail-Datenquelle
              gibt. Voraussetzung: laufende TWS/IB-Gateway-Instanz + ein
              aktives, kostenpflichtiges Marktdaten-Abo bei IB.
  None:       Keine Live-Kette verfuegbar - Instrument laeuft nur in Modus 1
              (Backtest) und Modus 2 (Analog-Matching/ML), nicht in Modus 3
              (Heutige Kette) oder Modus 4 (Depot-Live-Bewertung).
  Weitere Anbieter (z.B. ein bezahlter Feed) lassen sich spaeter per
  chain_providers.base.register_provider("name", factory) ergaenzen, OHNE
  dieses Modul oder chain_providers/base.py selbst anzufassen - dafuer nur
  hier bei einem Instrument chain_provider="name" setzen.

BEWUSSTE VEREINFACHUNG BEI ROHSTOFFEN (Gold, Oel): statt Kurshistorie vom
Futures-Kontrakt (GC=F/CL=F) UND Kette vom ETF-Proxy zu mischen, wird hier
DURCHGEHEND derselbe ETF (GLD/USO) fuer Kurs UND Kette verwendet. Vorteil:
keine Rollover-Spruenge in der HV-Berechnung, keine Preis/Ketten-Inkonsistenz.
Nachteil, den man kennen sollte: GLD/USO bilden den zugrunde liegenden
Rohstoff NICHT exakt 1:1 ab (bei USO wegen Rollkosten/Contango teils
spuerbar ueber laengere Zeitraeume, bei GLD sehr eng aber mit kleiner
Verwaltungsgebuehr-Drift) - fuer ein Backtest-/Analog-Tool ein akzeptabler
Kompromiss, aber kein Ersatz fuer eine echte Futures-Kurve.

DATENTIEFE: SPX und Gold haben eine zusaetzliche MONATLICHE Fruehgeschichte
vor Beginn der echten Tagesdaten (siehe has_shiller_history, long_history_label
und data_fetch.fetch_and_save() / MONTHLY_HISTORY_FETCHERS):
  - SPX:  Shiller-S&P-500-Monatsreihe (github.com/datasets/s-and-p-500),
          seit 1871. ^GSPC (Tagesdaten) und die Shiller-Reihe sind BEIDE echte
          Indexpunkte auf derselben Skala - werden nahtlos aneinandergehaengt,
          OHNE Skalierung.
  - Gold: Monatliche Weltbank-/Timothy-Green-Goldpreisreihe (github.com/
          datasets/gold-prices), seit 1833, in absoluten USD/Feinunze.
          WICHTIG: das ist NICHT dieselbe Skala wie der GLD-ETF-Kurs (die
          echten Tagesdaten dieses Projekts fuer Gold, siehe oben) - GLD
          bildet historisch ca. 1/10 Feinunze Gold ab, nicht exakt konstant
          ueber die Jahrzehnte. data_fetch.py skaliert die Monatsreihe daher
          beim Aufbau EINMALIG auf den tatsaechlichen GLD-Kurs am ersten
          echten GLD-Handelstag (siehe data_fetch.scale_monthly_to_daily()) -
          sonst entstuende an der Nahtstelle 2004 ein kuenstlicher ~10x-Sprung
          (dieselbe Fehlerklasse wie ein Skalen-Mismatch zwischen chain_ticker
          und price_ticker, hier aber in der Historie statt live).
  Andere Instrumente (Russell 2000, Nasdaq-100, Nikkei, EuroStoxx50, Oel)
  haben KEINE solche Fruehgeschichte - ihre Historie beginnt schlicht dort,
  wo yfinance echte Tagesdaten liefert:
  ^RUT   ab ca. 1987/88          ^NDX  ab ca. 1985
  ^N225  ab ca. 1965 (Yahoo)     USO   ab 2006 (Auflage des ETF)
  ^STOXX50E ab ca. 1998 (der Index existiert erst seit 1998)

DIVIDEND_YIELD wird bei Rohstoff-ETFs zur Naeherung fuer Verwaltungsgebuehr/
Convenience Yield in forward_price() (siehe black76.py) - kann dort auch 0
oder negativ sein. Fuer GLD/USO ist der Effekt klein und wird hier
vereinfachend auf 0 gesetzt.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(unsafe_hash=True)
class InstrumentConfig:
    key: str                        # interner Schluessel, z.B. "spx" - wird Teil jedes Dateinamens
    label: str                      # Anzeigename im Dashboard
    price_ticker: str               # yfinance-Ticker fuer die taegliche OHLC-Historie
    currency: str                   # "USD" | "EUR" | "JPY" - nur Anzeige, KEINE FX-Umrechnung im Projekt
    multiplier: float                # Punkte -> Geldbetrag je Kontrakt (Options-Kontraktgroesse)
    dividend_yield: float            # Default fuer forward_price(); bei Rohstoff-ETFs Naeherung, siehe oben
    has_shiller_history: bool        # True = es gibt eine monatliche Fruehgeschichte (siehe Moduldokstring
                                      # "DATENTIEFE") - Name aus Zeiten, als das nur fuer SPX/Shiller galt,
                                      # gilt inzwischen generisch fuer jedes Instrument mit Eintrag in
                                      # data_fetch.MONTHLY_HISTORY_FETCHERS.
    settlement: str                  # "cash_index" | "etf" | "future" (nur Doku/Info)
    chain_provider: str | None       # "yfinance" | "ib" | None | eigener registrierter Name
    chain_ticker: str | None = None       # yfinance-/Provider-Symbol der Kette, falls != price_ticker
    long_history_label: str | None = None  # Anzeigetext fuer die monatliche Fruehgeschichte (dashboard_common.py) -
                                            # None, wenn has_shiller_history False ist.
    ib_index_symbol: str | None = None    # IB-Symbol des Index (z.B. "ESTX50") - nur falls chain_provider="ib"
    ib_exchange: str | None = None        # IB-Boerse (z.B. "EUREX")
    ib_currency: str | None = None        # IB-Kontraktwaehrung, falls != currency
    ib_trading_class: str | None = None   # IB Trading-Class der Option (z.B. "OESX")


INSTRUMENTS: dict[str, InstrumentConfig] = {
    "spx": InstrumentConfig(
        key="spx", label="S&P 500", price_ticker="^GSPC", currency="USD",
        multiplier=100.0, dividend_yield=0.015, has_shiller_history=True,
        settlement="cash_index", chain_provider="yfinance", chain_ticker="^SPX",
        long_history_label="Robert Shillers Monatsdaten (seit 1871)",
        # WICHTIG: Kurshistorie ^GSPC, Kette aber ^SPX - Yahoo fuehrt die
        # SPX-Optionskette (CBOE-Feed) unter einem anderen Ticker als die
        # Index-Kurshistorie. Das war im urspruenglichen fetch_today_chain.py
        # (TICKER = "^SPX") bereits richtig geloest; hier uebernommen.
    ),
    "rut": InstrumentConfig(
        key="rut", label="Russell 2000", price_ticker="^RUT", currency="USD",
        multiplier=100.0, dividend_yield=0.013, has_shiller_history=False,
        settlement="cash_index", chain_provider="yfinance", chain_ticker="IWM",
        # HINWEIS: IWM ist ein ETF (Standard-Equity-Kontraktgroesse 100), NICHT
        # identisch mit einer echten ^RUT-Indexoption bei CBOE. price_ticker
        # (^RUT, Indexpunkte ~2000-3000) und chain_ticker (IWM, ETF-Kurs
        # ~1/10 davon) sind bewusst UNTERSCHIEDLICH skaliert - siehe
        # dashboard_livechain._build_daily_df_with_today() fuer die
        # Skalen-Korrektur beim synthetischen 'Heute'-Balken.
    ),
    "ndx": InstrumentConfig(
        key="ndx", label="Nasdaq-100", price_ticker="^NDX", currency="USD",
        multiplier=100.0, dividend_yield=0.007, has_shiller_history=False,
        settlement="cash_index", chain_provider="yfinance", chain_ticker="^NDX",
    ),
    "nikkei": InstrumentConfig(
        key="nikkei", label="Nikkei 225", price_ticker="^N225", currency="JPY",
        multiplier=1000.0, dividend_yield=0.02, has_shiller_history=False,
        settlement="cash_index", chain_provider=None, chain_ticker=None,
        # HINWEIS: keine kostenlose, verlaessliche Optionskette verfuegbar (an
        # der OSE/JPX gehandelt, kein Yahoo-Feed) - daher chain_provider=None.
        # Backtest (Modus 1) und Analog-Matching/ML (Modus 2) funktionieren mit
        # der ^N225-Kurshistorie trotzdem; 'Heutige Kette' und Depot-Live-
        # Bewertung stehen fuer Nikkei nicht zur Verfuegung. multiplier=1000
        # ist der reale OSE-Kontraktwert (JPY 1000 je Indexpunkt) - wird
        # aktuell nirgends live genutzt, da keine Kette angebunden ist.
    ),
    "sx5e": InstrumentConfig(
        key="sx5e", label="EuroStoxx 50", price_ticker="^STOXX50E", currency="EUR",
        multiplier=10.0, dividend_yield=0.030, has_shiller_history=False,
        settlement="cash_index", chain_provider=None,
        # ABSICHTLICH None (nicht "ib"): Eurex-Anbindung ueber Interactive
        # Brokers ist implementiert (chain_providers/ib_provider.py) und kann
        # jederzeit durch chain_provider="ib" aktiviert werden, sobald IB
        # TWS/Gateway + ein aktives Eurex-Marktdaten-Abo vorhanden sind. Ohne
        # das laesst sich diese Anbindung von hier aus nicht testen, deshalb
        # standardmaessig deaktiviert - SX5E laeuft bis dahin nur in Backtest/ML.
        ib_index_symbol="ESTX50", ib_exchange="EUREX", ib_currency="EUR",
        ib_trading_class="OESX",
    ),
    "gold": InstrumentConfig(
        key="gold", label="Gold (via GLD)", price_ticker="GLD", currency="USD",
        multiplier=100.0, dividend_yield=0.0, has_shiller_history=True,
        settlement="etf", chain_provider="yfinance", chain_ticker="GLD",
        long_history_label="Weltbank-/Timothy-Green-Goldpreise (seit 1833, auf GLD-Skala umgerechnet)",
        # Kurs UND Kette durchgehend GLD (siehe Modul-Docstring) - kein GC=F mehr.
        # has_shiller_history=True (GEAENDERT): erweitert die Historie ueber
        # GLDs Auflegung (2004) hinaus mit monatlichen Goldpreisen seit 1833
        # (siehe Moduldokstring "DATENTIEFE" und data_fetch.fetch_gold_monthly()/
        # scale_monthly_to_daily() fuer die noetige Skalierung auf GLD-Niveau).
    ),
    "oil": InstrumentConfig(
        key="oil", label="Rohoel WTI (via USO)", price_ticker="USO", currency="USD",
        multiplier=100.0, dividend_yield=0.0, has_shiller_history=False,
        settlement="etf", chain_provider="yfinance", chain_ticker="USO",
        # Kurs UND Kette durchgehend USO (siehe Modul-Docstring) - kein CL=F.
        # USO bildet WTI wegen Rollkosten/Contango NICHT exakt 1:1 ab.
    ),
}

DEFAULT_INSTRUMENT = "spx"


def get_instrument(key: str) -> InstrumentConfig:
    try:
        return INSTRUMENTS[key]
    except KeyError:
        raise ValueError(f"Unbekanntes Instrument '{key}'. Verfuegbar: {sorted(INSTRUMENTS)}") from None


def data_file(instr: InstrumentConfig, kind: str, data_dir: Path) -> Path:
    """Einheitlicher, instrumentenspezifischer Dateiname.
    Beispiel: data_file(get_instrument('spx'), 'daily', DATA_DIR)
              -> data/spx_daily.parquet
              data_file(get_instrument('oil'), 'training_table', DATA_DIR)
              -> data/oil_training_table.parquet"""
    return Path(data_dir) / f"{instr.key}_{kind}.parquet"


def models_dir(instr: InstrumentConfig, models_dir_base: Path) -> Path:
    """Instrumentenspezifischer Modell-Ordner, z.B. models/oil/vintage_2026-01-01/."""
    d = Path(models_dir_base) / instr.key
    d.mkdir(parents=True, exist_ok=True)
    return d