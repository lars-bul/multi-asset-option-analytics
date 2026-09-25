"""
Holt Kursdaten fuer ein Instrument (siehe instruments.py) und speichert sie
lokal als Parquet - so frueh wie moeglich beginnend.

Einfach ausfuehren mit:
    python src/data_fetch.py            # laedt ALLE Instrumente aus instruments.py
    python src/data_fetch.py spx        # nur SPX
    python src/data_fetch.py gold rut   # mehrere gezielt

Kann jederzeit erneut laufen gelassen werden, um die lokale Historie zu
aktualisieren (ueberschreibt data/{instrument}_daily.parquet komplett - das
ist bei dieser Datenmenge unproblematisch).

DATENAUFLOESUNG - WICHTIG FUER SPAETERE AUSWERTUNGEN:
Nur fuer SPX (has_shiller_history=True) gibt es die zusaetzliche, sehr lange
Historie: echte TAEGLICHE OHLC-Daten fuer den S&P gibt es realistisch nicht vor
Ende 1927 (Yahoo Finance / ^GSPC beginnt am 1927-12-30) - das ist eine harte
Grenze. Was es aber gibt: Robert Shillers monatliche S&P-Composite-Reihe, frei
verfuegbar, zurueck bis 1871 (siehe github.com/datasets/s-and-p-500). Das
deckt 1900, die Great Depression (1929-1932) und alles davor ab - nur eben
mit monatlicher statt taeglicher Aufloesung. Fuer SPX kombiniert dieses Skript
daher zwei Quellen zu EINER durchgehenden Reihe:
  - ab 1927-12-30: echte taegliche OHLC-Daten (yfinance)
  - 1871-01 bis 1927-12-29: monatliche Shiller-Werte, auf Handelstage
    vorwaertsgefuellt (open=high=low=close, da keine echte Tagesspanne
    existiert)

FUER ALLE ANDEREN INSTRUMENTE (Russell 2000, Nasdaq-100, EuroStoxx50, Gold)
gibt es KEIN vergleichbares Shiller-Aequivalent - hier beginnt die Historie
schlicht dort, wo yfinance echte Daten liefert (siehe instruments.py fuer die
grobe Erwartung je Instrument). Insbesondere bei Gold (GC=F, Front-Month-
Future) ist zu beachten: das ist EIN durchgehend gehandelter Kontrakt-Strom
mit moeglichen Rollover-Spruengen, keine ueber Jahrzehnte "geglaettete" Reihe
wie ein Aktienindex.

Jede Zeile bekommt eine Spalte "resolution" ("daily" oder
"monthly_interpolated" - Letzteres nur bei SPX moeglich), damit nachgelagerter
Code (z.B. Realized-Vol-Berechnung fuers ML-Modell) weiss, welche Tage echte
Tagesbewegungen sind und welche nur grob interpoliert Platzhalter aus einem
Monatswert.
"""

from pathlib import Path
import sys

import pandas as pd
import numpy as np

from instruments import INSTRUMENTS, InstrumentConfig, data_file

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

SHILLER_CSV_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"


def fetch_via_yfinance(instr: InstrumentConfig) -> pd.DataFrame:
    """Echte taegliche Historie via yfinance fuer instr.price_ticker. Bei SPX
    beginnt das real erst 1927-12-30, auch wenn hier ein frueheres Startdatum
    angefragt wird; bei anderen Instrumenten gilt die jeweils fruehestmoegliche
    yfinance-Historie (siehe instruments.py)."""
    import yfinance as yf

    df = yf.download(instr.price_ticker, start="1900-01-01", auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"yfinance hat keine Daten fuer '{instr.price_ticker}' geliefert.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df = df.rename(columns={
        "Date": "date", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })
    df["resolution"] = "daily"
    return df[["date", "open", "high", "low", "close", "volume", "resolution"]]


def fetch_shiller_monthly() -> pd.DataFrame:
    """Monatliche S&P-Composite-Reihe seit 1871 (Robert Shiller, via die
    gepflegte CSV-Kopie im 'datasets'-GitHub-Repo). Nur ein Kurswert pro
    Monat - keine echte Tagesspanne. NUR fuer SPX relevant (siehe Modul-Docstring)."""
    df = pd.read_csv(SHILLER_CSV_URL, usecols=["Date", "SP500"])
    df = df.rename(columns={"Date": "date", "SP500": "close"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    return df.sort_values("date").reset_index(drop=True)


def expand_monthly_to_bdays(monthly: pd.DataFrame, until: pd.Timestamp) -> pd.DataFrame:
    """Vorwaertsfuellen der monatlichen Werte auf Handelstage, bis zu dem
    Datum, ab dem echte Tagesdaten uebernehmen ('until', exklusiv)."""
    monthly = monthly[monthly["date"] < until]
    if monthly.empty:
        return monthly.assign(open=[], high=[], low=[], volume=[], resolution=[])

    bdays = pd.DataFrame({
        "date": pd.bdate_range(start=monthly["date"].min(), end=until - pd.Timedelta(days=1))
    })
    expanded = pd.merge_asof(bdays, monthly, on="date")
    expanded["open"] = expanded["close"]
    expanded["high"] = expanded["close"]
    expanded["low"] = expanded["close"]
    expanded["volume"] = 0
    expanded["resolution"] = "monthly_interpolated"
    return expanded[["date", "open", "high", "low", "close", "volume", "resolution"]]


def make_synthetic_fallback(n_days: int = 3000) -> pd.DataFrame:
    """
    Letzter Notfall-Fallback, falls yfinance (und bei SPX zusaetzlich die
    Shiller-Quelle) nicht erreichbar sind (z.B. komplett offline). Erzeugt
    einen plausiblen synthetischen Kursverlauf, nur damit das Dashboard sofort
    etwas anzeigt. Sobald ein echter Fetch klappt, wird diese Datei automatisch
    ersetzt.
    """
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)

    daily_ret = rng.normal(loc=0.0003, scale=0.011, size=n_days)
    close = 100 * np.exp(np.cumsum(daily_ret))

    high = close * (1 + np.abs(rng.normal(0, 0.004, n_days)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n_days)))
    open_ = low + (high - low) * rng.random(n_days)
    volume = rng.integers(1_000_000, 5_000_000, n_days)

    return pd.DataFrame({
        "date": dates, "open": open_, "high": high, "low": low,
        "close": close, "volume": volume, "resolution": "synthetic",
    })


def fetch_and_save(instr: InstrumentConfig, out_file: Path | None = None) -> pd.DataFrame:
    """Holt/kombiniert die Historie fuer EIN Instrument und speichert sie unter
    data/{instr.key}_daily.parquet (oder out_file, falls angegeben). Gibt das
    gespeicherte DataFrame zurueck."""
    out_file = out_file or data_file(instr, "daily", DATA_DIR)
    print(f"\n=== {instr.label} ({instr.key}) ===")

    daily_df = pd.DataFrame()
    try:
        daily_df = fetch_via_yfinance(instr)
        print(f"[ok] {len(daily_df)} echte Tagesdaten via yfinance geladen "
              f"(ab {daily_df['date'].min().date()}).")
    except Exception as e:
        print(f"[warnung] yfinance-Abruf fuer '{instr.price_ticker}' fehlgeschlagen ({e}).")

    monthly_df = pd.DataFrame()
    if instr.has_shiller_history:
        try:
            monthly_df = fetch_shiller_monthly()
            print(f"[ok] {len(monthly_df)} Monatswerte via Shiller-Datensatz geladen "
                  f"(ab {monthly_df['date'].min().date()}).")
        except Exception as e:
            print(f"[warnung] Shiller-Abruf fehlgeschlagen ({e}).")

    if daily_df.empty and monthly_df.empty:
        print("[warnung] Keine Quelle erreichbar (vermutlich offline).")
        print("[info] Erzeuge stattdessen synthetische Platzhalterdaten, "
              "damit das Dashboard sofort lauffaehig ist.")
        df = make_synthetic_fallback()

    elif daily_df.empty:
        # keine Tagesdaten erreichbar -> komplette Reihe bleibt monatlich (nur SPX moeglich)
        df = monthly_df.copy()
        df["open"] = df["close"]
        df["high"] = df["close"]
        df["low"] = df["close"]
        df["volume"] = 0
        df["resolution"] = "monthly_interpolated"
        df = df[["date", "open", "high", "low", "close", "volume", "resolution"]]

    elif monthly_df.empty:
        df = daily_df

    else:
        cutoff = daily_df["date"].min()
        early_part = expand_monthly_to_bdays(monthly_df, until=cutoff)
        df = pd.concat([early_part, daily_df], ignore_index=True)
        df = df.sort_values("date").drop_duplicates(subset="date", keep="last")
        df = df.reset_index(drop=True)

    df.to_parquet(out_file, index=False)
    n_daily = (df["resolution"] == "daily").sum()
    n_monthly = (df["resolution"] == "monthly_interpolated").sum()
    print(f"[ok] gespeichert unter: {out_file}")
    print(f"     Zeitraum: {df['date'].min().date()} bis {df['date'].max().date()}")
    print(f"     davon {n_daily} echte Tagesdaten, {n_monthly} monatlich interpolierte Tage")
    return df


def main():
    keys = sys.argv[1:] or list(INSTRUMENTS.keys())
    unknown = [k for k in keys if k not in INSTRUMENTS]
    if unknown:
        print(f"[fehler] unbekannte Instrument-Schluessel: {unknown}. "
              f"Verfuegbar: {sorted(INSTRUMENTS)}")
        return 1

    for key in keys:
        fetch_and_save(INSTRUMENTS[key])
    return 0


if __name__ == "__main__":
    sys.exit(main())