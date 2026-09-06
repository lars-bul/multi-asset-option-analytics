"""
Holt SPX-Kursdaten und speichert sie lokal als Parquet - so frueh wie
moeglich beginnend.

Einfach ausfuehren mit:
    python src/data_fetch.py

Kann jederzeit erneut laufen gelassen werden, um die lokale Historie zu
aktualisieren (ueberschreibt data/spx_daily.parquet komplett - das ist bei
dieser Datenmenge unproblematisch).

DATENAUFLOESUNG - WICHTIG FUER SPAETERE AUSWERTUNGEN:
Echte TAEGLICHE OHLC-Daten fuer den S&P gibt es realistisch nicht vor Ende
1927 (Yahoo Finance / ^GSPC beginnt am 1927-12-30) - das ist eine harte
Grenze, an der auch zusaetzliches Geld/mehr Anbieter nichts aendern, weil
diese Granularitaet aus der Zeit schlicht nicht ueberliefert ist.

Was es aber gibt: Robert Shillers monatliche S&P-Composite-Reihe, frei
verfuegbar, zurueck bis 1871 (siehe github.com/datasets/s-and-p-500). Das
deckt 1900, die Great Depression (1929-1932) und alles davor ab - nur eben
mit monatlicher statt taeglicher Aufloesung.

Dieses Skript kombiniert daher zwei Quellen zu EINER durchgehenden Reihe:
  - ab 1927-12-30: echte taegliche OHLC-Daten (yfinance)
  - 1871-01 bis 1927-12-29: monatliche Shiller-Werte, auf Handelstage
    vorwaertsgefuellt (open=high=low=close, da keine echte Tagesspanne
    existiert)

Jede Zeile bekommt eine Spalte "resolution" ("daily" oder
"monthly_interpolated"), damit nachgelagerter Code (z.B. Realized-Vol-
Berechnung fuers ML-Modell) weiss, welche Tage echte Tagesbewegungen sind
und welche nur grob interpoliert Platzhalter aus einem Monatswert.

KORREKTUR: die Funktion make_synthetic_fallback() war zuvor ZWEIMAL
definiert (identischer Name, leicht unterschiedlicher Koerper) - die zweite
Definition hat die erste stillschweigend ueberschrieben (reiner Totcode ohne
funktionale Auswirkung, aber Verwirrungspotenzial). Jetzt gibt es nur noch
eine Version.
"""

from pathlib import Path
import sys

import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
OUT_FILE = DATA_DIR / "spx_daily.parquet"

SHILLER_CSV_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"


def fetch_via_yfinance() -> pd.DataFrame:
    """Echte taegliche SPX-Historie via yfinance. Beginnt real erst 1927-12-30,
    auch wenn hier ein frueheres Startdatum angefragt wird."""
    import yfinance as yf

    df = yf.download("^GSPC", start="1900-01-01", auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError("yfinance hat keine Daten geliefert.")

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
    Monat - keine echte Tagesspanne."""
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
    Letzter Notfall-Fallback, falls WEDER yfinance NOCH die Shiller-Quelle
    erreichbar sind (z.B. komplett offline). Erzeugt einen plausiblen
    synthetischen Kursverlauf, nur damit das Dashboard sofort etwas anzeigt.
    Sobald ein echter Fetch klappt, wird diese Datei automatisch ersetzt.
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


def main():
    daily_df = pd.DataFrame()
    monthly_df = pd.DataFrame()

    try:
        daily_df = fetch_via_yfinance()
        print(f"[ok] {len(daily_df)} echte Tagesdaten via yfinance geladen "
              f"(ab {daily_df['date'].min().date()}).")
    except Exception as e:
        print(f"[warnung] yfinance-Abruf fehlgeschlagen ({e}).")

    try:
        monthly_df = fetch_shiller_monthly()
        print(f"[ok] {len(monthly_df)} Monatswerte via Shiller-Datensatz geladen "
              f"(ab {monthly_df['date'].min().date()}).")
    except Exception as e:
        print(f"[warnung] Shiller-Abruf fehlgeschlagen ({e}).")

    if daily_df.empty and monthly_df.empty:
        print("[warnung] Keine der beiden echten Quellen erreichbar (vermutlich offline).")
        print("[info] Erzeuge stattdessen synthetische Platzhalterdaten, "
              "damit das Dashboard sofort lauffaehig ist.")
        df = make_synthetic_fallback()

    elif daily_df.empty:
        # keine Tagesdaten erreichbar -> komplette Reihe bleibt monatlich
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

    df.to_parquet(OUT_FILE, index=False)
    n_daily = (df["resolution"] == "daily").sum()
    n_monthly = (df["resolution"] == "monthly_interpolated").sum()
    print(f"[ok] gespeichert unter: {OUT_FILE}")
    print(f"     Zeitraum: {df['date'].min().date()} bis {df['date'].max().date()}")
    print(f"     davon {n_daily} echte Tagesdaten, {n_monthly} monatlich interpolierte Tage")


if __name__ == "__main__":
    sys.exit(main())