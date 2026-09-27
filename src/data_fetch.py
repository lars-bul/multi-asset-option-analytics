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
Fuer SPX und Gold gibt es eine zusaetzliche, sehr lange Historie ueber die
echten yfinance-Tagesdaten hinaus (siehe instruments.py, has_shiller_history/
long_history_label und MONTHLY_HISTORY_FETCHERS unten):
  - SPX:  echte taegliche OHLC-Daten (yfinance, ^GSPC) gibt es realistisch
          nicht vor Ende 1927 (Yahoo Finance beginnt am 1927-12-30) - das ist
          eine harte Grenze. Robert Shillers monatliche S&P-Composite-Reihe
          (github.com/datasets/s-and-p-500) deckt zusaetzlich 1871-1927 ab,
          in DERSELBEN Einheit (S&P-Indexpunkte) wie ^GSPC - daher OHNE
          Skalierung direkt anschliessbar.
  - Gold: echte taegliche OHLC-Daten (yfinance, GLD) gibt es erst ab Auflage
          des ETF (2004). Eine monatliche Weltbank-/Timothy-Green-Goldpreis-
          reihe (github.com/datasets/gold-prices) deckt zusaetzlich 1833-2004
          ab - ABER in absoluten USD/Feinunze, NICHT auf GLDs Skala (GLD
          bildet historisch ca. 1/10 Feinunze ab, nicht exakt konstant ueber
          die Jahrzehnte). scale_monthly_to_daily() skaliert die Monatsreihe
          deshalb EINMALIG beim Aufbau auf den echten GLD-Kurs am ersten
          echten GLD-Handelstag, bevor sie an die Tagesdaten angehaengt wird -
          sonst entstuende an der Nahtstelle 2004 ein kuenstlicher ~10x-
          Sprung (dieselbe Fehlerklasse, die als Live-Bug schon einmal bei
          Russell 2000/IWM aufgetreten ist, siehe dashboard_livechain.py).

FUER ALLE ANDEREN INSTRUMENTE (Russell 2000, Nasdaq-100, EuroStoxx50, Oel)
gibt es KEINE solche Fruehgeschichte - hier beginnt die Historie schlicht
dort, wo yfinance echte Daten liefert (siehe instruments.py fuer die grobe
Erwartung je Instrument). Insbesondere bei Oel (USO) ist zu beachten: das ist
EIN durchgehend gehandelter ETF mit moeglichen Rollover-Spruengen im
zugrunde liegenden Futures-Kontrakt, keine ueber Jahrzehnte "geglaettete"
Reihe wie ein Aktienindex.

Jede Zeile bekommt eine Spalte "resolution" ("daily" oder
"monthly_interpolated"), damit nachgelagerter Code (z.B. Realized-Vol-
Berechnung fuers ML-Modell) weiss, welche Tage echte Tagesbewegungen sind und
welche nur grob interpoliert Platzhalter aus einem Monatswert.
"""

from pathlib import Path
import sys
from typing import Callable

import pandas as pd
import numpy as np

from instruments import INSTRUMENTS, InstrumentConfig, data_file

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

SHILLER_CSV_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"

# github.com/datasets/gold-prices: monatliche Goldpreise (USD/Feinunze) 1833-heute,
# 1833-1959 aus Timothy Greens historischer Tabelle, ab 1960 von der Weltbank
# ("Pink Sheet"). Der Standard-Branch dieses Repos war zum Zeitpunkt der
# Implementierung nicht mit letzter Sicherheit bekannt (Repos wandern von
# 'master' zu 'main') - beide werden deshalb der Reihe nach versucht.
GOLD_MONTHLY_CSV_URLS = [
    "https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly-processed.csv",
    "https://raw.githubusercontent.com/datasets/gold-prices/master/data/monthly-processed.csv",
]


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
    Monat - keine echte Tagesspanne. Werte sind ECHTE S&P-500-Indexpunkte,
    identische Skala wie ^GSPC - siehe MONTHLY_HISTORY_FETCHERS (needs_scaling=False)."""
    df = pd.read_csv(SHILLER_CSV_URL, usecols=["Date", "SP500"])
    df = df.rename(columns={"Date": "date", "SP500": "close"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    return df.sort_values("date").reset_index(drop=True)


def fetch_gold_monthly() -> pd.DataFrame:
    """Monatliche Goldpreisreihe seit 1833 (github.com/datasets/gold-prices,
    Timothy Green 1833-1959 + Weltbank Commodity Markets ab 1960), in
    absoluten USD/Feinunze. NICHT auf GLD-Skala - siehe
    MONTHLY_HISTORY_FETCHERS (needs_scaling=True) und scale_monthly_to_daily().
    Probiert mehrere Branch-Namen (main/master), da der Standard-Branch des
    Quell-Repos sich im Lauf der Zeit geaendert haben kann."""
    last_err: Exception | None = None
    for url in GOLD_MONTHLY_CSV_URLS:
        try:
            df = pd.read_csv(url, usecols=["Date", "Price"])
            df = df.rename(columns={"Date": "date", "Price": "close"})
            df["date"] = pd.to_datetime(df["date"])
            df = df.dropna(subset=["close"])
            df = df[df["close"] > 0]
            return df.sort_values("date").reset_index(drop=True)
        except Exception as e:
            last_err = e
    raise RuntimeError(
        "Monatliche Goldpreise (github.com/datasets/gold-prices) unter keiner der "
        f"bekannten Branch-URLs abrufbar: {last_err}"
    )


# Registry: welches Instrument bekommt eine monatliche Fruehgeschichte, ueber
# welche Funktion, und muss diese vor dem Anhaengen auf die Skala der echten
# Tagesdaten umgerechnet werden? (True bei Gold: GLD-ETF-Kurs != absoluter
# USD/oz-Goldpreis. False bei SPX: ^GSPC-Punkte == Shiller-S&P500-Punkte,
# direkt anschliessbar.) Ein neues Instrument mit eigener Fruehgeschichte wird
# hier an EINER Stelle ergaenzt, siehe auch instruments.py has_shiller_history.
MONTHLY_HISTORY_FETCHERS: dict[str, tuple[Callable[[], pd.DataFrame], bool]] = {
    "spx": (fetch_shiller_monthly, False),
    "gold": (fetch_gold_monthly, True),
}


def scale_monthly_to_daily(monthly_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """Skaliert eine monatliche Preisreihe (absolute Einheit, z.B. USD/Feinunze
    Gold) auf die Skala der ECHTEN Tagesdaten (z.B. GLD-ETF-Kurs) - noetig,
    wenn beide Reihen NICHT dieselbe Grundeinheit haben (siehe
    MONTHLY_HISTORY_FETCHERS). Ein einfaches Aneinanderhaengen ohne Skalierung
    wuerde an der Nahtstelle einen kuenstlichen Preis-Sprung erzeugen - exakt
    die Fehlerklasse, die als Live-Bug schon einmal bei Russell 2000/IWM
    (unterschiedliche Skala von price_ticker und chain_ticker) aufgetreten ist.

    Skalierungsfaktor = echter erster Tagesschlusskurs / monatlicher
    Referenzpreis IM SELBEN Kalendermonat (oder, falls kein Monatswert exakt
    in diesem Monat liegt, der naechstliegende Monatswert VOR dem ersten
    Tagesdatum). Gibt eine KOPIE von monthly_df mit skalierter 'close'-Spalte
    zurueck. Wirft ValueError, falls sich keine gueltige Referenz finden laesst
    (dann bleibt die Fruehgeschichte fuer dieses Instrument besser weg, statt
    mit einer geratenen Skalierung falsche Daten zu erzeugen)."""
    if monthly_df.empty or daily_df.empty:
        raise ValueError("Monatliche oder taegliche Reihe ist leer - keine Skalierung moeglich.")

    first_daily_date = pd.Timestamp(daily_df["date"].min())
    first_daily_close = float(daily_df.loc[daily_df["date"] == first_daily_date, "close"].iloc[0])
    if not np.isfinite(first_daily_close) or first_daily_close <= 0:
        raise ValueError(f"Erster echter Tagesschlusskurs ist ungueltig ({first_daily_close}).")

    same_month = monthly_df[
        (monthly_df["date"].dt.year == first_daily_date.year)
        & (monthly_df["date"].dt.month == first_daily_date.month)
    ]
    if same_month.empty:
        before = monthly_df[monthly_df["date"] < first_daily_date]
        if before.empty:
            raise ValueError(
                "Keine monatliche Referenz vor/im selben Monat wie der erste "
                f"echte Tagesschlusskurs ({first_daily_date.date()}) gefunden."
            )
        same_month = before.tail(1)

    ref_price = float(same_month["close"].iloc[-1])
    if not np.isfinite(ref_price) or ref_price <= 0:
        raise ValueError(f"Ungueltiger monatlicher Referenzpreis ({ref_price}) fuer die Skalierung.")

    ratio = first_daily_close / ref_price
    out = monthly_df.copy()
    out["close"] = out["close"] * ratio
    return out


def expand_monthly_to_bdays(monthly: pd.DataFrame, until: pd.Timestamp) -> pd.DataFrame:
    """Vorwaertsfuellen der monatlichen Werte auf Handelstage, bis zu dem
    Datum, ab dem echte Tagesdaten uebernehmen ('until', exklusiv). Erwartet
    monthly['close'] bereits auf der ZIELSKALA der echten Tagesdaten (bei
    Instrumenten mit needs_scaling=True vorher scale_monthly_to_daily()
    aufrufen)."""
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
    Letzter Notfall-Fallback, falls yfinance (und ggf. die monatliche
    Fruehgeschichte) nicht erreichbar sind (z.B. komplett offline). Erzeugt
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
        entry = MONTHLY_HISTORY_FETCHERS.get(instr.key)
        if entry is None:
            print(f"[warnung] has_shiller_history=True fuer '{instr.key}', aber kein Eintrag "
                  "in MONTHLY_HISTORY_FETCHERS registriert - ueberspringe Fruehgeschichte.")
        else:
            fetch_fn, needs_scaling = entry
            try:
                monthly_df = fetch_fn()
                print(f"[ok] {len(monthly_df)} Monatswerte geladen "
                      f"(ab {monthly_df['date'].min().date()}, roh/unskaliert).")
                if needs_scaling and not daily_df.empty:
                    monthly_df = scale_monthly_to_daily(monthly_df, daily_df)
                    print(f"[ok] Monatsreihe auf {instr.price_ticker}-Skala umgerechnet "
                          f"(Referenz: erster echter Tagesschlusskurs am {daily_df['date'].min().date()}).")
                elif needs_scaling and daily_df.empty:
                    print("[warnung] Skalierung noetig, aber keine echten Tagesdaten vorhanden - "
                          "ueberspringe Fruehgeschichte (keine Referenz fuer die Skala).")
                    monthly_df = pd.DataFrame()
            except Exception as e:
                print(f"[warnung] Monatsdaten-Abruf/-Skalierung fuer '{instr.key}' fehlgeschlagen ({e}).")
                monthly_df = pd.DataFrame()

    if daily_df.empty and monthly_df.empty:
        print("[warnung] Keine Quelle erreichbar (vermutlich offline).")
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