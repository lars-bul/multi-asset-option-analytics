"""
Holt Kursdaten fuer ein Instrument (siehe instruments.py) und speichert sie
lokal als Parquet.

Ausfuehren mit:
    python src/data_fetch.py            # laedt ALLE Instrumente
    python src/data_fetch.py spx        # nur SPX
    python src/data_fetch.py gold rut   # mehrere gezielt

WICHTIG:
- Echte Tagesdaten via yfinance werden bevorzugt.
- SPX und Gold koennen zusaetzlich eine sehr lange monatliche Fruehgeschichte
  bekommen.
- Eine bereits vorhandene echte Daily-Historie wird NIEMALS durch
  monthly_interpolated oder synthetische Fallback-Daten ueberschrieben,
  wenn yfinance temporaer ausfaellt.
"""

from pathlib import Path
import sys
from typing import Callable

import pandas as pd
import numpy as np

from instruments import INSTRUMENTS, InstrumentConfig, data_file


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

SHILLER_CSV_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"
)

GOLD_MONTHLY_CSV_URLS = [
    "https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly-processed.csv",
    "https://raw.githubusercontent.com/datasets/gold-prices/master/data/monthly-processed.csv",
]


def fetch_via_yfinance(instr: InstrumentConfig) -> pd.DataFrame:
    """
    Laedt echte taegliche Kursdaten via yfinance.

    Der Yahoo-Abruf kann sporadisch mit "possibly delisted" bzw.
    leerem Ergebnis antworten, obwohl der Ticker verfuegbar ist.
    Deshalb werden mehrere Versuche mit threads=False durchgefuehrt.

    Erwartete Spalten:
        date, open, high, low, close, volume, resolution
    """

    import time
    import yfinance as yf

    last_error = None

    for attempt in range(1, 4):
        try:
            print(
                f"[info] yfinance Versuch {attempt}/3 "
                f"fuer '{instr.price_ticker}'..."
            )

            df = yf.download(
                instr.price_ticker,
                start="1900-01-01",
                auto_adjust=False,
                progress=False,
                threads=False,
                timeout=30,
            )

            if df.empty:
                raise RuntimeError(
                    f"yfinance hat keine Daten fuer "
                    f"'{instr.price_ticker}' geliefert."
                )

            # yfinance 1.x kann MultiIndex-Spalten liefern:
            #
            # Price         Adj Close        Close ...
            # Ticker            ^GSPC        ^GSPC ...
            #
            # Wir reduzieren diese auf einfache Spaltennamen.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()

            df = df.rename(
                columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )

            required = [
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]

            missing = [
                col
                for col in required
                if col not in df.columns
            ]

            if missing:
                raise RuntimeError(
                    f"yfinance-Daten fuer "
                    f"'{instr.price_ticker}' enthalten "
                    f"nicht die erwarteten Spalten: {missing}. "
                    f"Vorhandene Spalten: {list(df.columns)}"
                )

            df = df[required].copy()

            df["date"] = pd.to_datetime(
                df["date"],
                errors="coerce",
            )

            for col in [
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]:
                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce",
                )

            # Nur gueltige Zeilen behalten.
            df = df.dropna(
                subset=[
                    "date",
                    "close",
                ]
            )

            if df.empty:
                raise RuntimeError(
                    f"Nach der Bereinigung sind keine "
                    f"gueltigen Tagesdaten fuer "
                    f"'{instr.price_ticker}' uebrig."
                )

            df = (
                df.sort_values("date")
                .drop_duplicates(
                    subset="date",
                    keep="last",
                )
                .reset_index(drop=True)
            )

            df["resolution"] = "daily"

            result = df[
                [
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "resolution",
                ]
            ]

            print(
                f"[ok] yfinance erfolgreich: "
                f"{len(result)} Tageswerte geladen."
            )

            return result

        except Exception as e:
            last_error = e

            print(
                f"[warnung] yfinance Versuch "
                f"{attempt}/3 fehlgeschlagen: {e}"
            )

            if attempt < 3:
                print(
                    "[info] Neuer Versuch in 2 Sekunden..."
                )
                time.sleep(2)

    raise RuntimeError(
        f"yfinance konnte '{instr.price_ticker}' "
        f"nach 3 Versuchen nicht laden: {last_error}"
    )


def fetch_shiller_monthly() -> pd.DataFrame:
    """
    Monatliche S&P-Composite-Reihe seit 1871.

    Die Reihe liegt in derselben Indexpunkt-Skala wie ^GSPC vor und muss
    deshalb fuer SPX nicht skaliert werden.
    """

    df = pd.read_csv(
        SHILLER_CSV_URL,
        usecols=["Date", "SP500"],
    )

    df = df.rename(
        columns={
            "Date": "date",
            "SP500": "close",
        }
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "date",
            "close",
        ]
    )

    df = df[df["close"] > 0]

    return (
        df.sort_values("date")
        .drop_duplicates(subset="date", keep="last")
        .reset_index(drop=True)
    )


def fetch_gold_monthly() -> pd.DataFrame:
    """
    Monatliche Goldpreisreihe seit 1833.

    Die Quelle liefert absolute USD/Feinunze und ist deshalb nicht direkt
    auf die GLD-ETF-Skala uebertragbar. Die Skalierung erfolgt spaeter anhand
    des ersten echten GLD-Tageskurses.
    """

    last_err: Exception | None = None

    for url in GOLD_MONTHLY_CSV_URLS:
        try:
            df = pd.read_csv(
                url,
                usecols=["Date", "Price"],
            )

            df = df.rename(
                columns={
                    "Date": "date",
                    "Price": "close",
                }
            )

            df["date"] = pd.to_datetime(
                df["date"],
                errors="coerce",
            )

            df["close"] = pd.to_numeric(
                df["close"],
                errors="coerce",
            )

            df = df.dropna(
                subset=[
                    "date",
                    "close",
                ]
            )

            df = df[df["close"] > 0]

            return (
                df.sort_values("date")
                .drop_duplicates(
                    subset="date",
                    keep="last",
                )
                .reset_index(drop=True)
            )

        except Exception as e:
            last_err = e

    raise RuntimeError(
        "Monatliche Goldpreise konnten unter keiner der bekannten "
        f"URLs geladen werden: {last_err}"
    )


# Registry:
#
# key -> (Fetcher, muss auf Daily-Skala skaliert werden?)
#
# SPX:
#   Shiller und ^GSPC liegen auf derselben Indexpunkt-Skala.
#
# Gold:
#   historische Goldpreise sind USD/oz,
#   GLD ist ein ETF mit anderer Skala.
MONTHLY_HISTORY_FETCHERS: dict[
    str,
    tuple[Callable[[], pd.DataFrame], bool],
] = {
    "spx": (
        fetch_shiller_monthly,
        False,
    ),
    "gold": (
        fetch_gold_monthly,
        True,
    ),
}


def scale_monthly_to_daily(
    monthly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Skaliert eine monatliche Preisreihe auf die Skala der echten Tagesdaten.

    Wird aktuell insbesondere fuer Gold benoetigt.
    """

    if monthly_df.empty or daily_df.empty:
        raise ValueError(
            "Monatliche oder taegliche Reihe ist leer - "
            "keine Skalierung moeglich."
        )

    first_daily_date = pd.Timestamp(
        daily_df["date"].min()
    )

    first_daily_rows = daily_df[
        daily_df["date"] == first_daily_date
    ]

    if first_daily_rows.empty:
        raise ValueError(
            "Erster echter Tagesdatensatz konnte nicht gefunden werden."
        )

    first_daily_close = float(
        first_daily_rows["close"].iloc[0]
    )

    if (
        not np.isfinite(first_daily_close)
        or first_daily_close <= 0
    ):
        raise ValueError(
            f"Erster echter Tagesschlusskurs ist ungueltig "
            f"({first_daily_close})."
        )

    same_month = monthly_df[
        (monthly_df["date"].dt.year == first_daily_date.year)
        & (
            monthly_df["date"].dt.month
            == first_daily_date.month
        )
    ]

    if same_month.empty:
        before = monthly_df[
            monthly_df["date"] < first_daily_date
        ]

        if before.empty:
            raise ValueError(
                "Keine monatliche Referenz vor/im selben Monat "
                f"wie der erste echte Tagesschlusskurs "
                f"({first_daily_date.date()}) gefunden."
            )

        same_month = before.tail(1)

    ref_price = float(
        same_month["close"].iloc[-1]
    )

    if (
        not np.isfinite(ref_price)
        or ref_price <= 0
    ):
        raise ValueError(
            f"Ungueltiger monatlicher Referenzpreis "
            f"({ref_price})."
        )

    ratio = first_daily_close / ref_price

    out = monthly_df.copy()

    out["close"] = out["close"] * ratio

    return out


def expand_monthly_to_bdays(
    monthly: pd.DataFrame,
    until: pd.Timestamp,
) -> pd.DataFrame:
    """
    Fuellt monatliche Werte auf Werktage auf, bis zum Beginn der echten
    Tagesdaten.

    Die monatlichen Werte bekommen resolution='monthly_interpolated'.
    """

    monthly = monthly[
        monthly["date"] < until
    ].copy()

    if monthly.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "resolution",
            ]
        )

    bdays = pd.DataFrame(
        {
            "date": pd.bdate_range(
                start=monthly["date"].min(),
                end=until - pd.Timedelta(days=1),
            )
        }
    )

    expanded = pd.merge_asof(
        bdays,
        monthly,
        on="date",
    )

    expanded["open"] = expanded["close"]
    expanded["high"] = expanded["close"]
    expanded["low"] = expanded["close"]
    expanded["volume"] = 0
    expanded["resolution"] = "monthly_interpolated"

    return expanded[
        [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "resolution",
        ]
    ]


def make_synthetic_fallback(
    n_days: int = 3000,
) -> pd.DataFrame:
    """
    Letzter Notfall-Fallback.

    Wird nur verwendet, wenn weder echte Tagesdaten noch monatliche
    historische Daten verfuegbar sind UND keine brauchbare vorhandene
    Datei existiert.
    """

    rng = np.random.default_rng(42)

    dates = pd.bdate_range(
        end=pd.Timestamp.today().normalize(),
        periods=n_days,
    )

    daily_ret = rng.normal(
        loc=0.0003,
        scale=0.011,
        size=n_days,
    )

    close = 100 * np.exp(
        np.cumsum(daily_ret)
    )

    high = close * (
        1 + np.abs(
            rng.normal(
                0,
                0.004,
                n_days,
            )
        )
    )

    low = close * (
        1 - np.abs(
            rng.normal(
                0,
                0.004,
                n_days,
            )
        )
    )

    open_ = (
        low
        + (high - low)
        * rng.random(n_days)
    )

    volume = rng.integers(
        1_000_000,
        5_000_000,
        n_days,
    )

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "resolution": "synthetic",
        }
    )


def read_existing_data(
    out_file: Path,
) -> pd.DataFrame:
    """
    Liest eine bereits vorhandene Parquet-Datei.

    Bei einem Lesefehler wird ein leerer DataFrame zurueckgegeben.
    """

    if not out_file.exists():
        return pd.DataFrame()

    try:
        df = pd.read_parquet(out_file)

        if df.empty:
            return pd.DataFrame()

        if "date" in df.columns:
            df["date"] = pd.to_datetime(
                df["date"],
                errors="coerce",
            )

        return df

    except Exception as e:
        print(
            f"[warnung] Vorhandene Datei konnte nicht gelesen werden: "
            f"{e}"
        )
        return pd.DataFrame()


def has_real_daily_data(
    df: pd.DataFrame,
) -> bool:
    """
    Prueft, ob ein DataFrame echte Daily-Daten enthaelt.
    """

    if df.empty:
        return False

    if "resolution" not in df.columns:
        return False

    return bool(
        (
            df["resolution"] == "daily"
        ).any()
    )


def fetch_and_save(
    instr: InstrumentConfig,
    out_file: Path | None = None,
) -> pd.DataFrame:
    """
    Holt/kombiniert die Historie fuer EIN Instrument.

    Sicherheitsregel:
    Wenn bereits echte Daily-Daten vorhanden sind und yfinance temporaer
    ausfaellt, wird die bestehende Datei NICHT mit Fallback-Daten
    ueberschrieben.
    """

    out_file = (
        out_file
        or data_file(
            instr,
            "daily",
            DATA_DIR,
        )
    )

    print(
        f"\n=== {instr.label} ({instr.key}) ==="
    )

    # ---------------------------------------------------------
    # Vorhandene Datei einlesen
    # ---------------------------------------------------------

    existing_df = read_existing_data(
        out_file
    )

    existing_has_daily = has_real_daily_data(
        existing_df
    )

    if existing_has_daily:
        existing_daily = existing_df[
            existing_df["resolution"] == "daily"
        ]

        print(
            f"[info] vorhandene Datei enthaelt "
            f"{len(existing_daily)} echte Tagesdaten "
            f"({existing_daily['date'].min().date()} "
            f"bis {existing_daily['date'].max().date()})."
        )

    # ---------------------------------------------------------
    # Echte Tagesdaten laden
    # ---------------------------------------------------------

    daily_df = pd.DataFrame()

    try:
        daily_df = fetch_via_yfinance(
            instr
        )

        print(
            f"[ok] {len(daily_df)} echte Tagesdaten "
            f"via yfinance geladen "
            f"(ab {daily_df['date'].min().date()} "
            f"bis {daily_df['date'].max().date()})."
        )

    except Exception as e:
        print(
            f"[warnung] yfinance-Abruf fuer "
            f"'{instr.price_ticker}' fehlgeschlagen: {e}"
        )

    # ---------------------------------------------------------
    # SICHERHEIT:
    # Wenn yfinance ausfaellt, vorhandene echte Daily-Historie
    # niemals mit Monats- oder Synthetic-Daten ueberschreiben.
    # ---------------------------------------------------------

    if daily_df.empty and existing_has_daily:
        print(
            "[schutz] yfinance liefert aktuell keine "
            "Tagesdaten."
        )

        print(
            "[schutz] Vorhandene echte Daily-Historie "
            "wird NICHT ueberschrieben."
        )

        return existing_df

    # ---------------------------------------------------------
    # Monatliche Fruehgeschichte
    # ---------------------------------------------------------

    monthly_df = pd.DataFrame()

    if instr.has_shiller_history:

        entry = MONTHLY_HISTORY_FETCHERS.get(
            instr.key
        )

        if entry is None:

            print(
                f"[warnung] has_shiller_history=True fuer "
                f"'{instr.key}', aber kein Eintrag in "
                f"MONTHLY_HISTORY_FETCHERS registriert."
            )

        else:

            fetch_fn, needs_scaling = entry

            try:

                monthly_df = fetch_fn()

                print(
                    f"[ok] {len(monthly_df)} Monatswerte geladen "
                    f"(ab {monthly_df['date'].min().date()}, "
                    f"roh/unskaliert)."
                )

                if (
                    needs_scaling
                    and not daily_df.empty
                ):

                    monthly_df = (
                        scale_monthly_to_daily(
                            monthly_df,
                            daily_df,
                        )
                    )

                    print(
                        f"[ok] Monatsreihe auf "
                        f"{instr.price_ticker}-Skala "
                        f"umgerechnet."
                    )

                elif (
                    needs_scaling
                    and daily_df.empty
                ):

                    print(
                        "[warnung] Skalierung noetig, aber "
                        "keine echten Tagesdaten vorhanden."
                    )

                    print(
                        "[warnung] Fruehgeschichte wird "
                        "deshalb uebersprungen."
                    )

                    monthly_df = (
                        pd.DataFrame()
                    )

            except Exception as e:

                print(
                    f"[warnung] Monatsdaten-Abruf/-Skalierung "
                    f"fuer '{instr.key}' fehlgeschlagen: {e}"
                )

                monthly_df = (
                    pd.DataFrame()
                )

    # ---------------------------------------------------------
    # Daten zusammenbauen
    # ---------------------------------------------------------

    if (
        daily_df.empty
        and monthly_df.empty
    ):

        # Dieser Fall sollte bei einer bestehenden echten
        # Daily-Datei wegen des Schutzes oben nicht mehr
        # erreicht werden.

        print(
            "[warnung] Keine Quelle erreichbar."
        )

        if not existing_df.empty:

            print(
                "[schutz] Es existiert eine vorhandene Datei. "
                "Diese wird NICHT durch synthetische Daten "
                "ersetzt."
            )

            return existing_df

        print(
            "[info] Erzeuge synthetische "
            "Platzhalterdaten."
        )

        df = make_synthetic_fallback()

    elif daily_df.empty:

        # Keine echten Tagesdaten, aber monatliche Historie
        # vorhanden.
        #
        # Dieser Fall ist jetzt nur noch relevant, wenn es keine
        # vorhandene echte Daily-Datei gibt.

        print(
            "[info] Keine echten Tagesdaten verfuegbar."
        )

        print(
            "[info] Verwende monatliche Historie."
        )

        df = monthly_df.copy()

        df["open"] = df["close"]
        df["high"] = df["close"]
        df["low"] = df["close"]
        df["volume"] = 0
        df["resolution"] = (
            "monthly_interpolated"
        )

        df = df[
            [
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "resolution",
            ]
        ]

    elif monthly_df.empty:

        # Normale Situation fuer Instrumente ohne
        # Fruehgeschichte.
        df = daily_df

    else:

        # Monatliche Fruehgeschichte vor den echten
        # Tagesdaten.

        cutoff = daily_df[
            "date"
        ].min()

        early_part = (
            expand_monthly_to_bdays(
                monthly_df,
                until=cutoff,
            )
        )

        df = pd.concat(
            [
                early_part,
                daily_df,
            ],
            ignore_index=True,
        )

        df = (
            df.sort_values("date")
            .drop_duplicates(
                subset="date",
                keep="last",
            )
            .reset_index(drop=True)
        )

    # ---------------------------------------------------------
    # Letzte Validierung
    # ---------------------------------------------------------

    if df.empty:
        raise RuntimeError(
            f"Fuer '{instr.key}' wurden keine Daten erzeugt."
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["date"]
    )

    df = (
        df.sort_values("date")
        .drop_duplicates(
            subset="date",
            keep="last",
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Speichern
    # ---------------------------------------------------------

    df.to_parquet(
        out_file,
        index=False,
    )

    n_daily = int(
        (
            df["resolution"] == "daily"
        ).sum()
    )

    n_monthly = int(
        (
            df["resolution"]
            == "monthly_interpolated"
        ).sum()
    )

    n_synthetic = int(
        (
            df["resolution"]
            == "synthetic"
        ).sum()
    )

    print(
        f"[ok] gespeichert unter: {out_file}"
    )

    print(
        f"     Zeitraum: "
        f"{df['date'].min().date()} "
        f"bis "
        f"{df['date'].max().date()}"
    )

    print(
        f"     davon {n_daily} echte Tagesdaten, "
        f"{n_monthly} monatlich interpolierte Tage, "
        f"{n_synthetic} synthetische Tage"
    )

    return df


def main():
    keys = (
        sys.argv[1:]
        or list(INSTRUMENTS.keys())
    )

    unknown = [
        k
        for k in keys
        if k not in INSTRUMENTS
    ]

    if unknown:

        print(
            f"[fehler] unbekannte "
            f"Instrument-Schluessel: {unknown}. "
            f"Verfuegbar: "
            f"{sorted(INSTRUMENTS)}"
        )

        return 1

    for key in keys:

        fetch_and_save(
            INSTRUMENTS[key]
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())