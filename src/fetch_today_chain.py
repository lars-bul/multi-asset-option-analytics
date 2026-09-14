"""
Holt die AKTUELLE (heutige) SPX-Optionskette - Bid/Ask/Mid je Strike/Expiry,
gefiltert auf DTE zwischen 0 und 90 Tagen. Keine Historie, nur der heutige
Snapshot (Yahoo-Daten sind ca. 15 Minuten verzoegert, aber kostenlos und
ohne Account nutzbar).

Nutzung:
    pip install yfinance --break-system-packages
    python fetch_today_chain.py

Ausgabe: eine CSV-Datei 'spx_chain_<heutiges_datum>.csv' mit Spalten
date, expiry, dte, strike, option_type, bid, ask, mid, last, volume,
open_interest, implied_vol (von Yahoo, falls vorhanden - kann von eurer
eigenen black76.py-Berechnung abweichen, dient nur als Referenz).

HINWEISE / GRENZEN dieses kostenlosen Wegs:
- Nur "heute" - Yahoo speichert keine Bid/Ask-Historie, das laesst sich
  damit nicht rueckwirkend abrufen.
- 15 Minuten Verzoegerung, keine Garantie auf Vollstaendigkeit aller
  Strikes/Serien (v.a. sehr illiquide, weit OTM Serien koennen bid=ask=0
  zeigen).
- Yahoo fasst SPX und SPXW (wöchentliche/0DTE-Serien) unter demselben
  Ticker "^SPX" zusammen - dieses Skript unterscheidet die Roots NICHT
  extra, sondern nimmt einfach alle von Yahoo gelisteten Expiries im
  gewuenschten DTE-Fenster.
"""
import sys
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

TICKER = "^SPX"
MIN_DTE = 0.0
MAX_DTE = 90.0
ET = ZoneInfo("America/New_York")


def _now_et() -> datetime:
    return datetime.now(ET)


def dte_exact(expiry: pd.Timestamp, now_et: datetime | None = None) -> float:
    """Praezise Resttage bis zum SPX-Verfall (fraktional, z.B. 5.5 statt 5) -
    Restlaufzeit spielt bei Black-76 v.a. bei kurzen DTEs (0DTE!) eine
    spuerbare Rolle, ganze Kalendertage waeren hier zu ungenau.

    ANNAHME: Settlement um 16:00 ET (PM-Settlement) - trifft exakt auf die
    SPXW-Wochenserien/0DTE-Serien zu, die den Grossteil des DTE-0-90-Fensters
    ausmachen. Die klassischen monatlichen AM-settled Serien (dritter Freitag,
    Ticker-Root reines 'SPX') verfallen tatsaechlich schon zum Handelsstart
    (~6.5h frueher) - fuer diese Serien ist dte_exact folglich leicht ZU HOCH.
    Yahoo unterscheidet die Roots in den zurueckgegebenen Expiries nicht
    explizit, daher wird hier bewusst einheitlich PM-Settlement angenommen
    und diese Ungenauigkeit fuer die Monats-Serien in Kauf genommen (im
    Dashboard als Hinweis vermerkt)."""
    if now_et is None:
        now_et = _now_et()
    settle = datetime(expiry.year, expiry.month, expiry.day, 16, 0, 0, tzinfo=ET)
    return (settle - now_et).total_seconds() / 86400.0


def fetch_today_spot(ticker: str = TICKER) -> float:
    """Aktueller (15 Min. verzoegerter) Spot-Preis - wird fuer die
    Forward-Umrechnung und damit fuer Delta/IV je Serie benoetigt
    (siehe chain_delta.enrich_chain_with_delta)."""
    tk = yf.Ticker(ticker)
    fast = tk.fast_info
    price = fast.get("lastPrice") or fast.get("last_price")
    if price is None:
        raise RuntimeError(f"Konnte aktuellen Spot-Preis fuer {ticker} nicht ermitteln.")
    return float(price)


def fetch_today_open(ticker: str = TICKER) -> float | None:
    """Heutiger Eroeffnungskurs, FALLS die Sitzung bereits eroeffnet hat -
    wird fuer den synthetischen 'Heute'-Tagesbalken in der Analog-Tage-
    Simulation benoetigt (siehe dashboard_livechain._build_daily_df_with_today).

    HINTERGRUND: daily_df enthaelt nur ABGESCHLOSSENE Handelstage. Ohne
    diesen Eroeffnungskurs wuerde eine Abfrage waehrend eines laufenden
    Handelstags fuer 'heute' auf den letzten abgeschlossenen Tag zurueckfallen
    - montags waere das der Freitag (3 Tage alt). Der Eroeffnungskurs
    schliesst diese Luecke ohne auf den EOD-Abschluss warten zu muessen.

    Gibt None zurueck, wenn kein Eroeffnungskurs verfuegbar ist (vor
    Handelsbeginn, Wochenende, Feiertag) - Aufrufer soll dann auf den
    letzten echten Schlusskurs zurueckfallen (mit Warnhinweis)."""
    tk = yf.Ticker(ticker)
    fast = tk.fast_info
    open_price = fast.get("open") or fast.get("regularMarketOpen")
    if open_price is None or open_price <= 0:
        return None
    return float(open_price)


def fetch_today_chain(ticker: str = TICKER, min_dte: float = MIN_DTE, max_dte: float = MAX_DTE) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    now_et = _now_et()
    today = pd.Timestamp(date.today())

    try:
        expiries = tk.options
    except Exception as e:
        raise RuntimeError(
            f"Konnte Expiries fuer {ticker} nicht laden - Yahoo-Endpoint nicht "
            f"erreichbar oder Ticker falsch? Fehler: {e}"
        )

    if not expiries:
        raise RuntimeError(f"Keine Expiries fuer {ticker} gefunden.")

    rows = []
    for exp_str in expiries:
        exp = pd.Timestamp(exp_str)
        dte_precise = dte_exact(exp, now_et)
        if dte_precise < min_dte or dte_precise > max_dte:
            continue

        try:
            chain = tk.option_chain(exp_str)
        except Exception as e:
            print(f"  Warnung: Expiry {exp_str} uebersprungen ({e})", file=sys.stderr)
            continue

        for option_type, df_leg in (("call", chain.calls), ("put", chain.puts)):
            if df_leg.empty:
                continue
            sub = df_leg.copy()
            sub["date"] = today
            sub["expiry"] = exp
            sub["dte"] = round(dte_precise)
            sub["dte_exact"] = dte_precise
            sub["option_type"] = option_type
            sub["mid"] = (sub["bid"] + sub["ask"]) / 2.0
            rows.append(sub[[
                "date", "expiry", "dte", "dte_exact", "strike", "option_type",
                "bid", "ask", "mid", "lastPrice", "volume", "openInterest", "impliedVolatility",
            ]].rename(columns={
                "lastPrice": "last", "openInterest": "open_interest", "impliedVolatility": "implied_vol",
            }))

    if not rows:
        raise RuntimeError(
            f"Keine Serien im DTE-Fenster [{min_dte}, {max_dte}] gefunden - "
            "Fenster erweitern oder Handelstag pruefen (Wochenende/Feiertag?)."
        )

    return pd.concat(rows, ignore_index=True).sort_values(["expiry", "option_type", "strike"]).reset_index(drop=True)


def compare_to_model(chain: pd.DataFrame, spot: float, r_free: float, dividend_yield: float,
                      current_iv: float, vol_surface) -> pd.DataFrame:
    """SYSTEMATISCHER Abgleich: fuer JEDE Serie der heutigen Kette wird der
    theoretische Preis aus eurer eigenen black76.leg_price() berechnet und
    der Differenz zum echten Markt-Mid gegenuebergestellt. Das ist der
    eigentliche Zweck - nicht nur Daten anschauen, sondern automatisiert
    pruefen, wie gut euer HV-/Skew-Modell (VolSurfaceConfig) die reale
    heutige Kette trifft, als Programm-Output (DataFrame), nicht als
    manueller Blick auf eine Webseite."""
    from black76 import forward_price, leg_price

    rows = chain.copy()
    rows["model_price"] = rows.apply(
        lambda r: leg_price(
            r["strike"], r["option_type"] == "put", r["dte"] / 365.0,
            forward_price(spot, r_free, dividend_yield, r["dte"] / 365.0),
            current_iv, r_free, vol_surface,
        ),
        axis=1,
    )
    rows["model_minus_mid"] = rows["model_price"] - rows["mid"]
    rows["model_minus_mid_pct"] = rows["model_minus_mid"] / rows["mid"].replace(0, pd.NA)
    return rows


if __name__ == "__main__":
    chain = fetch_today_chain()
    print(f"{len(chain)} Serien in der heutigen Kette (DTE {MIN_DTE}-{MAX_DTE}).")
    print(chain.head(10))

    # Beispiel-Abgleich gegen euer eigenes Modell (current_iv hier grob per
    # HV(30) ersetzen - siehe backtest.compute_hv() fuer den echten Wert aus
    # euren Kursdaten):
    #
    # from black76 import VolSurfaceConfig
    # spot = chain.iloc[0]["strike"]  # besser: echten Spot separat holen
    # checked = compare_to_model(chain, spot, r_free=0.0, dividend_yield=0.015,
    #                             current_iv=0.13, vol_surface=VolSurfaceConfig())
    # print(checked[["expiry", "strike", "option_type", "mid", "model_price", "model_minus_mid_pct"]])