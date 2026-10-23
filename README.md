# Multi-Asset Options Dashboard - Decision Support for Derivative Traders

A **local-only** quantitative Decision Support dashboard system for Derivative Traders, trading index and commodity options, built with Python and Streamlit. It combines a model-based iron condor backtester, an "analog day" matching engine (find historically similar market regimes and replay them against *today's real option quotes*), a live option-chain scanner, a portfolio analyzer and a cross-instrument pair-trade search - all with autocorrelation-robust statistics instead of naive averages.

> **Disclaimer:** This is a research and education project. It does not place orders, does not connect to a broker for execution, and nothing here is investment advice or a promise of profitability. See [Limitations](#limitations-and-honest-caveats).

---

## What it does

The app has a persistent price overview (candlestick chart for the selected instrument) and five analysis modes.

| Mode | What it does |
|---|---|
| **Iron Condor Backtest** | Daily overlapping ("ladder") multi-leg backtest. Each of 4 legs is freely configurable (put/call, long/short, target delta *or* fixed strike, quantity). Black-76 pricing with optional skew/term-structure vol surface, historical-volatility IV source, optional "fixed IV rank" mode, stop-on-touch and early exit. Reports win rate, profit factor, expectation, Newey-West t-stat, 95% CI and sub-period consistency. |
| **Analog Day Matching** | Builds a PCA-whitened similarity space from ~40 price-derived features, finds all historical days within a distance threshold of a chosen day, re-prices a put/call leg pair on those days at a fixed IV rank, and screens short vs. long strangles with a chronological train/validation split. |
| **Live Chain** | Loads today's option chain, validates quotes, and builds a symmetric distance table (strikes at x% from spot). For every row it replays the *percentage moves* of kernel-weighted analog days against today's real strikes and bid prices. On top: a model-free **iron condor search**, a **reverse iron condor** search, and a **best-trades-across-all-expiries** ranking, including a "limit price" (the minimum credit / maximum debit at which the criteria still hold). One-click transfer of a trade into the portfolio. |
| **Portfolio Analysis** | Store your own 1-4 leg option combos (persisted in `data/depot.json`, atomic writes with `.bak` backup, corrupt-file protection). Evaluates them with the same analog-day statistics, shows exit levels today (Black-76, sticky strike) and break-evens at expiry, plus P&L curves. |
| **Pair Trade** | Credit spread on a *lead* instrument vs. debit spread on a *follower* instrument, evaluated on the same calendar days. Exact structural max-loss (separable piecewise-linear payoff) rather than "worst observed day". |

### Supported instruments

S&P 500, Russell 2000 (price: `^RUT`, chain: IWM ETF), Nasdaq-100, Nikkei 225, EuroStoxx 50, Gold (via GLD), WTI crude (via USO).

| Capability | SPX, NDX, RUT, Gold, Oil | Nikkei, EuroStoxx 50 |
|---|---|---|
| Backtest, analog matching, ML pipeline | yes | yes |
| Live chain, portfolio live valuation, pair trade | yes (Yahoo Finance, delayed) | no free data source configured* |

\* An Interactive Brokers provider for Eurex (`ib_async`) is implemented but **disabled by default and untested** - it requires a running TWS/Gateway and a paid market-data subscription.

---

## Technical highlights

- **Robust statistics.** Analog days that are close in time have overlapping payoffs. The code uses Bartlett-kernel HAC standard errors on the calendar time axis, Kish effective sample size and a design-effect-adjusted `n_eff`, t-distribution CIs, and sub-period sign consistency - plus a train/validation split to guard against multiple-comparison hits.
- **Model-free scoring on real quotes.** The live scanner uses today's actual bid/ask, projects historical % moves onto today's spot, and applies a configurable safety haircut to all sold legs. No volatility model is involved in that path.
- **Similarity search.** StandardScaler + PCA (whitened, components chosen for 80% explained variance), hard cutoff or Gaussian kernel weighting with a minimum-candidates guarantee, causality filters (no analog whose exit lies after the query date), minimum gap in trading days, and a clustering diagnostic.
- **Quote plausibility filter.** Provider-independent checks: crossed/missing bid-ask, excessive spread, put-call-parity violations (implied forward vs. median), and smile outliers.
- **Pricing.** Black-76 with explicit spot-to-forward conversion, a delta/skew-consistent strike solver (fixed-point iteration), implied-vol and "package IV" solvers on real chain data.
- **Multi-instrument architecture.** A single `instruments.py` config drives data paths, tickers, multipliers and chain providers. Chain providers use a registry pattern (`register_provider`) so new data sources plug in without touching core modules.
- **Feature engineering without look-ahead.** Multi-window HV, HV term structure, HV rank, vol-of-vol, Parkinson / Garman-Klass vol, MA distance/slope, return horizons up to 5y, rolling autocorrelation, drawdown, skew/kurtosis, days-since-large-move.
- **ML pipeline (optional).** Walk-forward training of LightGBM / Random Forest / logistic regression ensembles on strangle outcome classes, out-of-sample evaluation with Newey-West reliability per predicted class, and SHAP + PCA feature analysis.
- **Data depth.** Daily history from Yahoo Finance, extended with monthly early history for SPX (Shiller series, from 1871) and Gold (from 1833, rescaled to the GLD level). Early monthly data is flagged (`resolution = monthly_interpolated`) and excluded from backtests.
- **Hedge search backend.** `hedge_search.py` implements scenario-based hedge candidate generation, CVaR/delta/gamma-aware quantity optimization and greedy hedge combination (backend only - see status below).

---

## Quick start (step by step)

### 1. Prerequisites

- **Python 3.10 or newer** (the code uses modern type-hint syntax).
- Internet connection (price history and option chains come from Yahoo Finance).
- Windows users: if you get a time-zone error at startup, run `pip install tzdata`.

### 2. Get the code and install

```bash
git clone <your-repo-url>
cd <repo-folder>

python -m venv venv

# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Download price history

```bash
python src/data_fetch.py spx            # just the S&P 500
python src/data_fetch.py spx gold rut   # several at once
python src/data_fetch.py                # all instruments
```

Valid keys: `spx`, `rut`, `ndx`, `nikkei`, `sx5e`, `gold`, `oil`. Files are written to `data/<key>_daily.parquet`. If Yahoo is temporarily unavailable, an existing real daily history is never overwritten by fallback data.

### 4. Build the training table (needed for everything except the plain backtest)

```bash
python src/build_dataset.py spx
```

This computes features and strangle outcomes for expiries of 1-90 days and writes `data/<key>_training_table.parquet`. It uses multiple CPU cores and can take several minutes. You can also trigger it from inside the dashboard (button in the "Live Chain" tab).

### 5. Start the dashboard

```bash
streamlit run src/dashboard.py
```

Your browser opens `http://localhost:8501`. Pick an instrument at the top, then a mode.

**Tips for first use**
- The app opens in **Live Chain** mode, which loads the chain automatically. Outside US market hours Yahoo often returns bid/ask = 0 - choose price basis **"Mid (approximation)"** in the condor search section.
- If you only want to try the backtest, step 3 is enough.

### Optional: one-command launcher

The repository contains `run.py` / `start.bat` / `start.sh`, intended to create the virtual environment, install packages, refresh data and start the app. These predate the multi-instrument rework, so the data-refresh step may only cover SPX; if in doubt, use the manual steps above.

### Optional: ML training and evaluation

```bash
python src/train.py               # walk-forward training, saves models/vintage_*/
python src/evaluate.py            # out-of-sample reliability per predicted class
python src/feature_importance.py  # SHAP + PCA analysis
```

---

## Project structure

```
├── data/                      # parquet files, depot.json (created at runtime)
├── models/                    # trained ML vintages and reports
├── requirements.txt
└── src/
    ├── dashboard.py           # entry point: header, instrument selector, mode router
    ├── dashboard_common.py    # data loading, candlestick overview, UI helpers
    ├── dashboard_ladder.py    # mode 1: iron condor / multi-leg backtest
    ├── dashboard_analog.py    # mode 2: analog day matching
    ├── dashboard_livechain.py # mode 3: live chain, condor / reverse condor / all-expiry search
    ├── dashboard_depot.py     # mode 4: portfolio analysis
    ├── dashboard_pair.py      # mode 5: pair trade
    ├── instruments.py         # central instrument configuration
    ├── data_fetch.py          # price history download (+ monthly early history)
    ├── features.py            # leak-free feature engineering
    ├── labels.py / build_dataset.py   # strangle labels and training table
    ├── black76.py / backtest.py       # pricing, vol surface, ladder backtest
    ├── analog_matching.py     # PCA similarity space, analog search, HAC statistics
    ├── condor_search.py       # model-free condor / reverse condor grid search
    ├── pair_trade_search.py   # lead/follower pair evaluation
    ├── hedge_search.py        # hedge finder (backend)
    ├── chain_delta.py         # implied vol / delta from real quotes
    ├── quote_filter.py        # quote plausibility checks
    ├── market_data.py         # shared per-instrument market snapshot
    ├── depot_engine.py        # portfolio data model and persistence
    ├── chain_providers/       # Yahoo Finance + Interactive Brokers providers, registry
    ├── stats.py               # Newey-West / sub-period reliability
    └── train.py / evaluate.py / feature_importance.py   # optional ML pipeline
```

---

## Limitations and honest caveats

- **No historical option prices.** The backtest and Analog mode price options with Black-76 on a historical-volatility proxy, not with real historical quotes. Results show how a *modelled* structure would have behaved, not what a real fill would have been. Only the Live Chain / Portfolio / Pair modes use real quotes - and only today's.
- **Hold-to-expiry, simplified risk.** Live-chain statistics value positions at expiry. Path dependence, vega, early assignment and margin are not modelled. Costs are approximated by a configurable haircut, not real commissions or slippage.
- **Data quality.** Yahoo Finance data is delayed (~15 min), can have stale or zero quotes and occasionally fails. ETF proxies (IWM, GLD, USO) do not track their underlyings exactly. Early history for SPX/Gold is monthly and interpolated.
- **Statistics are not guarantees.** Analog days come from a limited number of historical episodes, many are clustered in time, and many strikes/structures are tested at once. HAC corrections and train/validation splits reduce, but do not remove, the risk of false positives. Treat "candidates" as hypotheses to examine, not signals to trade.
- **Known open items (work in progress):**
  - Analog Day Matching and the Iron Condor Backtest have not yet been fully migrated to the multi-instrument setup (e.g. instrument-specific training table, dividend yield default); Analog mode is currently only reliable after this is finished.
  - The Interactive Brokers provider is untested; the hedge finder has no dashboard UI yet.

---

## About the author

Contact: www.linkedin.com/in/lars-bullmahn-a578bb425

## License

_Add your license here (e.g. MIT), or state "All rights reserved"._
