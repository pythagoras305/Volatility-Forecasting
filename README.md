# Equity Volatility Forecaster

A research-grade ML system that forecasts next-5-day realized volatility for S&P 500 equities. Four models are compared: two classical baselines (rolling historical vol, GARCH(1,1)) and two ML approaches (LightGBM with Optuna HPO, PyTorch LSTM). All experiments are tracked with MLflow and fully reproducible from a single command.

---

## Results

> _Run `make all` to reproduce. Results populated after training._

| Model | QLIKE (↓) | RMSE log-RV (↓) | MAE log-RV (↓) | Val QLIKE (↓) |
|---|---|---|---|---|
| Rolling HV (10d) | 0.1926 | 0.703 | 0.562 | 0.185 |
| GARCH(1,1) | 0.1589 | 0.625 | 0.497 | 0.164 |
| **LightGBM** ✓ | **0.1259** | **0.469** | **0.363** | 0.100 |
| **LSTM** ✓ | **0.1274** | **0.470** | 0.361 | 0.114 |

Both ML models beat both baselines on test QLIKE (the primary metric). LightGBM wins by a slim margin over LSTM. GARCH outperforms rolling HV, confirming volatility clustering is a real effect worth modeling. See [`reports/comparison.md`](reports/comparison.md) for the full regime breakdown and bootstrap CIs, generated automatically by `make compare`.

---

## Quickstart

**Requirements:** Python 3.11+, [`uv`](https://github.com/astral-sh/uv) (or pip).

```bash
# 1. Clone and create environment
git clone <repo-url> && cd equity-vol-forecaster
uv venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# 2. Run the full pipeline (~2–4 hours, mostly GARCH and data download)
make all

# 3. View results
cat reports/comparison.md

# 4. Launch MLflow UI
mlflow ui --backend-store-uri mlruns/
```

Individual steps:

```bash
make universe          # Scrape S&P 500 constituents from Wikipedia
make ingest            # Download OHLCV from yfinance (2010–2024)
make features          # Build feature panel → data/processed/features.parquet
make train-baselines   # Rolling HV + GARCH(1,1)
make train-lgbm        # LightGBM + Optuna (50 trials)
make train-lstm        # PyTorch LSTM
make compare           # Query MLflow → reports/comparison.md
make test              # Run pytest
```

---

## Project Structure

```
equity-vol-forecaster/
├── src/
│   ├── config.py              # Paths, constants, date ranges
│   ├── data/
│   │   ├── universe.py        # S&P 500 constituent scraper
│   │   ├── ingest.py          # yfinance OHLCV download
│   │   ├── storage.py         # SQLite read/write
│   │   └── features.py        # Point-in-time feature engineering
│   ├── models/
│   │   ├── base.py            # Abstract VolModel interface
│   │   ├── rolling_hist.py    # Rolling historical vol baseline
│   │   ├── garch.py           # GARCH(1,1) per ticker
│   │   ├── lgbm.py            # LightGBM (global, Optuna)
│   │   └── lstm.py            # PyTorch LSTM
│   ├── eval/
│   │   ├── splits.py          # Purged walk-forward CV
│   │   ├── metrics.py         # QLIKE, RMSE, MAE, bootstrap CI
│   │   └── runner.py          # Fit → evaluate → MLflow log
│   └── viz/
│       └── plots.py           # Comparison figures
├── scripts/                   # 01_build_universe … 07_compare_models
├── tests/                     # test_features, test_splits, test_metrics
├── reports/comparison.md      # Auto-generated after make compare
└── mlruns/                    # MLflow tracking (gitignored)
```

---

## Methodology

### Target variable

`log_rv_5_next` — log of realized volatility over the **next 5 trading days**:

```
RV_5(t) = sqrt( sum_{k=1}^{5} r_{t+k}^2 )
target  = log( RV_5(t) )
```

Using log-RV stabilizes the distribution (approximately Gaussian) and makes RMSE/MAE meaningful. The 5-day horizon matches common risk-management horizons and is less noisy than 1-day.

### Features (point-in-time as of date t)

| Category | Features |
|---|---|
| Lagged returns | log returns at lags 1, 2, 3, 5, 10, 21 days |
| Realized vol | trailing RV over 5, 10, 21, 63 days (+ log-transformed) |
| Range vol | Parkinson estimator over 5, 10, 21 days |
| Volume | log-volume z-score (21-day window) |
| Calendar | day-of-week indicator |
| Price level | distance from 21-day and 63-day MA (z-score) |
| Cross-section | decile rank of trailing 21-day return within S&P 500 |

All features are constructed so that only data through date _t_ is used. The target uses dates _t+1_ through _t+5_. Leakage is verified in `tests/test_features.py`.

### Train / Val / Test splits

```
Training:   2014-01-01 → 2019-12-31   (6 years)
Validation: 2020-01-01 → 2021-12-31   (2 years, includes COVID vol spike)
Test:       2022-01-01 → 2024-12-31   (3 years, held out)
Purge gap:  5 trading days between splits (matches forecast horizon)
```

The validation set intentionally includes the March 2020 COVID shock — this is documented as a known distribution shift, not corrected for.

### Models

**Rolling Historical Vol** — log of trailing realized vol; window (5/10/21/63 days) tuned on validation QLIKE.

**GARCH(1,1)** — per-ticker GARCH fitted with `arch` on a rolling 2-year window, refit weekly. 5-day forecast = sum of 1-day variance forecasts.

**LightGBM** — global model across all tickers (ticker as categorical feature). 50 Optuna trials optimizing QLIKE on validation set. Early stopping per trial.

**LSTM** — 2-layer LSTM (hidden=64) → dropout(0.2) → linear head. Sequence length 60 days. Feature standardization computed on training fold only. Early stopping on validation MSE.

### Evaluation

**Primary metric:** QLIKE = mean(RV_actual/RV_pred − log(RV_actual/RV_pred) − 1). Standard for vol forecasting; penalizes underprediction of risk more than overprediction.

**Secondary:** RMSE and MAE on log-RV.

**Regime analysis:** test set split into 5 vol quintiles; metrics reported per quintile.

**Statistical significance:** 1 000 bootstrap resamples of test-set predictions; 95% CI on QLIKE difference vs best baseline. A result is considered significant if the CI does not cross zero.

---

## Limitations

1. **Survivorship bias** — The universe is the *current* S&P 500 constituent list. Companies delisted or removed between 2014 and 2024 (bankruptcies, acquisitions, index removals) are excluded. This biases the sample toward large, stable firms and likely overstates model performance relative to a live trading universe.

2. **Lookback** — Ten years of daily data (2014–2024) includes one major vol regime shift (COVID-19). The model has seen only a limited number of market regimes. Out-of-sample performance in future regime shifts is unknown.

3. **No transaction costs** — All metrics measure forecast quality, not strategy P&L. Turning a good vol forecast into a profitable strategy requires slippage, spread, and margin cost assumptions not modelled here.

4. **Daily bars only** — No intraday microstructure. Overnight gaps and intraday vol patterns are not captured.

5. **No news or sentiment features** — LLM-extracted earnings call and news sentiment features are planned for Phase 1.

6. **GARCH convergence** — GARCH(1,1) occasionally fails to converge on short windows; affected tickers/dates fall back to the nearest prior forecast.

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| **0** | ✅ Complete | Data pipeline, 4 models, evaluation, reproducible repo |
| **1** | Planned | LLM news/earnings sentiment features (Claude API) |
| **2** | Planned | Research agent, backtested vol-targeting strategy |
| **3** | Planned | Alpaca paper trading, web dashboard |

---

## Reproducibility

All results in the Results table above are produced by:

```bash
make all
```

from a clean clone with no pre-existing `data/` or `mlruns/` directories.

**Expected runtimes** (M1 MacBook / modern x86, no GPU):
- `make universe + ingest`: ~30–60 min (network-bound, yfinance rate limits)
- `make features`: ~5 min
- `make train-baselines`: ~20–40 min (GARCH per-ticker)
- `make train-lgbm`: ~10–20 min (50 Optuna trials)
- `make train-lstm`: ~15–30 min (CPU; ~5 min with CUDA GPU)
- `make compare`: <1 min

Random seeds are fixed (LightGBM: `random_state=42`, LSTM: deterministic DataLoader, Optuna: `seed=42`). Minor floating-point differences across platforms are expected.

MLflow tracking is stored locally in `mlruns/`. Launch the UI with:

```bash
mlflow ui --backend-store-uri mlruns/
```
