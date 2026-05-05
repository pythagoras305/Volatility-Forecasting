# Equity Volatility Forecaster

## Project goal
A research and production-style ML system that forecasts next-day realized volatility for S&P 500 equities, comparing classical baselines (rolling historical vol, GARCH(1,1)) against ML approaches (LightGBM, LSTM). Future phases add LLM-extracted news features, a research agent, backtested vol-targeting strategy, and Alpaca paper trading.

This document defines Phase 0: data pipeline + four volatility models + evaluation + clean repo.
Future phases will be added to this document as they are reached.

## Audience for this codebase
This is a portfolio project. The primary audiences are recruiters and engineers evaluating the author for AI engineering, ML engineering, quant, and data science roles, in that order of priority. Code quality, repo organization, README clarity, and reproducibility matter as much as model performance.

## Non-goals for Phase 0
- No LLM integration yet (Phase 1).
- No agents (Phase 2).
- No backtesting or trading strategy (Phase 2).
- No paper trading integration (Phase 3).
- No web dashboard (Phase 3).
- No intraday data. Daily bars only.

## Constraints
- Free data sources only (yfinance for OHLCV; Wikipedia for current S&P 500 constituents).
- SQLite for storage. Postgres can come later if needed.
- Python 3.11+. Use `uv` for environment management if available, otherwise venv + pip.
- All experiments tracked with MLflow (local backend, no cloud).
- Reproducibility: any result in the README must be reproducible by running a single make target or script.
- Survivorship bias: use current S&P 500 constituents only. Document this limitation explicitly in README and in the writeup. Do not silently ignore it.

## Repo structure (target)
equity-vol-forecaster/
├── README.md                    # Top-level: what, why, how, results, limitations
├── CLAUDE.md                    # This file
├── pyproject.toml               # uv/pip config
├── Makefile                     # One-command targets for full pipeline
├── .gitignore
├── data/
│   ├── raw/                     # gitignored, populated by scripts
│   └── processed/               # gitignored
├── src/
│   ├── __init__.py
│   ├── config.py                # Paths, constants, date ranges
│   ├── data/
│   │   ├── __init__.py
│   │   ├── universe.py          # S&P 500 constituent list
│   │   ├── ingest.py            # yfinance OHLCV download
│   │   ├── storage.py           # SQLite read/write
│   │   └── features.py          # Feature engineering (returns, vol, technicals)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract VolModel interface
│   │   ├── rolling_hist.py      # Baseline: rolling historical vol
│   │   ├── garch.py             # Baseline: GARCH(1,1) via arch package
│   │   ├── lgbm.py              # LightGBM regressor
│   │   └── lstm.py              # PyTorch LSTM
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── splits.py            # Purged walk-forward CV
│   │   ├── metrics.py           # QLIKE, RMSE on log-vol, MAE
│   │   └── runner.py            # Run a model across all splits, log to MLflow
│   └── viz/
│       ├── __init__.py
│       └── plots.py             # Equity curves, error distributions, regime plots
├── scripts/
│   ├── 01_build_universe.py     # Output: data/processed/universe.parquet
│   ├── 02_ingest_ohlcv.py       # Output: data/raw/ohlcv.db (SQLite)
│   ├── 03_build_features.py     # Output: data/processed/features.parquet
│   ├── 04_train_baselines.py    # Logs to MLflow
│   ├── 05_train_lgbm.py         # Logs to MLflow
│   ├── 06_train_lstm.py         # Logs to MLflow
│   └── 07_compare_models.py     # Output: reports/comparison.md
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis on returns/vol
│   └── 02_results_writeup.ipynb # Final results notebook for README/writeup
├── tests/
│   ├── test_features.py
│   ├── test_splits.py
│   └── test_metrics.py
├── reports/
│   └── comparison.md            # Auto-generated model comparison
└── mlruns/                      # MLflow tracking, gitignored

## Modeling spec

### Target variable
- `log_rv_5_next`: log of realized volatility computed over the next 5 trading days, where realized volatility is the square root of the sum of squared daily log returns.
- Rationale: 5-day RV is less noisy than 1-day, still actionable, matches typical risk-management horizons.
- Predict log(RV) not RV: makes the distribution roughly Gaussian and stabilizes training.

### Features (per ticker, per date — point-in-time, no leakage)
- Lagged log returns: 1, 2, 3, 5, 10, 21 days
- Realized vol over trailing windows: 5, 10, 21, 63 days
- High-low range vol (Parkinson estimator) over 5, 10, 21 days
- Log volume z-score over 21-day window
- Day-of-week indicator
- Distance from 21-day and 63-day moving average (z-score)
- Cross-sectional rank of trailing 21-day return within S&P 500 (decile)

### Models

**1. Rolling historical vol baseline**
- Predict `log_rv_5_next` as the log of realized vol over a trailing window.
- Tune window length on validation set: candidates [5, 10, 21, 63].

**2. GARCH(1,1) baseline**
- Per-ticker GARCH(1,1) on daily log returns using `arch` package.
- 5-day forecast aggregated from 1-day forecasts.
- Refit weekly on rolling 2-year window to keep compute reasonable.

**3. LightGBM regressor**
- Single global model across all tickers (ticker as categorical feature).
- Hyperparameter search via Optuna, ~50 trials, optimizing QLIKE on validation folds.
- Early stopping on validation loss.

**4. LSTM**
- PyTorch. Per-sequence input of last 60 days of features, predict next-day log_rv_5_next.
- Simple architecture: 2-layer LSTM (hidden size 64) → dropout → linear head.
- Same train/val/test splits as LGBM.
- Train on GPU if available, CPU acceptable for daily data.

### Evaluation

**Splits: purged walk-forward**
- Train: 2014-01-01 to 2019-12-31
- Validation: 2020-01-01 to 2021-12-31 (includes COVID vol shock — keep, document)
- Test: 2022-01-01 to 2024-12-31 (held out, only touched after final model selection)
- Purge: 5-day gap between train end and val start, and val end and test start, to avoid label leakage.
- For walk-forward refits during evaluation: 6-month step.

**Metrics**
- Primary: QLIKE on RV (not log RV). QLIKE = mean(RV_actual / RV_pred − log(RV_actual / RV_pred) − 1). Standard for vol forecasting; penalizes underprediction more than overprediction.
- Secondary: RMSE on log_rv_5_next, MAE on log_rv_5_next.
- Reported per-regime: split test set into vol quintiles based on trailing 21-day cross-sectional median vol; report metrics per quintile.
- Sharpe-style significance: bootstrap 1000 resamples of test predictions, report 95% CI on QLIKE difference vs best baseline.

**Success criterion**
- At least one ML model (LGBM or LSTM) beats both baselines on QLIKE on the test set with the bootstrap CI not crossing zero.
- If neither beats baselines, report that honestly and analyze why. A well-documented null result is acceptable for the portfolio — confirmation-bias-free reporting is itself a quality signal.

## MLflow conventions
- One experiment per model family: `baseline_rolling`, `baseline_garch`, `lgbm`, `lstm`.
- Every run logs: all hyperparameters, all metrics (per fold + aggregate), feature list, code git SHA, full config as YAML artifact.
- Final-model runs are tagged `tag:final=true` for easy filtering.

## README sections (target)
1. What this is (3-4 sentences)
2. Results table — QLIKE/RMSE for all four models, baseline-vs-ML
3. Quickstart — `make all` reproduces everything
4. Project structure
5. Methodology — splits, target, features, models
6. Limitations — survivorship bias, lookback, no transaction costs at this stage, daily-only, etc.
7. Roadmap — phases 1-3
8. Reproducibility notes
