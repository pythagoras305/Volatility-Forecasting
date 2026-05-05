"""Central configuration: paths, date ranges, and constants."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"
MLRUNS_DIR = ROOT / "mlruns"

UNIVERSE_PATH = PROCESSED_DIR / "universe.parquet"
OHLCV_DB_PATH = RAW_DIR / "ohlcv.db"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"

# ---------------------------------------------------------------------------
# Date ranges
# ---------------------------------------------------------------------------
INGEST_START = "2010-01-01"
INGEST_END = "2024-12-31"

TRAIN_START = "2014-01-01"
TRAIN_END = "2019-12-31"
VAL_START = "2020-01-01"
VAL_END = "2021-12-31"
TEST_START = "2022-01-01"
TEST_END = "2024-12-31"

# Days to purge between splits to prevent label leakage (matches forecast horizon)
PURGE_DAYS = 5

# Walk-forward refit step in months
WF_STEP_MONTHS = 6

# ---------------------------------------------------------------------------
# Target variable
# ---------------------------------------------------------------------------
# Forecast horizon: next N trading days of realized volatility
RV_HORIZON = 5
TARGET_COL = f"log_rv_{RV_HORIZON}_next"

# ---------------------------------------------------------------------------
# Feature engineering constants
# ---------------------------------------------------------------------------
RETURN_LAGS = [1, 2, 3, 5, 10, 21]
RV_WINDOWS = [5, 10, 21, 63]
PARKINSON_WINDOWS = [5, 10, 21]
VOL_ZSCORE_WINDOW = 21
MA_WINDOWS = [21, 63]
CS_RETURN_WINDOW = 21  # window for cross-sectional return rank

# ---------------------------------------------------------------------------
# GARCH
# ---------------------------------------------------------------------------
GARCH_REFIT_WINDOW_YEARS = 2
GARCH_REFIT_FREQ = "W"  # weekly refit

# ---------------------------------------------------------------------------
# LightGBM / Optuna
# ---------------------------------------------------------------------------
LGBM_OPTUNA_TRIALS = 50
LGBM_EXPERIMENT = "lgbm"

# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------
LSTM_SEQ_LEN = 60
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2
LSTM_BATCH_SIZE = 512
LSTM_MAX_EPOCHS = 50
LSTM_LR = 1e-3
LSTM_EXPERIMENT = "lstm"

# ---------------------------------------------------------------------------
# MLflow experiments
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = MLRUNS_DIR.as_uri()  # file:///... — required on Windows
EXPERIMENT_BASELINE_ROLLING = "baseline_rolling"
EXPERIMENT_BASELINE_GARCH = "baseline_garch"
EXPERIMENT_LGBM = "lgbm"
EXPERIMENT_LSTM = "lstm"

# ---------------------------------------------------------------------------
# Rolling historical vol baseline
# ---------------------------------------------------------------------------
ROLLING_VOL_WINDOW_CANDIDATES = [5, 10, 21, 63]

# ---------------------------------------------------------------------------
# Evaluation / bootstrap
# ---------------------------------------------------------------------------
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_CI = 0.95
VOL_REGIME_QUINTILES = 5
