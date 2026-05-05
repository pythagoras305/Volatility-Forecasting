.PHONY: all universe ingest features train-baselines train-lgbm train-lstm compare test lint format clean

PYTHON := python

# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
all: universe ingest features train-baselines train-lgbm train-lstm compare

# ---------------------------------------------------------------------------
# Individual pipeline steps
# ---------------------------------------------------------------------------
universe:
	$(PYTHON) scripts/01_build_universe.py

ingest: universe
	$(PYTHON) scripts/02_ingest_ohlcv.py

features: ingest
	$(PYTHON) scripts/03_build_features.py

train-baselines: features
	$(PYTHON) scripts/04_train_baselines.py

train-lgbm: features
	$(PYTHON) scripts/05_train_lgbm.py

train-lstm: features
	$(PYTHON) scripts/06_train_lstm.py

compare: train-baselines train-lgbm train-lstm
	$(PYTHON) scripts/07_compare_models.py

# ---------------------------------------------------------------------------
# Dev tooling
# ---------------------------------------------------------------------------
test:
	pytest tests/ -v

lint:
	ruff check src/ scripts/ tests/

format:
	black src/ scripts/ tests/
	ruff check --fix src/ scripts/ tests/

# ---------------------------------------------------------------------------
# Cleanup (never removes raw data or mlruns — use clean-all for that)
# ---------------------------------------------------------------------------
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

clean-all: clean
	rm -rf data/raw/ data/processed/ mlruns/ reports/comparison.md
