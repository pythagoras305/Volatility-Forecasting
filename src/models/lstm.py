"""PyTorch LSTM for volatility forecasting.

Architecture: 2-layer LSTM (hidden 64) → dropout → linear head.
Input: last SEQ_LEN days of features per ticker.
Output: log_rv_5_next scalar.

Feature standardization uses mean/std computed on the training fold only
(no leakage from val/test into scaler parameters).
"""

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.config import (
    LSTM_BATCH_SIZE,
    LSTM_DROPOUT,
    LSTM_HIDDEN_SIZE,
    LSTM_LR,
    LSTM_MAX_EPOCHS,
    LSTM_NUM_LAYERS,
    LSTM_SEQ_LEN,
)
from src.models.base import VolModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class VolSequenceDataset(Dataset):
    """Sliding-window sequence dataset, one sequence per (ticker, date)."""

    def __init__(
        self,
        panel: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        seq_len: int,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ):
        sequences, targets = [], []
        panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

        # Precompute scaler stats on the provided data if not given
        feat_matrix = panel[feature_cols].values.astype(np.float32)
        if mean is None:
            mean = np.nanmean(feat_matrix, axis=0)
        if std is None:
            std = np.nanstd(feat_matrix, axis=0)
        std = np.where(std < 1e-8, 1.0, std)

        self.mean = mean
        self.std = std

        feat_norm = (feat_matrix - mean) / std
        feat_norm = np.nan_to_num(feat_norm, nan=0.0)

        targets_arr = panel[target_col].values.astype(np.float32)

        for ticker, grp in panel.groupby("ticker", sort=False):
            idx = grp.index.values
            if len(idx) < seq_len + 1:
                continue
            feat_grp = feat_norm[idx]
            tgt_grp = targets_arr[idx]

            for end in range(seq_len, len(idx)):
                seq = feat_grp[end - seq_len: end]  # (seq_len, n_features)
                tgt = tgt_grp[end]
                if not np.isfinite(tgt):
                    continue
                sequences.append(seq)
                targets.append(tgt)

        self.X = torch.tensor(np.array(sequences), dtype=torch.float32)
        self.y = torch.tensor(np.array(targets), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------


class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # (batch, hidden)
        return self.head(self.dropout(last)).squeeze(-1)


# ---------------------------------------------------------------------------
# VolModel wrapper
# ---------------------------------------------------------------------------


class LSTMVolModel(VolModel):
    """PyTorch LSTM volatility model."""

    def __init__(
        self,
        seq_len: int = LSTM_SEQ_LEN,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
        dropout: float = LSTM_DROPOUT,
        batch_size: int = LSTM_BATCH_SIZE,
        max_epochs: int = LSTM_MAX_EPOCHS,
        lr: float = LSTM_LR,
    ):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lr = lr

        self._net: _LSTMNet | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._feature_cols: list[str] = []
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def name(self) -> str:
        return f"lstm_h{self.hidden_size}_l{self.num_layers}"

    def fit(
        self,
        train: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        val: pd.DataFrame | None = None,
    ) -> None:
        self._feature_cols = feature_cols
        logger.info("Building LSTM dataset (device=%s)...", self._device)

        # Compute scaler on training data only
        train_feats = train[feature_cols].values.astype(np.float32)
        self._mean = np.nanmean(train_feats, axis=0)
        self._std = np.nanstd(train_feats, axis=0)

        train_ds = VolSequenceDataset(
            train, feature_cols, target_col, self.seq_len, self._mean, self._std
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)

        val_loader = None
        if val is not None and len(val) > 0:
            val_ds = VolSequenceDataset(
                val, feature_cols, target_col, self.seq_len, self._mean, self._std
            )
            val_loader = DataLoader(val_ds, batch_size=self.batch_size * 2, shuffle=False, num_workers=0)

        n_features = len(feature_cols)
        self._net = _LSTMNet(n_features, self.hidden_size, self.num_layers, self.dropout).to(self._device)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        criterion = nn.MSELoss()

        best_val_loss = np.inf
        best_state = None
        patience_count = 0
        early_stop_patience = 7

        for epoch in range(1, self.max_epochs + 1):
            self._net.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self._device), y_batch.to(self._device)
                optimizer.zero_grad()
                preds = self._net(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            train_loss = np.mean(train_losses)

            if val_loader is not None:
                self._net.eval()
                val_losses = []
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(self._device), y_batch.to(self._device)
                        preds = self._net(X_batch)
                        val_losses.append(criterion(preds, y_batch).item())
                val_loss = np.mean(val_losses)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self._net.state_dict().items()}
                    patience_count = 0
                else:
                    patience_count += 1

                if epoch % 5 == 0 or epoch == 1:
                    logger.info("Epoch %3d/%d — train_loss=%.4f val_loss=%.4f", epoch, self.max_epochs, train_loss, val_loss)

                if patience_count >= early_stop_patience:
                    logger.info("Early stopping at epoch %d (best val=%.4f)", epoch, best_val_loss)
                    break
            else:
                if epoch % 5 == 0 or epoch == 1:
                    logger.info("Epoch %3d/%d — train_loss=%.4f", epoch, self.max_epochs, train_loss)

        if best_state is not None:
            self._net.load_state_dict(best_state)
        logger.info("LSTM training complete. Best val loss: %.4f", best_val_loss)

    def predict(self, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Model has not been fit. Call fit() first.")

        # Build inference dataset — sequences may not cover all rows (first seq_len rows per ticker will be missing)
        ds = VolSequenceDataset(
            df, feature_cols, list(df.columns)[0], self.seq_len, self._mean, self._std
        )
        # We need to map predictions back to df rows. VolSequenceDataset iterates
        # ticker-sorted, so we reconstruct the index mapping.
        df_sorted = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        preds_dict: dict[tuple, float] = {}
        loader = DataLoader(ds, batch_size=self.batch_size * 2, shuffle=False, num_workers=0)

        self._net.eval()
        all_preds = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self._device)
                all_preds.append(self._net(X_batch).cpu().numpy())

        if all_preds:
            flat_preds = np.concatenate(all_preds)
        else:
            return np.full(len(df), np.nan)

        # Rebuild index: same logic as dataset construction
        feat_cols_dummy = feature_cols
        seq_len = self.seq_len
        idx_order = []
        for ticker, grp in df_sorted.groupby("ticker", sort=False):
            grp_idx = grp.index.values
            if len(grp_idx) < seq_len + 1:
                continue
            for end in range(seq_len, len(grp_idx)):
                idx_order.append(grp_idx[end])

        result = np.full(len(df_sorted), np.nan)
        for i, row_idx in enumerate(idx_order):
            if i < len(flat_preds):
                result[row_idx] = flat_preds[i]

        # Re-align to original df order
        orig_order = df.index if hasattr(df, "index") else range(len(df))
        # df_sorted was reset_index'd; map back via (ticker, date)
        df_sorted["_pred"] = result
        merged = df.reset_index(drop=True).merge(
            df_sorted[["ticker", "date", "_pred"]],
            on=["ticker", "date"],
            how="left",
        )
        return merged["_pred"].values

    def get_params(self) -> dict:
        return {
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "lr": self.lr,
        }
