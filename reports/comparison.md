# Model Comparison Report
_Generated on 2026-05-05_

## Overall metrics (test set: 2022-01-01 → 2024-12-31)

| Model      |   QLIKE (↓) |   RMSE log-RV (↓) |   MAE log-RV (↓) |   Val QLIKE (↓) |
|:-----------|------------:|------------------:|-----------------:|----------------:|
| Rolling HV |      0.1926 |            0.7026 |           0.562  |          0.1854 |
| GARCH(1,1) |      0.1589 |            0.6247 |           0.4966 |          0.1641 |
| LightGBM   |      0.1259 |            0.4691 |           0.3631 |          0.1001 |
| LSTM       |      0.1274 |            0.4695 |           0.3613 |          0.1144 |

_Primary metric: QLIKE. Lower is better. QLIKE = mean(RV_actual/RV_pred − log(RV_actual/RV_pred) − 1)._

## QLIKE by volatility regime (test set)

Quintile 0 = lowest vol, Quintile 4 = highest vol.

| Model      |      0 |      1 |      2 |      3 |      4 |
|:-----------|-------:|-------:|-------:|-------:|-------:|
| GARCH(1,1) | 0.3596 | 0.1469 | 0.0786 | 0.0511 | 0.1581 |
| LSTM       | 0.13   | 0.0346 | 0.0431 | 0.0866 | 0.3428 |
| LightGBM   | 0.1308 | 0.0349 | 0.0425 | 0.0885 | 0.333  |
| Rolling HV | 0.3707 | 0.1886 | 0.1333 | 0.103  | 0.1675 |

### Bootstrap CI: QLIKE difference (ML − best baseline)

Negative = ML is better. CI computed over 1 000 bootstrap resamples of test predictions.

- **LightGBM vs Rolling HV**: Δ=-0.0666 (95% CI [-0.0692, -0.0646]) — ✓ beats baseline

- **LightGBM vs GARCH(1,1)**: Δ=-0.0666 (95% CI [-0.0692, -0.0646]) — ✓ beats baseline

- **LSTM vs Rolling HV**: Δ=-0.0652 (95% CI [-0.0678, -0.0632]) — ✓ beats baseline

- **LSTM vs GARCH(1,1)**: Δ=-0.0652 (95% CI [-0.0678, -0.0632]) — ✓ beats baseline


## Notes

- **Survivorship bias**: Universe is current S&P 500 constituents. Companies that were delisted or removed between 2014–2024 are excluded. This likely overstates model quality on stable large-caps.
- **COVID shock (2020–2021)**: The validation period includes the March 2020 vol spike. Models saw this regime during validation but not training.
- **No transaction costs**: These are forecast quality metrics, not strategy returns.
- **Daily only**: No intraday microstructure, no overnight gaps modelled separately.
