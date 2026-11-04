# House Price Prediction with Random Forest Regression

Predicting house prices (in €) with a scikit-learn Random Forest, including tuning, validation, feature importance and error analysis. Demonstrated on a synthetic German-style dataset of 2,000 houses; swap in your own CSV with a `price` column.

## Files

| File | Purpose |
|---|---|
| `haus_price_rf.py` | Training, cross-validation, hyperparameter search, feature importance |
| `haus_plots.py` | Generates the dataset and six analysis plots |
| `haus_synthetic.csv` | Synthetic data (2,000 rows) |
| `plots/` | Output figures |

## Usage

```bash
pip install numpy pandas scikit-learn matplotlib
python haus_price_rf.py                # synthetic data
python haus_price_rf.py your_data.csv  # your own data (needs a 'price' column)
python haus_plots.py                   # regenerate plots
```

## Features

`living_area_m2`, `plot_area_m2`, `rooms`, `year_built`, `dist_center_km`, `energy_class` (1 = A+, 8 = H). Categorical columns are one-hot encoded automatically.

## Results (80/20 split, random_state=42)

| Model | R² | MAE | RMSE | MAPE |
|---|---|---|---|---|
| Baseline (300 trees) | 0.8761 | €40,151 | €51,324 | 11.48% |
| Tuned (400 trees, max_depth 20, min_samples_leaf 2, max_features 0.5) | 0.8778 | €39,770 | €50,971 | 11.67% |

Validation: OOB R² 0.8866, 5-fold CV R² 0.8830 ± 0.0066.

### Feature importance (permutation)

`living_area_m2` 1.1622 > `year_built` 0.2508 > `dist_center_km` 0.2458 > `plot_area_m2` 0.0320 > `rooms` 0.0120 > `energy_class` 0.0064

### Error by price quintile (residual = actual − predicted)

| Price band | Mean residual | MAPE |
|---|---|---|
| €80,000 – €303,220 | −€37,874 | 26.8% |
| €303,220 – €401,520 | −€12,714 | 10.2% |
| €401,520 – €477,740 | −€2,835 | 7.9% |
| €477,740 – €549,960 | +€10,624 | 6.4% |
| €549,960 – €795,100 | +€35,502 | 7.0% |

The model overpredicts cheap houses and underpredicts expensive ones (regression to the mean). A log-price target is a recommended next step.

## Plots

1. Actual vs predicted
2. Residual distribution and residuals vs predicted
3. Permutation importance
4. Partial dependence
5. Error by price quintile
6. Correlation matrix

## Notes

Results are from synthetic data and will differ on real listings.
