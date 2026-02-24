# Project Guidelines

## Running Scripts

Do NOT run training scripts directly. Instead, prompt the user to run them manually. Training and hyperparameter tuning (grid search) can take a long time.

Example - instead of running:
```bash
python src/model.py --tune
```

Say: "Ready to run. Execute with: `python src/model.py --tune`"

## CLI Arguments (`src/model.py`)

| Argument | Type | Description |
|----------|------|-------------|
| `--model` | choice | Model backend: `lightgbm`, `xgboost`, `catboost`, `ridge`, `elasticnet`, `stacking` |
| `--tune` | flag | Enable hyperparameter tuning via RandomizedSearchCV |
| `--tune-iterations` | int | Number of tuning iterations (default: 30) |
| `--random-cv` | flag | Use random KFold CV instead of time-based CV |
| `--no-inflation` | flag | Disable CPI inflation adjustment |
| `--inflation-lag` | int | Months to lag inflation data (0 = same month) |
| `--contractor-history` | flag | Add contractor prior wins count at bid time |
| `--competition-intensity` | flag | Add number of bidders per job as feature |

### Default Config
- Model: lightgbm
- Inflation: enabled, no lag
- Contractor history: disabled
- Competition intensity: disabled
- CV: time-based, 5 splits
- Tuning: disabled
- num_boost_round: 2000, learning_rate: 0.05

## MLflow Tracking

All runs are tracked under the `construction-price-prediction` experiment.

**Logged parameters:** All config values, plus `tuned_*` prefixed params if tuning is enabled.

**Logged metrics:**
- `cv_rmse_mean`, `cv_rmse_std`
- `cv_rmse_fold_1` through `cv_rmse_fold_N`

**Logged artifacts:**
- `feature_importance.csv`
- `submission.csv`

---

## Project Status: Diminishing Returns Analysis

### Model Selection - SATURATED
| Model | Status | Notes |
|-------|--------|-------|
| LightGBM | Tuned | Primary model, well-optimized |
| XGBoost | Tuned | Similar performance to LightGBM |
| CatBoost | Tuned | Similar performance |
| Ridge/ElasticNet | Tested | Linear baselines, significantly worse |
| Stacking | Implemented | Ensemble of above; marginal gains over single best |

**Verdict:** Switching models unlikely to yield significant improvement. All major gradient boosting frameworks explored.

### Hyperparameter Tuning - SATURATED
- RandomizedSearchCV with 30+ iterations per model
- Ranges are well-calibrated (learning rate 0.03-0.15, depth 4-10, regularization 0-2)
- Early stopping prevents overfitting
- Time-based CV ensures proper temporal validation

**Verdict:** Further tuning iterations yield diminishing returns. Current params are near-optimal.

### Feature Engineering - MODERATE POTENTIAL REMAINING

#### Implemented (tested)
| Feature | Impact |
|---------|--------|
| Inflation adjustment (materials PPI + labor ECI) | Moderate |
| Line-item unit price estimation | High - core predictive signal |
| Date features (month, year, day of week) | Low-moderate |
| Category/location encodings | Moderate |
| Contractor prior wins (total, by category, by location) | Low |
| Competition intensity (# bidders) | Low |

#### Potential ideas not yet explored
- **Geographic cost indices** - regional labor/material cost multipliers
- **Seasonality interactions** - category × month interactions
- **Item co-occurrence** - which pay items appear together
- **Bid spread features** - how does this contractor's estimate compare to their typical markup
- **Project complexity** - ratio of unique items to total items, variance in quantities
- **Historical accuracy** - contractor's past bid-to-actual ratios (if outcome data available)

**Verdict:** Most obvious features implemented. Remaining ideas require additional data sources or more complex feature engineering with uncertain payoff.

### Data Quality / Leakage - CHECK COMPLETE
- Price lookup uses only training fold data (no leakage)
- Time-based CV respects temporal ordering
- Contractor history computed only from prior wins

### Recommended Next Steps (if pursuing further)
1. **Error analysis** - examine worst predictions to identify systematic patterns
2. **External data** - regional cost indices, commodity prices, weather
3. **Target engineering** - predict log-ratio to estimated cost instead of raw bid
