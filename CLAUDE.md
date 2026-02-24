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
