# Project Guidelines

## Running Scripts

Do NOT run training scripts directly. Instead, prompt the user to run them manually. Training and hyperparameter tuning (grid search) can take a long time.

Example - instead of running:
```bash
python src/lightgbm_lineitem.py --tune
```

Say: "Ready to run. Execute with: `python src/lightgbm_lineitem.py --tune`"
