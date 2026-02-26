"""
Stacking Sweep v2 — Even Coverage with Random Sampling

Context: stacking_sweep.py used Optuna TPE which concentrated ~60% of trials
on one stack combo and pinned inflation=True. The best Kaggle run (a932d990,
0.2858) used no inflation — a config never tested in v1.

Changes from v1:
  1. RandomSampler instead of TPE — even coverage of discrete categorical space
  2. use_inflation now a variable: [True, False]
     When True: log_inflation [True, False], inflation_lag_months [0, 3, 6]
     When False: both pinned to defaults
  3. Stack composition via curated index (8 combos) instead of 4 binary flags
  4. Per-model hyperparams pinned to best known (not tuned per trial)
  5. Duplicate detection — skips combos already run in any prior sweep

Discrete space size:
  8 stacks × 2 markup × 4 recency × 3 train_from × 2 contractor × 2 competition
    × [1 (no-inflation) + 2×3 (inflation variants)]
  = 8 × 2 × 4 × 3 × 2 × 2 × 7 = 5,376 combinations

MLflow experiment: stacking-sweep-v2
Default trials: 50 (runs faster with pinned hyperparams)
"""

import os
import sys
import urllib.request

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import optuna
from optuna.integration.mlflow import MLflowCallback

import mlflow
from model import main, DEFAULT_CONFIG, _PROJECT_ROOT

optuna.logging.set_verbosity(optuna.logging.WARNING)

TRACKING_URI = f"sqlite:///{os.path.join(_PROJECT_ROOT, 'mlflow.db')}"
STUDY_NAME = "stacking-sweep-v2"

# Curated stack combinations — covers promising subsets without over-representing any one
STACK_OPTIONS = [
    ["lightgbm", "catboost", "elasticnet"],             # 0: best stacking-sweep run (3d67adaf)
    ["lightgbm", "xgboost", "catboost"],                # 1: best Kaggle run (a932d990)
    ["lightgbm", "xgboost", "catboost", "elasticnet"],  # 2: all 4 models
    ["xgboost", "catboost", "elasticnet"],              # 3: underexplored 3-combo
    ["lightgbm", "xgboost", "elasticnet"],              # 4: underexplored 3-combo
    ["lightgbm", "catboost"],                           # 5: lean 2-model
    ["lightgbm", "xgboost"],                            # 6: lean 2-model
    ["xgboost", "catboost"],                            # 7: lean 2-model
]

# Pinned per-model hyperparams from best known runs
# LightGBM: from 3d67adaf (best stacking-sweep run, cv_rmse=0.3290)
LGBM_PARAMS = {
    "learning_rate": 0.0708,
    "num_leaves": 91,
    "max_depth": 10,
    "n_estimators": 700,
    "subsample": 0.857,
    "colsample_bytree": 0.768,
    "reg_lambda": 0.800,
}

# CatBoost: from 3d67adaf
CATBOOST_PARAMS = {
    "learning_rate": 0.147,
    "depth": 8,
    "iterations": 400,
    "l2_leaf_reg": 6.30,
}

# XGBoost: from 604d70b9 (best xgb-in-stack run)
XGBOOST_PARAMS = {
    "learning_rate": 0.098,
    "max_depth": 7,
    "n_estimators": 700,
    "subsample": 0.695,
    "colsample_bytree": 0.872,
    "reg_lambda": 0.897,
}

# Keys used to identify a unique combo for duplicate detection
COMBO_KEYS = [
    "stack_idx",
    "use_inflation",
    "log_inflation",
    "inflation_lag_months",
    "use_markup_target",
    "recency_weight",
    "train_from",
    "use_contractor_history",
    "use_competition_intensity",
]


def make_combo(params: dict) -> tuple:
    return tuple(params[k] for k in COMBO_KEYS)


def _stack_idx_for_models(models: list) -> int | None:
    """Return STACK_OPTIONS index for a list of model names, or None if not found."""
    models_set = set(models)
    for i, option in enumerate(STACK_OPTIONS):
        if set(option) == models_set:
            return i
    return None


def load_existing_combos() -> set:
    """
    Load all previously run combos from all MLflow experiments.
    Returns a set of combo tuples keyed by COMBO_KEYS.
    """
    existing = set()
    client = mlflow.tracking.MlflowClient(tracking_uri=TRACKING_URI)

    all_experiments = client.search_experiments()

    for exp in all_experiments:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            max_results=10000,
        )
        for run in runs:
            p = run.data.params
            # Only process runs that have stack-related params
            if "stack_idx" not in p and "use_lgbm" not in p:
                continue

            # Reconstruct stack_idx for v1 runs that used binary flags
            if "stack_idx" in p:
                try:
                    stack_idx = int(p["stack_idx"])
                except (ValueError, TypeError):
                    continue
            elif "use_lgbm" in p:
                # v1-style: reconstruct model list from binary flags
                models = []
                if p.get("use_lgbm") == "True":
                    models.append("lightgbm")
                if p.get("use_xgb") == "True":
                    models.append("xgboost")
                if p.get("use_catboost") == "True":
                    models.append("catboost")
                if p.get("use_elasticnet") == "True":
                    models.append("elasticnet")
                stack_idx = _stack_idx_for_models(models)
                if stack_idx is None:
                    # Combo not in our curated list — can't collide
                    continue
            else:
                continue

            # Parse inflation params
            use_inflation = p.get("use_inflation", "True") == "True"
            log_inflation = p.get("log_inflation", "False") == "True"
            try:
                inflation_lag = int(p.get("inflation_lag_months", "0"))
            except (ValueError, TypeError):
                inflation_lag = 0

            # Parse feature flags
            use_markup = p.get("use_markup_target", "False") == "True"
            try:
                recency = float(p.get("recency_weight", "0.0"))
            except (ValueError, TypeError):
                recency = 0.0
            train_from = p.get("train_from", "None")
            if train_from == "None":
                train_from = None
            use_contractor = p.get("use_contractor_history", "False") == "True"
            use_competition = p.get("use_competition_intensity", "False") == "True"

            combo = make_combo({
                "stack_idx": stack_idx,
                "use_inflation": use_inflation,
                "log_inflation": log_inflation,
                "inflation_lag_months": inflation_lag,
                "use_markup_target": use_markup,
                "recency_weight": recency,
                "train_from": train_from,
                "use_contractor_history": use_contractor,
                "use_competition_intensity": use_competition,
            })
            existing.add(combo)

    return existing


# Shared mutable state for duplicate detection across trials
_seen_combos: set = set()


mlflc = MLflowCallback(
    tracking_uri=TRACKING_URI,
    metric_name="cv_rmse_mean",
    mlflow_kwargs={"experiment_name": STUDY_NAME},
)


def notify(title, message):
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        print("[ntfy] NTFY_TOPIC not set, skipping notification")
        return
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{topic}",
            data=message.encode(),
            headers={"Title": title},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"[ntfy] notification failed: {e}")


def objective(trial):
    # --- Stack composition (curated index) ---
    stack_idx = trial.suggest_categorical("stack_idx", list(range(len(STACK_OPTIONS))))
    stack_models = STACK_OPTIONS[stack_idx]

    # --- Inflation config ---
    # Conditional params: only call suggest_* when inflation=True.
    # Calling the same param name with different distributions raises ValueError in Optuna.
    # model.py's mlflow.log_params(config) records the actual values regardless.
    use_inflation = trial.suggest_categorical("use_inflation", [True, False])
    if use_inflation:
        log_inflation = trial.suggest_categorical("log_inflation", [True, False])
        inflation_lag = trial.suggest_categorical("inflation_lag_months", [0, 3, 6])
    else:
        log_inflation = False
        inflation_lag = 0

    # --- Feature flags ---
    use_markup = trial.suggest_categorical("use_markup_target", [True, False])
    recency = trial.suggest_categorical("recency_weight", [0.0, 0.5, 1.0, 2.0])
    train_from = trial.suggest_categorical("train_from", [None, "2022-01-01", "2022-07-01"])
    use_contractor = trial.suggest_categorical("use_contractor_history", [True, False])
    use_competition = trial.suggest_categorical("use_competition_intensity", [True, False])

    # --- Duplicate detection ---
    combo = make_combo({
        "stack_idx": stack_idx,
        "use_inflation": use_inflation,
        "log_inflation": log_inflation,
        "inflation_lag_months": inflation_lag,
        "use_markup_target": use_markup,
        "recency_weight": recency,
        "train_from": train_from,
        "use_contractor_history": use_contractor,
        "use_competition_intensity": use_competition,
    })

    if combo in _seen_combos:
        raise optuna.exceptions.TrialPruned()

    _seen_combos.add(combo)

    # --- Per-model hyperparams: pinned to best known ---
    stacking_model_params = {}
    for model in stack_models:
        if model == "lightgbm":
            stacking_model_params["lightgbm"] = LGBM_PARAMS.copy()
        elif model == "xgboost":
            stacking_model_params["xgboost"] = XGBOOST_PARAMS.copy()
        elif model == "catboost":
            stacking_model_params["catboost"] = CATBOOST_PARAMS.copy()
        # elasticnet has no hyperparams to pin

    config = DEFAULT_CONFIG.copy()
    config.update({
        "model": "stacking",
        "stack_models": stack_models,
        "stacking_model_params": stacking_model_params,
        "use_markup_target": use_markup,
        "train_from": train_from,
        "use_contractor_history": use_contractor,
        "use_competition_intensity": use_competition,
        "recency_weight": recency,
        "use_inflation": use_inflation,
        "inflation_lag_months": inflation_lag,
        "log_inflation": log_inflation,
        "use_gpu": True,
        "tune": False,
    })

    return main(config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stacking sweep v2 — even random coverage")
    parser.add_argument(
        "--trials", type=int, default=50,
        help="Number of Optuna trials (default: 50; runs faster with pinned hyperparams)"
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(STUDY_NAME)

    print("Loading existing combos from all MLflow experiments...")
    _seen_combos.update(load_existing_combos())
    print(f"  {len(_seen_combos)} existing combos loaded — these will be skipped")
    print()

    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
        sampler=optuna.samplers.RandomSampler(),
    )

    print(f"Starting stacking sweep v2: up to {args.trials} trials (RandomSampler)")
    print("Sweep axes:")
    print("  Stack combos (8):  see STACK_OPTIONS in script")
    print("  use_inflation:     [True, False]")
    print("    log_inflation:   [True, False]  (when inflation=True)")
    print("    inflation_lag:   [0, 3, 6]      (when inflation=True)")
    print("  use_markup_target: [True, False]")
    print("  recency_weight:    [0.0, 0.5, 1.0, 2.0]")
    print("  train_from:        [None, 2022-01-01, 2022-07-01]")
    print("  use_contractor_history:    [True, False]")
    print("  use_competition_intensity: [True, False]")
    print("Pinned: per-model hyperparams from best known runs; gpu=True; tune=False")
    print(f"Total discrete space: 5,376 combinations (~{args.trials/5376*100:.1f}% coverage)")
    print()

    study.optimize(objective, n_trials=args.trials)

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])

    print("\n" + "=" * 60)
    print("SWEEP COMPLETE")
    print(f"Trials completed: {completed}")
    print(f"Trials pruned (duplicates): {pruned}")

    if study.best_trial is not None:
        print(f"Best CV RMSE: {study.best_value:.4f}")
        print("Best params:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        best_params_str = ", ".join(f"{k}={v}" for k, v in study.best_params.items())
        notify(
            title="Stacking sweep v2 complete",
            message=f"Best RMSE: {study.best_value:.4f}\n{best_params_str}",
        )
    else:
        print("No trials completed (all pruned as duplicates).")
        notify(
            title="Stacking sweep v2 complete",
            message=f"No new trials — all {pruned} pruned as duplicates.",
        )
