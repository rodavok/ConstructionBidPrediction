"""
Optuna sweep over stacking model combinations, per-model hyperparameters,
and feature flags.

Motivation: the previous sweep (model-drift-target-sweep) found that stacking
outperforms single models, and that markup-ratio + train_from are the strongest
levers. This sweep attacks three dimensions not yet systematically explored:

  1. Stack composition  — which subset of lgbm/xgb/catboost/elasticnet to include
  2. Base-model tuning  — per-model learning rate, depth, n_estimators, regularization
                          (conditional on whether that model is in the stack)
  3. Feature flags      — markup-ratio, train_from, contractor-history,
                          competition-intensity, and recency-weight interact with
                          the stack; sweep them jointly

Pinned for all trials (best config from model-drift-target-sweep):
  use_inflation:           True
  inflation_lag_months:    3
  log_inflation:           True
  use_gpu:                 True
  tune:                    False

Sampler: TPE (default) — appropriate for continuous + conditional hyperparams.
MLflowCallback logs each trial into the 'stacking-sweep' MLflow experiment.
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
STUDY_NAME = "stacking-sweep"

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


@mlflc.track_in_mlflow()
def objective(trial):
    # --- Stack composition ---
    use_lgbm = trial.suggest_categorical("use_lgbm", [True, False])
    use_xgb = trial.suggest_categorical("use_xgb", [True, False])
    use_catboost = trial.suggest_categorical("use_catboost", [True, False])
    use_elasticnet = trial.suggest_categorical("use_elasticnet", [True, False])

    stack_models = []
    if use_lgbm:
        stack_models.append("lightgbm")
    if use_xgb:
        stack_models.append("xgboost")
    if use_catboost:
        stack_models.append("catboost")
    if use_elasticnet:
        stack_models.append("elasticnet")

    if len(stack_models) < 2:
        raise optuna.exceptions.TrialPruned()

    # --- Feature flags ---
    use_markup = trial.suggest_categorical("use_markup_target", [True, False])
    train_from = trial.suggest_categorical(
        "train_from", [None, "2022-01-01", "2022-07-01"]
    )
    use_contractor = trial.suggest_categorical("use_contractor_history", [True, False])
    use_competition = trial.suggest_categorical("use_competition_intensity", [True, False])
    recency = trial.suggest_categorical("recency_weight", [0.0, 0.5, 1.0, 2.0])

    # --- Per-model hyperparameters (only sampled when model is in the stack) ---
    stacking_model_params = {}

    if use_lgbm:
        stacking_model_params["lightgbm"] = {
            "learning_rate": trial.suggest_float("lgbm_lr", 0.03, 0.15, log=True),
            "num_leaves": trial.suggest_int("lgbm_num_leaves", 16, 128),
            "max_depth": trial.suggest_int("lgbm_max_depth", 4, 10),
            "n_estimators": trial.suggest_int("lgbm_n_estimators", 200, 1000, step=100),
            "subsample": trial.suggest_float("lgbm_subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("lgbm_colsample", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("lgbm_reg_lambda", 0.0, 2.0),
        }

    if use_xgb:
        stacking_model_params["xgboost"] = {
            "learning_rate": trial.suggest_float("xgb_lr", 0.03, 0.15, log=True),
            "max_depth": trial.suggest_int("xgb_max_depth", 4, 10),
            "n_estimators": trial.suggest_int("xgb_n_estimators", 200, 1000, step=100),
            "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_colsample", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("xgb_reg_lambda", 0.0, 2.0),
        }

    if use_catboost:
        stacking_model_params["catboost"] = {
            "learning_rate": trial.suggest_float("cat_lr", 0.03, 0.15, log=True),
            "depth": trial.suggest_int("cat_depth", 4, 10),
            "iterations": trial.suggest_int("cat_iterations", 200, 1000, step=100),
            "l2_leaf_reg": trial.suggest_float("cat_l2_leaf_reg", 1.0, 10.0),
        }

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
        # Pinned from model-drift-target-sweep
        "use_inflation": True,
        "inflation_lag_months": 3,
        "log_inflation": True,
        "use_gpu": True,
        "tune": False,
    })

    return main(config)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100, help="Number of Optuna trials (default: 100)")
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(STUDY_NAME)

    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
    )

    print(f"Starting stacking sweep: up to {args.trials} trials (TPE sampler)")
    print("Sweep axes:")
    print("  Stack composition: any ≥2 subset of {lightgbm, xgboost, catboost, elasticnet}")
    print("  Per-model params:  lr, depth/num_leaves, n_estimators, subsample, colsample, reg_lambda")
    print("  use_markup_target: [True, False]")
    print("  train_from:        [None, 2022-01-01, 2022-07-01]")
    print("  use_contractor_history: [True, False]")
    print("  use_competition_intensity: [True, False]")
    print("  recency_weight:    [0.0, 0.5, 1.0, 2.0]")
    print("Pinned: inflation=True, lag=3, log_inflation=True, gpu=True, tune=False")
    print()

    study.optimize(objective, n_trials=args.trials, callbacks=[mlflc])

    print("\n" + "=" * 60)
    print("SWEEP COMPLETE")
    print(f"Best CV RMSE: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    best_params_str = ", ".join(f"{k}={v}" for k, v in study.best_params.items())
    notify(
        title="Stacking sweep complete",
        message=f"Best RMSE: {study.best_value:.4f}\n{best_params_str}",
    )
