"""
Random Forest Sweep — Standalone RF Characterization

Goal: Understand how RandomForest behaves across its own hyperparameter space
and across the same feature/context flags we test with GBM models.

Sweep axes:
  RF hyperparams:
    n_estimators:      [200, 300, 500, 700]
    max_depth:         [None, 10, 15, 20]
    min_samples_split: [2, 5, 10]
    min_samples_leaf:  [1, 2, 5]
    max_features:      ['sqrt', 'log2', 0.5, 0.7]
    bootstrap:         [True, False]

  Feature/context flags (same axes as stacking-sweep-v2):
    use_inflation:     [True, False]
      log_inflation:   [True, False]   (only when inflation=True)
      inflation_lag:   [0, 3, 6]       (only when inflation=True)
    use_markup_target: [True, False]
    recency_weight:    [0.0, 0.5, 1.0, 2.0]
    train_from:        [None, '2022-01-01', '2022-07-01']
    use_contractor_history:    [True, False]
    use_competition_intensity: [True, False]

Discrete space: 4×4×3×3×4×2 × 7×2×4×3×2×2 = 1,152 × 672 = 774,144 combinations
Default trials: 80 (RandomSampler for even coverage)

MLflow experiment: randomforest-sweep
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
STUDY_NAME = "randomforest-sweep"

# Keys used to identify a unique combo for duplicate detection
COMBO_KEYS = [
    "n_estimators",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "max_features",
    "bootstrap",
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


def load_existing_combos() -> set:
    """Load previously run RF combos from the randomforest-sweep experiment."""
    existing = set()
    client = mlflow.tracking.MlflowClient(tracking_uri=TRACKING_URI)

    try:
        exp = client.get_experiment_by_name(STUDY_NAME)
    except Exception:
        return existing

    if exp is None:
        return existing

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        max_results=10000,
    )

    for run in runs:
        p = run.data.params
        # Only process RF runs
        if p.get("model") != "randomforest":
            continue
        # Skip runs missing key RF params
        if "n_estimators" not in p:
            continue

        try:
            combo = make_combo({
                "n_estimators": int(p.get("n_estimators", 500)),
                "max_depth": None if p.get("max_depth") in ("None", None) else int(p["max_depth"]),
                "min_samples_split": int(p.get("min_samples_split", 5)),
                "min_samples_leaf": int(p.get("min_samples_leaf", 2)),
                "max_features": p.get("max_features", "sqrt"),
                "bootstrap": p.get("bootstrap", "True") == "True",
                "use_inflation": p.get("use_inflation", "True") == "True",
                "log_inflation": p.get("log_inflation", "False") == "True",
                "inflation_lag_months": int(p.get("inflation_lag_months", "0")),
                "use_markup_target": p.get("use_markup_target", "False") == "True",
                "recency_weight": float(p.get("recency_weight", "0.0")),
                "train_from": None if p.get("train_from") in ("None", None) else p["train_from"],
                "use_contractor_history": p.get("use_contractor_history", "False") == "True",
                "use_competition_intensity": p.get("use_competition_intensity", "False") == "True",
            })
            existing.add(combo)
        except (ValueError, TypeError):
            continue

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
    # --- RF hyperparameters ---
    n_estimators = trial.suggest_categorical("n_estimators", [200, 300, 500, 700])
    max_depth = trial.suggest_categorical("max_depth", [None, 10, 15, 20])
    min_samples_split = trial.suggest_categorical("min_samples_split", [2, 5, 10])
    min_samples_leaf = trial.suggest_categorical("min_samples_leaf", [1, 2, 5])
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7])
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])

    # --- Inflation config ---
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
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "bootstrap": bootstrap,
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

    # RF model params injected directly (bypasses on-disk tuned_params.json)
    rf_params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "bootstrap": bootstrap,
        "n_jobs": -1,
        "random_state": 42,
    }

    config = DEFAULT_CONFIG.copy()
    config.update({
        "model": "randomforest",
        "model_params": rf_params,
        # Also expose RF params as top-level config keys so they are logged
        # cleanly by mlflow.log_params(config) and readable in load_existing_combos
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "bootstrap": bootstrap,
        "use_markup_target": use_markup,
        "train_from": train_from,
        "use_contractor_history": use_contractor,
        "use_competition_intensity": use_competition,
        "recency_weight": recency,
        "use_inflation": use_inflation,
        "inflation_lag_months": inflation_lag,
        "log_inflation": log_inflation,
        "use_gpu": False,  # RF has no GPU path
        "tune": False,
    })

    return main(config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Random Forest sweep — standalone RF characterization")
    parser.add_argument(
        "--trials", type=int, default=80,
        help="Number of Optuna trials (default: 80)"
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(STUDY_NAME)

    print("Loading existing combos from randomforest-sweep experiment...")
    _seen_combos.update(load_existing_combos())
    print(f"  {len(_seen_combos)} existing combos loaded — these will be skipped")
    print()

    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
        sampler=optuna.samplers.RandomSampler(),
    )

    print(f"Starting randomforest sweep: up to {args.trials} trials (RandomSampler)")
    print("RF hyperparams swept:")
    print("  n_estimators:      [200, 300, 500, 700]")
    print("  max_depth:         [None, 10, 15, 20]")
    print("  min_samples_split: [2, 5, 10]")
    print("  min_samples_leaf:  [1, 2, 5]")
    print("  max_features:      ['sqrt', 'log2', 0.5, 0.7]")
    print("  bootstrap:         [True, False]")
    print("Feature/context flags swept:")
    print("  use_inflation:     [True, False]")
    print("    log_inflation:   [True, False]  (when inflation=True)")
    print("    inflation_lag:   [0, 3, 6]      (when inflation=True)")
    print("  use_markup_target: [True, False]")
    print("  recency_weight:    [0.0, 0.5, 1.0, 2.0]")
    print("  train_from:        [None, 2022-01-01, 2022-07-01]")
    print("  use_contractor_history:    [True, False]")
    print("  use_competition_intensity: [True, False]")
    print(f"Total discrete space: 774,144 combinations ({args.trials/774144*100:.2f}% coverage)")
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
            title="RF sweep complete",
            message=f"Best RMSE: {study.best_value:.4f}\n{best_params_str}",
        )
    else:
        print("No trials completed (all pruned as duplicates).")
        notify(
            title="RF sweep complete",
            message=f"No new trials — all {pruned} pruned as duplicates.",
        )
