"""
Grid sweep over model choice, drift-mitigation, and target framing.

Motivation: the previous sweep (feature-grid-sweep) exhausted feature-flag
combinations and found only 0.009 RMSE spread across 172 trials — those
dimensions are saturated.  This sweep attacks three dimensions that have real
signal in the main-experiment runs but were never systematically tested:

  1. Model         — catboost/xgboost outperform lightgbm by ~0.020 RMSE
  2. Drift control — recency_weight and train_from interact; fold-5 degrades on
                     catboost/stacking but not lightgbm, suggesting concept
                     drift that weighting/windowing may mitigate
  3. Target frame  — use_markup_target predicts log(bid/estimated_cost) instead
                     of log(bid); never tested in any prior run

Search space (72 combinations):
  model:            [lightgbm, catboost, xgboost]
  recency_weight:   [0.0, 0.5, 1.0, 2.0]
  train_from:       [None, '2022-01-01', '2023-01-01']
  use_markup_target:[True, False]

Pinned for all trials (best config from feature-grid-sweep):
  use_inflation:           True
  inflation_lag_months:    3
  log_inflation:           True
  use_competition_intensity: True
  use_contractor_history:  True
  use_gpu:                 True
  tune:                    False

Note: MLflowCallback logs each trial into a new MLflow experiment whose name
matches the study_name ('model-drift-target-sweep'), not the main experiment.
"""

import os
import sys
import urllib.request

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


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

import optuna
from optuna.integration.mlflow import MLflowCallback

import mlflow
from model import main, DEFAULT_CONFIG, _PROJECT_ROOT

optuna.logging.set_verbosity(optuna.logging.WARNING)

TRACKING_URI = f"sqlite:///{os.path.join(_PROJECT_ROOT, 'mlflow.db')}"
STUDY_NAME = "model-drift-target-sweep"

mlflc = MLflowCallback(
    tracking_uri=TRACKING_URI,
    metric_name="cv_rmse_mean",
    mlflow_kwargs={"experiment_name": STUDY_NAME},
)


@mlflc.track_in_mlflow()
def objective(trial):
    model = trial.suggest_categorical("model", ["lightgbm", "catboost", "xgboost"])
    recency_weight = trial.suggest_categorical("recency_weight", [0.0, 0.5, 1.0, 2.0])
    train_from = trial.suggest_categorical("train_from", [None, "2022-01-01", "2023-01-01"])
    use_markup_target = trial.suggest_categorical("use_markup_target", [True, False])

    config = DEFAULT_CONFIG.copy()
    config["model"] = model
    config["use_gpu"] = True
    config["tune"] = False
    # Pinned best feature config from feature-grid-sweep
    config["use_inflation"] = True
    config["inflation_lag_months"] = 3
    config["log_inflation"] = True
    config["use_competition_intensity"] = True
    config["use_contractor_history"] = True
    # Sweep axes
    config["recency_weight"] = recency_weight
    config["train_from"] = train_from
    config["use_markup_target"] = use_markup_target

    return main(config)


if __name__ == "__main__":
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(STUDY_NAME)

    search_space = {
        "model": ["lightgbm", "catboost", "xgboost"],
        "recency_weight": [0.0, 0.5, 1.0, 2.0],
        "train_from": [None, "2022-01-01", "2023-01-01"],
        "use_markup_target": [True, False],
    }

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.GridSampler(search_space),
        study_name=STUDY_NAME,
    )

    n_trials = 3 * 4 * 3 * 2  # 72
    print(f"Starting grid sweep: {n_trials} trials")
    print(f"Axes: model × recency_weight × train_from × use_markup_target")
    print(f"Pinned: inflation=True, lag=3, log_inflation=True, competition=True, contractor=True, gpu=True, tune=False")
    print()

    study.optimize(objective, n_trials=n_trials, callbacks=[mlflc])

    print("\n" + "=" * 60)
    print("SWEEP COMPLETE")
    print(f"Best CV RMSE: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    best_params_str = ", ".join(f"{k}={v}" for k, v in study.best_params.items())
    notify(
        title="Optuna sweep complete",
        message=f"Best RMSE: {study.best_value:.4f}\n{best_params_str}",
    )
