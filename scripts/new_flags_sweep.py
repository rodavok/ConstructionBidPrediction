"""
New Flags Sweep — Bias Correction × Location-Category Interaction

Pins the best config from stacking-sweep-v2 (run cd196af4, CV RMSE 0.3283):
  stack_models:             lightgbm, catboost, randomforest
  use_inflation:            False
  use_markup_target:        True
  recency_weight:           1.0
  train_from:               2022-01-01
  use_contractor_history:   True
  use_competition_intensity: True

Exhaustively tests all 4 combinations of the two new flags:
  use_bias_correction:      [True, False]
  use_loc_cat_interaction:  [True, False]

MLflow experiment: new-flags-sweep
Total runs: 4
"""
import itertools
import os
import sys
import urllib.request

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mlflow
from model import main, DEFAULT_CONFIG, _PROJECT_ROOT

TRACKING_URI = f"sqlite:///{os.path.join(_PROJECT_ROOT, 'mlflow.db')}"
EXPERIMENT_NAME = "new-flags-sweep"

# Pinned per-model hyperparams from stacking_sweep_v2 (best known runs)
LGBM_PARAMS = {
    "learning_rate": 0.0708,
    "num_leaves": 91,
    "max_depth": 10,
    "n_estimators": 700,
    "subsample": 0.857,
    "colsample_bytree": 0.768,
    "reg_lambda": 0.800,
}
CATBOOST_PARAMS = {
    "learning_rate": 0.147,
    "depth": 8,
    "iterations": 400,
    "l2_leaf_reg": 6.30,
}

# Everything pinned to cd196af4 (best CV result from stacking-sweep-v2)
PINNED_CONFIG = {
    "model": "stacking",
    "stack_models": ["lightgbm", "catboost", "randomforest"],
    "stacking_model_params": {
        "lightgbm": LGBM_PARAMS,
        "catboost": CATBOOST_PARAMS,
    },
    "use_inflation": False,
    "log_inflation": False,
    "inflation_lag_months": 0,
    "use_markup_target": True,
    "recency_weight": 1.0,
    "train_from": "2022-01-01",
    "use_contractor_history": True,
    "use_competition_intensity": True,
    "use_gpu": True,
    "tune": False,
}


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


if __name__ == "__main__":
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    combos = list(itertools.product([False, True], [False, True]))

    print("New Flags Sweep")
    print("Pinned config: lgbm+cat+rf, no inflation, markup=True, recency=1.0, train_from=2022-01-01")
    print(f"Search space: use_bias_correction × use_loc_cat_interaction = {len(combos)} runs")
    print()

    results = []
    for bias_correction, loc_cat in combos:
        config = DEFAULT_CONFIG.copy()
        config.update(PINNED_CONFIG)
        config["use_bias_correction"] = bias_correction
        config["use_loc_cat_interaction"] = loc_cat

        print(f"\n{'='*60}")
        print(f"RUN: bias_correction={bias_correction}  loc_cat_interaction={loc_cat}")
        print(f"{'='*60}")

        cv_rmse = main(config)
        results.append((cv_rmse, bias_correction, loc_cat))

    results.sort()

    print("\n" + "=" * 60)
    print("SWEEP COMPLETE — Results (sorted by CV RMSE):")
    print("=" * 60)
    header = f"  {'CV RMSE':>8}  {'bias_correction':>16}  {'loc_cat_interaction':>20}"
    print(header)
    for rmse, bc, lc in results:
        print(f"  {rmse:>8.4f}  {str(bc):>16}  {str(lc):>20}")

    best_rmse, best_bc, best_lc = results[0]
    summary = (
        f"Best CV RMSE: {best_rmse:.4f}\n"
        f"bias_correction={best_bc}, loc_cat_interaction={best_lc}\n"
        f"Baseline (neither flag): "
        + next(f"{r:.4f}" for r, bc, lc in results if not bc and not lc)
    )
    print(f"\n{summary}")
    notify(title="New flags sweep complete", message=summary)
