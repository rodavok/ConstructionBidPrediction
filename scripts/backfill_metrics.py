"""
Retroactively log derived metrics to existing MLflow runs.

Computes and logs:
  - cv_fold_trend   : cv_rmse_fold_last - cv_rmse_fold_1 (for runs with fold metrics)
  - test_pred_mean  : mean of log1p(total_bid) from submission CSV
  - test_pred_std   : std  of log1p(total_bid) from submission CSV
  - test_pred_iqr   : IQR  of log1p(total_bid) from submission CSV

MAE cannot be backfilled (fold predictions were not saved).
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import mlflow

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
mlflow.set_tracking_uri(f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}")

EXPERIMENT_NAMES = [
    "construction-price-prediction",
    "stacking-sweep",
    "model-drift-target-sweep",
    "feature-grid-sweep",
]

SUBMISSIONS_DIR = os.path.join(PROJECT_ROOT, "submissions")

# Build map: run_id_short -> submission file path
sub_map = {}
for path in glob.glob(os.path.join(SUBMISSIONS_DIR, "submission_*.csv")):
    basename = os.path.basename(path)
    short_id = basename.replace("submission_", "").replace(".csv", "")
    sub_map[short_id] = path

client = mlflow.tracking.MlflowClient()

trend_updated = 0
pred_updated = 0
skipped = 0

for exp_name in EXPERIMENT_NAMES:
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        print(f"  Experiment not found: {exp_name}")
        continue

    runs = client.search_runs(exp.experiment_id, max_results=1000)
    print(f"\n{exp_name}: {len(runs)} runs")

    for run in runs:
        rid = run.info.run_id
        short = rid[:8]
        metrics = run.data.metrics
        updated = False

        # --- cv_fold_trend ---
        if "cv_fold_trend" not in metrics:
            # Find highest fold index available
            fold_scores = {}
            for k, v in metrics.items():
                if k.startswith("cv_rmse_fold_"):
                    try:
                        idx = int(k.split("_")[-1])
                        fold_scores[idx] = v
                    except ValueError:
                        pass

            if fold_scores and len(fold_scores) >= 2:
                first = fold_scores[min(fold_scores)]
                last = fold_scores[max(fold_scores)]
                trend = last - first
                client.log_metric(rid, "cv_fold_trend", trend)
                trend_updated += 1
                updated = True

        # --- test prediction distribution from submission CSV ---
        if "test_pred_mean" not in metrics and short in sub_map:
            try:
                df = pd.read_csv(sub_map[short])
                preds = df["total_bid"].values
                log_preds = np.log1p(np.maximum(preds, 0))
                client.log_metric(rid, "test_pred_mean", float(np.mean(log_preds)))
                client.log_metric(rid, "test_pred_std", float(np.std(log_preds)))
                client.log_metric(rid, "test_pred_iqr",
                                  float(np.percentile(log_preds, 75) - np.percentile(log_preds, 25)))
                pred_updated += 1
                updated = True
            except Exception as e:
                print(f"  Warning: could not process {sub_map[short]}: {e}")

        if not updated:
            skipped += 1

print(f"\nDone.")
print(f"  cv_fold_trend logged:        {trend_updated} runs")
print(f"  test_pred_* logged:          {pred_updated} runs")
print(f"  Runs with nothing to update: {skipped}")
