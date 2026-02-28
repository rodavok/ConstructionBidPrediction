"""
OOF error analysis: analyze where the model fails.

Loads OOF predictions from a MLflow run artifact (oof_predictions.csv).
Falls back to re-training with the best run's config if the artifact is missing:
  stacking (lgbm + xgboost + catboost), no inflation, recency=1.0, train_from=2022-01-01

Usage:
    python scripts/error_analysis.py                          # default: run a932d990
    python scripts/error_analysis.py --run-id <full_run_id>
    python scripts/error_analysis.py --retrain               # skip MLflow lookup, always retrain
"""
import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import add_dummy_inflation_features, extract_date_features
from model import (
    build_unit_price_lookup,
    estimate_line_item_cost,
    aggregate_job_features,
    compute_contractor_win_history,
    compute_competition_intensity,
    prepare_features,
    get_backend,
    load_tuned_params,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..')

# Config that matches a932d990 (best Kaggle run)
BEST_RUN_CONFIG = {
    'model': 'stacking',
    'stack_models': ['lightgbm', 'xgboost', 'catboost'],
    'use_inflation': False,
    'recency_weight': 1.0,
    'train_from': '2022-01-01',
    'time_based_cv': True,
    'cv_splits': 5,
    'use_markup_target': False,
    'use_gpu': False,
    'use_loc_cat_interaction': False,
    'stacking_model_params': {},
}


def resolve_run_id(run_id_or_prefix):
    """Resolve a full run ID from a prefix or full ID. Returns None if not found."""
    import mlflow
    client = mlflow.tracking.MlflowClient()
    if len(run_id_or_prefix) == 32:
        return run_id_or_prefix
    # Search across all experiments for a run matching the prefix
    experiments = client.search_experiments()
    for exp in experiments:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"attributes.run_id LIKE '{run_id_or_prefix}%'",
            max_results=1,
        )
        if runs:
            return runs[0].info.run_id
    return None


def load_oof_from_mlflow(run_id_or_prefix):
    """Try to download oof_predictions.csv artifact from an MLflow run."""
    try:
        import mlflow
        client = mlflow.tracking.MlflowClient()
        run_id = resolve_run_id(run_id_or_prefix)
        if run_id is None:
            print(f"Run with prefix '{run_id_or_prefix}' not found in MLflow.")
            return None
        with tempfile.TemporaryDirectory() as tmp:
            local_path = client.download_artifacts(run_id, "oof_predictions.csv", tmp)
            df = pd.read_csv(local_path)
        print(f"Loaded OOF predictions from MLflow run {run_id[:8]} ({len(df):,} rows)")
        return df
    except Exception as e:
        print(f"Could not load OOF artifact from run '{run_id_or_prefix}': {e}")
        return None


def build_features_for_analysis(train_raw):
    """
    Build aggregated job-level features for the analysis breakdowns.
    Uses full dataset (slight leakage acceptable for diagnostics).
    """
    print("Building unit price lookup (full dataset — acceptable leakage for diagnostics)...")
    item_unit_prices, cat_unit_prices, unit_prices = build_unit_price_lookup(train_raw)

    print("Estimating line-item costs...")
    train = estimate_line_item_cost(train_raw, item_unit_prices, cat_unit_prices, unit_prices)

    print("Aggregating to job level...")
    agg = aggregate_job_features(train, is_train=True, log_inflation=False)

    agg = agg.rename(columns={'bid_date_first': 'bid_date'})
    agg = extract_date_features(agg)
    agg['bid_date'] = pd.to_datetime(agg['bid_date'])
    agg = agg.sort_values('bid_date').reset_index(drop=True)
    return agg


def retrain_and_get_oof(train_raw):
    """
    Re-generate OOF predictions by re-training with BEST_RUN_CONFIG.
    Returns a DataFrame with columns: job_id, contractor_id, actual_log, pred_log.
    """
    config = BEST_RUN_CONFIG
    print(f"\nRe-training with best-run config: {config['model']}, "
          f"no inflation, recency={config['recency_weight']}, "
          f"train_from={config['train_from']}")

    # Full dataset for price lookup; filtered for training
    train_raw = train_raw.copy()
    train_raw['bid_date_parsed'] = pd.to_datetime(train_raw['bid_date'])
    cutoff = pd.to_datetime(config['train_from'])
    price_lookup_data = train_raw.copy()
    train_model = train_raw[train_raw['bid_date_parsed'] >= cutoff].reset_index(drop=True)
    print(f"  Training on {train_model.groupby(['job_id','contractor_id']).ngroups:,} jobs "
          f"(from {config['train_from']}), price lookup uses all "
          f"{train_raw.groupby(['job_id','contractor_id']).ngroups:,}")

    # Inflation: no-op dummy columns (mirrors --no-inflation flag)
    train_model = add_dummy_inflation_features(train_model)
    price_lookup_data = add_dummy_inflation_features(price_lookup_data)

    # Contractor history and competition (computed on train_model only)
    print("Adding contractor history...")
    train_model = compute_contractor_win_history(train_model)
    print("Adding competition intensity...")
    train_model = compute_competition_intensity(train_model)

    # Job keys for CV splitting (sorted by date)
    job_keys = (train_model
                .groupby(['job_id', 'contractor_id'])
                .agg(bid_date_parsed=('bid_date_parsed', 'first'))
                .reset_index()
                .sort_values('bid_date_parsed')
                .reset_index(drop=True))
    job_keys['key'] = job_keys['job_id'] + '__' + job_keys['contractor_id']
    price_lookup_data['key'] = price_lookup_data['job_id'] + '__' + price_lookup_data['contractor_id']
    train_model['key'] = train_model['job_id'] + '__' + train_model['contractor_id']

    backend = get_backend('stacking', config)
    params = backend['get_default_params'](config)

    # Load tuned params if available (uses tuned lightgbm/xgboost/catboost params inside stacking)
    tuned_all = load_tuned_params()
    if tuned_all.get('stacking'):
        params = backend['get_tuned_params'](tuned_all['stacking'])

    cv = TimeSeriesSplit(n_splits=config['cv_splits'])
    print(f"\nGenerating OOF predictions with {config['cv_splits']}-fold time-based CV...")

    oof_job_ids, oof_contractor_ids, oof_actuals, oof_preds = [], [], [], []

    for fold, (tr_idx, val_idx) in enumerate(cv.split(job_keys)):
        tr_keys_df = job_keys.iloc[tr_idx]
        val_keys_df = job_keys.iloc[val_idx]

        tr_key_set = set(tr_keys_df['key'])
        val_key_set = set(val_keys_df['key'])

        train_fold = train_model[train_model['key'].isin(tr_key_set)].copy()
        val_fold = train_model[train_model['key'].isin(val_key_set)].copy()

        # Price lookup excludes the validation fold (no leakage)
        lookup_fold = price_lookup_data[~price_lookup_data['key'].isin(val_key_set)].copy()
        item_prices, cat_prices, unit_prices = build_unit_price_lookup(lookup_fold)

        train_fold = estimate_line_item_cost(train_fold, item_prices, cat_prices, unit_prices)
        val_fold = estimate_line_item_cost(val_fold, item_prices, cat_prices, unit_prices)

        train_agg = aggregate_job_features(train_fold, is_train=True, log_inflation=False)
        val_agg = aggregate_job_features(val_fold, is_train=True, log_inflation=False)

        X_tr, y_tr, _, _, feature_cols = prepare_features(train_agg.copy(), val_agg.copy())
        X_val, y_val, _, _, _ = prepare_features(val_agg.copy(), train_agg.copy())
        X_val = X_val.reindex(columns=feature_cols, fill_value=0)

        model = backend['train'](X_tr.values, y_tr, X_val.values, y_val, params, None)
        y_pred = backend['predict'](model, X_val.values)

        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
        fold_dates = val_keys_df['bid_date_parsed']
        print(f"  Fold {fold+1}: RMSE={rmse:.4f}  n={len(val_idx):,}  "
              f"dates={fold_dates.min().date()} to {fold_dates.max().date()}")

        oof_job_ids.extend(val_keys_df['job_id'].tolist())
        oof_contractor_ids.extend(val_keys_df['contractor_id'].tolist())
        oof_actuals.extend(y_val.tolist())
        oof_preds.extend(y_pred.tolist())

    oof_df = pd.DataFrame({
        'job_id': oof_job_ids,
        'contractor_id': oof_contractor_ids,
        'actual_log': oof_actuals,
        'pred_log': oof_preds,
    })

    overall_rmse = np.sqrt(np.mean((oof_df['actual_log'] - oof_df['pred_log']) ** 2))
    print(f"\n  Overall OOF RMSE: {overall_rmse:.4f}")
    return oof_df


def run_analysis(oof_df, agg):
    """
    Join OOF predictions with aggregated features and print all breakdowns.
    agg: job-level features DataFrame (from build_features_for_analysis)
    oof_df: DataFrame with job_id, contractor_id, actual_log, pred_log
    """
    # Join OOF predictions onto feature DataFrame
    analyzed = agg.merge(oof_df, on=['job_id', 'contractor_id'], how='inner')
    print(f"\nMatched {len(analyzed):,} of {len(oof_df):,} OOF rows to features")

    analyzed['error'] = analyzed['pred_log'] - analyzed['actual_log']
    analyzed['abs_error'] = analyzed['error'].abs()
    analyzed['pred'] = np.expm1(analyzed['pred_log'])
    analyzed['actual'] = np.expm1(analyzed['actual_log'])
    analyzed['pct_error'] = (analyzed['pred'] - analyzed['actual']) / analyzed['actual'] * 100

    print(f"\n{'='*60}")
    print("ERROR ANALYSIS")
    print(f"{'='*60}")
    print(f"Total bids analyzed: {len(analyzed):,}")
    print(f"Overall RMSE (log): {np.sqrt(np.mean(analyzed['error']**2)):.4f}")
    print(f"Median absolute % error: {analyzed['pct_error'].abs().median():.1f}%")

    # ---- ERROR BY YEAR ----
    print(f"\n{'='*60}")
    print("ERROR BY YEAR")
    print(f"{'='*60}")
    year_errors = analyzed.groupby('year').agg(
        mean_abs_error=('abs_error', 'mean'),
        median_pct_error=('pct_error', lambda x: x.abs().median()),
        count=('job_id', 'count'),
    ).round(3)
    print(year_errors.to_string())

    # ---- WORST PREDICTIONS ----
    print(f"\n{'='*60}")
    print("WORST PREDICTIONS (highest absolute log error)")
    print(f"{'='*60}")
    worst = analyzed.nlargest(20, 'abs_error')[
        ['job_id', 'job_category_description_first', 'primary_location_first',
         'actual', 'pred', 'pct_error', 'estimated_cost_sum', 'year']
    ]
    worst.columns = ['job_id', 'category', 'location', 'actual', 'predicted', 'pct_error', 'est_cost', 'year']
    print(worst.to_string(index=False))

    # ---- ERROR BY CATEGORY ----
    print(f"\n{'='*60}")
    print("ERROR BY CATEGORY")
    print(f"{'='*60}")
    cat_errors = analyzed.groupby('job_category_description_first').agg(
        mean_abs_error=('abs_error', 'mean'),
        median_pct_error=('pct_error', lambda x: x.abs().median()),
        count=('abs_error', 'count'),
    ).round(3).sort_values('mean_abs_error', ascending=False)
    print(cat_errors.to_string())

    # ---- ERROR BY LOCATION ----
    print(f"\n{'='*60}")
    print("ERROR BY LOCATION (min 20 bids)")
    print(f"{'='*60}")
    loc_errors = analyzed.groupby('primary_location_first').agg(
        mean_abs_error=('abs_error', 'mean'),
        median_pct_error=('pct_error', lambda x: x.abs().median()),
        count=('abs_error', 'count'),
    ).round(3)
    loc_errors = loc_errors[loc_errors['count'] >= 20].sort_values('mean_abs_error', ascending=False)
    print(loc_errors.head(15).to_string())

    # ---- ERROR BY BID SIZE ----
    print(f"\n{'='*60}")
    print("ERROR BY BID SIZE (quintiles)")
    print(f"{'='*60}")
    analyzed['bid_quintile'] = pd.qcut(
        analyzed['actual'], 5,
        labels=['Q1 (smallest)', 'Q2', 'Q3', 'Q4', 'Q5 (largest)']
    )
    size_errors = analyzed.groupby('bid_quintile', observed=True).agg(
        mean_abs_error=('abs_error', 'mean'),
        median_pct_error=('pct_error', lambda x: x.abs().median()),
        min_bid=('actual', 'min'),
        max_bid=('actual', 'max'),
    ).round(3)
    print(size_errors.to_string())

    # ---- BIAS ----
    print(f"\n{'='*60}")
    print("PREDICTION BIAS")
    print(f"{'='*60}")
    over = (analyzed['error'] > 0).sum()
    under = (analyzed['error'] < 0).sum()
    mean_err = analyzed['error'].mean()
    print(f"Over-predictions:  {over:,} ({over/len(analyzed)*100:.1f}%)")
    print(f"Under-predictions: {under:,} ({under/len(analyzed)*100:.1f}%)")
    print(f"Mean error (log):  {mean_err:.4f} ({'over' if mean_err > 0 else 'under'}-predicting on average)")

    # ---- ESTIMATED COST ACCURACY ----
    print(f"\n{'='*60}")
    print("ESTIMATED COST vs ACTUAL")
    print(f"{'='*60}")
    analyzed['est_ratio'] = analyzed['estimated_cost_sum'] / analyzed['actual']
    print(f"Median estimate/actual ratio: {analyzed['est_ratio'].median():.2f}")
    print(f"Correlation (est_cost, actual): {analyzed['estimated_cost_sum'].corr(analyzed['actual']):.3f}")

    bad_est = analyzed[analyzed['est_ratio'] < 0.5]
    print(f"\nBids where estimate < 50% of actual: {len(bad_est):,}")
    if len(bad_est) > 0:
        print("Categories:")
        print(bad_est['job_category_description_first'].value_counts().head(10).to_string())

    # ---- MARKUP DISTRIBUTION ----
    print(f"\n{'='*60}")
    print("MARKUP DISTRIBUTION (bid / estimated_cost)")
    print(f"{'='*60}")
    analyzed['markup'] = analyzed['actual'] / analyzed['estimated_cost_sum'].clip(lower=1)
    print(analyzed['markup'].describe().round(3).to_string())
    print(f"\nMarkup by category:")
    markup_by_cat = analyzed.groupby('job_category_description_first')['markup'].median().sort_values(ascending=False)
    print(markup_by_cat.head(15).round(3).to_string())

    # ---- ERROR BY CONTRACTOR ----
    print(f"\n{'='*60}")
    print("ERROR BY CONTRACTOR (top 15 by error, min 10 bids)")
    print(f"{'='*60}")
    contractor_errors = analyzed.groupby('contractor_id').agg(
        mean_abs_error=('abs_error', 'mean'),
        median_pct_error=('pct_error', lambda x: x.abs().median()),
        count=('abs_error', 'count'),
        mean_bid=('actual', 'mean'),
    ).round(3)
    contractor_errors = (
        contractor_errors[contractor_errors['count'] >= 10]
        .sort_values('mean_abs_error', ascending=False)
    )
    print(contractor_errors.head(15).to_string())

    # ---- ERROR BY COMPETITION (num_bidders) ----
    print(f"\n{'='*60}")
    print("ERROR BY COMPETITION INTENSITY (num_bidders)")
    print(f"{'='*60}")
    analyzed['bidder_bucket'] = pd.cut(
        analyzed['num_bidders_first'],
        bins=[0, 2, 4, 6, 10, 999],
        labels=['1-2', '3-4', '5-6', '7-10', '10+']
    )
    comp_errors = analyzed.groupby('bidder_bucket', observed=True).agg(
        mean_abs_error=('abs_error', 'mean'),
        median_pct_error=('pct_error', lambda x: x.abs().median()),
        count=('abs_error', 'count'),
    ).round(3)
    print(comp_errors.to_string())

    # ---- ERROR BY CONTRACTOR EXPERIENCE ----
    print(f"\n{'='*60}")
    print("ERROR BY CONTRACTOR PRIOR WINS")
    print(f"{'='*60}")
    analyzed['wins_bucket'] = pd.cut(
        analyzed['contractor_prior_wins_first'],
        bins=[-1, 0, 5, 15, 40, 9999],
        labels=['0 (new)', '1-5', '6-15', '16-40', '40+']
    )
    wins_errors = analyzed.groupby('wins_bucket', observed=True).agg(
        mean_abs_error=('abs_error', 'mean'),
        median_pct_error=('pct_error', lambda x: x.abs().median()),
        count=('abs_error', 'count'),
    ).round(3)
    print(wins_errors.to_string())


def main():
    parser = argparse.ArgumentParser(description="OOF error analysis")
    parser.add_argument(
        '--run-id', default='a932d990',
        help='MLflow run ID or prefix to load OOF predictions from (default: a932d990)'
    )
    parser.add_argument(
        '--retrain', action='store_true',
        help='Skip MLflow lookup and always re-train with best-run config'
    )
    args = parser.parse_args()

    print("Loading raw training data...")
    train_raw = pd.read_csv(os.path.join(DATA_DIR, 'raw_train.csv'))

    # --- Load or generate OOF predictions ---
    oof_df = None
    if not args.retrain:
        oof_df = load_oof_from_mlflow(args.run_id)

    if oof_df is None:
        oof_df = retrain_and_get_oof(train_raw.copy())

    # Build aggregated features for analysis breakdowns (always from full dataset)
    print("\nBuilding features for analysis...")
    train_enriched = add_dummy_inflation_features(train_raw.copy())
    train_enriched = compute_contractor_win_history(train_enriched)
    train_enriched = compute_competition_intensity(train_enriched)
    agg = build_features_for_analysis(train_enriched)

    run_analysis(oof_df, agg)


if __name__ == "__main__":
    main()
