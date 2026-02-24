"""
OOF error analysis: generate out-of-fold predictions and analyze where the model fails.

Uses CatBoost with tuned params (best single model per MLflow runs).
Feature config: inflation=True, lag=0, history=False, competition=False
(mirrors the top illustrious-grub-960 config, simplified for interpretability).

Run with: python scripts/error_analysis.py
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from catboost import CatBoostRegressor
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import add_inflation_features

TUNED_PARAMS_FILE = os.path.join(os.path.dirname(__file__), '..', 'tuned_params.json')


def load_tuned_catboost():
    if os.path.exists(TUNED_PARAMS_FILE):
        with open(TUNED_PARAMS_FILE) as f:
            params = json.load(f).get('catboost', {})
        if params:
            return params
    return {}


def main():
    print("Loading data...")
    train = pd.read_csv("raw_train.csv")
    train = add_inflation_features(train)

    # Build unit price lookup (slight leakage — acceptable for error analysis)
    train['unit_price'] = train['amount']
    train['unit_price_adj'] = train['unit_price'] * train['inflation_factor']

    valid = train['unit_price_adj'].notna() & (train['unit_price_adj'] > 0) & (train['unit_price_adj'] < 1e7)
    item_prices = (
        train[valid]
        .groupby(['pay_item_description', 'unit_english_id'])['unit_price_adj']
        .median()
    )

    train = train.merge(
        item_prices.rename('est_unit_price'),
        left_on=['pay_item_description', 'unit_english_id'],
        right_index=True,
        how='left',
    )
    train['estimated_cost'] = train['quantity'] * train['est_unit_price'].fillna(
        train['unit_price_adj'].median()
    )

    # Aggregate to job level
    agg = train.groupby(['job_id', 'contractor_id']).agg({
        'total_bid': 'first',
        'bid_date': 'first',
        'job_category_description': 'first',
        'primary_location': 'first',
        'inflation_factor': 'first',
        'estimated_cost': ['sum', 'mean', 'std'],
        'quantity': ['sum', 'mean', 'std'],
        'num_pay_items': 'first',
    }).reset_index()
    agg.columns = ['_'.join(c).strip('_') for c in agg.columns]

    agg['bid_date'] = pd.to_datetime(agg['bid_date_first'])
    agg['year'] = agg['bid_date'].dt.year
    agg['month'] = agg['bid_date'].dt.month
    agg['day_of_week'] = agg['bid_date'].dt.dayofweek
    agg['day_of_year'] = agg['bid_date'].dt.dayofyear
    agg['log_bid'] = np.log1p(agg['total_bid_first'])

    agg['category_code'] = agg['job_category_description_first'].astype('category').cat.codes
    agg['location_code'] = agg['primary_location_first'].astype('category').cat.codes

    feature_cols = [
        'estimated_cost_sum', 'estimated_cost_mean', 'estimated_cost_std',
        'quantity_sum', 'quantity_mean', 'quantity_std',
        'num_pay_items_first', 'inflation_factor_first',
        'year', 'month', 'day_of_week', 'day_of_year',
        'category_code', 'location_code',
    ]

    agg = agg.sort_values('bid_date').reset_index(drop=True)
    X = agg[feature_cols].fillna(0)
    y = agg['log_bid'].values

    # Load tuned CatBoost params
    tuned = load_tuned_catboost()
    cb_params = dict(
        loss_function='RMSE',
        depth=int(tuned.get('depth', 6)),
        learning_rate=float(tuned.get('learning_rate', 0.05)),
        iterations=int(tuned.get('iterations', 500)),
        l2_leaf_reg=float(tuned.get('l2_leaf_reg', 3.0)),
        bagging_temperature=float(tuned.get('bagging_temperature', 1.0)),
        random_seed=42,
        verbose=False,
    )
    print(f"CatBoost params: {cb_params}")

    # OOF predictions with time-based CV
    print("\nGenerating OOF predictions with time-based CV...")
    cv = TimeSeriesSplit(n_splits=5)
    oof_preds = np.zeros(len(y))
    fold_rmses = []

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X)):
        X_tr, X_val = X.iloc[tr_idx].values, X.iloc[val_idx].values
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = CatBoostRegressor(**cb_params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50)
        oof_preds[val_idx] = model.predict(X_val)

        rmse = np.sqrt(np.mean((y_val - oof_preds[val_idx]) ** 2))
        fold_rmses.append(rmse)

        # Show time range for this fold
        fold_dates = agg.iloc[val_idx]['bid_date']
        print(
            f"  Fold {fold+1}: RMSE={rmse:.4f}  "
            f"n={len(val_idx):,}  "
            f"dates={fold_dates.min().date()} to {fold_dates.max().date()}"
        )

    print(f"\n  Overall OOF RMSE: {np.sqrt(np.mean((y[oof_preds != 0] - oof_preds[oof_preds != 0])**2)):.4f}")

    # Attach predictions — only analyze rows that got OOF predictions
    agg['pred_log'] = oof_preds
    agg['pred'] = np.expm1(oof_preds)
    agg['actual'] = agg['total_bid_first']
    agg['error'] = agg['pred_log'] - agg['log_bid']
    agg['abs_error'] = np.abs(agg['error'])
    agg['pct_error'] = (agg['pred'] - agg['actual']) / agg['actual'] * 100
    analyzed = agg[agg['pred_log'] != 0].copy()

    print(f"\n{'='*60}")
    print("ERROR ANALYSIS")
    print(f"{'='*60}")
    print(f"Total bids analyzed: {len(analyzed):,}")
    print(f"Overall RMSE (log): {np.sqrt(np.mean(analyzed['error']**2)):.4f}")
    print(f"Median absolute % error: {analyzed['pct_error'].abs().median():.1f}%")

    # ---- TEMPORAL: error by fold (CV fold maps to time period) ----
    print(f"\n{'='*60}")
    print("ERROR BY CV FOLD (temporal trend)")
    print(f"{'='*60}")
    fold_splits = list(cv.split(X))
    for fold_i, (_, val_idx) in enumerate(fold_splits):
        fold_data = analyzed.loc[analyzed.index.isin(val_idx)]
        if fold_data.empty:
            continue
        rmse = fold_rmses[fold_i]
        dates = agg.iloc[val_idx]['bid_date']
        print(
            f"  Fold {fold_i+1}: RMSE={rmse:.4f}  "
            f"median_%err={fold_data['pct_error'].abs().median():.1f}%  "
            f"n={len(fold_data):,}  "
            f"({dates.min().date()} to {dates.max().date()})"
        )

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

    # ---- ERROR BY CONTRACTOR (top contractors by volume) ----
    print(f"\n{'='*60}")
    print("ERROR BY CONTRACTOR (top 15 by bid count, min 10 bids)")
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


if __name__ == "__main__":
    main()
