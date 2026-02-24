"""
Quick error analysis: generate OOF predictions and analyze where the model fails.
Run with: python scripts/error_analysis.py
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import sys
sys.path.insert(0, 'src')
from utils import add_inflation_features

def main():
    # Load line-item data
    print("Loading data...")
    train = pd.read_csv("raw_train.csv")
    train = add_inflation_features(train)

    # Build unit price lookup from ALL training data (for simplicity - slight leakage but OK for error analysis)
    train['unit_price'] = train['amount'] / train['quantity'].replace(0, np.nan)
    train['unit_price_adj'] = train['unit_price'] * train['inflation_factor']

    valid = train['unit_price_adj'].notna() & (train['unit_price_adj'] > 0) & (train['unit_price_adj'] < 1e7)
    item_prices = train[valid].groupby(['pay_item_description', 'unit_english_id'])['unit_price_adj'].median()

    # Estimate costs
    train = train.merge(item_prices.rename('est_unit_price'),
                        left_on=['pay_item_description', 'unit_english_id'],
                        right_index=True, how='left')
    train['estimated_cost'] = train['quantity'] * train['est_unit_price'].fillna(train['unit_price_adj'].median())

    # Aggregate to job level
    agg = train.groupby(['job_id', 'contractor_id']).agg({
        'total_bid': 'first',
        'bid_date': 'first',
        'job_category_description': 'first',
        'primary_location': 'first',
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

    # Encode categoricals
    agg['category_code'] = agg['job_category_description_first'].astype('category').cat.codes
    agg['location_code'] = agg['primary_location_first'].astype('category').cat.codes

    feature_cols = ['estimated_cost_sum', 'estimated_cost_mean', 'estimated_cost_std',
                    'quantity_sum', 'quantity_mean', 'quantity_std',
                    'num_pay_items_first', 'year', 'month', 'day_of_week', 'day_of_year',
                    'category_code', 'location_code']

    X = agg[feature_cols].fillna(0)
    y = agg['log_bid'].values

    # Time-based CV to get OOF predictions
    print("\nGenerating OOF predictions with time-based CV...")
    agg = agg.sort_values('bid_date').reset_index(drop=True)
    X = X.loc[agg.index]
    y = y[agg.index]

    oof_preds = np.zeros(len(y))
    cv = TimeSeriesSplit(n_splits=5)

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63, verbose=-1)
        model.fit(X_tr, y_tr)
        oof_preds[val_idx] = model.predict(X_val)

        rmse = np.sqrt(np.mean((y_val - oof_preds[val_idx])**2))
        print(f"  Fold {fold+1}: RMSE = {rmse:.4f}")

    # Add predictions to dataframe
    agg['pred_log'] = oof_preds
    agg['pred'] = np.expm1(oof_preds)
    agg['actual'] = agg['total_bid_first']
    agg['error'] = agg['pred_log'] - agg['log_bid']
    agg['abs_error'] = np.abs(agg['error'])
    agg['pct_error'] = (agg['pred'] - agg['actual']) / agg['actual'] * 100

    # Only analyze rows that got OOF predictions (not first fold's training data)
    analyzed = agg[agg['pred_log'] != 0].copy()

    print(f"\n{'='*60}")
    print("ERROR ANALYSIS")
    print(f"{'='*60}")
    print(f"Total bids analyzed: {len(analyzed):,}")
    print(f"Overall RMSE (log): {np.sqrt(np.mean(analyzed['error']**2)):.4f}")
    print(f"Median absolute % error: {analyzed['pct_error'].abs().median():.1f}%")

    # Worst predictions
    print(f"\n{'='*60}")
    print("WORST PREDICTIONS (highest absolute error)")
    print(f"{'='*60}")
    worst = analyzed.nlargest(20, 'abs_error')[
        ['job_id', 'job_category_description_first', 'primary_location_first',
         'actual', 'pred', 'pct_error', 'estimated_cost_sum']
    ]
    worst.columns = ['job_id', 'category', 'location', 'actual', 'predicted', 'pct_error', 'est_cost']
    print(worst.to_string(index=False))

    # Error by category
    print(f"\n{'='*60}")
    print("ERROR BY CATEGORY")
    print(f"{'='*60}")
    cat_errors = analyzed.groupby('job_category_description_first').agg({
        'abs_error': ['mean', 'count'],
        'pct_error': lambda x: x.abs().median()
    }).round(3)
    cat_errors.columns = ['mean_abs_error', 'count', 'median_pct_error']
    cat_errors = cat_errors.sort_values('mean_abs_error', ascending=False)
    print(cat_errors.to_string())

    # Error by location (top 15 locations)
    print(f"\n{'='*60}")
    print("ERROR BY LOCATION (top 15 by error)")
    print(f"{'='*60}")
    loc_errors = analyzed.groupby('primary_location_first').agg({
        'abs_error': ['mean', 'count'],
        'pct_error': lambda x: x.abs().median()
    }).round(3)
    loc_errors.columns = ['mean_abs_error', 'count', 'median_pct_error']
    loc_errors = loc_errors[loc_errors['count'] >= 20].sort_values('mean_abs_error', ascending=False)
    print(loc_errors.head(15).to_string())

    # Error by year
    print(f"\n{'='*60}")
    print("ERROR BY YEAR")
    print(f"{'='*60}")
    year_errors = analyzed.groupby('year').agg({
        'abs_error': 'mean',
        'pct_error': lambda x: x.abs().median(),
        'job_id': 'count'
    }).round(3)
    year_errors.columns = ['mean_abs_error', 'median_pct_error', 'count']
    print(year_errors.to_string())

    # Error by bid size quintile
    print(f"\n{'='*60}")
    print("ERROR BY BID SIZE (quintiles)")
    print(f"{'='*60}")
    analyzed['bid_quintile'] = pd.qcut(analyzed['actual'], 5, labels=['Q1 (smallest)', 'Q2', 'Q3', 'Q4', 'Q5 (largest)'])
    size_errors = analyzed.groupby('bid_quintile').agg({
        'abs_error': 'mean',
        'pct_error': lambda x: x.abs().median(),
        'actual': ['min', 'max']
    }).round(3)
    size_errors.columns = ['mean_abs_error', 'median_pct_error', 'min_bid', 'max_bid']
    print(size_errors.to_string())

    # Bias check: over vs under prediction
    print(f"\n{'='*60}")
    print("PREDICTION BIAS")
    print(f"{'='*60}")
    over = (analyzed['error'] > 0).sum()
    under = (analyzed['error'] < 0).sum()
    print(f"Over-predictions: {over:,} ({over/len(analyzed)*100:.1f}%)")
    print(f"Under-predictions: {under:,} ({under/len(analyzed)*100:.1f}%)")
    print(f"Mean error (log): {analyzed['error'].mean():.4f} ({'over-predicting' if analyzed['error'].mean() > 0 else 'under-predicting'} on average)")

    # Check if error correlates with estimated cost accuracy
    print(f"\n{'='*60}")
    print("ESTIMATED COST vs ACTUAL")
    print(f"{'='*60}")
    analyzed['est_ratio'] = analyzed['estimated_cost_sum'] / analyzed['actual']
    print(f"Median estimate/actual ratio: {analyzed['est_ratio'].median():.2f}")
    print(f"Correlation (est_cost, actual): {analyzed['estimated_cost_sum'].corr(analyzed['actual']):.3f}")

    # Cases where estimate is way off
    bad_estimates = analyzed[analyzed['est_ratio'] < 0.5].copy()
    print(f"\nBids where estimate < 50% of actual: {len(bad_estimates):,}")
    if len(bad_estimates) > 0:
        print("Categories of under-estimated bids:")
        print(bad_estimates['job_category_description_first'].value_counts().head(10).to_string())

if __name__ == "__main__":
    main()
