"""
LightGBM Model with Line-Item Features
Uses quantity data and learned unit prices to estimate job costs
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, RandomizedSearchCV
from scipy.stats import randint, uniform
import warnings

from utils import extract_date_features, create_submission

warnings.filterwarnings('ignore')


def load_raw_data():
    """Load raw line-item data."""
    train = pd.read_csv("../raw_train.csv")
    test = pd.read_csv("../raw_test.csv")
    return train, test


def build_unit_price_lookup(train_df):
    """
    Build lookup table of unit prices from training data.
    Groups by pay_item_description + unit_english_id for granular pricing.
    """
    train_df = train_df.copy()
    train_df['unit_price'] = train_df['amount'] / train_df['quantity'].replace(0, np.nan)

    # Filter out extreme outliers and invalid prices
    valid = train_df['unit_price'].notna() & (train_df['unit_price'] > 0) & (train_df['unit_price'] < 1e7)
    train_df = train_df[valid]

    # Granular: by pay_item + unit
    item_unit_prices = train_df.groupby(['pay_item_description', 'unit_english_id'])['unit_price'].agg(['median', 'count'])
    item_unit_prices.columns = ['price_item_unit', 'count_item_unit']

    # Fallback: by category + unit
    cat_unit_prices = train_df.groupby(['category_description', 'unit_english_id'])['unit_price'].median()
    cat_unit_prices.name = 'price_cat_unit'

    # Global fallback: by unit only
    unit_prices = train_df.groupby('unit_english_id')['unit_price'].median()
    unit_prices.name = 'price_unit'

    return item_unit_prices, cat_unit_prices, unit_prices


def estimate_line_item_cost(df, item_unit_prices, cat_unit_prices, unit_prices, min_count=5):
    """
    Estimate cost for each line item using learned unit prices.
    Uses hierarchical fallback: item+unit -> category+unit -> unit only
    """
    df = df.copy()

    # Merge item+unit prices
    df = df.merge(item_unit_prices, left_on=['pay_item_description', 'unit_english_id'],
                  right_index=True, how='left')

    # Use item+unit price only if we have enough samples
    df.loc[df['count_item_unit'] < min_count, 'price_item_unit'] = np.nan

    # Merge category+unit fallback
    df = df.merge(cat_unit_prices, left_on=['category_description', 'unit_english_id'],
                  right_index=True, how='left')

    # Merge unit fallback
    df = df.merge(unit_prices, left_on='unit_english_id', right_index=True, how='left')

    # Hierarchical price selection
    df['estimated_unit_price'] = df['price_item_unit'].fillna(df['price_cat_unit']).fillna(df['price_unit'])

    # Estimate cost
    df['estimated_cost'] = df['quantity'] * df['estimated_unit_price']

    return df


def aggregate_job_features(df, is_train=True):
    """
    Aggregate line-item features to job level.
    """
    # Group by job+contractor (each bid is unique)
    group_cols = ['job_id', 'contractor_id']

    agg_dict = {
        # Basic job info (take first since constant per job+contractor)
        'job_category_description': 'first',
        'primary_location': 'first',
        'bid_date': 'first',
        'num_pay_items': 'first',

        # Quantity aggregations
        'quantity': ['sum', 'mean', 'std', 'max'],

        # Estimated cost aggregations
        'estimated_cost': ['sum', 'mean', 'std'],
        'estimated_unit_price': ['mean', 'std'],
    }

    if is_train:
        agg_dict['total_bid'] = 'first'

    agg = df.groupby(group_cols).agg(agg_dict)
    agg.columns = ['_'.join(col).strip('_') for col in agg.columns]
    agg = agg.reset_index()

    # Category counts
    cat_counts = df.groupby(group_cols)['category_description'].value_counts().unstack(fill_value=0)
    cat_counts.columns = [f'cat_count_{c}' for c in cat_counts.columns]
    cat_counts = cat_counts.reset_index()
    agg = agg.merge(cat_counts, on=group_cols)

    # Unit counts
    unit_counts = df.groupby(group_cols)['unit_english_id'].value_counts().unstack(fill_value=0)
    unit_counts.columns = [f'unit_count_{c}' for c in unit_counts.columns]
    unit_counts = unit_counts.reset_index()
    agg = agg.merge(unit_counts, on=group_cols)

    # Quantity by unit type
    qty_by_unit = df.groupby(group_cols + ['unit_english_id'])['quantity'].sum().unstack(fill_value=0)
    qty_by_unit.columns = [f'qty_{c}' for c in qty_by_unit.columns]
    qty_by_unit = qty_by_unit.reset_index()
    agg = agg.merge(qty_by_unit, on=group_cols)

    return agg


def prepare_features(train_agg, test_agg):
    """Prepare final feature matrices."""
    # Rename bid_date column for extract_date_features
    train_agg = train_agg.rename(columns={'bid_date_first': 'bid_date'})
    test_agg = test_agg.rename(columns={'bid_date_first': 'bid_date'})

    # Extract date features
    train_agg = extract_date_features(train_agg)
    test_agg = extract_date_features(test_agg)

    # Target
    y = np.log1p(train_agg['total_bid_first'])

    # Identify feature columns
    exclude_cols = {'job_id', 'contractor_id', 'total_bid_first', 'bid_date',
                    'job_category_description_first', 'primary_location_first'}

    # Encode categoricals
    cat_cols = ['job_category_description_first', 'primary_location_first']
    for col in cat_cols:
        combined = pd.concat([train_agg[col], test_agg[col]]).astype('category')
        categories = combined.cat.categories
        new_col = col + '_cat'
        train_agg[new_col] = train_agg[col].astype('category').cat.set_categories(categories).cat.codes
        test_agg[new_col] = test_agg[col].astype('category').cat.set_categories(categories).cat.codes

    # Get all numeric feature columns
    feature_cols = [c for c in train_agg.columns if c not in exclude_cols
                    and c in test_agg.columns
                    and train_agg[c].dtype in ['int64', 'float64', 'int32', 'float32']]

    # Ensure columns exist in both
    feature_cols = [c for c in feature_cols if c in test_agg.columns]

    X_train = train_agg[feature_cols].copy()
    X_test = test_agg[feature_cols].copy()

    # Fill NaN with 0 for count/quantity columns
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Row IDs for submission
    row_ids = test_agg['job_id'] + '__' + test_agg['contractor_id']

    return X_train, y.values, X_test, row_ids, feature_cols


def tune_hyperparameters(X_train, y_train, n_iter=30, sample_frac=0.2):
    """
    Tune LightGBM hyperparameters using RandomizedSearchCV.
    Uses a subsample of data for faster tuning.
    Returns the best parameters found.
    """
    # Subsample for faster tuning
    if sample_frac < 1.0:
        n_samples = int(len(X_train) * sample_frac)
        np.random.seed(42)
        idx = np.random.choice(len(X_train), n_samples, replace=False)
        X_tune = X_train.iloc[idx]
        y_tune = y_train[idx]
        print(f"\nTuning on {n_samples:,} samples ({sample_frac:.0%} of data)")
    else:
        X_tune, y_tune = X_train, y_train

    print(f"Running {n_iter} iterations × 3 folds = {n_iter * 3} fits...")

    param_dist = {
        'num_leaves': randint(31, 96),
        'max_depth': randint(4, 10),
        'learning_rate': uniform(0.03, 0.12),  # 0.03 to 0.15
        'n_estimators': randint(200, 500),
        'min_child_samples': randint(10, 50),
        'subsample': uniform(0.6, 0.35),  # 0.6 to 0.95
        'colsample_bytree': uniform(0.6, 0.35),  # 0.6 to 0.95
        'reg_alpha': uniform(0, 2),
        'reg_lambda': uniform(0, 2),
    }

    base_model = lgb.LGBMRegressor(
        objective='regression',
        boosting_type='gbdt',
        verbose=-1,
        n_jobs=-1,
        random_state=42
    )

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=3,
        scoring='neg_root_mean_squared_error',
        random_state=42,
        verbose=2,
        n_jobs=1  # Avoid nested parallelism issues
    )

    search.fit(X_tune, y_tune)

    print(f"\nBest CV RMSE: {-search.best_score_:.4f}")
    print("\nBest parameters:")
    for k, v in search.best_params_.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    return search.best_params_


def train_and_predict_cv(train_raw, test_raw, tuned_params=None):
    """
    Train LightGBM with proper CV - build price lookup only from training fold.
    This avoids data leakage from validation fold contributing to price estimates.
    """
    if tuned_params:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'seed': 42,
            'n_jobs': -1,
            # Map sklearn API params to lgb.train params
            'num_leaves': tuned_params.get('num_leaves', 63),
            'max_depth': tuned_params.get('max_depth', -1),
            'learning_rate': tuned_params.get('learning_rate', 0.05),
            'min_child_samples': tuned_params.get('min_child_samples', 20),
            'feature_fraction': tuned_params.get('colsample_bytree', 0.8),
            'bagging_fraction': tuned_params.get('subsample', 0.8),
            'bagging_freq': 5,
            'lambda_l1': tuned_params.get('reg_alpha', 0),
            'lambda_l2': tuned_params.get('reg_lambda', 0),
        }
        num_boost_round = tuned_params.get('n_estimators', 1000)
    else:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42,
            'n_jobs': -1
        }
        num_boost_round = 2000

    # Get unique job+contractor combinations for CV splitting
    job_keys = train_raw.groupby(['job_id', 'contractor_id']).size().reset_index()[['job_id', 'contractor_id']]
    job_keys['key'] = job_keys['job_id'] + '__' + job_keys['contractor_id']

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    models = []
    feature_cols = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(job_keys)):
        train_keys = set(job_keys.iloc[train_idx]['key'])
        val_keys = set(job_keys.iloc[val_idx]['key'])

        # Split raw data by job keys
        train_raw['key'] = train_raw['job_id'] + '__' + train_raw['contractor_id']
        train_fold = train_raw[train_raw['key'].isin(train_keys)].copy()
        val_fold = train_raw[train_raw['key'].isin(val_keys)].copy()

        # Build price lookup ONLY from training fold (no leakage!)
        item_prices, cat_prices, unit_prices = build_unit_price_lookup(train_fold)

        # Estimate costs using training-fold prices
        train_fold = estimate_line_item_cost(train_fold, item_prices, cat_prices, unit_prices)
        val_fold = estimate_line_item_cost(val_fold, item_prices, cat_prices, unit_prices)

        # Aggregate to job level
        train_agg = aggregate_job_features(train_fold, is_train=True)
        val_agg = aggregate_job_features(val_fold, is_train=True)

        # Prepare features
        X_tr, y_tr, _, _, cols = prepare_features(train_agg.copy(), val_agg.copy())
        X_val, y_val, _, _, _ = prepare_features(val_agg.copy(), train_agg.copy())

        # Align columns
        if feature_cols is None:
            feature_cols = cols
        X_tr = X_tr.reindex(columns=feature_cols, fill_value=0)
        X_val = X_val.reindex(columns=feature_cols, fill_value=0)

        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        models.append(model)

        y_pred_val = model.predict(X_val)
        rmse = np.sqrt(np.mean((y_val - y_pred_val) ** 2))
        cv_scores.append(rmse)
        print(f"  Fold {fold+1}: RMSE = {rmse:.4f}")

    print(f"\nCV RMSE: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    # For final predictions, use all training data
    print("\nTraining final model on all data...")
    item_prices, cat_prices, unit_prices = build_unit_price_lookup(train_raw)
    train_raw_est = estimate_line_item_cost(train_raw.copy(), item_prices, cat_prices, unit_prices)
    test_raw_est = estimate_line_item_cost(test_raw.copy(), item_prices, cat_prices, unit_prices)

    train_agg = aggregate_job_features(train_raw_est, is_train=True)
    test_agg = aggregate_job_features(test_raw_est, is_train=False)

    X_train, y_train, X_test, row_ids, _ = prepare_features(train_agg, test_agg)
    X_train = X_train.reindex(columns=feature_cols, fill_value=0)
    X_test = X_test.reindex(columns=feature_cols, fill_value=0)

    # Feature importance (from last CV model)
    print("\nTop 20 Feature Importance:")
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': models[-1].feature_importance()
    }).sort_values('importance', ascending=False)
    print(importance.head(20).to_string(index=False))

    # Train final model
    train_data = lgb.Dataset(X_train, label=y_train)
    final_model = lgb.train(params, train_data, num_boost_round=num_boost_round)

    predictions_log = final_model.predict(X_test)
    predictions = np.expm1(predictions_log)
    predictions = np.maximum(predictions, 0)

    return predictions, row_ids


def main(tune=False, n_iter=50):
    print("Loading raw data...")
    train_raw, test_raw = load_raw_data()
    print(f"  Train: {train_raw.shape}, Test: {test_raw.shape}")

    tuned_params = None
    if tune:
        # Prepare data for tuning (uses all training data for price lookup)
        print("\nPreparing data for hyperparameter tuning...")
        item_prices, cat_prices, unit_prices = build_unit_price_lookup(train_raw)
        train_est = estimate_line_item_cost(train_raw.copy(), item_prices, cat_prices, unit_prices)
        test_est = estimate_line_item_cost(test_raw.copy(), item_prices, cat_prices, unit_prices)

        train_agg = aggregate_job_features(train_est, is_train=True)
        test_agg = aggregate_job_features(test_est, is_train=False)

        X_train, y_train, _, _, _ = prepare_features(train_agg, test_agg)
        tuned_params = tune_hyperparameters(X_train, y_train, n_iter=n_iter, sample_frac=0.2)

    print("\nTraining with proper CV (no leakage)...")
    predictions, row_ids = train_and_predict_cv(train_raw, test_raw, tuned_params=tuned_params)

    print("\nCreating submission...")
    create_submission(row_ids, predictions, filename="submission_lgbm_lineitem.csv")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning')
    parser.add_argument('--n-iter', type=int, default=30, help='Number of tuning iterations')
    args = parser.parse_args()
    main(tune=args.tune, n_iter=args.n_iter)
