"""
Gradient Boosting Model with Line-Item Features
Uses quantity data and learned unit prices to estimate job costs
Supports LightGBM, XGBoost, and CatBoost
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, KFold
from scipy.stats import randint, uniform
import warnings
import mlflow

from utils import extract_date_features, create_submission, add_inflation_features, load_cpi_data, add_dummy_inflation_features

warnings.filterwarnings('ignore')

mlflow.set_experiment("construction-price-prediction")


# =============================================================================
# MODEL BACKENDS
# =============================================================================

def get_lightgbm_backend():
    import lightgbm as lgb

    def get_default_params(config):
        return {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': config['num_leaves'],
            'learning_rate': config['learning_rate'],
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42,
            'n_jobs': -1
        }

    def get_tuned_params(tuned):
        return {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'seed': 42,
            'n_jobs': -1,
            'num_leaves': tuned.get('num_leaves', 63),
            'max_depth': tuned.get('max_depth', -1),
            'learning_rate': tuned.get('learning_rate', 0.05),
            'min_child_samples': tuned.get('min_child_samples', 20),
            'feature_fraction': tuned.get('colsample_bytree', 0.8),
            'bagging_fraction': tuned.get('subsample', 0.8),
            'bagging_freq': 5,
            'lambda_l1': tuned.get('reg_alpha', 0),
            'lambda_l2': tuned.get('reg_lambda', 0),
        }

    def get_tune_distributions():
        return {
            'num_leaves': randint(31, 96),
            'max_depth': randint(4, 10),
            'learning_rate': uniform(0.03, 0.12),
            'n_estimators': randint(200, 500),
            'min_child_samples': randint(10, 50),
            'subsample': uniform(0.6, 0.35),
            'colsample_bytree': uniform(0.6, 0.35),
            'reg_alpha': uniform(0, 2),
            'reg_lambda': uniform(0, 2),
        }

    def get_tuning_model():
        return lgb.LGBMRegressor(
            objective='regression',
            boosting_type='gbdt',
            verbose=-1,
            n_jobs=-1,
            random_state=42
        )

    def train(X_tr, y_tr, X_val, y_val, params, num_boost_round):
        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val)
        model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        return model

    def train_final(X_train, y_train, params, num_boost_round):
        train_data = lgb.Dataset(X_train, label=y_train)
        return lgb.train(params, train_data, num_boost_round=num_boost_round)

    def predict(model, X):
        return model.predict(X)

    def get_feature_importance(model, feature_cols):
        return pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importance()
        }).sort_values('importance', ascending=False)

    return {
        'name': 'lightgbm',
        'get_default_params': get_default_params,
        'get_tuned_params': get_tuned_params,
        'get_tune_distributions': get_tune_distributions,
        'get_tuning_model': get_tuning_model,
        'train': train,
        'train_final': train_final,
        'predict': predict,
        'get_feature_importance': get_feature_importance,
    }


def get_xgboost_backend():
    import xgboost as xgb

    def get_default_params(config):
        return {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': config['learning_rate'],
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42,
            'n_jobs': -1,
        }

    def get_tuned_params(tuned):
        return {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'seed': 42,
            'n_jobs': -1,
            'max_depth': tuned.get('max_depth', 6),
            'learning_rate': tuned.get('learning_rate', 0.05),
            'min_child_weight': tuned.get('min_child_weight', 1),
            'subsample': tuned.get('subsample', 0.8),
            'colsample_bytree': tuned.get('colsample_bytree', 0.8),
            'reg_alpha': tuned.get('reg_alpha', 0),
            'reg_lambda': tuned.get('reg_lambda', 1),
        }

    def get_tune_distributions():
        return {
            'max_depth': randint(4, 10),
            'learning_rate': uniform(0.03, 0.12),
            'n_estimators': randint(200, 500),
            'min_child_weight': randint(1, 10),
            'subsample': uniform(0.6, 0.35),
            'colsample_bytree': uniform(0.6, 0.35),
            'reg_alpha': uniform(0, 2),
            'reg_lambda': uniform(0, 2),
        }

    def get_tuning_model():
        return xgb.XGBRegressor(
            objective='reg:squarederror',
            n_jobs=-1,
            random_state=42
        )

    def train(X_tr, y_tr, X_val, y_val, params, num_boost_round):
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dval, 'val')],
            early_stopping_rounds=100,
            verbose_eval=False
        )
        return model

    def train_final(X_train, y_train, params, num_boost_round):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        return xgb.train(params, dtrain, num_boost_round=num_boost_round)

    def predict(model, X):
        import xgboost as xgb
        dtest = xgb.DMatrix(X)
        return model.predict(dtest)

    def get_feature_importance(model, feature_cols):
        scores = model.get_score(importance_type='gain')
        importance = [scores.get(f, 0) for f in feature_cols]
        return pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)

    return {
        'name': 'xgboost',
        'get_default_params': get_default_params,
        'get_tuned_params': get_tuned_params,
        'get_tune_distributions': get_tune_distributions,
        'get_tuning_model': get_tuning_model,
        'train': train,
        'train_final': train_final,
        'predict': predict,
        'get_feature_importance': get_feature_importance,
    }


def get_catboost_backend():
    from catboost import CatBoostRegressor, Pool

    def get_default_params(config):
        return {
            'loss_function': 'RMSE',
            'depth': 6,
            'learning_rate': config['learning_rate'],
            'random_seed': 42,
            'verbose': False,
        }

    def get_tuned_params(tuned):
        return {
            'loss_function': 'RMSE',
            'random_seed': 42,
            'verbose': False,
            'depth': tuned.get('depth', 6),
            'learning_rate': tuned.get('learning_rate', 0.05),
            'l2_leaf_reg': tuned.get('l2_leaf_reg', 3),
            'bagging_temperature': tuned.get('bagging_temperature', 1),
        }

    def get_tune_distributions():
        return {
            'depth': randint(4, 10),
            'learning_rate': uniform(0.03, 0.12),
            'iterations': randint(200, 500),
            'l2_leaf_reg': uniform(1, 9),
            'bagging_temperature': uniform(0, 1),
        }

    def get_tuning_model():
        return CatBoostRegressor(
            loss_function='RMSE',
            random_seed=42,
            verbose=False
        )

    def train(X_tr, y_tr, X_val, y_val, params, num_boost_round):
        model = CatBoostRegressor(**params, iterations=num_boost_round)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=100,
            verbose=False
        )
        return model

    def train_final(X_train, y_train, params, num_boost_round):
        model = CatBoostRegressor(**params, iterations=num_boost_round)
        model.fit(X_train, y_train, verbose=False)
        return model

    def predict(model, X):
        return model.predict(X)

    def get_feature_importance(model, feature_cols):
        return pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

    return {
        'name': 'catboost',
        'get_default_params': get_default_params,
        'get_tuned_params': get_tuned_params,
        'get_tune_distributions': get_tune_distributions,
        'get_tuning_model': get_tuning_model,
        'train': train,
        'train_final': train_final,
        'predict': predict,
        'get_feature_importance': get_feature_importance,
    }


def get_backend(name):
    backends = {
        'lightgbm': get_lightgbm_backend,
        'xgboost': get_xgboost_backend,
        'catboost': get_catboost_backend,
    }
    if name not in backends:
        raise ValueError(f"Unknown model: {name}. Choose from: {list(backends.keys())}")
    return backends[name]()


# =============================================================================
# EXPERIMENT CONFIG
# =============================================================================
DEFAULT_CONFIG = {
    # Model selection
    'model': 'lightgbm',  # 'lightgbm', 'xgboost', 'catboost'

    # Data source
    'use_aggregated': False,  # True = train_summary.csv/test.csv, False = raw_train.csv/raw_test.csv

    # Feature engineering
    'use_inflation': True,
    'inflation_lag_months': 0,  # 0 = same month, 3 = use price from 3 months prior
    'use_contractor_history': False,  # Add contractor win count at bid time
    'use_competition_intensity': False,  # Add number of bidders per job

    # Cross-validation
    'time_based_cv': True,
    'cv_splits': 5,

    # Hyperparameter tuning
    'tune': False,
    'tune_iterations': 30,

    # Model params (used when not tuning)
    'num_leaves': 63,  # LightGBM
    'learning_rate': 0.05,
    'num_boost_round': 2000,
}


def get_config(**overrides):
    """Get config with optional overrides."""
    config = DEFAULT_CONFIG.copy()
    config.update(overrides)
    return config


def load_and_prepare_data(config):
    """
    Load data and apply inflation adjustment upfront.
    This centralizes the inflation decision - all downstream code just uses the columns.
    """
    if config.get('use_aggregated', False):
        train = pd.read_csv("./train_summary.csv")
        test = pd.read_csv("./test.csv")
        print("Using AGGREGATED dataset (train_summary.csv, test.csv)")
    else:
        train = pd.read_csv("./raw_train.csv")
        test = pd.read_csv("./raw_test.csv")
        print("Using LINE-ITEM dataset (raw_train.csv, raw_test.csv)")

    if config['use_inflation']:
        cpi_data = load_cpi_data()
        reference_date = cpi_data.index.max()

        # Apply lag if specified
        lag = config.get('inflation_lag_months', 0)
        if lag > 0:
            reference_date = reference_date - lag
            print(f"Inflation: {lag}-month lag, reference {reference_date}")
        else:
            print(f"Inflation reference date: {reference_date}")

        train = add_inflation_features(train, str(reference_date))
        test = add_inflation_features(test, str(reference_date))
    else:
        print("Inflation adjustment: DISABLED")
        train = add_dummy_inflation_features(train)
        test = add_dummy_inflation_features(test)

    if config.get('use_contractor_history', False):
        print("Adding contractor win history feature...")
        train, test = compute_contractor_win_history(train, test)
    else:
        # Add dummy column so downstream code doesn't break
        train['contractor_prior_wins'] = 0
        test['contractor_prior_wins'] = 0

    if config.get('use_competition_intensity', False):
        print("Adding competition intensity feature...")
        train, test = compute_competition_intensity(train, test)
    else:
        train['num_bidders'] = 0
        test['num_bidders'] = 0

    return train, test


def compute_contractor_win_history(train_df, test_df=None):
    """
    Compute how many jobs each contractor has won at the time of each bid.
    Winner = contractor with lowest total_bid for each job.

    For training: uses only wins strictly before each bid's date (no leakage).
    For test: uses all training wins before each test bid's date.
    """
    # Get unique bids (job_id + contractor_id level)
    train_bids = train_df.groupby(['job_id', 'contractor_id']).agg({
        'bid_date': 'first',
        'total_bid': 'first'
    }).reset_index()
    train_bids['bid_date'] = pd.to_datetime(train_bids['bid_date'])

    # Identify winner for each job (lowest bid)
    winners = train_bids.loc[train_bids.groupby('job_id')['total_bid'].idxmin()]
    winners = winners[['job_id', 'contractor_id', 'bid_date']].copy()
    winners.columns = ['job_id', 'winner_contractor', 'win_date']

    # For each bid, count contractor's wins strictly before bid_date
    def count_prior_wins(row, win_history):
        contractor = row['contractor_id']
        bid_date = row['bid_date']
        contractor_wins = win_history[win_history['winner_contractor'] == contractor]
        prior_wins = (contractor_wins['win_date'] < bid_date).sum()
        return prior_wins

    # Compute for training data
    train_bids['contractor_prior_wins'] = train_bids.apply(
        lambda r: count_prior_wins(r, winners), axis=1
    )

    # Merge back to line-item level
    win_counts = train_bids[['job_id', 'contractor_id', 'contractor_prior_wins']]
    train_df = train_df.merge(win_counts, on=['job_id', 'contractor_id'], how='left')
    train_df['contractor_prior_wins'] = train_df['contractor_prior_wins'].fillna(0).astype(int)

    if test_df is not None:
        # For test, use all training wins
        test_bids = test_df.groupby(['job_id', 'contractor_id']).agg({
            'bid_date': 'first'
        }).reset_index()
        test_bids['bid_date'] = pd.to_datetime(test_bids['bid_date'])

        test_bids['contractor_prior_wins'] = test_bids.apply(
            lambda r: count_prior_wins(r, winners), axis=1
        )

        test_win_counts = test_bids[['job_id', 'contractor_id', 'contractor_prior_wins']]
        test_df = test_df.merge(test_win_counts, on=['job_id', 'contractor_id'], how='left')
        test_df['contractor_prior_wins'] = test_df['contractor_prior_wins'].fillna(0).astype(int)

        return train_df, test_df

    return train_df


def compute_competition_intensity(train_df, test_df=None):
    """
    Compute number of bidders per job (competition intensity).

    More bidders typically means more competitive pricing.
    For training: count unique contractors per job from training data.
    For test: use training bidder counts for known jobs, median for new jobs.
    """
    # Count bidders per job in training data
    train_bidders = train_df.groupby('job_id')['contractor_id'].nunique().reset_index()
    train_bidders.columns = ['job_id', 'num_bidders']

    # Merge to training data
    train_df = train_df.merge(train_bidders, on='job_id', how='left')

    if test_df is not None:
        # For test jobs that appear in training, use known count
        # For new test jobs, use median from training
        test_bidders = test_df.groupby('job_id')['contractor_id'].nunique().reset_index()
        test_bidders.columns = ['job_id', 'num_bidders']
        test_df = test_df.merge(test_bidders, on='job_id', how='left')

        # Fill any missing with training median
        median_bidders = train_bidders['num_bidders'].median()
        test_df['num_bidders'] = test_df['num_bidders'].fillna(median_bidders)

        return train_df, test_df

    return train_df


def build_unit_price_lookup(train_df):
    """
    Build lookup table of unit prices from training data.
    Groups by pay_item_description + unit_english_id for granular pricing.

    Expects inflation_factor column to already exist.
    """
    train_df = train_df.copy()

    train_df['unit_price'] = train_df['amount'] / train_df['quantity'].replace(0, np.nan)

    # Adjust unit prices for inflation (scale to reference date)
    train_df['unit_price_adjusted'] = train_df['unit_price'] * train_df['inflation_factor']

    # Filter out extreme outliers and invalid prices
    valid = train_df['unit_price_adjusted'].notna() & (train_df['unit_price_adjusted'] > 0) & (train_df['unit_price_adjusted'] < 1e7)
    train_df = train_df[valid]

    # Granular: by pay_item + unit (using inflation-adjusted prices)
    item_unit_prices = train_df.groupby(['pay_item_description', 'unit_english_id'])['unit_price_adjusted'].agg(['median', 'count'])
    item_unit_prices.columns = ['price_item_unit', 'count_item_unit']

    # Fallback: by category + unit
    cat_unit_prices = train_df.groupby(['category_description', 'unit_english_id'])['unit_price_adjusted'].median()
    cat_unit_prices.name = 'price_cat_unit'

    # Global fallback: by unit only
    unit_prices = train_df.groupby('unit_english_id')['unit_price_adjusted'].median()
    unit_prices.name = 'price_unit'

    return item_unit_prices, cat_unit_prices, unit_prices


def estimate_line_item_cost(df, item_unit_prices, cat_unit_prices, unit_prices, min_count=5):
    """
    Estimate cost for each line item using learned unit prices.
    Uses hierarchical fallback: item+unit -> category+unit -> unit only

    Expects inflation_factor column to already exist.
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

    # Hierarchical price selection (these are at reference_date price level)
    df['estimated_unit_price_ref'] = df['price_item_unit'].fillna(df['price_cat_unit']).fillna(df['price_unit'])

    # Scale estimate to the bid's actual date (divide by inflation_factor to go from reference -> bid date)
    # If inflation_factor > 1, bid was in past, so prices were lower -> divide
    df['estimated_unit_price'] = df['estimated_unit_price_ref'] / df['inflation_factor']

    # Estimate cost at bid date price level
    df['estimated_cost'] = df['quantity'] * df['estimated_unit_price']

    # Also keep the reference-date cost for comparison
    df['estimated_cost_ref'] = df['quantity'] * df['estimated_unit_price_ref']

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

        # Inflation-adjusted cost (at reference date prices)
        'estimated_cost_ref': ['sum', 'mean'],

        # Inflation features (materials PPI + labor ECI)
        'inflation_factor': 'first',  # Materials PPI adjustment factor
        'cpi_at_bid': 'first',  # Materials PPI value
        'labor_eci': 'first',  # Labor cost index
        'labor_inflation_factor': 'first',  # Labor adjustment factor

        # Contractor history
        'contractor_prior_wins': 'first',

        # Competition intensity
        'num_bidders': 'first',
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


def prepare_features_aggregated(train_df, test_df):
    """
    Prepare features for aggregated dataset (train_summary.csv, test.csv).
    Simpler pipeline without line-item features.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Extract date features
    train_df = extract_date_features(train_df)
    test_df = extract_date_features(test_df)

    # Target
    y = np.log1p(train_df['total_bid'])

    # Columns to exclude from features
    exclude_cols = {'job_id', 'contractor_id', 'total_bid', 'bid_date', 'row_id',
                    'job_category_description', 'primary_location'}

    # Encode categoricals
    cat_cols = ['job_category_description', 'primary_location']
    for col in cat_cols:
        combined = pd.concat([train_df[col], test_df[col]]).astype('category')
        categories = combined.cat.categories
        new_col = col + '_cat'
        train_df[new_col] = train_df[col].astype('category').cat.set_categories(categories).cat.codes
        test_df[new_col] = test_df[col].astype('category').cat.set_categories(categories).cat.codes

    # Get all numeric feature columns
    feature_cols = [c for c in train_df.columns if c not in exclude_cols
                    and c in test_df.columns
                    and train_df[c].dtype in ['int64', 'float64', 'int32', 'float32']]

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    # Fill NaN
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Row IDs - use existing row_id column if present, otherwise construct
    if 'row_id' in test_df.columns:
        row_ids = test_df['row_id']
    else:
        row_ids = test_df['job_id'].astype(str) + '__' + test_df['contractor_id'].astype(str)

    return X_train, y.values, X_test, row_ids, feature_cols


def tune_hyperparameters(X_train, y_train, backend, bid_dates=None, n_iter=30, n_splits=3, time_based_cv=True):
    """
    Tune hyperparameters using RandomizedSearchCV.

    Args:
        backend: Model backend dict from get_backend()
        time_based_cv: If True, use TimeSeriesSplit (requires bid_dates).
                       If False, use random KFold.
    """
    if time_based_cv and bid_dates is not None:
        # Sort by date for time-based CV
        bid_dates = pd.to_datetime(bid_dates)
        sort_idx = bid_dates.argsort()
        X_tune = X_train.iloc[sort_idx].reset_index(drop=True)
        y_tune = y_train[sort_idx]
        cv = TimeSeriesSplit(n_splits=n_splits)
        print(f"\nTuning {backend['name']} on {len(X_tune):,} samples with time-based CV")
        print(f"Date range: {bid_dates.min().date()} to {bid_dates.max().date()}")
    else:
        X_tune = X_train
        y_tune = y_train
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        print(f"\nTuning {backend['name']} on {len(X_tune):,} samples with random CV")

    print(f"Running {n_iter} iterations × {n_splits} folds = {n_iter * n_splits} fits...")

    param_dist = backend['get_tune_distributions']()
    base_model = backend['get_tuning_model']()

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
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


def train_and_predict_cv(train_raw, test_raw, config, backend, tuned_params=None):
    """
    Train model with proper CV - build price lookup only from training fold.
    This avoids data leakage from validation fold contributing to price estimates.

    Expects inflation columns to already exist in train_raw and test_raw.
    """
    if tuned_params:
        params = backend['get_tuned_params'](tuned_params)
        num_boost_round = tuned_params.get('n_estimators', tuned_params.get('iterations', 1000))
    else:
        params = backend['get_default_params'](config)
        num_boost_round = config['num_boost_round']

    # Get unique job+contractor combinations with their bid dates
    train_raw = train_raw.copy()
    train_raw['bid_date_parsed'] = pd.to_datetime(train_raw['bid_date'])
    job_keys = train_raw.groupby(['job_id', 'contractor_id']).agg({
        'bid_date_parsed': 'first'
    }).reset_index()
    job_keys['key'] = job_keys['job_id'] + '__' + job_keys['contractor_id']

    # Set up CV strategy
    n_splits = config['cv_splits']
    if config['time_based_cv']:
        # Sort by date for time-based CV
        job_keys = job_keys.sort_values('bid_date_parsed').reset_index(drop=True)
        cv = TimeSeriesSplit(n_splits=n_splits)
        print("Time-based CV folds:")
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        print("Random CV folds:")

    cv_scores = []
    models = []
    feature_cols = None

    for fold, (train_idx, val_idx) in enumerate(cv.split(job_keys)):
        train_keys_df = job_keys.iloc[train_idx]
        val_keys_df = job_keys.iloc[val_idx]

        if config['time_based_cv']:
            train_date_range = f"{train_keys_df['bid_date_parsed'].min().date()} to {train_keys_df['bid_date_parsed'].max().date()}"
            val_date_range = f"{val_keys_df['bid_date_parsed'].min().date()} to {val_keys_df['bid_date_parsed'].max().date()}"
            print(f"  Fold {fold+1}: Train {train_date_range} ({len(train_idx):,} jobs) -> Val {val_date_range} ({len(val_idx):,} jobs)")
        else:
            print(f"  Fold {fold+1}: Train {len(train_idx):,} jobs -> Val {len(val_idx):,} jobs")

        train_keys = set(train_keys_df['key'])
        val_keys = set(val_keys_df['key'])

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

        model = backend['train'](X_tr, y_tr, X_val, y_val, params, num_boost_round)
        models.append(model)

        y_pred_val = backend['predict'](model, X_val)
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
    importance = backend['get_feature_importance'](models[-1], feature_cols)
    print(importance.head(20).to_string(index=False))

    # Train final model
    final_model = backend['train_final'](X_train, y_train, params, num_boost_round)

    predictions_log = backend['predict'](final_model, X_test)
    predictions = np.expm1(predictions_log)
    predictions = np.maximum(predictions, 0)

    return predictions, row_ids, cv_scores, importance


def train_and_predict_cv_aggregated(train_df, test_df, config, backend, tuned_params=None):
    """
    Train model with CV for aggregated dataset.
    Simpler pipeline without line-item processing.
    """
    if tuned_params:
        params = backend['get_tuned_params'](tuned_params)
        num_boost_round = tuned_params.get('n_estimators', tuned_params.get('iterations', 1000))
    else:
        params = backend['get_default_params'](config)
        num_boost_round = config['num_boost_round']

    # Parse dates and create job keys
    train_df = train_df.copy()
    train_df['bid_date_parsed'] = pd.to_datetime(train_df['bid_date'])
    train_df['key'] = train_df['job_id'].astype(str) + '__' + train_df['contractor_id'].astype(str)

    # Set up CV strategy
    n_splits = config['cv_splits']
    if config['time_based_cv']:
        train_df = train_df.sort_values('bid_date_parsed').reset_index(drop=True)
        cv = TimeSeriesSplit(n_splits=n_splits)
        print("Time-based CV folds:")
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        print("Random CV folds:")

    cv_scores = []
    models = []
    feature_cols = None

    for fold, (train_idx, val_idx) in enumerate(cv.split(train_df)):
        train_fold = train_df.iloc[train_idx].copy()
        val_fold = train_df.iloc[val_idx].copy()

        if config['time_based_cv']:
            train_date_range = f"{train_fold['bid_date_parsed'].min().date()} to {train_fold['bid_date_parsed'].max().date()}"
            val_date_range = f"{val_fold['bid_date_parsed'].min().date()} to {val_fold['bid_date_parsed'].max().date()}"
            print(f"  Fold {fold+1}: Train {train_date_range} ({len(train_idx):,} bids) -> Val {val_date_range} ({len(val_idx):,} bids)")
        else:
            print(f"  Fold {fold+1}: Train {len(train_idx):,} bids -> Val {len(val_idx):,} bids")

        # Prepare features
        X_tr, y_tr, _, _, cols = prepare_features_aggregated(train_fold, val_fold)
        X_val, y_val, _, _, _ = prepare_features_aggregated(val_fold, train_fold)

        # Align columns
        if feature_cols is None:
            feature_cols = cols
        X_tr = X_tr.reindex(columns=feature_cols, fill_value=0)
        X_val = X_val.reindex(columns=feature_cols, fill_value=0)

        model = backend['train'](X_tr, y_tr, X_val, y_val, params, num_boost_round)
        models.append(model)

        y_pred_val = backend['predict'](model, X_val)
        rmse = np.sqrt(np.mean((y_val - y_pred_val) ** 2))
        cv_scores.append(rmse)
        print(f"  Fold {fold+1}: RMSE = {rmse:.4f}")

    print(f"\nCV RMSE: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    # Train final model on all data
    print("\nTraining final model on all data...")
    X_train, y_train, X_test, row_ids, _ = prepare_features_aggregated(train_df, test_df)
    X_train = X_train.reindex(columns=feature_cols, fill_value=0)
    X_test = X_test.reindex(columns=feature_cols, fill_value=0)

    # Feature importance
    print("\nTop 20 Feature Importance:")
    importance = backend['get_feature_importance'](models[-1], feature_cols)
    print(importance.head(20).to_string(index=False))

    # Train final model
    final_model = backend['train_final'](X_train, y_train, params, num_boost_round)

    predictions_log = backend['predict'](final_model, X_test)
    predictions = np.expm1(predictions_log)
    predictions = np.maximum(predictions, 0)

    return predictions, row_ids, cv_scores, importance


def main(config):
    with mlflow.start_run():
        # Log config
        mlflow.log_params(config)

        print("=" * 60)
        print("EXPERIMENT CONFIG:")
        for k, v in config.items():
            print(f"  {k}: {v}")
        print("=" * 60)

        # Get model backend
        backend = get_backend(config['model'])
        print(f"\nUsing model: {backend['name']}")

        print("\nLoading data...")
        train_data, test_data = load_and_prepare_data(config)
        print(f"  Train: {train_data.shape}, Test: {test_data.shape}")
        print(f"CV strategy: {'time-based' if config['time_based_cv'] else 'random'}")

        use_aggregated = config.get('use_aggregated', False)

        tuned_params = None
        if config['tune']:
            print("\nPreparing data for hyperparameter tuning...")
            if use_aggregated:
                # Aggregated path - simpler
                bid_dates = pd.to_datetime(train_data['bid_date'])
                X_train, y_train, _, _, _ = prepare_features_aggregated(train_data, test_data)
            else:
                # Line-item path - need to build price lookup and aggregate
                item_prices, cat_prices, unit_prices = build_unit_price_lookup(train_data)
                train_est = estimate_line_item_cost(train_data.copy(), item_prices, cat_prices, unit_prices)
                test_est = estimate_line_item_cost(test_data.copy(), item_prices, cat_prices, unit_prices)

                train_agg = aggregate_job_features(train_est, is_train=True)
                test_agg = aggregate_job_features(test_est, is_train=False)

                # Keep bid_date before prepare_features drops it
                bid_dates = train_agg['bid_date_first']

                X_train, y_train, _, _, _ = prepare_features(train_agg, test_agg)

            tuned_params = tune_hyperparameters(X_train, y_train, backend, bid_dates,
                                                n_iter=config['tune_iterations'],
                                                time_based_cv=config['time_based_cv'])
            mlflow.log_params({f"tuned_{k}": v for k, v in tuned_params.items()})

        print("\nTraining with CV...")
        if use_aggregated:
            predictions, row_ids, cv_scores, feature_importance = train_and_predict_cv_aggregated(
                train_data, test_data, config, backend, tuned_params=tuned_params
            )
        else:
            predictions, row_ids, cv_scores, feature_importance = train_and_predict_cv(
                train_data, test_data, config, backend, tuned_params=tuned_params
            )

        # Log metrics
        mlflow.log_metric("cv_rmse_mean", np.mean(cv_scores))
        mlflow.log_metric("cv_rmse_std", np.std(cv_scores))
        for i, score in enumerate(cv_scores):
            mlflow.log_metric(f"cv_rmse_fold_{i+1}", score)

        # Log feature importance as artifact
        feature_importance.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")

        print("\nCreating submission...")
        create_submission(row_ids, predictions, filename="submission.csv")
        mlflow.log_artifact("submission.csv")

        print(f"\nMLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['lightgbm', 'xgboost', 'catboost'], help='Model to use')
    parser.add_argument('--aggregated', action='store_true', help='Use aggregated dataset (train_summary.csv) instead of line-items')
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning')
    parser.add_argument('--tune-iterations', type=int, help='Number of tuning iterations')
    parser.add_argument('--random-cv', action='store_true', help='Use random CV instead of time-based CV')
    parser.add_argument('--cv-folds', type=int, help='Number of CV folds (default: 5)')
    parser.add_argument('--no-inflation', action='store_true', help='Disable inflation adjustment')
    parser.add_argument('--inflation-lag', type=int, help='Months to lag inflation data')
    parser.add_argument('--contractor-history', action='store_true', help='Add contractor prior wins feature')
    parser.add_argument('--competition-intensity', action='store_true', help='Add number of bidders per job feature')
    args = parser.parse_args()

    # Build config from defaults + CLI overrides
    overrides = {}
    if args.model:
        overrides['model'] = args.model
    if args.aggregated:
        overrides['use_aggregated'] = True
    if args.tune:
        overrides['tune'] = True
    if args.tune_iterations:
        overrides['tune_iterations'] = args.tune_iterations
    if args.random_cv:
        overrides['time_based_cv'] = False
    if args.cv_folds:
        overrides['cv_splits'] = args.cv_folds
    if args.no_inflation:
        overrides['use_inflation'] = False
    if args.inflation_lag:
        overrides['inflation_lag_months'] = args.inflation_lag
    if args.contractor_history:
        overrides['use_contractor_history'] = True
    if args.competition_intensity:
        overrides['use_competition_intensity'] = True

    config = get_config(**overrides)
    main(config)
