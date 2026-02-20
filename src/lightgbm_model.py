"""
LightGBM Model: Gradient Boosted Trees on Job-Level Features
Improvement over baseline Ridge regression
"""

import numpy as np
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import KFold
import warnings

from utils import (
    load_data,
    extract_date_features,
    encode_categoricals_as_int,
    create_submission,
)

warnings.filterwarnings('ignore')


def prepare_features(train, test):
    """Prepare feature matrices for LightGBM."""
    # Extract date features
    train = extract_date_features(train)
    test = extract_date_features(test)

    # Log transform target
    y = np.log1p(train['total_bid'])

    # Numeric date features
    date_cols = ['year', 'month', 'day_of_year', 'day_of_week']

    # Encode categoricals as integers for LightGBM native handling
    categorical_cols = ['job_category_description', 'primary_location', 'contractor_id']
    train, test, cat_feature_cols = encode_categoricals_as_int(train, test, categorical_cols)

    feature_cols = date_cols + cat_feature_cols

    X_train = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    return X_train, y.values, X_test, test['row_id'], cat_feature_cols


def train_and_predict(X_train, y_train, X_test, cat_features):
    """Train LightGBM and generate predictions."""

    # LightGBM parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }

    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_features)
        val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_features)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        y_pred_val = model.predict(X_val)
        rmse = np.sqrt(np.mean((y_val - y_pred_val) ** 2))
        cv_scores.append(rmse)

    print(f"CV RMSE (log space): {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    # Train final model on full data
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)

    final_model = lgb.train(
        params,
        train_data,
        num_boost_round=1000
    )

    # Feature importance
    print("\nFeature Importance:")
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': final_model.feature_importance()
    }).sort_values('importance', ascending=False)
    print(importance.to_string(index=False))

    # Predictions
    y_pred_log = final_model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.maximum(y_pred, 0)  # Ensure non-negative

    return y_pred


def main():
    print("Loading data...")
    train, test = load_data()
    print(f"  Train: {train.shape}, Test: {test.shape}")

    print("\nPreparing features...")
    X_train, y_train, X_test, row_ids, cat_features = prepare_features(train, test)
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Categorical features: {cat_features}")

    print("\nTraining model...")
    predictions = train_and_predict(X_train, y_train, X_test, cat_features)

    print("\nCreating submission...")
    create_submission(row_ids, predictions, filename="submission_lgbm.csv")


if __name__ == "__main__":
    main()
