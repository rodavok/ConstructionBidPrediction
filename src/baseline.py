"""
Baseline Model: Linear Regression on Job-Level Features
Phase 1 approach from claude.md
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings

from utils import (
    load_data,
    extract_date_features,
    target_encode,
    create_submission,
)

warnings.filterwarnings('ignore')


def prepare_features(train, test):
    """Prepare feature matrices."""
    # Extract date features
    train = extract_date_features(train)
    test = extract_date_features(test)

    # Log transform target
    y = np.log1p(train['total_bid'])

    # Numeric date features
    date_cols = ['year', 'month', 'day_of_year', 'day_of_week']

    # Target encode categorical columns
    train['category_encoded'], test['category_encoded'] = target_encode(
        train, test, 'job_category_description', 'total_bid'
    )
    train['location_encoded'], test['location_encoded'] = target_encode(
        train, test, 'primary_location', 'total_bid'
    )
    train['contractor_encoded'], test['contractor_encoded'] = target_encode(
        train, test, 'contractor_id', 'total_bid'
    )

    # Log transform encoded values for consistency
    for col in ['category_encoded', 'location_encoded', 'contractor_encoded']:
        train[col] = np.log1p(train[col])
        test[col] = np.log1p(test[col])

    feature_cols = date_cols + ['category_encoded', 'location_encoded', 'contractor_encoded']

    X_train = train[feature_cols].values
    X_test = test[feature_cols].values

    return X_train, y.values, X_test, test['row_id']


def train_and_predict(X_train, y_train, X_test):
    """Train Ridge regression and generate predictions."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_root_mean_squared_error')
    print(f"CV RMSE (log space): {-cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Predictions
    y_pred_log = model.predict(X_test_scaled)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.maximum(y_pred, 0)  # Ensure non-negative

    return y_pred


def main():
    print("Loading data...")
    train, test = load_data()
    print(f"  Train: {train.shape}, Test: {test.shape}")

    print("\nPreparing features...")
    X_train, y_train, X_test, row_ids = prepare_features(train, test)
    print(f"  Features: {X_train.shape[1]}")

    print("\nTraining model...")
    predictions = train_and_predict(X_train, y_train, X_test)

    print("\nCreating submission...")
    create_submission(row_ids, predictions)


if __name__ == "__main__":
    main()
