"""
Baseline Model: Linear Regression on Job-Level Features
Phase 1 approach from claude.md
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

DATA_DIR = "."


def load_data():
    """Load training and test data."""
    train = pd.read_csv(f"{DATA_DIR}/train_summary.csv")
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    return train, test


def extract_date_features(df):
    """Extract features from bid_date."""
    df = df.copy()
    df['bid_date'] = pd.to_datetime(df['bid_date'])
    df['year'] = df['bid_date'].dt.year
    df['month'] = df['bid_date'].dt.month
    df['day_of_year'] = df['bid_date'].dt.dayofyear
    df['day_of_week'] = df['bid_date'].dt.dayofweek
    return df


def target_encode(train, test, col, target, smoothing=10):
    """Apply target encoding with smoothing."""
    global_mean = train[target].mean()
    agg = train.groupby(col)[target].agg(['mean', 'count'])
    smooth = (agg['count'] * agg['mean'] + smoothing * global_mean) / (agg['count'] + smoothing)

    train_encoded = train[col].map(smooth).fillna(global_mean)
    test_encoded = test[col].map(smooth).fillna(global_mean)

    return train_encoded, test_encoded


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


def rmsle(y_true, y_pred):
    """Calculate RMSLE."""
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


def rmsle_cv_score(y_true, y_pred_log):
    """Calculate RMSLE from log predictions."""
    y_pred = np.expm1(y_pred_log)
    y_pred = np.maximum(y_pred, 0)
    return rmsle(y_true, y_pred)


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


def create_submission(row_ids, predictions, filename="submission.csv"):
    """Create submission file."""
    submission = pd.DataFrame({
        'row_id': row_ids,
        'total_bid': predictions
    })
    submission.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")
    print(f"  Shape: {submission.shape}")
    print(f"  Predictions range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    return submission


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
