"""
Shared utilities for construction price prediction models.
"""

import pandas as pd
import numpy as np

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


def rmsle(y_true, y_pred):
    """Calculate RMSLE."""
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


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


def target_encode(train, test, col, target, smoothing=10):
    """Apply target encoding with smoothing."""
    global_mean = train[target].mean()
    agg = train.groupby(col)[target].agg(['mean', 'count'])
    smooth = (agg['count'] * agg['mean'] + smoothing * global_mean) / (agg['count'] + smoothing)

    train_encoded = train[col].map(smooth).fillna(global_mean)
    test_encoded = test[col].map(smooth).fillna(global_mean)

    return train_encoded, test_encoded


def encode_categoricals_as_int(train, test, cols):
    """Encode categorical columns as integers for tree-based models."""
    train = train.copy()
    test = test.copy()
    cat_feature_cols = []

    for col in cols:
        combined = pd.concat([train[col], test[col]], axis=0).astype('category')
        categories = combined.cat.categories

        new_col = col + '_cat'
        train[new_col] = train[col].astype('category').cat.set_categories(categories).cat.codes
        test[new_col] = test[col].astype('category').cat.set_categories(categories).cat.codes
        cat_feature_cols.append(new_col)

    return train, test, cat_feature_cols
