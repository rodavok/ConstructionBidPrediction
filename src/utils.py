"""
Shared utilities for construction price prediction models.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = "."
CPI_FILE = Path(__file__).parent.parent / "data" / "cpi_monthly.csv"


def load_cpi_data():
    """Load monthly construction PPI data (WPUSI012011)."""
    cpi = pd.read_csv(CPI_FILE)
    cpi['date'] = pd.to_datetime(cpi['observation_date'])
    cpi['year_month'] = cpi['date'].dt.to_period('M')
    return cpi.set_index('year_month')['WPUSI012011']


def get_inflation_factor(df, reference_date=None):
    """
    Calculate inflation adjustment factor for each row.

    Adjusts prices from bid_date to reference_date (default: latest date in CPI data).
    Factor > 1 means prices need to be scaled up (older bids).

    Args:
        df: DataFrame with 'bid_date' column
        reference_date: Target date for adjustment (default: latest CPI date)

    Returns:
        Series of inflation factors
    """
    cpi_data = load_cpi_data()

    if reference_date is None:
        reference_date = cpi_data.index.max()
    else:
        reference_date = pd.Period(reference_date, freq='M')

    reference_cpi = cpi_data[reference_date]

    # Get year-month for each bid
    df = df.copy()
    df['bid_date'] = pd.to_datetime(df['bid_date'])
    bid_periods = df['bid_date'].dt.to_period('M')

    # Map to CPI values (forward-fill for any missing months)
    bid_cpi = bid_periods.map(cpi_data)

    # Handle missing CPI values by using nearest available
    if bid_cpi.isna().any():
        # For dates beyond our CPI data, use the last known value
        latest_cpi = cpi_data.iloc[-1]
        bid_cpi = bid_cpi.fillna(latest_cpi)

    # Inflation factor: reference_cpi / bid_cpi
    # Older bids (lower CPI) get factor > 1 to scale up
    inflation_factor = reference_cpi / bid_cpi

    return inflation_factor


def add_inflation_features(df, reference_date=None):
    """
    Add inflation-related features to DataFrame.

    Args:
        df: DataFrame with 'bid_date' column
        reference_date: Target date for adjustment

    Returns:
        DataFrame with added columns:
        - inflation_factor: multiplier to adjust prices to reference date
        - cpi_at_bid: CPI value at time of bid
    """
    cpi_data = load_cpi_data()

    if reference_date is None:
        reference_date = cpi_data.index.max()
    else:
        reference_date = pd.Period(reference_date, freq='M')

    reference_cpi = cpi_data[reference_date]

    df = df.copy()
    df['bid_date'] = pd.to_datetime(df['bid_date'])
    bid_periods = df['bid_date'].dt.to_period('M')

    # Map to CPI values
    df['cpi_at_bid'] = bid_periods.map(cpi_data)

    # Fill missing with latest known CPI
    if df['cpi_at_bid'].isna().any():
        latest_cpi = cpi_data.iloc[-1]
        df['cpi_at_bid'] = df['cpi_at_bid'].fillna(latest_cpi)

    df['inflation_factor'] = reference_cpi / df['cpi_at_bid']

    return df


def add_dummy_inflation_features(df):
    """
    Add neutral inflation features (no adjustment).

    Used when inflation adjustment is disabled.
    """
    df = df.copy()
    df['inflation_factor'] = 1.0
    df['cpi_at_bid'] = 1.0
    return df


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
