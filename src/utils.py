"""
Shared utilities for construction price prediction models.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = "."
CPI_FILE = Path(__file__).parent.parent / "data" / "cpi_monthly.csv"
ECI_FILE = Path(__file__).parent.parent / "data" / "eci-seasonal-dataset.xlsx"


def load_materials_ppi():
    """Load monthly construction materials PPI data (WPUSI012011)."""
    ppi = pd.read_csv(CPI_FILE)
    ppi['date'] = pd.to_datetime(ppi['observation_date'])
    ppi['year_month'] = ppi['date'].dt.to_period('M')
    return ppi.set_index('year_month')['WPUSI012011']


def load_labor_eci():
    """
    Load construction labor ECI data and interpolate to monthly.

    Source: BLS Employment Cost Index, Construction industry,
    Wages and salaries, seasonally adjusted.
    """
    df = pd.read_excel(ECI_FILE, sheet_name='Seasonal dataset')

    # Filter for construction wages index
    labor = df[
        (df['Industry'] == 'Construction') &
        (df['Estimate Type'] == 'Wages and salaries') &
        (df['Periodicity'] == 'Current dollar index number')
    ][['Year', 'Period', 'Estimate']].copy()

    # Convert quarter-end months to dates
    month_map = {'March': 3, 'June': 6, 'September': 9, 'December': 12}
    labor['month'] = labor['Period'].map(month_map)
    labor['date'] = pd.to_datetime(
        labor['Year'].astype(str) + '-' + labor['month'].astype(str) + '-01'
    )
    labor = labor.sort_values('date').set_index('date')['Estimate']

    # Interpolate to monthly frequency
    monthly_index = pd.date_range(labor.index.min(), labor.index.max(), freq='MS')
    labor_monthly = labor.reindex(monthly_index).interpolate(method='linear')
    labor_monthly.index = labor_monthly.index.to_period('M')

    return labor_monthly


def load_cpi_data():
    """Load monthly construction PPI data (WPUSI012011). Deprecated alias."""
    return load_materials_ppi()


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
        reference_date: Target date for adjustment (period string like '2025-12')

    Returns:
        DataFrame with added columns:
        - inflation_factor: materials PPI adjustment factor
        - materials_ppi: PPI value at time of bid
        - labor_eci: labor ECI value at time of bid
        - labor_inflation_factor: labor ECI adjustment factor
    """
    materials = load_materials_ppi()
    labor = load_labor_eci()

    if reference_date is None:
        # Use the latest date available in both indices
        ref_materials = materials.index.max()
        ref_labor = labor.index.max()
    else:
        ref_materials = pd.Period(reference_date, freq='M')
        ref_labor = pd.Period(reference_date, freq='M')

    # Clamp to available data range
    ref_materials = min(ref_materials, materials.index.max())
    ref_labor = min(ref_labor, labor.index.max())

    ref_materials_val = materials[ref_materials]
    ref_labor_val = labor[ref_labor]

    df = df.copy()
    df['bid_date'] = pd.to_datetime(df['bid_date'])
    bid_periods = df['bid_date'].dt.to_period('M')

    # Materials PPI
    df['materials_ppi'] = bid_periods.map(materials)
    if df['materials_ppi'].isna().any():
        df['materials_ppi'] = df['materials_ppi'].fillna(materials.iloc[-1])
    df['inflation_factor'] = ref_materials_val / df['materials_ppi']

    # Labor ECI
    df['labor_eci'] = bid_periods.map(labor)
    if df['labor_eci'].isna().any():
        df['labor_eci'] = df['labor_eci'].fillna(labor.iloc[-1])
    df['labor_inflation_factor'] = ref_labor_val / df['labor_eci']

    # Keep cpi_at_bid as alias for backwards compatibility
    df['cpi_at_bid'] = df['materials_ppi']

    return df


def add_dummy_inflation_features(df):
    """
    Add neutral inflation features (no adjustment).

    Used when inflation adjustment is disabled.
    """
    df = df.copy()
    df['inflation_factor'] = 1.0
    df['cpi_at_bid'] = 1.0
    df['materials_ppi'] = 1.0
    df['labor_eci'] = 1.0
    df['labor_inflation_factor'] = 1.0
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
