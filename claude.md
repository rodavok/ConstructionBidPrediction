# Construction Bid Price Prediction

## Project Overview
Kaggle competition: Predict contractor total bid amounts for construction projects using pre-letting job characteristics.

**Competition URL:** https://www.kaggle.com/competitions/bidding-predictions-for-construction/overview

## Objective
Predict `total_bid` for each contractor-job pair in test set.

**Evaluation Metric:** RMSLE (Root Mean Squared Logarithmic Error)
- Lower is better
- Penalizes underestimation more than overestimation
- Scale-invariant

## Data Structure

### Files
- `raw_train.csv` - pay-item level training (multiple rows per job)
- `train_summary.csv` - job-level training (one row per contractor-job)
- `raw_test.csv` - pay-item level test data
- `test.csv` - job-level test data
- `sample_submission.csv` - submission format

### Key Columns
- `row_id` - unique contractor-job identifier (format: "JOB_xxx_CON_xxx")
- `job_id` - project identifier
- `contractor_id` - bidder identifier
- `bid_date` - date job was bid
- `total_bid` - TARGET (sum of all pay items)
- `job_category_description` / `category_id` - job type
- `primary_location` - job location
- `pay_item_id` - line item identifier (raw data only)
- `quantity` - pay item quantity (raw data only)
- `amount` - pay item UNIT PRICE (raw data only)

### Data Granularity
- **Pay-item level:** Multiple pay items per contractor-job (can engineer features)
- **Job level:** One row per contractor-job (evaluation unit)

## Current Approach: Baseline Model

### Phase 1 - Simple Linear Regression (Job-Level Only)
Use only aggregated job-level features from `train_summary.csv` and `test.csv`.

**Features:**
- `bid_date` (extract: year, month, day_of_year)
- `category_id` (one-hot or target encoding)
- `primary_location` (one-hot or target encoding)
- `contractor_id` (target encoding or frequency)

**Modeling:**
- Log-transform target: `log(total_bid + 1)`
- Train linear regression
- Predict and exponentiate: `exp(prediction) - 1`

**Why log-transform?**
Aligns with RMSLE metric and handles wide range of bid amounts.

### Future Improvements
- Engineer features from pay-item level data (item mix, quantities, complexity)
- Try gradient boosting (XGBoost, LightGBM, CatBoost)
- Contractor-specific bidding patterns
- Time-series features (lag features, moving averages)
- External data (inflation, construction cost indices)

## Submission Format
CSV with header:
```
row_id,total_bid
JOB_000022_CON_000039,4014359.20
```

## Notes
- All predictions must be non-negative (negative values clipped to 0)
- External data allowed if public and disclosed
- Collaboration and ensembling permitted
