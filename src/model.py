"""
Gradient Boosting Model with Line-Item Features
Uses quantity data and learned unit prices to estimate job costs
Supports LightGBM, XGBoost, and CatBoost
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, KFold
from scipy.stats import randint, uniform
import warnings
import mlflow

from utils import extract_date_features, create_submission, add_inflation_features, load_cpi_data, add_dummy_inflation_features

warnings.filterwarnings('ignore')

# Resolve project root from this file's location so it works on any machine
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
mlflow.set_tracking_uri(f"sqlite:///{os.path.join(_PROJECT_ROOT, 'mlflow.db')}")


def _fix_mlflow_paths():
    """
    Patch all absolute paths in mlflow.db that point to a different machine.
    This happens when mlflow.db is synced between machines with different usernames/paths.
    Updates both the experiment artifact_location and all run artifact_uris.
    """
    import sqlite3
    exp = mlflow.get_experiment_by_name("construction-price-prediction")
    if exp is None:
        return

    old_root = None
    expected_exp_loc = os.path.join(_PROJECT_ROOT, "mlruns", exp.experiment_id)
    if os.path.normpath(exp.artifact_location) != os.path.normpath(expected_exp_loc):
        # Infer the old root by stripping the known suffix from artifact_location
        suffix = os.path.join("mlruns", exp.experiment_id)
        if exp.artifact_location.endswith(suffix):
            old_root = exp.artifact_location[: -len(suffix)].rstrip("/")

    if old_root is None:
        return  # Nothing to fix

    db_path = os.path.join(_PROJECT_ROOT, "mlflow.db")
    conn = sqlite3.connect(db_path)

    conn.execute(
        "UPDATE experiments SET artifact_location = replace(artifact_location, ?, ?) "
        "WHERE artifact_location LIKE ?",
        (old_root, _PROJECT_ROOT, f"{old_root}%"),
    )
    runs_updated = conn.execute(
        "UPDATE runs SET artifact_uri = replace(artifact_uri, ?, ?) "
        "WHERE artifact_uri LIKE ?",
        (old_root, _PROJECT_ROOT, f"{old_root}%"),
    ).rowcount

    conn.commit()
    conn.close()
    print(f"[mlflow] Repointed paths: {old_root} -> {_PROJECT_ROOT} ({runs_updated} runs updated)")


_fix_mlflow_paths()
mlflow.set_experiment("construction-price-prediction")

TUNED_PARAMS_FILE = "tuned_params.json"


def load_tuned_params():
    """Load previously tuned parameters from disk."""
    if os.path.exists(TUNED_PARAMS_FILE):
        with open(TUNED_PARAMS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_tuned_params(model_name, params):
    """Save tuned parameters for a model to disk."""
    all_params = load_tuned_params()
    # Convert numpy types to Python types for JSON serialization
    clean_params = {}
    for k, v in params.items():
        if hasattr(v, 'item'):  # numpy scalar
            clean_params[k] = v.item()
        else:
            clean_params[k] = v
    all_params[model_name] = clean_params
    with open(TUNED_PARAMS_FILE, 'w') as f:
        json.dump(all_params, f, indent=2)
    print(f"Saved tuned params for '{model_name}' to {TUNED_PARAMS_FILE}")


def compute_recency_weights(bid_dates, alpha):
    """
    Exponential decay sample weights that up-weight recent bids.

    alpha=0  → uniform weights (no decay, same as before)
    alpha=1  → a bid from 1 year ago gets weight exp(-1) ≈ 0.37x vs most recent
    alpha=2  → a bid from 1 year ago gets weight exp(-2) ≈ 0.14x vs most recent

    Weights are normalized to mean=1 so the effective dataset size and learning
    rate are unchanged; only the relative emphasis shifts toward recent data.
    """
    if alpha == 0:
        return None
    dates = pd.to_datetime(bid_dates)
    days_old = (dates.max() - dates).dt.days.values.astype(float)
    weights = np.exp(-alpha * days_old / 365.0)
    weights /= weights.mean()
    return weights


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

    def train(X_tr, y_tr, X_val, y_val, params, num_boost_round, sample_weight=None):
        train_data = lgb.Dataset(X_tr, label=y_tr, weight=sample_weight)
        val_data = lgb.Dataset(X_val, label=y_val)
        model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        return model

    def train_final(X_train, y_train, params, num_boost_round, sample_weight=None):
        train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
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

    def train(X_tr, y_tr, X_val, y_val, params, num_boost_round, sample_weight=None):
        dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=sample_weight)
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

    def train_final(X_train, y_train, params, num_boost_round, sample_weight=None):
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
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

    def train(X_tr, y_tr, X_val, y_val, params, num_boost_round, sample_weight=None):
        model = CatBoostRegressor(**params, iterations=num_boost_round)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=100,
            verbose=False,
            sample_weight=sample_weight,
        )
        return model

    def train_final(X_train, y_train, params, num_boost_round, sample_weight=None):
        model = CatBoostRegressor(**params, iterations=num_boost_round)
        model.fit(X_train, y_train, verbose=False, sample_weight=sample_weight)
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


def get_ridge_backend():
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    def get_default_params(_config):
        return {
            'alpha': 1.0,
        }

    def get_tuned_params(tuned):
        return {
            'alpha': tuned.get('ridge__alpha', 1.0),
        }

    def get_tune_distributions():
        from scipy.stats import loguniform
        return {
            'ridge__alpha': loguniform(1e-3, 1e3),
        }

    def get_tuning_model():
        return Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge())
        ])

    def train(X_tr, y_tr, _X_val, _y_val, params, _num_boost_round, sample_weight=None):
        # Validation data and num_boost_round ignored for linear models
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(**params))
        ])
        model.fit(X_tr, y_tr, ridge__sample_weight=sample_weight)
        return model

    def train_final(X_train, y_train, params, _num_boost_round, sample_weight=None):
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(**params))
        ])
        model.fit(X_train, y_train, ridge__sample_weight=sample_weight)
        return model

    def predict(model, X):
        return model.predict(X)

    def get_feature_importance(model, feature_cols):
        coeffs = model.named_steps['ridge'].coef_
        return pd.DataFrame({
            'feature': feature_cols,
            'importance': np.abs(coeffs)
        }).sort_values('importance', ascending=False)

    return {
        'name': 'ridge',
        'get_default_params': get_default_params,
        'get_tuned_params': get_tuned_params,
        'get_tune_distributions': get_tune_distributions,
        'get_tuning_model': get_tuning_model,
        'train': train,
        'train_final': train_final,
        'predict': predict,
        'get_feature_importance': get_feature_importance,
    }


def get_elasticnet_backend():
    from sklearn.linear_model import ElasticNet
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    def get_default_params(_config):
        return {
            'alpha': 1.0,
            'l1_ratio': 0.5,  # 0 = Ridge, 1 = Lasso
            'max_iter': 5000,
        }

    def get_tuned_params(tuned):
        return {
            'alpha': tuned.get('elasticnet__alpha', 1.0),
            'l1_ratio': tuned.get('elasticnet__l1_ratio', 0.5),
            'max_iter': 5000,
        }

    def get_tune_distributions():
        from scipy.stats import loguniform, uniform
        return {
            'elasticnet__alpha': loguniform(1e-3, 1e3),
            'elasticnet__l1_ratio': uniform(0.01, 0.98),  # Avoid pure Ridge/Lasso edge cases
        }

    def get_tuning_model():
        return Pipeline([
            ('scaler', StandardScaler()),
            ('elasticnet', ElasticNet(max_iter=5000))
        ])

    def train(X_tr, y_tr, _X_val, _y_val, params, _num_boost_round, sample_weight=None):
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('elasticnet', ElasticNet(**params))
        ])
        model.fit(X_tr, y_tr, elasticnet__sample_weight=sample_weight)
        return model

    def train_final(X_train, y_train, params, _num_boost_round, sample_weight=None):
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('elasticnet', ElasticNet(**params))
        ])
        model.fit(X_train, y_train, elasticnet__sample_weight=sample_weight)
        return model

    def predict(model, X):
        return model.predict(X)

    def get_feature_importance(model, feature_cols):
        coeffs = model.named_steps['elasticnet'].coef_
        return pd.DataFrame({
            'feature': feature_cols,
            'importance': np.abs(coeffs)
        }).sort_values('importance', ascending=False)

    return {
        'name': 'elasticnet',
        'get_default_params': get_default_params,
        'get_tuned_params': get_tuned_params,
        'get_tune_distributions': get_tune_distributions,
        'get_tuning_model': get_tuning_model,
        'train': train,
        'train_final': train_final,
        'predict': predict,
        'get_feature_importance': get_feature_importance,
    }


def get_stacking_backend(stack_models=None):
    """
    Stacking ensemble with configurable base models. Ridge as meta-learner.
    Uses out-of-fold predictions to train the meta-model (no leakage).

    Base model hyperparameters are loaded from tuned_params.json if available.
    Run each model with --tune first to populate the file:
        python src/model.py --model lightgbm --tune
        python src/model.py --model stacking --stack-models lightgbm elasticnet
    """
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRegressor
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.ensemble import StackingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    if stack_models is None:
        stack_models = ['lightgbm', 'xgboost', 'catboost']

    # Store for use in get_feature_importance
    _stack_models = stack_models

    def _make_estimator(name, saved_params):
        """Create a single base estimator by name."""
        if name == 'lightgbm':
            params = {
                'n_estimators': saved_params.get('n_estimators', 500),
                'num_leaves': saved_params.get('num_leaves', 63),
                'max_depth': saved_params.get('max_depth', -1),
                'learning_rate': saved_params.get('learning_rate', 0.05),
                'min_child_samples': saved_params.get('min_child_samples', 20),
                'subsample': saved_params.get('subsample', 0.8),
                'colsample_bytree': saved_params.get('colsample_bytree', 0.8),
                'reg_alpha': saved_params.get('reg_alpha', 0),
                'reg_lambda': saved_params.get('reg_lambda', 0),
                'verbose': -1,
                'n_jobs': -1,
                'random_state': 42,
            }
            return ('lgb', lgb.LGBMRegressor(**params))

        elif name == 'xgboost':
            params = {
                'n_estimators': saved_params.get('n_estimators', 500),
                'max_depth': saved_params.get('max_depth', 6),
                'learning_rate': saved_params.get('learning_rate', 0.05),
                'min_child_weight': saved_params.get('min_child_weight', 1),
                'subsample': saved_params.get('subsample', 0.8),
                'colsample_bytree': saved_params.get('colsample_bytree', 0.8),
                'reg_alpha': saved_params.get('reg_alpha', 0),
                'reg_lambda': saved_params.get('reg_lambda', 1),
                'n_jobs': -1,
                'random_state': 42,
            }
            return ('xgb', xgb.XGBRegressor(**params))

        elif name == 'catboost':
            params = {
                'iterations': saved_params.get('iterations', 500),
                'depth': saved_params.get('depth', 6),
                'learning_rate': saved_params.get('learning_rate', 0.05),
                'l2_leaf_reg': saved_params.get('l2_leaf_reg', 3),
                'bagging_temperature': saved_params.get('bagging_temperature', 1),
                'verbose': False,
                'random_seed': 42,
            }
            return ('catboost', CatBoostRegressor(**params))

        elif name == 'ridge':
            params = {
                'alpha': saved_params.get('alpha', 1.0),
            }
            # Linear models need scaling
            return ('ridge', Pipeline([
                ('scaler', StandardScaler()),
                ('ridge', Ridge(**params))
            ]))

        elif name == 'elasticnet':
            params = {
                'alpha': saved_params.get('alpha', 1.0),
                'l1_ratio': saved_params.get('l1_ratio', 0.5),
                'max_iter': 5000,
            }
            return ('elasticnet', Pipeline([
                ('scaler', StandardScaler()),
                ('elasticnet', ElasticNet(**params))
            ]))

        else:
            raise ValueError(f"Unknown model for stacking: {name}")

    def _make_base_estimators():
        """Create base estimators for stacking, using tuned params if available."""
        saved = load_tuned_params()
        estimators = []

        for name in _stack_models:
            model_params = saved.get(name, {})
            if model_params:
                print(f"  {name}: using tuned params")
            else:
                print(f"  {name}: using defaults (run --model {name} --tune to tune)")
            estimators.append(_make_estimator(name, model_params))

        return estimators

    def _make_stacking_model(final_alpha=1.0):
        """Create the stacking regressor."""
        return StackingRegressor(
            estimators=_make_base_estimators(),
            final_estimator=Ridge(alpha=final_alpha),
            cv=5,  # 5-fold CV for generating meta-features
            n_jobs=-1,
            passthrough=False,  # Only use base model predictions as meta-features
        )

    def get_default_params(_config):
        return {
            'final_alpha': 1.0,
        }

    def get_tuned_params(tuned):
        return {
            'final_alpha': tuned.get('final_estimator__alpha', 1.0),
        }

    def get_tune_distributions():
        from scipy.stats import loguniform
        return {
            'final_estimator__alpha': loguniform(1e-3, 1e3),
        }

    def get_tuning_model():
        return _make_stacking_model()

    def train(X_tr, y_tr, _X_val, _y_val, params, _num_boost_round, sample_weight=None):
        # StackingRegressor handles internal CV for meta-features.
        # sample_weight is not propagated: sklearn's StackingRegressor doesn't
        # reliably route it through internal cross-val to base estimators.
        if sample_weight is not None:
            print("  [stacking] recency_weight ignored (not supported for stacking)")
        model = _make_stacking_model(final_alpha=params.get('final_alpha', 1.0))
        model.fit(X_tr, y_tr)
        return model

    def train_final(X_train, y_train, params, _num_boost_round, sample_weight=None):
        model = _make_stacking_model(final_alpha=params.get('final_alpha', 1.0))
        model.fit(X_train, y_train)
        return model

    def predict(model, X):
        return model.predict(X)

    def _get_importance_from_estimator(estimator, name):
        """Extract feature importance/coefficients from an estimator."""
        # Tree-based models have feature_importances_
        if hasattr(estimator, 'feature_importances_'):
            return estimator.feature_importances_
        # Pipeline-wrapped linear models
        if hasattr(estimator, 'named_steps'):
            if 'ridge' in estimator.named_steps:
                return np.abs(estimator.named_steps['ridge'].coef_)
            if 'elasticnet' in estimator.named_steps:
                return np.abs(estimator.named_steps['elasticnet'].coef_)
        # Direct linear models
        if hasattr(estimator, 'coef_'):
            return np.abs(estimator.coef_)
        return None

    def get_feature_importance(model, feature_cols):
        importances = {}
        normalized = []

        # Map short names to full names used in estimators
        name_map = {'lgb': 'lightgbm', 'xgb': 'xgboost'}

        for short_name, estimator in model.named_estimators_.items():
            full_name = name_map.get(short_name, short_name)
            imp = _get_importance_from_estimator(estimator, full_name)
            if imp is not None:
                importances[f'importance_{short_name}'] = imp
                # Normalize to 0-1 scale
                norm = imp / (imp.max() + 1e-10)
                normalized.append(norm)

        # Average normalized importances
        if normalized:
            combined = np.mean(normalized, axis=0)
        else:
            combined = np.zeros(len(feature_cols))

        result = {'feature': feature_cols, 'importance': combined}
        result.update(importances)
        return pd.DataFrame(result).sort_values('importance', ascending=False)

    return {
        'name': 'stacking',
        'get_default_params': get_default_params,
        'get_tuned_params': get_tuned_params,
        'get_tune_distributions': get_tune_distributions,
        'get_tuning_model': get_tuning_model,
        'train': train,
        'train_final': train_final,
        'predict': predict,
        'get_feature_importance': get_feature_importance,
    }


def _get_gpu_params(model_name):
    """Return GPU-specific params to overlay on top of any model config."""
    if model_name == 'lightgbm':
        return {'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0}
    elif model_name == 'xgboost':
        return {'device': 'cuda', 'tree_method': 'hist'}
    elif model_name == 'catboost':
        return {'task_type': 'GPU', 'devices': '0'}
    return {}


def get_backend(name, config=None):
    backends = {
        'lightgbm': get_lightgbm_backend,
        'xgboost': get_xgboost_backend,
        'catboost': get_catboost_backend,
        'ridge': get_ridge_backend,
        'elasticnet': get_elasticnet_backend,
        'stacking': get_stacking_backend,
    }
    if name not in backends:
        raise ValueError(f"Unknown model: {name}. Choose from: {list(backends.keys())}")
    if name == 'stacking' and config is not None:
        return backends[name](stack_models=config.get('stack_models', ['lightgbm', 'xgboost', 'catboost']))
    return backends[name]()


# =============================================================================
# EXPERIMENT CONFIG
# =============================================================================
DEFAULT_CONFIG = {
    # Model selection
    'model': 'lightgbm',  # 'lightgbm', 'xgboost', 'catboost'
    'stack_models': ['lightgbm', 'xgboost', 'catboost'],  # Base models for stacking

    # Data source
    'use_aggregated': False,  # True = train_summary.csv/test.csv, False = raw_train.csv/raw_test.csv

    # Feature engineering
    'use_inflation': True,
    'inflation_lag_months': 0,  # 0 = same month, 3 = use price from 3 months prior
    'log_inflation': False,  # Log-transform inflation factors (better for linear models)
    'use_contractor_history': False,  # Add contractor win count at bid time
    'use_competition_intensity': False,  # Add number of bidders per job

    # Recency weighting: exponential decay factor (0 = uniform, 1 = sample from 1 year
    # ago weighted at exp(-1)≈0.37x relative to the most recent sample)
    'recency_weight': 0.0,

    # Markup ratio target: predict log(bid / estimated_cost) instead of log(bid).
    # Strips out the dominant cost signal so the model learns contractor markup patterns.
    # Final predictions are decoded as exp(markup_pred) * estimated_cost.
    # CV RMSE is always reported in log-bid space for comparability.
    'use_markup_target': False,

    # Restrict model training to bids on or after this date (ISO format, e.g. '2022-07-01').
    # The price lookup (estimated_cost features) still uses ALL training data to maximise
    # pay-item coverage; only the regression target and features fed to the GBM are filtered.
    # Useful for dropping pre-COVID bids whose market conditions no longer reflect current prices.
    'train_from': None,

    # Cross-validation
    'time_based_cv': True,
    'cv_splits': 5,

    # GPU acceleration
    'use_gpu': False,

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

        if config.get('log_inflation', False):
            print("Inflation factors will be log-transformed")
    else:
        print("Inflation adjustment: DISABLED")
        train = add_dummy_inflation_features(train)
        test = add_dummy_inflation_features(test)

    if config.get('use_contractor_history', False):
        print("Adding contractor win history feature...")
        train, test = compute_contractor_win_history(train, test)
    else:
        # Add dummy columns so downstream code doesn't break
        train['contractor_prior_wins'] = 0
        train['contractor_prior_wins_category'] = 0
        train['contractor_prior_wins_location'] = 0
        test['contractor_prior_wins'] = 0
        test['contractor_prior_wins_category'] = 0
        test['contractor_prior_wins_location'] = 0

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

    Computes three features (all using only wins strictly before bid_date):
    - contractor_prior_wins: total wins across all jobs
    - contractor_prior_wins_category: wins in the same job_category_description
    - contractor_prior_wins_location: wins in the same primary_location

    For training: uses only wins strictly before each bid's date (no leakage).
    For test: uses all training wins before each test bid's date.
    """
    # Get unique bids (job_id + contractor_id level) with category and location
    train_bids = train_df.groupby(['job_id', 'contractor_id']).agg({
        'bid_date': 'first',
        'total_bid': 'first',
        'job_category_description': 'first',
        'primary_location': 'first'
    }).reset_index()
    train_bids['bid_date'] = pd.to_datetime(train_bids['bid_date'])

    # Identify winner for each job (lowest bid) - include category and location
    winners = train_bids.loc[train_bids.groupby('job_id')['total_bid'].idxmin()]
    winners = winners[['job_id', 'contractor_id', 'bid_date', 'job_category_description', 'primary_location']].copy()
    winners.columns = ['job_id', 'winner_contractor', 'win_date', 'win_category', 'win_location']

    def count_prior_wins(row, win_history):
        """Count contractor's wins strictly before bid_date."""
        contractor = row['contractor_id']
        bid_date = row['bid_date']
        contractor_wins = win_history[
            (win_history['winner_contractor'] == contractor) &
            (win_history['win_date'] < bid_date)
        ]

        # Total wins
        total_wins = len(contractor_wins)

        # Wins in same category
        category_wins = (contractor_wins['win_category'] == row['job_category_description']).sum()

        # Wins in same location
        location_wins = (contractor_wins['win_location'] == row['primary_location']).sum()

        return total_wins, category_wins, location_wins

    # Compute for training data
    win_counts = train_bids.apply(lambda r: count_prior_wins(r, winners), axis=1)
    train_bids['contractor_prior_wins'] = win_counts.apply(lambda x: x[0])
    train_bids['contractor_prior_wins_category'] = win_counts.apply(lambda x: x[1])
    train_bids['contractor_prior_wins_location'] = win_counts.apply(lambda x: x[2])

    # Merge back to line-item level
    win_cols = ['job_id', 'contractor_id', 'contractor_prior_wins',
                'contractor_prior_wins_category', 'contractor_prior_wins_location']
    train_df = train_df.merge(train_bids[win_cols], on=['job_id', 'contractor_id'], how='left')
    for col in ['contractor_prior_wins', 'contractor_prior_wins_category', 'contractor_prior_wins_location']:
        train_df[col] = train_df[col].fillna(0).astype(int)

    if test_df is not None:
        # For test, use all training wins before each test bid's date
        test_bids = test_df.groupby(['job_id', 'contractor_id']).agg({
            'bid_date': 'first',
            'job_category_description': 'first',
            'primary_location': 'first'
        }).reset_index()
        test_bids['bid_date'] = pd.to_datetime(test_bids['bid_date'])

        win_counts = test_bids.apply(lambda r: count_prior_wins(r, winners), axis=1)
        test_bids['contractor_prior_wins'] = win_counts.apply(lambda x: x[0])
        test_bids['contractor_prior_wins_category'] = win_counts.apply(lambda x: x[1])
        test_bids['contractor_prior_wins_location'] = win_counts.apply(lambda x: x[2])

        test_df = test_df.merge(test_bids[win_cols], on=['job_id', 'contractor_id'], how='left')
        for col in ['contractor_prior_wins', 'contractor_prior_wins_category', 'contractor_prior_wins_location']:
            test_df[col] = test_df[col].fillna(0).astype(int)

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

    Note: The 'amount' column IS the unit price (quantity × amount = total_bid).

    Expects inflation_factor column to already exist.
    """
    train_df = train_df.copy()

    # amount is already the unit price (not line total)
    train_df['unit_price'] = train_df['amount']

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


def aggregate_job_features(df, is_train=True, log_inflation=False):
    """
    Aggregate line-item features to job level.

    Args:
        df: Line-item DataFrame
        is_train: Whether this is training data (includes total_bid)
        log_inflation: If True, log-transform inflation factors for model features
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

        # Inflation adjustment factors
        'inflation_factor': 'first',  # Materials PPI adjustment factor
        'labor_inflation_factor': 'first',  # Labor ECI adjustment factor

        # Contractor history
        'contractor_prior_wins': 'first',
        'contractor_prior_wins_category': 'first',
        'contractor_prior_wins_location': 'first',

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

    # Log-transform inflation factors if requested (better for linear models)
    if log_inflation:
        agg['inflation_factor_first'] = np.log(agg['inflation_factor_first'])
        agg['labor_inflation_factor_first'] = np.log(agg['labor_inflation_factor_first'])

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


def train_and_predict_cv(train_raw, test_raw, config, backend, tuned_params=None,
                         price_lookup_data=None):
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

    if config.get('use_gpu'):
        params.update(_get_gpu_params(backend['name']))

    # Get unique job+contractor combinations with their bid dates
    train_raw = train_raw.copy()
    train_raw['bid_date_parsed'] = pd.to_datetime(train_raw['bid_date'])
    job_keys = train_raw.groupby(['job_id', 'contractor_id']).agg({
        'bid_date_parsed': 'first'
    }).reset_index()
    job_keys['key'] = job_keys['job_id'] + '__' + job_keys['contractor_id']

    # Price lookup source: full historical data if provided, otherwise the (possibly filtered)
    # train_raw.  Using the full dataset maximises pay-item coverage so estimated_cost_sum
    # stays accurate even when model training is restricted to a recent window.
    lookup_base = price_lookup_data.copy() if price_lookup_data is not None else train_raw
    lookup_base['bid_date_parsed'] = pd.to_datetime(lookup_base['bid_date'])
    lookup_base['key'] = lookup_base['job_id'] + '__' + lookup_base['contractor_id']

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

        # Build price lookup from all historical data except the current validation fold.
        # This avoids leakage (val fold's own unit prices can't inform their cost estimate)
        # while maximising pay-item coverage when price_lookup_data spans more history.
        lookup_fold = lookup_base[~lookup_base['key'].isin(val_keys)].copy()
        item_prices, cat_prices, unit_prices = build_unit_price_lookup(lookup_fold)

        # Estimate costs using lookup prices (applied to both train and val folds)
        train_fold = estimate_line_item_cost(train_fold, item_prices, cat_prices, unit_prices)
        val_fold = estimate_line_item_cost(val_fold, item_prices, cat_prices, unit_prices)

        # Aggregate to job level
        log_inf = config.get('log_inflation', False)
        train_agg = aggregate_job_features(train_fold, is_train=True, log_inflation=log_inf)
        val_agg = aggregate_job_features(val_fold, is_train=True, log_inflation=log_inf)

        # Prepare features
        X_tr, y_tr, _, _, cols = prepare_features(train_agg.copy(), val_agg.copy())
        X_val, y_val, _, _, _ = prepare_features(val_agg.copy(), train_agg.copy())

        # Markup ratio target: replace log(bid) with log(bid / estimated_cost)
        if config.get('use_markup_target', False):
            cost_tr = np.maximum(train_agg['estimated_cost_sum'].values, 1.0)
            y_tr = np.log(np.maximum(train_agg['total_bid_first'].values, 1.0) / cost_tr)

        # Align columns
        if feature_cols is None:
            feature_cols = cols
        X_tr = X_tr.reindex(columns=feature_cols, fill_value=0)
        X_val = X_val.reindex(columns=feature_cols, fill_value=0)

        # Recency weights: computed from bid dates of the training fold
        alpha = config.get('recency_weight', 0.0)
        sample_weight = compute_recency_weights(train_agg['bid_date_first'], alpha)

        model = backend['train'](X_tr, y_tr, X_val, y_val, params, num_boost_round,
                                 sample_weight=sample_weight)
        models.append(model)

        y_pred_val = backend['predict'](model, X_val)
        if config.get('use_markup_target', False):
            # Decode markup prediction back to log-bid space for comparable RMSE
            cost_val = np.maximum(val_agg['estimated_cost_sum'].values, 1.0)
            y_pred_log_bid = y_pred_val + np.log(cost_val)
            rmse = np.sqrt(np.mean((y_val - y_pred_log_bid) ** 2))
        else:
            rmse = np.sqrt(np.mean((y_val - y_pred_val) ** 2))
        cv_scores.append(rmse)
        print(f"  Fold {fold+1}: RMSE = {rmse:.4f}")

    print(f"\nCV RMSE: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    # For final predictions, use all training data
    print("\nTraining final model on all data...")
    # Use lookup_base (full historical data if provided) for best price coverage on test set
    item_prices, cat_prices, unit_prices = build_unit_price_lookup(lookup_base)
    train_raw_est = estimate_line_item_cost(train_raw.copy(), item_prices, cat_prices, unit_prices)
    test_raw_est = estimate_line_item_cost(test_raw.copy(), item_prices, cat_prices, unit_prices)

    train_agg = aggregate_job_features(train_raw_est, is_train=True, log_inflation=log_inf)
    test_agg = aggregate_job_features(test_raw_est, is_train=False, log_inflation=log_inf)

    X_train, y_train, X_test, row_ids, _ = prepare_features(train_agg, test_agg)
    X_train = X_train.reindex(columns=feature_cols, fill_value=0)
    X_test = X_test.reindex(columns=feature_cols, fill_value=0)

    if config.get('use_markup_target', False):
        cost_train = np.maximum(train_agg['estimated_cost_sum'].values, 1.0)
        y_train = np.log(np.maximum(train_agg['total_bid_first'].values, 1.0) / cost_train)

    # Feature importance (from last CV model)
    print("\nTop 20 Feature Importance:")
    importance = backend['get_feature_importance'](models[-1], feature_cols)
    print(importance.head(20).to_string(index=False))

    # Train final model (apply recency weighting to full training set)
    final_weight = compute_recency_weights(train_agg['bid_date_first'], config.get('recency_weight', 0.0))
    final_model = backend['train_final'](X_train, y_train, params, num_boost_round,
                                         sample_weight=final_weight)

    predictions_log = backend['predict'](final_model, X_test)
    if config.get('use_markup_target', False):
        test_cost = np.maximum(test_agg['estimated_cost_sum'].values, 1.0)
        predictions = np.exp(predictions_log) * test_cost
    else:
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

    if config.get('use_gpu'):
        params.update(_get_gpu_params(backend['name']))

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

        alpha = config.get('recency_weight', 0.0)
        sample_weight = compute_recency_weights(train_fold['bid_date_parsed'], alpha)

        model = backend['train'](X_tr, y_tr, X_val, y_val, params, num_boost_round,
                                 sample_weight=sample_weight)
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

    # Train final model (apply recency weighting to full training set)
    final_weight = compute_recency_weights(train_df['bid_date_parsed'], config.get('recency_weight', 0.0))
    final_model = backend['train_final'](X_train, y_train, params, num_boost_round,
                                         sample_weight=final_weight)

    predictions_log = backend['predict'](final_model, X_test)
    predictions = np.expm1(predictions_log)
    predictions = np.maximum(predictions, 0)

    return predictions, row_ids, cv_scores, importance


def main(config):
    with mlflow.start_run(nested=bool(mlflow.active_run())):
        # Log config
        mlflow.log_params(config)

        print("=" * 60)
        print("EXPERIMENT CONFIG:")
        for k, v in config.items():
            print(f"  {k}: {v}")
        print("=" * 60)

        # Get model backend
        backend = get_backend(config['model'], config)
        print(f"\nUsing model: {backend['name']}")

        print("\nLoading data...")
        train_data, test_data = load_and_prepare_data(config)
        print(f"  Train: {train_data.shape}, Test: {test_data.shape}")

        # train_from: restrict the model training window while keeping the full
        # dataset available for price lookup (so estimated_cost features stay accurate).
        price_lookup_data = None
        if config.get('train_from'):
            cutoff = pd.to_datetime(config['train_from'])
            mask = pd.to_datetime(train_data['bid_date']) >= cutoff
            price_lookup_data = train_data          # full data for price lookup
            train_data = train_data[mask].reset_index(drop=True)
            n_jobs_full = price_lookup_data.groupby(['job_id', 'contractor_id']).ngroups
            n_jobs_filt = train_data.groupby(['job_id', 'contractor_id']).ngroups
            print(f"  train_from={config['train_from']}: {n_jobs_filt:,} of {n_jobs_full:,} unique jobs used for model training")
            print(f"  Price lookup still uses all {n_jobs_full:,} jobs for maximum pay-item coverage")

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

                log_inf = config.get('log_inflation', False)
                train_agg = aggregate_job_features(train_est, is_train=True, log_inflation=log_inf)
                test_agg = aggregate_job_features(test_est, is_train=False, log_inflation=log_inf)

                # Keep bid_date before prepare_features drops it
                bid_dates = train_agg['bid_date_first']

                X_train, y_train, _, _, _ = prepare_features(train_agg, test_agg)

            tuned_params = tune_hyperparameters(X_train, y_train, backend, bid_dates,
                                                n_iter=config['tune_iterations'],
                                                time_based_cv=config['time_based_cv'])
            mlflow.log_params({f"tuned_{k}": v for k, v in tuned_params.items()})

            # Save tuned params for reuse (e.g., in stacking)
            save_tuned_params(config['model'], tuned_params)

        print("\nTraining with CV...")
        if use_aggregated:
            predictions, row_ids, cv_scores, feature_importance = train_and_predict_cv_aggregated(
                train_data, test_data, config, backend, tuned_params=tuned_params
            )
        else:
            predictions, row_ids, cv_scores, feature_importance = train_and_predict_cv(
                train_data, test_data, config, backend, tuned_params=tuned_params,
                price_lookup_data=price_lookup_data,
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
        run_id = mlflow.active_run().info.run_id
        run_id_short = run_id[:8]  # First 8 chars for readability

        # Save to submissions directory with run ID
        import os
        os.makedirs("submissions", exist_ok=True)
        submission_path = f"submissions/submission_{run_id_short}.csv"

        create_submission(row_ids, predictions, filename=submission_path)
        mlflow.log_artifact(submission_path)

        print(f"\nMLflow run ID: {run_id}")
        print(f"To view run: mlflow runs get -r {run_id}")

        return np.mean(cv_scores)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['lightgbm', 'xgboost', 'catboost', 'ridge', 'elasticnet', 'stacking'], help='Model to use')
    parser.add_argument('--stack-models', nargs='+', choices=['lightgbm', 'xgboost', 'catboost', 'ridge', 'elasticnet'],
                        help='Base models for stacking (e.g., --stack-models lightgbm elasticnet)')
    parser.add_argument('--aggregated', action='store_true', help='Use aggregated dataset (train_summary.csv) instead of line-items')
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning')
    parser.add_argument('--tune-iterations', type=int, help='Number of tuning iterations')
    parser.add_argument('--random-cv', action='store_true', help='Use random CV instead of time-based CV')
    parser.add_argument('--cv-folds', type=int, help='Number of CV folds (default: 5)')
    parser.add_argument('--no-inflation', action='store_true', help='Disable inflation adjustment')
    parser.add_argument('--inflation-lag', type=int, help='Months to lag inflation data')
    parser.add_argument('--log-inflation', action='store_true', help='Log-transform inflation factors')
    parser.add_argument('--contractor-history', action='store_true', help='Add contractor prior wins feature')
    parser.add_argument('--competition-intensity', action='store_true', help='Add number of bidders per job feature')
    parser.add_argument('--gpu', action='store_true', help='Enable GPU acceleration (CUDA for XGBoost/CatBoost, OpenCL for LightGBM)')
    parser.add_argument('--recency-weight', type=float, help='Exponential decay factor for recency weighting (0=uniform, 1=~37%% weight to 1-year-old bids)')
    parser.add_argument('--markup-ratio', action='store_true', help='Predict log(bid/estimated_cost) instead of log(bid); model learns contractor markup patterns')
    parser.add_argument('--train-from', type=str, help='Only train model on bids from this date onward (ISO format, e.g. 2022-07-01). Price lookup still uses all data.')
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
    if args.log_inflation:
        overrides['log_inflation'] = True
    if args.contractor_history:
        overrides['use_contractor_history'] = True
    if args.competition_intensity:
        overrides['use_competition_intensity'] = True
    if args.gpu:
        overrides['use_gpu'] = True
    if args.recency_weight is not None:
        overrides['recency_weight'] = args.recency_weight
    if args.markup_ratio:
        overrides['use_markup_target'] = True
    if args.train_from:
        overrides['train_from'] = args.train_from
    if args.stack_models:
        overrides['stack_models'] = args.stack_models

    config = get_config(**overrides)
    main(config)
