import os
import joblib
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from common.evaluation import evaluate_model
from common.preprocessing import clean_infinite_values


def create_xgboost_model(objective='reg:squarederror'):
    """
    Create a new XGBoost model with optimized hyperparameters.

    :param objective: Objective function, defaults to 'reg:squarederror'
    :type objective: str
    :returns: Configured XGBoost model
    :rtype: xgboost.XGBRegressor
    """
    model = xgb.XGBRegressor(
        objective=objective,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric=['rmse', 'mae'],
        early_stopping_rounds=20,
        missing=np.nan
    )

    return model


def create_quantile_model(quantile_alpha=0.05):
    """
    Create a XGBoost model for quantile regression.

    :param quantile_alpha: Alpha value for quantile regression (0.05 for lower bound, 0.95 for upper bound)
    :type quantile_alpha: float
    :returns: Quantile XGBoost model
    :rtype: xgboost.XGBRegressor
    """
    model = xgb.XGBRegressor(
        objective='reg:quantileerror',
        quantile_alpha=quantile_alpha,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        missing=np.nan
    )

    return model


def load_models(model_dir='models'):
    """
    Load XGBoost models from disk.

    :param model_dir: Directory containing saved models
    :type model_dir: str
    :returns: Tuple of (models, quantile_models)
    :rtype: tuple(dict, dict)
    """
    models = {}
    quantile_models = {}

    for hours in [1, 2, 3]:
        enhanced_model_path = os.path.join(model_dir, f'enhanced_glucose_model_{hours}hr.pkl')
        enhanced_lower_path = os.path.join(model_dir, f'enhanced_glucose_model_{hours}hr_lower.pkl')
        enhanced_upper_path = os.path.join(model_dir, f'enhanced_glucose_model_{hours}hr_upper.pkl')

        if os.path.exists(enhanced_model_path) and os.path.exists(enhanced_lower_path) and os.path.exists(
                enhanced_upper_path):
            print(f"Loading enhanced models for {hours}-hour prediction")
            models[hours] = joblib.load(enhanced_model_path)
            quantile_models[hours] = {
                'lower': joblib.load(enhanced_lower_path),
                'upper': joblib.load(enhanced_upper_path)
            }
        else:
            model_path = os.path.join(model_dir, f'glucose_model_{hours}hr.pkl')
            lower_path = os.path.join(model_dir, f'glucose_model_{hours}hr_lower.pkl')
            upper_path = os.path.join(model_dir, f'glucose_model_{hours}hr_upper.pkl')

            if os.path.exists(model_path) and os.path.exists(lower_path) and os.path.exists(upper_path):
                print(f"Loading original models for {hours}-hour prediction")
                models[hours] = joblib.load(model_path)
                quantile_models[hours] = {
                    'lower': joblib.load(lower_path),
                    'upper': joblib.load(upper_path)
                }
            else:
                print(f"Warning: Models for {hours}-hour prediction not found")

    return models, quantile_models


def train_test_evaluate_model(X, y, model, model_name="Model"):
    """
    Train, test and evaluate a model.

    :param X: Features dataframe
    :type X: pandas.DataFrame
    :param y: Target values
    :type y: pandas.Series
    :param model: Model to train
    :type model: xgboost.XGBRegressor
    :param model_name: Name for reporting
    :type model_name: str
    :returns: Trained model and evaluation metrics
    :rtype: tuple(xgboost.XGBRegressor, dict)
    """
    from sklearn.model_selection import train_test_split

    # Clean data
    X = clean_infinite_values(X)
    X = X.fillna(method='ffill').fillna(0)

    try:
        # Split data with stratification if possible
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=X['SUBJECT_ID'] if len(X['SUBJECT_ID'].unique()) > 1 else None
        )
    except ValueError:
        # Fall back to simple split if stratification fails
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training data size: {X_train.shape[0]} samples")
    print(f"Test data size: {X_test.shape[0]} samples")

    # Final check for infinities and NaNs
    if np.isinf(X_train.values).any() or np.isnan(X_train.values).any():
        print("Warning: Infinities or NaNs found in training data after preparation!")
        X_train = pd.DataFrame(X_train, columns=X.columns).fillna(0)
        X_test = pd.DataFrame(X_test, columns=X.columns).fillna(0)

        X_train = X_train.replace([np.inf, -np.inf], [1e10, -1e10])
        X_test = X_test.replace([np.inf, -np.inf], [1e10, -1e10])

    # Train the model
    print(f"Training {model_name}...")
    try:
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=True)
    except Exception as e:
        print(f"Error during model fitting: {e}")
        print("Trying with simpler model configuration...")

        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            missing=np.nan
        )

        model.fit(X_train, y_train)

    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test, model_name)

    return model, metrics
