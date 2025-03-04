import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a trained model and print performance metrics.
    Works with both XGBoost and LSTM models.

    :param model: Trained model
    :param X_test: Test features
    :type X_test: pandas.DataFrame or numpy.ndarray
    :param y_test: True target values
    :type y_test: pandas.Series or numpy.ndarray
    :param model_name: Name to display in output
    :type model_name: str
    :returns: Dict with evaluation metrics
    :rtype: dict
    """
    y_pred = model.predict(X_test)

    # Reshape if needed
    if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()

    if isinstance(y_test, pd.Series):
        y_test = y_test.values

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Calculate MAPE safely (avoiding division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, np.nan))) * 100

    errors = y_test - y_pred
    mean_error = np.mean(errors)
    std_error = np.std(errors)

    print(f"\n{model_name} - Evaluation Metrics:")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R²): {r2:.3f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"Mean Error: {mean_error:.2f}")
    print(f"Standard Deviation of Error: {std_error:.2f}")

    if hasattr(model, 'feature_importances_'):
        # For tree-based models like XGBoost
        feature_importance = None

        if isinstance(X_test, pd.DataFrame):
            feature_cols = list(X_test.columns)
            importance = model.feature_importances_

            if len(feature_cols) == len(importance):
                feature_importance = dict(zip(feature_cols, importance))
                sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

                print("\nTop 10 Feature Importance:")
                for feature, imp in sorted_importance[:10]:
                    print(f"{feature}: {imp:.4f}")
            else:
                print(
                    f"Warning: Feature columns length ({len(feature_cols)}) doesn't match importance length ({len(importance)})")
        else:
            # For numpy arrays or other non-DataFrame inputs
            importance = model.feature_importances_
            print("\nFeature Importance Values (feature indices):")
            sorted_idx = np.argsort(importance)[::-1]
            for i in sorted_idx[:10]:  # Top 10
                print(f"Feature {i}: {importance[i]:.4f}")

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'mean_error': mean_error,
        'std_error': std_error
    }
