import os
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm

from common.data_loader import load_and_preprocess_training_data
from common.preprocessing import clean_infinite_values
from common.visualization import plot_model_evaluation
from xgboost_utils.feature_engineering import prepare_dataset
from xgboost_utils.model import create_xgboost_model, create_quantile_model, train_test_evaluate_model

def train_enhanced_glucose_models(features_df, save_dir='models'):
    """
    Train XGBoost models for glucose prediction at different time horizons.

    :param features_df: DataFrame with extracted features
    :type features_df: pandas.DataFrame
    :param save_dir: Directory to save models
    :type save_dir: str
    :returns: Tuple of (models, quantile_models)
    :rtype: tuple(dict, dict)
    """
    features_df = clean_infinite_values(features_df)

    models = {}
    quantile_models = {}

    all_feature_cols = [col for col in features_df.columns
                        if not col.lower().startswith('future_')
                        and col.upper() != 'SUBJECT_ID'
                        and col.lower() != 'subject_id']

    print(f"Training with {len(all_feature_cols)} features")
    all_metrics = {}

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    metrics_dir = os.path.join(save_dir, 'metrics')
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    print("\nTRAINING ENHANCED GLUCOSE PREDICTION MODELS")
    print("=" * 50)

    for hours in tqdm([1, 2, 3], desc="Training models for different time horizons"):
        target_col = f'future_glucose_{hours}hr'
        print(f"\nTraining Enhanced {hours}-hour Prediction Model")
        print("*" * 30)

        X = features_df[all_feature_cols]
        y = features_df[target_col]

        # Create and train main model
        model = create_xgboost_model()
        model, metrics = train_test_evaluate_model(X, y, model, f"Enhanced {hours}-hour Prediction Model")
        all_metrics[hours] = metrics

        # Create and train lower bound model (5th percentile)
        lower_model = create_quantile_model(quantile_alpha=0.05)
        print(f"Training lower bound {hours}-hour prediction model...")
        lower_model.fit(X, y, verbose=False)

        # Create and train upper bound model (95th percentile)
        upper_model = create_quantile_model(quantile_alpha=0.95)
        print(f"Training upper bound {hours}-hour prediction model...")
        upper_model.fit(X, y, verbose=False)

        # Make predictions and plot evaluation
        y_pred = model.predict(X)
        plot_model_evaluation(y, y_pred, title=f"Enhanced {hours}-hour Prediction", save_png=False)

        # Evaluate confidence interval coverage
        X_sample = X.sample(min(1000, len(X)))
        y_sample = y.loc[X_sample.index]
        lower_bound = lower_model.predict(X_sample)
        upper_bound = upper_model.predict(X_sample)
        within_ci = np.sum((y_sample >= lower_bound) & (y_sample <= upper_bound)) / len(y_sample) * 100
        print(f"90% CI coverage: {within_ci:.2f}%")

        # Save evaluation metrics
        metrics_df = pd.DataFrame({
            'Metric': ['RMSE', 'MAE', 'R²', 'MAPE', 'Mean Error', 'Std Error', 'CI Coverage'],
            'Value': [metrics['rmse'], metrics['mae'], metrics['r2'],
                     metrics['mape'], metrics['mean_error'], metrics['std_error'], within_ci]
        })
        metrics_df.to_csv(os.path.join(metrics_dir, f'metrics_{hours}hr.csv'), index=False)

        # Save feature importance
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(all_feature_cols, model.feature_importances_))
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

            print("\nTop 10 Feature Importance:")
            for feature, imp in sorted_importance[:10]:
                print(f"{feature}: {imp:.4f}")

            importance_df = pd.DataFrame({
                'Feature': all_feature_cols,
                'Importance': model.feature_importances_
            })
            importance_df = importance_df.sort_values('Importance', ascending=False)
            importance_df.to_csv(os.path.join(metrics_dir, f'feature_importance_{hours}hr.csv'), index=False)

        # Save models
        models[hours] = model
        quantile_models[hours] = {
            'lower': lower_model,
            'upper': upper_model
        }

        joblib.dump(model, os.path.join(save_dir, f'enhanced_glucose_model_{hours}hr.pkl'))
        joblib.dump(lower_model, os.path.join(save_dir, f'enhanced_glucose_model_{hours}hr_lower.pkl'))
        joblib.dump(upper_model, os.path.join(save_dir, f'enhanced_glucose_model_{hours}hr_upper.pkl'))

    # Print performance summary
    print("\nENHANCED MODEL PERFORMANCE SUMMARY")
    print("=" * 50)

    summary_data = []
    for hours, metrics in all_metrics.items():
        summary_data.append({
            'Prediction Horizon': f"{hours}-hour",
            'RMSE': f"{metrics['rmse']:.2f}",
            'MAE': f"{metrics['mae']:.2f}",
            'R²': f"{metrics['r2']:.3f}"
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(metrics_dir, 'enhanced_model_summary.csv'), index=False)

    return models, quantile_models


def train_models(training_file, model_dir='models'):
    """
    Train XGBoost models from training data.

    :param training_file: Path to training data CSV
    :type training_file: str
    :param model_dir: Directory to save models
    :type model_dir: str
    :returns: Tuple of (models, quantile_models)
    :rtype: tuple(dict, dict)
    """
    print(f"Loading training data from {training_file}...")
    df = load_and_preprocess_training_data(training_file)

    print("Creating features...")
    features_df = prepare_dataset(df)

    print("Training models...")
    models, quantile_models = train_enhanced_glucose_models(features_df, model_dir)

    print(f"Models saved to {model_dir} directory")
    return models, quantile_models
