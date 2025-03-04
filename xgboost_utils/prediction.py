import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm

from common.data_loader import load_patient_data, export_predictions
from common.preprocessing import clean_infinite_values
from xgboost_utils.feature_engineering import create_prediction_features
from xgboost_utils.model import load_models
from xgboost_utils.visualization import plot_patient_predictions


def predict_for_patient(patient_df, models, quantile_models):
    """
    Generate predictions for a patient using trained XGBoost models.

    :param patient_df: DataFrame with patient data
    :type patient_df: pandas.DataFrame
    :param models: Dict of trained models for each horizon
    :type models: dict
    :param quantile_models: Dict of quantile models for confidence intervals
    :type quantile_models: dict
    :returns: DataFrame with predictions
    :rtype: pandas.DataFrame
    """
    features_df = create_prediction_features(patient_df)

    if len(features_df) == 0:
        return pd.DataFrame()

    feature_cols = [col for col in features_df.columns if col != 'timestamp']

    X = features_df[feature_cols]
    timestamps = features_df['timestamp']
    predictions = []

    print(f"Generating predictions for {len(X)} time points...")

    for i, row in tqdm(X.iterrows(), total=len(X), desc="Making predictions"):
        row_df = pd.DataFrame([row])
        timestamp = timestamps.iloc[i]

        pred_row = {
            'timestamp': timestamp,
            'current_glucose': row['current_glucose']
        }

        prediction_made = False

        for hours in [1, 2, 3]:
            if hours not in models or hours not in quantile_models:
                continue

            model = models[hours]
            lower_model = quantile_models[hours]['lower']
            upper_model = quantile_models[hours]['upper']

            try:
                expected_features = model.feature_names_in_

                feature_mismatch = False
                if set(row_df.columns) != set(expected_features):
                    feature_mismatch = True
                    print(f"Warning: Feature mismatch for {hours}-hour model. Aligning features...")
                elif list(row_df.columns) != list(expected_features):
                    feature_mismatch = True
                    print(f"Warning: Feature order mismatch for {hours}-hour model. Reordering features...")

                if feature_mismatch:
                    missing_features = set(expected_features) - set(row_df.columns)
                    extra_features = set(row_df.columns) - set(expected_features)

                    if missing_features:
                        print(f"Missing features: {missing_features}")
                    if extra_features:
                        print(f"Extra features: {extra_features}")

                    aligned_row = pd.DataFrame(columns=expected_features)
                    for feat in expected_features:
                        if feat in row_df.columns:
                            aligned_row[feat] = row_df[feat].values
                        else:
                            print(f"Adding missing feature: {feat}, using 0")
                            aligned_row[feat] = 0

                    row_df = aligned_row

                if not np.array_equal(row_df.columns, expected_features):
                    raise ValueError(f"Feature alignment failed. Expected: {expected_features}, Got: {row_df.columns}")

                mean_pred = model.predict(row_df)[0]
                lower_bound = lower_model.predict(row_df)[0]
                upper_bound = upper_model.predict(row_df)[0]

                pred_time = timestamp + timedelta(hours=hours)
                pred_row[f'predicted_{hours}hr'] = mean_pred
                pred_row[f'lower_bound_{hours}hr'] = lower_bound
                pred_row[f'upper_bound_{hours}hr'] = upper_bound
                pred_row[f'prediction_time_{hours}hr'] = pred_time

                prediction_made = True
                print(f"Successfully made {hours}-hour prediction: {mean_pred:.1f} mg/dL at {pred_time}")

            except Exception as e:
                print(f"Error making {hours}-hour prediction: {str(e)}")
                continue

        if prediction_made:
            predictions.append(pred_row)
        else:
            print("Warning: No successful predictions for this time point")

    return pd.DataFrame(predictions)


def evaluate_predictions(predictions_df, actual_df):
    """
    Evaluate prediction accuracy against actual values.

    :param predictions_df: DataFrame with predictions
    :type predictions_df: pandas.DataFrame
    :param actual_df: DataFrame with actual values
    :type actual_df: pandas.DataFrame
    :returns: Dict with evaluation results
    :rtype: dict
    """
    evaluation_results = {}

    col_mapping = actual_df.attrs.get('column_mapping', {})

    time_col = col_mapping.get('time', 'time')
    glucose_col = col_mapping.get('glucose_level', 'glucose_level')

    if time_col not in actual_df.columns:
        print(
            f"Warning: Time column '{time_col}' not found in dataframe. Available columns: {actual_df.columns.tolist()}")
        time_col = actual_df.columns[0]
    if glucose_col not in actual_df.columns:
        print(
            f"Warning: Glucose column '{glucose_col}' not found in dataframe. Available columns: {actual_df.columns.tolist()}")
        glucose_col = actual_df.columns[1]

    for hours in [1, 2, 3]:
        if f'predicted_{hours}hr' not in predictions_df.columns:
            continue

        pred_times = predictions_df[f'prediction_time_{hours}hr']
        pred_values = predictions_df[f'predicted_{hours}hr']
        lower_bounds = predictions_df[f'lower_bound_{hours}hr']
        upper_bounds = predictions_df[f'upper_bound_{hours}hr']

        actual_values = []
        for pred_time in pred_times:
            time_diffs = abs(actual_df[time_col] - pred_time)
            closest_idx = time_diffs.idxmin()
            closest_time = actual_df.loc[closest_idx, time_col]

            if abs((closest_time - pred_time).total_seconds()) <= 600:
                actual_values.append(actual_df.loc[closest_idx, glucose_col])
            else:
                actual_values.append(np.nan)

        actual_values = np.array(actual_values)
        mask = ~np.isnan(actual_values)

        if sum(mask) > 0:
            actual_filtered = actual_values[mask]
            pred_filtered = pred_values.iloc[mask].values
            lower_filtered = lower_bounds.iloc[mask].values
            upper_filtered = upper_bounds.iloc[mask].values

            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

            rmse = np.sqrt(mean_squared_error(actual_filtered, pred_filtered))
            mae = mean_absolute_error(actual_filtered, pred_filtered)
            r2 = r2_score(actual_filtered, pred_filtered)

            within_ci = np.sum((actual_filtered >= lower_filtered) &
                               (actual_filtered <= upper_filtered)) / len(actual_filtered) * 100

            evaluation_results[hours] = {
                'num_points': len(actual_filtered),
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'ci_coverage': within_ci
            }

            print(f"\n{hours}-hour Prediction Evaluation:")
            print(f"Number of evaluation points: {len(actual_filtered)}")
            print(f"RMSE: {rmse:.2f}")
            print(f"MAE: {mae:.2f}")
            print(f"R²: {r2:.3f}")
            print(f"90% CI coverage: {within_ci:.2f}%")

    return evaluation_results


def process_patient_data(patient_file, model_dir='models', output_file='glucose_predictions.csv'):
    """
    Process patient data to generate predictions with XGBoost models.

    :param patient_file: Path to patient data CSV
    :type patient_file: str
    :param model_dir: Directory containing models
    :type model_dir: str
    :param output_file: Path to save predictions
    :type output_file: str
    :returns: DataFrame with predictions
    :rtype: pandas.DataFrame
    """
    print(f"Loading patient data from {patient_file}...")
    patient_df = load_patient_data(patient_file)

    if len(patient_df) < 5:
        print("Error: Need at least 5 glucose readings to make predictions")
        return None

    print("Loading prediction models...")
    models, quantile_models = load_models(model_dir)

    if not models:
        print("Error: No models found. Please train models first.")
        return None

    print("Making predictions...")
    predictions_df = predict_for_patient(patient_df, models, quantile_models)

    if len(predictions_df) == 0:
        print("Error: Could not generate any predictions")
        return None

    export_predictions(predictions_df, output_file)

    print("Plotting results...")
    try:
        plot_patient_predictions(patient_df, predictions_df)
    except Exception as e:
        print(f"Error plotting predictions: {str(e)}")
        print("Continuing without visualization...")

    try:
        future_end_time = None
        for hours in [3, 2, 1]:
            time_col = f'prediction_time_{hours}hr'
            if time_col in predictions_df.columns and not predictions_df[time_col].empty:
                future_end_time = predictions_df[time_col].max()
                print(f"Using {hours}-hour prediction horizon for evaluation")
                break

        if future_end_time is None:
            print("No valid prediction time horizons found for evaluation")
        else:
            col_mapping = patient_df.attrs.get('column_mapping', {})
            time_col = col_mapping.get('time', 'time')

            if patient_df[time_col].max() >= future_end_time:
                print("\nEvaluating prediction accuracy...")
                evaluation_results = evaluate_predictions(predictions_df, patient_df)

                if evaluation_results:
                    eval_df = pd.DataFrame([
                        {
                            'Prediction_Horizon': f"{hours}-hour",
                            'RMSE': results['rmse'],
                            'MAE': results['mae'],
                            'R2': results['r2'],
                            'CI_Coverage': results['ci_coverage'],
                            'Num_Points': results['num_points']
                        }
                        for hours, results in evaluation_results.items()
                    ])
                    eval_df.to_csv('prediction_evaluation.csv', index=False)
                    print("Evaluation results exported to prediction_evaluation.csv")
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        print("Continuing without evaluation...")

    return predictions_df
