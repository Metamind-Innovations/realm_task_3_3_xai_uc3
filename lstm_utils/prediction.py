from datetime import timedelta

import numpy as np
import pandas as pd

from common.data_loader import load_patient_data, export_predictions
from lstm_utils.data_processor import prepare_prediction_sequences
from lstm_utils.model import load_lstm_models
from lstm_utils.visualization import plot_predictions


def predict_glucose_with_lstm(patient_df, models, scalers, sequence_length=10):
    """
    Generate glucose predictions with LSTM models.

    Creates predictions for multiple time horizons with confidence
    intervals for each prediction.

    :param patient_df: Patient data with glucose readings
    :type patient_df: pandas.DataFrame
    :param models: Trained LSTM models for each prediction horizon
    :type models: dict
    :param scalers: Fitted scalers for features and targets
    :type scalers: dict
    :param sequence_length: Length of input sequences
    :type sequence_length: int
    :returns: DataFrame with predictions and timestamps
    :rtype: pandas.DataFrame
    """
    sequences, timestamps = prepare_prediction_sequences(patient_df, sequence_length)
    if sequences is None or timestamps is None:
        return pd.DataFrame()

    sequences_reshaped = sequences.reshape(-1, sequences.shape[2])
    sequences_scaled_2d = scalers['X'].transform(sequences_reshaped)
    sequences_scaled = sequences_scaled_2d.reshape(sequences.shape)

    predictions_df = pd.DataFrame({'timestamp': timestamps})

    # Assuming the first feature is glucose level
    glucose_idx = 0
    predictions_df['current_glucose'] = scalers['X'].inverse_transform(
        sequences_scaled[:, -1, :])[:, glucose_idx]

    for hours in models.keys():
        if hours not in scalers['y']:
            print(f"Scaler for {hours}-hour prediction not found, skipping")
            continue

        print(f"Making {hours}-hour predictions...")

        y_pred_scaled = models[hours].predict(sequences_scaled)

        y_pred = scalers['y'][hours].inverse_transform(y_pred_scaled)

        pred_times = [t + timedelta(hours=hours) for t in timestamps]

        predictions_df[f'predicted_{hours}hr'] = y_pred
        predictions_df[f'prediction_time_{hours}hr'] = pred_times

        # Calculate confidence intervals
        std_dev = np.std(y_pred) * 1.96  # ~95% confidence
        predictions_df[f'lower_bound_{hours}hr'] = y_pred - std_dev
        predictions_df[f'upper_bound_{hours}hr'] = y_pred + std_dev

    return predictions_df


def process_patient_data(patient_file, model_dir='lstm_models',
                         output_file='lstm_glucose_predictions.csv', sequence_length=None):
    """
    End-to-end process to load patient data, generate predictions, and export results.

    Handles adaptive sequence length selection based on input data size.

    :param patient_file: Path to CSV file with patient data
    :type patient_file: str
    :param model_dir: Directory containing trained models
    :type model_dir: str
    :param output_file: Path for saving prediction output
    :type output_file: str
    :param sequence_length: Override for sequence length, useful for very short inputs
    :type sequence_length: int
    :returns: DataFrame with predictions
    :rtype: pandas.DataFrame
    """
    print(f"Loading patient data from {patient_file}...")
    patient_df = load_patient_data(patient_file)

    row_count = len(patient_df)
    print(f"Input CSV has {row_count} rows")

    # Set adaptive sequence length if not specified
    if sequence_length is None:
        if row_count < 5:
            sequence_length = max(2, row_count)
            print(f"Using reduced sequence length of {sequence_length} for short input")
        else:
            sequence_length = min(10, row_count)  # Default max is 10
            print(f"Using sequence length of {sequence_length}")
    else:
        print(f"Using user-specified sequence length of {sequence_length}")

    print("Loading LSTM models...")
    models, scalers = load_lstm_models(model_dir)

    if not models or not scalers:
        print("Error: No models or scalers found. Please train models first.")
        return None

    print("Making predictions...")
    sequences, timestamps = prepare_prediction_sequences(patient_df, sequence_length=sequence_length)

    if sequences is None or timestamps is None:
        print("Error: Could not prepare sequences for prediction")
        return None

    predictions_df = predict_glucose_with_lstm(patient_df, models, scalers, sequence_length)

    if len(predictions_df) == 0:
        print("Error: Could not generate any predictions")
        return None

    export_predictions(predictions_df, output_file)
    print(f"Predictions exported to {output_file}")

    print("Plotting results...")
    try:
        plot_predictions(patient_df, predictions_df)
    except Exception as e:
        print(f"Error plotting predictions: {str(e)}")
        print("Continuing without visualization...")

    return predictions_df


def evaluate_predictions(predictions_df, actual_df):
    """
    Evaluate LSTM prediction accuracy against actual values.

    :param predictions_df: DataFrame with predictions
    :type predictions_df: pandas.DataFrame
    :param actual_df: DataFrame with actual values
    :type actual_df: pandas.DataFrame
    :returns: Dict with evaluation results
    :rtype: dict
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    evaluation_results = {}

    col_mapping = actual_df.attrs.get('column_mapping', {})
    time_col = col_mapping.get('time', 'time')
    glucose_col = col_mapping.get('glucose_level', 'glucose_level')

    if time_col not in actual_df.columns:
        print(f"Warning: Time column '{time_col}' not found. Available columns: {actual_df.columns.tolist()}")
        time_col = actual_df.columns[0]
    if glucose_col not in actual_df.columns:
        print(f"Warning: Glucose column '{glucose_col}' not found. Available columns: {actual_df.columns.tolist()}")
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

            # Only consider matches within 10 minutes
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
            print(f"95% CI coverage: {within_ci:.2f}%")

    return evaluation_results
