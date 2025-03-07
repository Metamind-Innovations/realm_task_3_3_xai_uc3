import os
from datetime import timedelta

import numpy as np
import pandas as pd

from common.data_loader import load_patient_data, export_predictions
from config import PREDICTION_HORIZONS
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

        # For single predictions, use a percentage-based range
        if len(y_pred) == 1:
            ci_percentage = 0.15  # Typical glucose reading fluctuation is around 10-15%
            base_value = y_pred[0]
            lower_bound = base_value * (1 - ci_percentage)
            upper_bound = base_value * (1 + ci_percentage)

            predictions_df[f'lower_bound_{hours}hr'] = lower_bound
            predictions_df[f'upper_bound_{hours}hr'] = upper_bound
        else:
            # For multiple predictions, use a combination of:
            # 1. Model uncertainty estimate
            # 2. Historical glucose variability
            # 3. Time horizon (longer horizon = wider interval)
            base_uncertainty = 0.05 + (hours * 0.03)  # 5% + 3% per hour

            lower_bounds = []
            upper_bounds = []

            for i, pred in enumerate(y_pred):
                seq_glucose = sequences[i, :, glucose_idx]
                if len(seq_glucose) > 1:
                    cv = np.std(seq_glucose) / np.mean(seq_glucose) if np.mean(seq_glucose) > 0 else 0.1
                    cv = min(max(cv, 0.05), 0.3)
                else:
                    cv = 0.1  # Default value

                total_uncertainty = base_uncertainty + cv
                lower_bounds.append(pred * (1 - total_uncertainty))
                upper_bounds.append(pred * (1 + total_uncertainty))

            predictions_df[f'lower_bound_{hours}hr'] = lower_bounds
            predictions_df[f'upper_bound_{hours}hr'] = upper_bounds

    return predictions_df


def process_patient_data(patient_file, model_dir='lstm_models',
                         output_file='lstm_glucose_predictions.csv',
                         sequence_length=None,
                         skip_plotting=False):
    """
    End-to-end process to load patient data, generate predictions, and export results.
    Now with support for demographic information from filename.

    :param patient_file: Path to CSV file with patient data
    :type patient_file: str
    :param model_dir: Directory containing trained models
    :type model_dir: str
    :param output_file: Path for saving prediction output
    :type output_file: str
    :param sequence_length: Override for sequence length, useful for very short inputs
    :type sequence_length: int
    :param skip_plotting: Whether to skip automatic plotting
    :type skip_plotting: bool
    :returns: DataFrame with predictions
    :rtype: pandas.DataFrame
    """
    print(f"Loading patient data from {patient_file}...")
    patient_df = load_patient_data(patient_file)

    row_count = len(patient_df)
    print(f"Input CSV has {row_count} rows")

    # Check if demographics were extracted from filename
    filename_demographics = patient_df.attrs.get('demographics_from_filename', {})
    if filename_demographics:
        if 'age' in filename_demographics:
            print(f"Using age {filename_demographics['age']} from filename")
        if 'gender' in filename_demographics:
            print(f"Using gender '{filename_demographics['gender']}' from filename")

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

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    export_predictions(predictions_df, output_file)

    # Only plot if not skipped
    if not skip_plotting:
        print("Plotting results...")
        try:
            # Use the updated plot_predictions function with output_dir parameter
            plot_predictions(patient_df, predictions_df, output_dir=output_dir)
        except Exception as e:
            print(f"Error plotting predictions: {str(e)}")
            print("Continuing without visualization...")

    return predictions_df
