import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

from xai_utils.lstm_explainer import LSTMExplainer
from xai_utils.xgboost_explainer import XGBoostExplainer


def generate_explanation_report(
        model_type,
        models,
        patient_data=None,
        features_data=None,
        prediction_results=None,
        output_dir='xai_reports',
        include_global_explanation=True,
        include_local_explanations=True
):
    """
    Generate a comprehensive explanation report for model predictions.

    :param model_type: Type of model ('xgboost' or 'lstm')
    :param models: Dict of trained models for different prediction horizons
    :param patient_data: Original patient data
    :param features_data: Features used for predictions
    :param prediction_results: Results of predictions
    :param output_dir: Directory to save reports
    :param include_global_explanation: Whether to include global model explanation
    :param include_local_explanations: Whether to include explanations for individual predictions
    :returns: Path to generated report
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Timestamp for report file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"{model_type}_explanation_{timestamp}.json")

    # Create report data structure
    report = {
        'metadata': {
            'model_type': model_type,
            'timestamp': datetime.now().isoformat(),
            'horizons': list(models.keys()) if isinstance(models, dict) else [1, 2, 3]
        },
        'global_explanations': {},
        'prediction_explanations': []
    }

    # Feature names
    feature_names = None
    if features_data is not None:
        if isinstance(features_data, pd.DataFrame):
            feature_names = features_data.columns.tolist()
        elif model_type.lower() == 'lstm' and hasattr(features_data, 'shape') and len(features_data.shape) == 3:
            # For 3D LSTM data, define feature names
            feature_names = [f"feature_{i}" for i in range(features_data.shape[2])]
            print(f"Created feature names for LSTM model: {feature_names}")

    # Create appropriate explainer based on model type
    if model_type.lower() == 'xgboost':
        explainer_class = XGBoostExplainer
    elif model_type.lower() == 'lstm':
        explainer_class = LSTMExplainer
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Generate global explanations for each prediction horizon model
    if include_global_explanation and isinstance(models, dict):
        for horizon, model in models.items():
            print(f"Generating global explanation for {horizon}-hour {model_type} model...")

            # Create explainer
            explainer = explainer_class(model, feature_names)

            # Skip if no feature data
            if features_data is None:
                print(f"Skipping global explanation for {horizon}-hour model - no feature data provided")
                continue

            # Create output directory for visualizations
            viz_dir = os.path.join(output_dir, f"{model_type}_{horizon}hr_global")
            os.makedirs(viz_dir, exist_ok=True)

            try:
                # Generate explanation
                start_time = time.time()

                # Debug information for LSTM
                if model_type.lower() == 'lstm':
                    print(f"Features data type: {type(features_data)}")
                    if hasattr(features_data, 'shape'):
                        print(f"Features data shape: {features_data.shape}")

                    # For LSTM, ensure we have 3D data
                    if isinstance(features_data, pd.DataFrame):
                        print("WARNING: LSTM explainer requires 3D input data. Attempting to convert DataFrame.")
                        try:
                            # Very simplified - in a real application you'd need proper sequence creation
                            values = features_data.values
                            seq_len = 10  # Default sequence length
                            n_features = values.shape[1]

                            # Create simple sequences (this is just for demonstration)
                            if len(values) >= seq_len:
                                sequences = []
                                for i in range(0, min(100, len(values) - seq_len + 1), seq_len):
                                    sequences.append(values[i:i + seq_len])
                                X_sample = np.array(sequences)
                                print(f"Created {len(sequences)} sequences with shape {X_sample.shape}")
                            else:
                                # Not enough data, duplicate what we have
                                print("Not enough data for sequences, duplicating available data")
                                X_sample = np.array([values] * (seq_len // len(values) + 1))[:seq_len]
                                X_sample = np.expand_dims(X_sample, axis=0)
                                print(f"Created data with shape {X_sample.shape}")
                        except Exception as e:
                            print(f"Failed to convert DataFrame to sequences: {str(e)}")
                            report['global_explanations'][f'{horizon}hr'] = {
                                'error': f"Failed to prepare LSTM input data: {str(e)}"
                            }
                            continue
                    else:
                        X_sample = features_data
                else:
                    X_sample = features_data

                try:
                    explanation = explainer.explain_model(X_sample, viz_dir)
                except Exception as e:
                    print(f"Error in explainer.explain_model: {str(e)}")
                    explanation = {'error': str(e)}

                # Add to report
                report['global_explanations'][f'{horizon}hr'] = explanation

                # Log result
                elapsed = time.time() - start_time
                print(f"Generated global explanation for {horizon}-hour model in {elapsed:.2f} seconds")

            except Exception as e:
                print(f"Error generating global explanation for {horizon}-hour model: {str(e)}")
                report['global_explanations'][f'{horizon}hr'] = {'error': str(e)}

    # Generate explanations for individual predictions
    if include_local_explanations and prediction_results is not None:
        print("Generating explanations for individual predictions...")

        # Limit the number of predictions to explain to avoid excessive computation
        max_predictions = 3
        sample_indices = np.linspace(0, len(prediction_results) - 1,
                                     min(max_predictions, len(prediction_results))).astype(int)

        # Iterate through selected prediction results
        for idx in sample_indices:
            pred_row = prediction_results.iloc[idx]
            timestamp = pred_row.get('timestamp')

            for horizon in [1, 2, 3]:
                pred_col = f'predicted_{horizon}hr'
                time_col = f'prediction_time_{horizon}hr'

                if pred_col not in prediction_results.columns or time_col not in prediction_results.columns:
                    continue

                prediction = pred_row.get(pred_col)
                pred_time = pred_row.get(time_col)

                if pd.isna(prediction) or pd.isna(pred_time):
                    continue

                # Get corresponding model
                if isinstance(models, dict) and horizon in models:
                    model = models[horizon]
                else:
                    print(f"Model for {horizon}-hour horizon not available, skipping explanation")
                    continue

                # Create explainer
                explainer = explainer_class(model, feature_names)

                # Get features for this prediction
                if features_data is not None:
                    if model_type.lower() == 'lstm':
                        if hasattr(features_data, 'shape') and len(features_data.shape) == 3 and idx < \
                                features_data.shape[0]:
                            # For 3D LSTM data, get appropriate sequence
                            X = features_data[idx:idx + 1]
                            print(f"Using LSTM sequence with shape {X.shape}")
                        else:
                            print(f"LSTM feature data not in expected format, skipping explanation")
                            continue
                    else:
                        # For DataFrame features
                        if idx < len(features_data):
                            X = features_data.iloc[idx:idx + 1]
                            print(f"Using feature data with shape {X.shape if hasattr(X, 'shape') else 'unknown'}")
                        else:
                            print(f"Feature index {idx} out of range, skipping explanation")
                            continue
                else:
                    print("No feature data provided, skipping individual explanation")
                    continue

                try:
                    # Generate explanation
                    print(f"Explaining prediction {idx}, horizon {horizon}hr")
                    explanation = explainer.explain_prediction(X, horizon)

                    # Add metadata
                    explanation['prediction_index'] = int(idx)
                    explanation['timestamp'] = timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp
                    explanation['prediction_time'] = pred_time.isoformat() if isinstance(pred_time,
                                                                                         datetime) else pred_time

                    # Add to report
                    report['prediction_explanations'].append(explanation)

                except Exception as e:
                    print(f"Error explaining prediction {idx}, {horizon}hr: {str(e)}")
                    error_explanation = {
                        'prediction_index': int(idx),
                        'prediction_horizon': horizon,
                        'timestamp': timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
                        'error': str(e)
                    }
                    report['prediction_explanations'].append(error_explanation)

    # Save report to JSON file
    try:
        # Make report JSON serializable
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='records')
            elif isinstance(obj, (pd.Series, pd.Index)):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            else:
                return str(obj)

        # Recursively convert dictionary
        def convert_dict(d):
            result = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    result[k] = convert_dict(v)
                elif isinstance(v, list):
                    result[k] = [convert_dict(item) if isinstance(item, dict) else convert_to_serializable(item) for
                                 item in v]
                else:
                    result[k] = convert_to_serializable(v)
            return result

        serializable_report = convert_dict(report)

        with open(report_file, 'w') as f:
            json.dump(serializable_report, f, indent=2)

        print(f"Explanation report saved to {report_file}")

    except Exception as e:
        print(f"Error saving report: {str(e)}")

    return report_file
