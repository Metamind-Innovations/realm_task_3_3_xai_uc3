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
    if features_data is not None and isinstance(features_data, pd.DataFrame):
        feature_names = features_data.columns.tolist()

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
                X_sample = features_data

                # For LSTM, need to reshape if it's a DataFrame
                if model_type.lower() == 'lstm' and isinstance(features_data, pd.DataFrame):
                    # This is a placeholder - actual implementation depends on data structure
                    print("Preparing sequences for LSTM explanation...")

                explanation = explainer.explain_model(X_sample, viz_dir)

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

        # Iterate through prediction results
        for i, pred_row in enumerate(prediction_results.iloc):
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
                    if i < len(features_data):
                        X = features_data.iloc[i:i + 1]

                        # For LSTM, need to reshape
                        if model_type.lower() == 'lstm':
                            # This is a placeholder - actual implementation depends on data structure
                            print(f"Preparing sequence for LSTM explanation of prediction {i}, {horizon}hr...")
                    else:
                        print(f"Feature index {i} out of range, skipping explanation")
                        continue
                else:
                    print("No feature data provided, skipping individual explanation")
                    continue

                try:
                    # Generate explanation
                    explanation = explainer.explain_prediction(X, horizon)

                    # Add metadata
                    explanation['prediction_index'] = i
                    explanation['timestamp'] = timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp
                    explanation['prediction_time'] = pred_time.isoformat() if isinstance(pred_time,
                                                                                         datetime) else pred_time

                    # Add to report
                    report['prediction_explanations'].append(explanation)

                except Exception as e:
                    print(f"Error explaining prediction {i}, {horizon}hr: {str(e)}")
                    error_explanation = {
                        'prediction_index': i,
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
