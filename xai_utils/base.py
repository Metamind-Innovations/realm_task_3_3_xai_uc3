import datetime
import json
import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseExplainer(ABC):
    """
    Base class for model explainers.

    Provides common functionality for explaining model predictions
    and standardized interface for different model types.
    """

    def __init__(self, model, model_type, feature_names=None):
        """
        Initialize explainer with a model.

        :param model: Trained model to explain
        :param model_type: Type of model (e.g., 'xgboost', 'lstm')
        :param feature_names: Names of features, optional
        """
        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names

    @abstractmethod
    def explain_prediction(self, X, prediction_horizon=None):
        """
        Generate explanation for a prediction.

        :param X: Input data to explain
        :param prediction_horizon: Optional prediction horizon (1hr, 2hr, 3hr)
        :returns: Dictionary with explanation data
        """
        pass

    @abstractmethod
    def explain_model(self, X_sample, output_dir=None):
        """
        Generate global explanation for the model.

        :param X_sample: Sample data for generating explanations
        :param output_dir: Directory to save visualizations
        :returns: Dictionary with explanation data
        """
        pass

    def save_explanation(self, explanation, output_file, pretty=True):
        """
        Save explanation to a JSON file.

        :param explanation: Dictionary with explanation data
        :param output_file: File path to save explanation
        :param pretty: Whether to format the JSON for readability
        """
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Make explanation JSON serializable
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
            elif isinstance(obj, datetime.datetime):
                return obj.isoformat()
            else:
                return str(obj)

        # Convert explanation to serializable format
        serializable_explanation = {}
        for key, value in explanation.items():
            if isinstance(value, dict):
                serializable_explanation[key] = {k: convert_to_serializable(v) for k, v in value.items()}
            else:
                serializable_explanation[key] = convert_to_serializable(value)

        # Add metadata
        serializable_explanation['metadata'] = {
            'timestamp': datetime.datetime.now().isoformat(),
            'model_type': self.model_type
        }

        # Save to file
        with open(output_file, 'w') as f:
            if pretty:
                json.dump(serializable_explanation, f, indent=2)
            else:
                json.dump(serializable_explanation, f)

        print(f"Explanation saved to {output_file}")

        return output_file
