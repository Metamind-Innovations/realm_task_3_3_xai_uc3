import os

import matplotlib.pyplot as plt
import numpy as np
import shap
import tensorflow as tf

from xai_utils.base import BaseExplainer


class LSTMExplainer(BaseExplainer):
    """
    Explainer for LSTM models.

    Provides explanations using:
    - SHAP values (DeepExplainer)
    - Feature attribution
    - Input perturbation analysis
    """

    def __init__(self, model, feature_names=None, sequence_length=None):
        """
        Initialize LSTM explainer.

        :param model: Trained LSTM model (Keras/TensorFlow)
        :param feature_names: Names of features, optional
        :param sequence_length: Length of input sequence
        """
        super().__init__(model, 'lstm', feature_names)
        self.sequence_length = sequence_length

    def explain_prediction(self, X, prediction_horizon=None):
        """
        Generate explanation for a specific prediction.

        :param X: Input data to explain (3D array with shape [batch, seq_len, features])
        :param prediction_horizon: Prediction horizon in hours (1, 2, or 3)
        :returns: Dictionary with explanation data
        """
        # Ensure X is the right shape
        if len(X.shape) == 2:
            # Add batch dimension if needed
            X = np.expand_dims(X, axis=0)

        # Make prediction
        prediction = self.model.predict(X)

        # Calculate feature attribution using input × gradient
        try:
            # Create a function to get gradients of output with respect to input
            grad_function = tf.keras.backend.function(
                [self.model.input],
                tf.gradients(self.model.output, self.model.input)
            )

            # Get gradients
            gradients = grad_function([X])[0]

            # Calculate attribution as input * gradient
            attribution = X * gradients

            # Aggregate attribution across sequence dimension (time)
            sequence_attribution = np.sum(attribution, axis=1)

            # Get average attribution for each feature
            feature_attribution = np.mean(np.abs(sequence_attribution), axis=0)

            if self.feature_names is None:
                self.feature_names = [f"feature_{i}" for i in range(X.shape[2])]

            # Create attribution dictionary
            attribution_dict = dict(zip(self.feature_names, feature_attribution.flatten()))
            sorted_attribution = sorted(attribution_dict.items(), key=lambda x: x[1], reverse=True)
            feature_attribution_dict = dict(sorted_attribution)

        except Exception as e:
            print(f"Feature attribution calculation error: {str(e)}")
            feature_attribution_dict = {"error": str(e)}

        # Perform input perturbation analysis
        try:
            perturbation_results = {}

            for i, feature in enumerate(self.feature_names):
                # Create perturbed inputs with this feature zeroed out
                perturbed_X = X.copy()
                perturbed_X[:, :, i] = 0

                # Get prediction with perturbed input
                perturbed_pred = self.model.predict(perturbed_X)

                # Calculate effect as difference in prediction
                effect = np.mean(np.abs(prediction - perturbed_pred))
                perturbation_results[feature] = float(effect)

            # Sort by effect
            sorted_perturbation = sorted(perturbation_results.items(), key=lambda x: x[1], reverse=True)
            perturbation_dict = dict(sorted_perturbation)

        except Exception as e:
            print(f"Perturbation analysis error: {str(e)}")
            perturbation_dict = {"error": str(e)}

        # Prepare the explanation
        explanation = {
            'prediction': prediction.tolist() if isinstance(prediction, np.ndarray) else prediction,
            'prediction_horizon': prediction_horizon,
            'feature_attribution': feature_attribution_dict,
            'perturbation_analysis': perturbation_dict,
            'top_features_by_attribution': {
                name: float(value) for name, value in list(feature_attribution_dict.items())[:10]
                if not isinstance(value, str)  # Skip error entries
            },
            'top_features_by_perturbation': {
                name: float(value) for name, value in list(perturbation_dict.items())[:10]
                if not isinstance(value, str)  # Skip error entries
            }
        }

        return explanation

    def explain_model(self, X_sample, output_dir=None):
        """
        Generate global explanation for the model.

        :param X_sample: Sample data for generating explanations (3D array)
        :param output_dir: Directory to save visualizations
        :returns: Dictionary with explanation data
        """
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Ensure feature names are set
        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(X_sample.shape[2])]

        # Calculate feature attribution using DeepExplainer from SHAP
        try:
            # Create a background dataset for SHAP
            if len(X_sample) > 100:
                background = X_sample[:100]  # Use first 100 samples as background
            else:
                background = X_sample

            # Create SHAP explainer
            explainer = shap.DeepExplainer(self.model, background)

            # Calculate SHAP values
            test_samples = X_sample[:10] if len(X_sample) > 10 else X_sample
            shap_values = explainer.shap_values(test_samples)

            # Aggregate SHAP values across sequence dimension (time)
            if isinstance(shap_values, list):
                # For multi-output models, use the first output
                aggregated_shap = np.mean(np.abs(shap_values[0]), axis=(0, 1))
            else:
                aggregated_shap = np.mean(np.abs(shap_values), axis=(0, 1))

            # Create SHAP dictionary
            shap_dict = dict(zip(self.feature_names, aggregated_shap))
            sorted_shap = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)

            # Create SHAP plots if output directory is provided
            if output_dir:
                # Summary bar plot
                plt.figure(figsize=(10, 8))
                plt.barh(range(len(sorted_shap)), [x[1] for x in sorted_shap], align='center')
                plt.yticks(range(len(sorted_shap)), [x[0] for x in sorted_shap])
                plt.title('Feature Importance (SHAP values)')
                plt.tight_layout()

                shap_bar_path = os.path.join(output_dir, 'shap_importance.png')
                plt.savefig(shap_bar_path, dpi=300, bbox_inches='tight')
                plt.close()

        except Exception as e:
            print(f"SHAP calculation error: {str(e)}")
            sorted_shap = [("error", str(e))]

        # Perform input perturbation analysis
        try:
            perturbation_results = {}

            for i, feature in enumerate(self.feature_names):
                # Create perturbed inputs with this feature zeroed out
                perturbed_X = X_sample.copy()
                perturbed_X[:, :, i] = 0

                # Get predictions
                original_preds = self.model.predict(X_sample)
                perturbed_preds = self.model.predict(perturbed_X)

                # Calculate effect as mean absolute difference
                effect = np.mean(np.abs(original_preds - perturbed_preds))
                perturbation_results[feature] = float(effect)

            # Sort by effect
            sorted_perturbation = sorted(perturbation_results.items(), key=lambda x: x[1], reverse=True)

            # Create perturbation plot if output directory is provided
            if output_dir:
                plt.figure(figsize=(10, 8))
                plt.barh(range(len(sorted_perturbation)), [x[1] for x in sorted_perturbation], align='center')
                plt.yticks(range(len(sorted_perturbation)), [x[0] for x in sorted_perturbation])
                plt.title('Feature Importance (Perturbation Analysis)')
                plt.tight_layout()

                perturbation_path = os.path.join(output_dir, 'perturbation_importance.png')
                plt.savefig(perturbation_path, dpi=300, bbox_inches='tight')
                plt.close()

        except Exception as e:
            print(f"Perturbation analysis error: {str(e)}")
            sorted_perturbation = [("error", str(e))]

        # Analyze temporal importance by perturbing different time steps
        try:
            temporal_importance = []

            for t in range(X_sample.shape[1]):  # For each time step
                # Create perturbed inputs with this time step zeroed out
                perturbed_X = X_sample.copy()
                perturbed_X[:, t, :] = 0

                # Get predictions
                original_preds = self.model.predict(X_sample)
                perturbed_preds = self.model.predict(perturbed_X)

                # Calculate effect as mean absolute difference
                effect = np.mean(np.abs(original_preds - perturbed_preds))
                temporal_importance.append((t, float(effect)))

            # Sort by effect
            sorted_temporal = sorted(temporal_importance, key=lambda x: x[1], reverse=True)

            # Create temporal importance plot if output directory is provided
            if output_dir:
                plt.figure(figsize=(10, 6))
                plt.bar(range(len(sorted_temporal)), [x[1] for x in sorted_temporal])
                plt.xticks(range(len(sorted_temporal)), [f"t-{x[0]}" for x in sorted_temporal])
                plt.title('Temporal Importance (Time Step Analysis)')
                plt.xlabel('Time Step')
                plt.ylabel('Importance')
                plt.tight_layout()

                temporal_path = os.path.join(output_dir, 'temporal_importance.png')
                plt.savefig(temporal_path, dpi=300, bbox_inches='tight')
                plt.close()

        except Exception as e:
            print(f"Temporal importance analysis error: {str(e)}")
            sorted_temporal = [("error", str(e))]

        # Prepare the explanation
        explanation = {
            'model_type': 'lstm',
            'shap_importance': dict(sorted_shap),
            'perturbation_importance': dict(sorted_perturbation),
            'temporal_importance': dict(sorted_temporal),
            'top_features_by_shap': {
                name: float(value) for name, value in sorted_shap[:10] if not isinstance(value, str)
            },
            'top_features_by_perturbation': {
                name: float(value) for name, value in sorted_perturbation[:10] if not isinstance(value, str)
            },
            'most_important_time_steps': {
                f"t-{t}": float(value) for t, value in sorted_temporal[:5] if not isinstance(value, str)
            },
            'visualization_paths': {
                'shap_importance': 'shap_importance.png' if output_dir else None,
                'perturbation_importance': 'perturbation_importance.png' if output_dir else None,
                'temporal_importance': 'temporal_importance.png' if output_dir else None
            }
        }

        return explanation
