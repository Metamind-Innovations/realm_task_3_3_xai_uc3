import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from xai_utils.base import BaseExplainer


class XGBoostExplainer(BaseExplainer):
    """
    Explainer for XGBoost models.

    Provides explanations using:
    - Feature importance
    - SHAP values
    - Partial dependence plots
    """

    def __init__(self, model, feature_names=None):
        """
        Initialize XGBoost explainer.

        :param model: Trained XGBoost model
        :param feature_names: Names of features, optional
        """
        super().__init__(model, 'xgboost', feature_names)

    def explain_prediction(self, X, prediction_horizon=None):
        """
        Generate explanation for a specific prediction.

        :param X: Input data to explain (DataFrame or array)
        :param prediction_horizon: Prediction horizon in hours (1, 2, or 3)
        :returns: Dictionary with explanation data
        """
        if isinstance(X, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = X.columns.tolist()
            X_values = X.values
        else:
            X_values = X
            if self.feature_names is None:
                self.feature_names = [f"feature_{i}" for i in range(X_values.shape[1])]

        # Make prediction
        prediction = self.model.predict(X)

        # Calculate SHAP values
        explainer = shap.Explainer(self.model)
        shap_values = explainer(X)

        # Get feature importances for this prediction
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            importance_dict = dict(zip(self.feature_names, importances))
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            feature_importance = dict(sorted_importance)
        else:
            feature_importance = {}

        # Prepare the explanation
        explanation = {
            'prediction': prediction.tolist() if isinstance(prediction, np.ndarray) else prediction,
            'prediction_horizon': prediction_horizon,
            'feature_importance': feature_importance,
            'shap_values': {
                'values': shap_values.values,
                'base_values': shap_values.base_values,
                'feature_names': self.feature_names
            },
            'top_features': {
                name: float(value) for name, value in list(feature_importance.items())[:10]
            }
        }

        return explanation

    def explain_model(self, X_sample, output_dir=None):
        """
        Generate global explanation for the model.

        :param X_sample: Sample data for generating explanations (DataFrame)
        :param output_dir: Directory to save visualizations
        :returns: Dictionary with explanation data
        """
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if isinstance(X_sample, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = X_sample.columns.tolist()
            X_values = X_sample.values
        else:
            X_values = X_sample
            if self.feature_names is None:
                self.feature_names = [f"feature_{i}" for i in range(X_values.shape[1])]

        # Get feature importances
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            importance_dict = dict(zip(self.feature_names, importances))
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            feature_importance = dict(sorted_importance)

            # Plot feature importance
            if output_dir:
                plt.figure(figsize=(10, 8))
                top_features = dict(list(sorted_importance)[:20])
                plt.barh(range(len(top_features)), list(top_features.values()), align='center')
                plt.yticks(range(len(top_features)), list(top_features.keys()))
                plt.title('Feature Importance')
                plt.xlabel('Importance')
                plt.tight_layout()

                importance_plot_path = os.path.join(output_dir, 'feature_importance.png')
                plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
        else:
            feature_importance = {}

        # Calculate SHAP values for global explanation
        try:
            explainer = shap.Explainer(self.model)
            shap_values = explainer(X_sample)

            # Create summary plot
            if output_dir:
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, show=False)
                shap_summary_path = os.path.join(output_dir, 'shap_summary.png')
                plt.savefig(shap_summary_path, dpi=300, bbox_inches='tight')
                plt.close()

                # Create dependence plots for top features
                top_features = list(feature_importance.keys())[:5]
                for feature in top_features:
                    plt.figure(figsize=(10, 6))
                    feature_idx = self.feature_names.index(feature)
                    shap.dependence_plot(feature_idx, shap_values.values, X_sample, show=False)
                    dependence_path = os.path.join(output_dir, f'dependence_{feature}.png')
                    plt.savefig(dependence_path, dpi=300, bbox_inches='tight')
                    plt.close()

            shap_data = {
                'mean_abs_shap': np.abs(shap_values.values).mean(0).tolist(),
                'feature_names': self.feature_names
            }
        except Exception as e:
            print(f"SHAP calculation error: {str(e)}")
            shap_data = {"error": str(e)}

        # Prepare the explanation
        explanation = {
            'model_type': 'xgboost',
            'feature_importance': feature_importance,
            'shap_data': shap_data,
            'top_features': {
                name: float(value) for name, value in list(feature_importance.items())[:10]
            },
            'visualization_paths': {
                'feature_importance': 'feature_importance.png' if output_dir else None,
                'shap_summary': 'shap_summary.png' if output_dir else None,
                'dependence_plots': [f'dependence_{feature}.png' for feature in
                                     list(feature_importance.keys())[:5]] if output_dir else None
            }
        }

        return explanation
