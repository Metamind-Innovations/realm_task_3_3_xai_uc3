"""
This module contains utilities for explaining model predictions using various
XAI (Explainable AI) techniques. It will be expanded in future updates to provide
comprehensive model explanations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import permutation_importance

# Define colors for consistent visualization
COLORS = {
    'xgboost': '#ff7f0e',  # Orange
    'lstm': '#1f77b4',  # Blue
    'ensemble': '#2ca02c'  # Green
}


def plot_feature_importance(model, feature_names, model_type='xgboost',
                            top_n=10, save_path=None):
    """
    Plot feature importance for a trained model.

    :param model: Trained model with feature_importances_ attribute
    :param feature_names: List of feature names
    :type feature_names: list
    :param model_type: Type of model ('xgboost', 'lstm')
    :type model_type: str
    :param top_n: Number of top features to show
    :type top_n: int
    :param save_path: Path to save the plot
    :type save_path: str
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"Warning: Model does not have feature_importances_ attribute")
        return

    # Get feature importance
    importance = model.feature_importances_

    # Sort features by importance
    indices = np.argsort(importance)[::-1]

    # Select top N features
    top_indices = indices[:top_n]
    top_importance = importance[top_indices]
    top_features = [feature_names[i] for i in top_indices]

    # Create plot
    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(top_features))

    plt.barh(y_pos, top_importance, align='center', color=COLORS.get(model_type, 'blue'))
    plt.yticks(y_pos, top_features)
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importance - {model_type.upper()}')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")

    plt.show()


def generate_permutation_importance(model, X_test, y_test, feature_names,
                                    model_type='xgboost', n_repeats=10,
                                    random_state=42, top_n=10, save_path=None):
    """
    Calculate and plot permutation feature importance.

    :param model: Trained model
    :param X_test: Test features
    :type X_test: numpy.ndarray or pandas.DataFrame
    :param y_test: Test targets
    :type y_test: numpy.ndarray or pandas.Series
    :param feature_names: List of feature names
    :type feature_names: list
    :param model_type: Type of model ('xgboost', 'lstm')
    :type model_type: str
    :param n_repeats: Number of times to permute each feature
    :type n_repeats: int
    :param random_state: Random seed
    :type random_state: int
    :param top_n: Number of top features to show
    :type top_n: int
    :param save_path: Path to save the plot
    :type save_path: str
    :returns: DataFrame with importance results
    :rtype: pandas.DataFrame
    """
    # Calculate permutation importance
    print("Calculating permutation importance (this may take some time)...")
    perm_importance = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=random_state
    )

    # Create DataFrame for results
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance_Mean': perm_importance.importances_mean,
        'Importance_Std': perm_importance.importances_std
    })

    # Sort by importance
    importance_df = importance_df.sort_values('Importance_Mean', ascending=False)

    # Plot top N features
    top_results = importance_df.head(top_n)

    plt.figure(figsize=(10, 8))
    plt.barh(
        np.arange(len(top_results)),
        top_results['Importance_Mean'],
        xerr=top_results['Importance_Std'],
        align='center',
        color=COLORS.get(model_type, 'blue'),
        ecolor='black',
        capsize=5
    )
    plt.yticks(np.arange(len(top_results)), top_results['Feature'])
    plt.xlabel('Mean Decrease in Accuracy')
    plt.title(f'Top {top_n} Permutation Feature Importance - {model_type.upper()}')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Permutation importance plot saved to {save_path}")

    plt.show()

    return importance_df


def generate_shap_summary(model, X_sample, feature_names, model_type='xgboost',
                          max_display=10, save_path=None):
    """
    Generate and plot SHAP summary for model.

    :param model: Trained model
    :param X_sample: Sample of data for SHAP analysis
    :type X_sample: numpy.ndarray or pandas.DataFrame
    :param feature_names: List of feature names
    :type feature_names: list
    :param model_type: Type of model ('xgboost', 'lstm')
    :type model_type: str
    :param max_display: Maximum number of features to display
    :type max_display: int
    :param save_path: Path to save the plot
    :type save_path: str
    """
    try:
        # Define explainer based on model type
        if model_type == 'xgboost':
            explainer = shap.TreeExplainer(model)
        else:
            # Use KernelExplainer for other model types
            explainer = shap.KernelExplainer(model.predict, X_sample)

        # Calculate SHAP values
        print("Calculating SHAP values (this may take some time)...")
        shap_values = explainer.shap_values(X_sample)

        # Prepare feature names if needed
        if isinstance(X_sample, pd.DataFrame):
            feature_names = X_sample.columns

        # Create summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=feature_names,
            max_display=max_display,
            show=False
        )

        plt.title(f'SHAP Feature Importance - {model_type.upper()}')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP summary plot saved to {save_path}")

        plt.show()

    except Exception as e:
        print(f"Error generating SHAP analysis: {str(e)}")


def explain_single_prediction(model, X_instance, feature_names, model_type='xgboost',
                              output_dir=None):
    """
    Explain a single prediction in detail.

    :param model: Trained model
    :param X_instance: Single instance to explain
    :type X_instance: numpy.ndarray or pandas.Series
    :param feature_names: List of feature names
    :type feature_names: list
    :param model_type: Type of model ('xgboost', 'lstm')
    :type model_type: str
    :param output_dir: Directory to save outputs
    :type output_dir: str
    """
    try:
        # Reshape input if needed
        if len(X_instance.shape) == 1:
            X_instance = X_instance.reshape(1, -1)

        # Make prediction
        prediction = model.predict(X_instance)[0]

        # Define explainer based on model type
        if model_type == 'xgboost':
            explainer = shap.TreeExplainer(model)
        else:
            # Use KernelExplainer for other model types
            explainer = shap.KernelExplainer(model.predict, X_instance)

        # Calculate SHAP values
        shap_values = explainer.shap_values(X_instance)

        # Create force plot
        plt.figure(figsize=(20, 3))
        force_plot = shap.force_plot(
            explainer.expected_value,
            shap_values[0, :],
            X_instance[0],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )

        if output_dir:
            force_plot_path = os.path.join(output_dir, f'{model_type}_force_plot.png')
            plt.savefig(force_plot_path, dpi=300, bbox_inches='tight')
            print(f"Force plot saved to {force_plot_path}")

        plt.show()

        # Create decision plot
        plt.figure(figsize=(12, 8))
        shap.decision_plot(
            explainer.expected_value,
            shap_values[0, :],
            feature_names=feature_names,
            show=False
        )

        plt.title(f'Decision Plot (Prediction: {prediction:.2f})')
        plt.tight_layout()

        if output_dir:
            decision_plot_path = os.path.join(output_dir, f'{model_type}_decision_plot.png')
            plt.savefig(decision_plot_path, dpi=300, bbox_inches='tight')
            print(f"Decision plot saved to {decision_plot_path}")

        plt.show()

        # Create waterfall plot
        plt.figure(figsize=(10, 12))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0, :],
                base_values=explainer.expected_value,
                data=X_instance[0],
                feature_names=feature_names
            ),
            show=False
        )

        plt.title(f'Waterfall Plot (Prediction: {prediction:.2f})')
        plt.tight_layout()

        if output_dir:
            waterfall_plot_path = os.path.join(output_dir, f'{model_type}_waterfall_plot.png')
            plt.savefig(waterfall_plot_path, dpi=300, bbox_inches='tight')
            print(f"Waterfall plot saved to {waterfall_plot_path}")

        plt.show()

    except Exception as e:
        print(f"Error explaining prediction: {str(e)}")


def compare_model_explanations(xgboost_model, lstm_model, X_sample,
                               feature_names, output_dir=None):
    """
    Compare explanations between XGBoost and LSTM models.

    :param xgboost_model: Trained XGBoost model
    :param lstm_model: Trained LSTM model
    :param X_sample: Sample data for analysis
    :type X_sample: numpy.ndarray or pandas.DataFrame
    :param feature_names: List of feature names
    :type feature_names: list
    :param output_dir: Directory to save outputs
    :type output_dir: str
    """
    print("This function will be implemented in a future update.")
    print("It will provide side-by-side comparisons of model explanations.")


def generate_explanation_report(model, X_train, X_test, y_test, feature_names,
                                model_type='xgboost', output_dir='explanations'):
    """
    Generate a comprehensive explanation report for a trained model.

    :param model: Trained model
    :param X_train: Training features
    :type X_train: numpy.ndarray or pandas.DataFrame
    :param X_test: Test features
    :type X_test: numpy.ndarray or pandas.DataFrame
    :param y_test: Test targets
    :type y_test: numpy.ndarray or pandas.Series
    :param feature_names: List of feature names
    :type feature_names: list
    :param model_type: Type of model ('xgboost', 'lstm')
    :type model_type: str
    :param output_dir: Directory to save outputs
    :type output_dir: str
    """
    print("Complete explanation report generation will be implemented in a future update.")
    print("This will include feature importance, SHAP analysis, permutation importance,")
    print("and detailed prediction explanations for example cases.")


# Placeholder for future implementation of temporal explanations for time series
def analyze_temporal_contributions(model, time_series_data, feature_names,
                                   time_window=5, output_dir=None):
    """
    Analyze how features contribute over time in time series predictions.

    This is a placeholder for future implementation.

    :param model: Trained model
    :param time_series_data: Time series data for analysis
    :type time_series_data: numpy.ndarray or pandas.DataFrame
    :param feature_names: List of feature names
    :type feature_names: list
    :param time_window: Window size for temporal analysis
    :type time_window: int
    :param output_dir: Directory to save outputs
    :type output_dir: str
    """
    print("Temporal contribution analysis will be implemented in a future update.")
    print("This will provide insights into how features influence predictions over time.")