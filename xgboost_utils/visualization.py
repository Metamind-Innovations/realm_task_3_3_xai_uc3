from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd


def plot_patient_predictions(patient_df, predictions_df, dir, save_png=False):
    """
    Plot patient glucose data with predictions and confidence intervals.

    :param patient_df: DataFrame with patient data
    :type patient_df: pandas.DataFrame
    :param predictions_df: DataFrame with predictions
    :type predictions_df: pandas.DataFrame
    :param save_png: Whether to save plot as PNG
    :type save_png: bool
    """
    plt.figure(figsize=(12, 8))

    col_mapping = patient_df.attrs.get('column_mapping', {})

    time_col = col_mapping.get('time', 'time')
    glucose_col = col_mapping.get('glucose_level', 'glucose_level')

    if time_col not in patient_df.columns:
        print(
            f"Warning: Time column '{time_col}' not found in dataframe. Available columns: {patient_df.columns.tolist()}")
        time_col = patient_df.columns[0]
    if glucose_col not in patient_df.columns:
        print(
            f"Warning: Glucose column '{glucose_col}' not found in dataframe. Available columns: {patient_df.columns.tolist()}")
        glucose_col = patient_df.columns[1]

    plt.plot(patient_df[time_col], patient_df[glucose_col], 'b-', label='Actual Glucose')
    plt.scatter(patient_df[time_col], patient_df[glucose_col], color='blue', s=30)

    colors = ['red', 'green', 'purple']
    hours_list = [1, 2, 3]

    has_predictions = False

    for i, hours in enumerate(hours_list):
        pred_col = f'predicted_{hours}hr'
        time_col_pred = f'prediction_time_{hours}hr'
        lower_col = f'lower_bound_{hours}hr'
        upper_col = f'upper_bound_{hours}hr'

        if (pred_col not in predictions_df.columns or
                time_col_pred not in predictions_df.columns or
                lower_col not in predictions_df.columns or
                upper_col not in predictions_df.columns):
            print(f"Skipping {hours}-hour predictions (columns not found)")
            continue

        if predictions_df[pred_col].isna().all():
            print(f"Skipping {hours}-hour predictions (all values are NaN)")
            continue

        pred_times = predictions_df[time_col_pred]
        pred_values = predictions_df[pred_col]
        lower_bounds = predictions_df[lower_col]
        upper_bounds = predictions_df[upper_col]

        has_predictions = True

        plt.scatter(pred_times, pred_values, color=colors[i],
                    label=f'{hours}-hour Prediction', s=40, marker='s')

        for j in range(len(pred_times)):
            plt.fill_between([pred_times.iloc[j], pred_times.iloc[j]],
                             [lower_bounds.iloc[j]], [upper_bounds.iloc[j]],
                             color=colors[i], alpha=0.2)

            plt.plot([pred_times.iloc[j], pred_times.iloc[j]],
                     [lower_bounds.iloc[j], upper_bounds.iloc[j]],
                     color=colors[i], linestyle='-', alpha=0.5)

    if not has_predictions:
        plt.text(0.5, 0.5, "No predictions available",
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=14, color='red')

    plt.title('XGBoost Glucose Predictions with 90% Confidence Intervals')
    plt.xlabel('Time')
    plt.ylabel('Glucose Level (mg/dL)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_png:
        plt.savefig(f'{dir}/xgboost_predictions_{timestamp}.png', dpi=300, bbox_inches='tight')

    plt.show()


def plot_prediction_comparison(actual_values, predictions, title="Prediction Comparison", save_path=None):
    """
    Plot comparison between actual and predicted values.

    :param actual_values: Actual glucose values
    :type actual_values: array-like
    :param predictions: Predicted glucose values
    :type predictions: array-like
    :param title: Plot title
    :type title: str
    :param save_path: Path to save the plot
    :type save_path: str
    """
    plt.figure(figsize=(10, 10))

    # Scatter plot
    plt.scatter(actual_values, predictions, alpha=0.5)

    # Add identity line (perfect predictions)
    min_val = min(min(actual_values), min(predictions))
    max_val = max(max(actual_values), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.title(title)
    plt.xlabel('Actual Glucose (mg/dL)')
    plt.ylabel('Predicted Glucose (mg/dL)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")

    plt.show()
