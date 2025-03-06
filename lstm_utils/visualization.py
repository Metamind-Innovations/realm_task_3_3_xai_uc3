from datetime import datetime

import matplotlib.pyplot as plt


def plot_predictions(patient_df, predictions_df, dir, save_png=False):
    """
    Visualize original glucose data alongside LSTM predictions with confidence intervals.

    Creates a plot showing actual glucose readings and predictions for
    each time horizon with confidence intervals.

    :param patient_df: Original patient data
    :type patient_df: pandas.DataFrame
    :param predictions_df: Prediction results from LSTM models
    :type predictions_df: pandas.DataFrame
    :param save_png: Whether to save the plot as PNG
    :type save_png: bool
    """
    plt.figure(figsize=(12, 8))

    col_mapping = patient_df.attrs.get('column_mapping', {})

    time_col = col_mapping.get('time', 'time')
    glucose_col = col_mapping.get('glucose_level', 'glucose_level')

    if time_col not in patient_df.columns:
        print(
            f"Warning: Time column '{time_col}' not found in dataframe. Available columns: {patient_df.columns.tolist()}")
        time_col = patient_df.columns[0]  # Use first column as fallback
    if glucose_col not in patient_df.columns:
        print(
            f"Warning: Glucose column '{glucose_col}' not found in dataframe. Available columns: {patient_df.columns.tolist()}")
        glucose_col = patient_df.columns[1]  # Use second column as fallback

    plt.plot(patient_df[time_col], patient_df[glucose_col], 'b-', label='Actual Glucose')
    plt.scatter(patient_df[time_col], patient_df[glucose_col], color='blue', s=30)

    colors = ['red', 'green', 'purple']
    hours_list = [1, 2, 3]  # PREDICTION_HORIZONS

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

    plt.title('LSTM Glucose Predictions with Confidence Intervals')
    plt.xlabel('Time')
    plt.ylabel('Glucose Level (mg/dL)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_png:
        plt.savefig(f'{dir}/lstm_predictions_{timestamp}.png', dpi=300, bbox_inches='tight')
        print("Plot saved as lstm_glucose_predictions.png")

    plt.show()
