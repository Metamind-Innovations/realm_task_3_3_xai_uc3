import os
from datetime import timedelta

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def load_and_preprocess_training_data(file_path):
    df = pd.read_csv(file_path)
    time_columns = ['TIMER', 'STARTTIME', 'GLCTIMER', 'ENDTIME']
    for col in time_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True)
    df = df.sort_values(by=['SUBJECT_ID', 'TIMER'])
    return df


def load_patient_data(file_path):
    df = pd.read_csv(file_path)

    if 'time' not in df.columns or 'glucose_level' not in df.columns:
        raise ValueError("CSV file must contain 'time' and 'glucose_level' columns")
    df['time'] = pd.to_datetime(df['time'], format='%d/%m/%Y %H:%M:%S')
    df = df.sort_values(by='time')
    return df


def create_patient_features(patient_df):
    df = patient_df.copy()

    features_df = pd.DataFrame()
    glucose_df = df[df['GLC'].notna()].copy()

    if len(glucose_df) <= 1:
        return None

    # For each glucose reading, we want to predict future values
    for i in range(len(glucose_df) - 1):
        current_time = glucose_df.iloc[i]['TIMER']
        current_glucose = glucose_df.iloc[i]['GLC']

        # Find all glucose readings in the next 1-3 hours
        future_cutoff_1hr = current_time + timedelta(hours=1)
        future_cutoff_2hr = current_time + timedelta(hours=2)
        future_cutoff_3hr = current_time + timedelta(hours=3)

        # Get future readings
        future_1hr = glucose_df[(glucose_df['TIMER'] > current_time) &
                                (glucose_df['TIMER'] <= future_cutoff_1hr)]['GLC'].mean()
        future_2hr = glucose_df[(glucose_df['TIMER'] > future_cutoff_1hr) &
                                (glucose_df['TIMER'] <= future_cutoff_2hr)]['GLC'].mean()
        future_3hr = glucose_df[(glucose_df['TIMER'] > future_cutoff_2hr) &
                                (glucose_df['TIMER'] <= future_cutoff_3hr)]['GLC'].mean()

        # Get previous readings if available
        past_readings = glucose_df[glucose_df['TIMER'] < current_time].tail(5)
        past_glucose_values = past_readings['GLC'].tolist()

        # Pad with NaN if we don't have 5 previous readings
        while len(past_glucose_values) < 5:
            past_glucose_values.insert(0, np.nan)

        # Find insulin doses between current time and previous glucose reading
        if i > 0:
            prev_time = glucose_df.iloc[i - 1]['TIMER']
            recent_insulin = df[(df['TIMER'] >= prev_time) &
                                (df['TIMER'] <= current_time) &
                                (df['EVENT'] == 'BOLUS_INYECTION')]
        else:
            # For the first reading, look back 6 hours
            prev_time = current_time - timedelta(hours=6)
            recent_insulin = df[(df['TIMER'] >= prev_time) &
                                (df['TIMER'] <= current_time) &
                                (df['EVENT'] == 'BOLUS_INYECTION')]

        # Calculate insulin features
        short_insulin = recent_insulin[recent_insulin['INSULINTYPE'] == 'Short']['INPUT'].sum()
        long_insulin = recent_insulin[recent_insulin['INSULINTYPE'] == 'Long']['INPUT'].sum()

        # Create row with features
        row = {
            'SUBJECT_ID': patient_df['SUBJECT_ID'].iloc[0],
            'current_glucose': current_glucose,
            'glucose_source': 1 if glucose_df.iloc[i]['GLCSOURCE'] == 'BLOOD' else 0,
            'prev_glucose_1': past_glucose_values[-1] if not np.isnan(past_glucose_values[-1]) else current_glucose,
            'prev_glucose_2': past_glucose_values[-2] if len(past_glucose_values) > 1 and not np.isnan(
                past_glucose_values[-2]) else current_glucose,
            'prev_glucose_3': past_glucose_values[-3] if len(past_glucose_values) > 2 and not np.isnan(
                past_glucose_values[-3]) else current_glucose,
            'prev_glucose_4': past_glucose_values[-4] if len(past_glucose_values) > 3 and not np.isnan(
                past_glucose_values[-4]) else current_glucose,
            'prev_glucose_5': past_glucose_values[-5] if len(past_glucose_values) > 4 and not np.isnan(
                past_glucose_values[-5]) else current_glucose,
            'short_insulin_dose': short_insulin,
            'long_insulin_dose': long_insulin,
            'future_glucose_1hr': future_1hr,
            'future_glucose_2hr': future_2hr,
            'future_glucose_3hr': future_3hr
        }

        features_df = pd.concat([features_df, pd.DataFrame([row])], ignore_index=True)

    return features_df


# Create features from the patient's CSV data for prediction
def create_prediction_features(patient_df):
    """Create features from a simple time + glucose CSV for prediction"""

    df = patient_df.copy()
    features_list = []

    # Need at least 5 readings to make a prediction
    if len(df) < 5:
        raise ValueError("Need at least 5 glucose readings to make predictions")

    # For each point (except the last 4), we'll create a feature row using the next 5 values
    for i in range(len(df) - 4):
        current_time = df.iloc[i]['time']
        sequence = df.iloc[i:i + 5]

        # Get the glucose values
        glucose_values = sequence['glucose_level'].tolist()

        # Create feature row
        feature_row = {
            'current_glucose': glucose_values[-1],  # Most recent reading
            'glucose_source': 0,  # Default to FINGERSTICK
            'prev_glucose_1': glucose_values[-2],
            'prev_glucose_2': glucose_values[-3],
            'prev_glucose_3': glucose_values[-4],
            'prev_glucose_4': glucose_values[-5],
            'prev_glucose_5': glucose_values[-5],  # Just duplicate the earliest
            'short_insulin_dose': 0,  # No insulin info in simple format
            'long_insulin_dose': 0,
            'timestamp': sequence.iloc[-1]['time']  # Most recent time
        }

        features_list.append(feature_row)

    features_df = pd.DataFrame(features_list)
    return features_df


# Prepare dataset for training
def prepare_dataset(df):
    all_features = pd.DataFrame()

    # Process each patient
    for subject_id, patient_data in df.groupby('SUBJECT_ID'):
        patient_features = create_patient_features(patient_data)
        if patient_features is not None:
            all_features = pd.concat([all_features, patient_features], ignore_index=True)

    # Fill missing values
    all_features = all_features.fillna(method='ffill')

    # Drop rows with NaN target values
    for target in ['future_glucose_1hr', 'future_glucose_2hr', 'future_glucose_3hr']:
        all_features = all_features[~all_features[target].isna()]

    return all_features


# Evaluate model and print detailed metrics
def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model and print detailed metrics"""
    y_pred = model.predict(X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Calculate additional metrics
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error

    # Calculate errors
    errors = y_test - y_pred
    mean_error = np.mean(errors)
    std_error = np.std(errors)

    # Print metrics
    print(f"\n{model_name} - Evaluation Metrics:")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R²): {r2:.3f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"Mean Error: {mean_error:.2f}")
    print(f"Standard Deviation of Error: {std_error:.2f}")

    # Print feature importance
    if hasattr(model, 'feature_importances_'):
        feature_cols = ['current_glucose', 'glucose_source',
                        'prev_glucose_1', 'prev_glucose_2', 'prev_glucose_3',
                        'prev_glucose_4', 'prev_glucose_5',
                        'short_insulin_dose', 'long_insulin_dose']

        importance = model.feature_importances_
        sorted_idx = np.argsort(importance)

        print("\nFeature Importance:")
        for i in sorted_idx[::-1]:
            print(f"{feature_cols[i]}: {importance[i]:.4f}")

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'mean_error': mean_error,
        'std_error': std_error
    }


# Plot model evaluation metrics
def plot_model_evaluation(y_test, y_pred, title="Model Evaluation"):
    """Plot actual vs predicted values and error distribution"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Actual vs Predicted
    ax1.scatter(y_test, y_pred, alpha=0.5)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax1.set_xlabel('Actual')
    ax1.set_ylabel('Predicted')
    ax1.set_title('Actual vs Predicted Values')
    ax1.grid(True, alpha=0.3)

    # Error Histogram
    errors = y_test - y_pred
    ax2.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--')
    ax2.set_xlabel('Prediction Error')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Error Distribution')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()


# Train XGBoost models with quantile regression for confidence intervals
def train_glucose_prediction_models(features_df, save_dir='models'):
    models = {}
    quantile_models = {}
    feature_cols = ['current_glucose', 'glucose_source',
                    'prev_glucose_1', 'prev_glucose_2', 'prev_glucose_3',
                    'prev_glucose_4', 'prev_glucose_5',
                    'short_insulin_dose', 'long_insulin_dose']

    # Store metrics for all models
    all_metrics = {}

    # Create directory for saving models if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create directory for metrics
    metrics_dir = os.path.join(save_dir, 'metrics')
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    print("\n" + "=" * 50)
    print("TRAINING GLUCOSE PREDICTION MODELS")
    print("=" * 50)

    for hours in [1, 2, 3]:
        target_col = f'future_glucose_{hours}hr'
        print(f"\n{'*' * 30}")
        print(f"Training {hours}-hour Prediction Model")
        print(f"{'*' * 30}")

        # Split data
        X = features_df[feature_cols]
        y = features_df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"Training data size: {X_train.shape[0]} samples")
        print(f"Test data size: {X_test.shape[0]} samples")

        # Train main model
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='rmse'
        )

        print(f"Training main {hours}-hour prediction model...")
        model.fit(X_train, y_train,
                  eval_set=[(X_train, y_train), (X_test, y_test)],
                  verbose=True)

        # Train lower bound model (5th percentile)
        lower_model = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=0.05,  # 5th percentile for 90% CI
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        print(f"Training lower bound {hours}-hour prediction model...")
        lower_model.fit(X_train, y_train)

        # Train upper bound model (95th percentile)
        upper_model = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=0.95,  # 95th percentile for 90% CI
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        print(f"Training upper bound {hours}-hour prediction model...")
        upper_model.fit(X_train, y_train)

        # Evaluate main model
        model_name = f"{hours}-hour Prediction Model"
        metrics = evaluate_model(model, X_test, y_test, model_name)
        all_metrics[hours] = metrics

        # Plot evaluation
        y_pred = model.predict(X_test)
        plot_model_evaluation(y_test, y_pred, title=f"{hours}-hour Prediction Evaluation")

        # Get confidence intervals on test data
        lower_bound = lower_model.predict(X_test)
        upper_bound = upper_model.predict(X_test)

        # Calculate percentage of true values that fall within the CI
        within_ci = np.sum((y_test >= lower_bound) & (y_test <= upper_bound)) / len(y_test) * 100

        print(f"90% CI coverage: {within_ci:.2f}%")

        # Export metrics to CSV
        metrics_df = pd.DataFrame({
            'Metric': ['RMSE', 'MAE', 'R²', 'MAPE', 'Mean Error', 'Std Error', 'CI Coverage'],
            'Value': [metrics['rmse'], metrics['mae'], metrics['r2'],
                      metrics['mape'], metrics['mean_error'], metrics['std_error'], within_ci]
        })
        metrics_df.to_csv(os.path.join(metrics_dir, f'metrics_{hours}hr.csv'), index=False)

        # Export feature importance to CSV if available
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            })
            importance_df = importance_df.sort_values('Importance', ascending=False)
            importance_df.to_csv(os.path.join(metrics_dir, f'feature_importance_{hours}hr.csv'), index=False)

        # Save models
        models[hours] = model
        quantile_models[hours] = {
            'lower': lower_model,
            'upper': upper_model
        }

        # Save models to disk
        joblib.dump(model, os.path.join(save_dir, f'glucose_model_{hours}hr.pkl'))
        joblib.dump(lower_model, os.path.join(save_dir, f'glucose_model_{hours}hr_lower.pkl'))
        joblib.dump(upper_model, os.path.join(save_dir, f'glucose_model_{hours}hr_upper.pkl'))

    # Print summary of all models
    print("\n" + "=" * 50)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 50)

    summary_data = []
    for hours, metrics in all_metrics.items():
        summary_data.append({
            'Prediction Horizon': f"{hours}-hour",
            'RMSE': f"{metrics['rmse']:.2f}",
            'MAE': f"{metrics['mae']:.2f}",
            'R²': f"{metrics['r2']:.3f}"
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # Save summary to CSV
    summary_df.to_csv(os.path.join(metrics_dir, 'model_summary.csv'), index=False)

    return models, quantile_models


# Load models from disk
def load_models(model_dir='models'):
    models = {}
    quantile_models = {}

    for hours in [1, 2, 3]:
        model_path = os.path.join(model_dir, f'glucose_model_{hours}hr.pkl')
        lower_path = os.path.join(model_dir, f'glucose_model_{hours}hr_lower.pkl')
        upper_path = os.path.join(model_dir, f'glucose_model_{hours}hr_upper.pkl')

        if os.path.exists(model_path) and os.path.exists(lower_path) and os.path.exists(upper_path):
            models[hours] = joblib.load(model_path)
            quantile_models[hours] = {
                'lower': joblib.load(lower_path),
                'upper': joblib.load(upper_path)
            }
        else:
            print(f"Warning: Models for {hours}-hour prediction not found")

    return models, quantile_models


# Prediction function for patient data
def predict_for_patient(patient_df, models, quantile_models):
    """Make predictions with confidence intervals for patient data"""

    # Create features for prediction
    features_df = create_prediction_features(patient_df)

    if len(features_df) == 0:
        return pd.DataFrame()  # Not enough data

    # Extract feature columns for prediction
    feature_cols = ['current_glucose', 'glucose_source',
                    'prev_glucose_1', 'prev_glucose_2', 'prev_glucose_3',
                    'prev_glucose_4', 'prev_glucose_5',
                    'short_insulin_dose', 'long_insulin_dose']

    X = features_df[feature_cols]
    timestamps = features_df['timestamp']

    # Make predictions for each time horizon
    predictions = []

    for i, row in X.iterrows():
        row_df = pd.DataFrame([row])
        timestamp = timestamps.iloc[i]

        pred_row = {
            'timestamp': timestamp,
            'current_glucose': row['current_glucose']
        }

        for hours in [1, 2, 3]:
            # Skip if model not available
            if hours not in models or hours not in quantile_models:
                continue

            # Get predictions
            model = models[hours]
            lower_model = quantile_models[hours]['lower']
            upper_model = quantile_models[hours]['upper']

            mean_pred = model.predict(row_df)[0]
            lower_bound = lower_model.predict(row_df)[0]
            upper_bound = upper_model.predict(row_df)[0]

            # Add to results
            pred_time = timestamp + timedelta(hours=hours)
            pred_row[f'predicted_{hours}hr'] = mean_pred
            pred_row[f'lower_bound_{hours}hr'] = lower_bound
            pred_row[f'upper_bound_{hours}hr'] = upper_bound
            pred_row[f'prediction_time_{hours}hr'] = pred_time

        predictions.append(pred_row)

    return pd.DataFrame(predictions)


# Plot predictions with confidence intervals
def plot_patient_predictions(patient_df, predictions_df):
    """Plot the original glucose data and predictions with confidence intervals"""

    plt.figure(figsize=(12, 8))

    # Plot original data
    plt.plot(patient_df['time'], patient_df['glucose_level'], 'b-', label='Actual Glucose')
    plt.scatter(patient_df['time'], patient_df['glucose_level'], color='blue', s=30)

    # Plot predictions for each time horizon with different colors
    colors = ['red', 'green', 'purple']
    hours_list = [1, 2, 3]

    for i, hours in enumerate(hours_list):
        if f'predicted_{hours}hr' not in predictions_df.columns:
            continue

        # Get prediction times and values
        pred_times = predictions_df[f'prediction_time_{hours}hr']
        pred_values = predictions_df[f'predicted_{hours}hr']
        lower_bounds = predictions_df[f'lower_bound_{hours}hr']
        upper_bounds = predictions_df[f'upper_bound_{hours}hr']

        # Plot mean predictions
        plt.scatter(pred_times, pred_values, color=colors[i],
                    label=f'{hours}-hour Prediction', s=40, marker='s')

        # Plot confidence intervals
        for j in range(len(pred_times)):
            plt.fill_between([pred_times.iloc[j], pred_times.iloc[j]],
                             [lower_bounds.iloc[j]], [upper_bounds.iloc[j]],
                             color=colors[i], alpha=0.2)

            plt.plot([pred_times.iloc[j], pred_times.iloc[j]],
                     [lower_bounds.iloc[j], upper_bounds.iloc[j]],
                     color=colors[i], linestyle='-', alpha=0.5)

    plt.title('Glucose Predictions with 90% Confidence Intervals')
    plt.xlabel('Time')
    plt.ylabel('Glucose Level (mg/dL)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Format x-axis dates
    plt.gcf().autofmt_xdate()

    # Save the plot
    plt.savefig('glucose_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()


# Export predictions to CSV
def export_predictions(predictions_df, output_file='glucose_predictions.csv'):
    """Export predictions to CSV file"""
    predictions_df.to_csv(output_file, index=False)
    print(f"Predictions exported to {output_file}")


# Evaluate predictions against actual values (if available)
def evaluate_predictions(predictions_df, actual_df):
    """Evaluate prediction accuracy if actual future values are available"""

    evaluation_results = {}

    # For each prediction horizon
    for hours in [1, 2, 3]:
        if f'predicted_{hours}hr' not in predictions_df.columns:
            continue

        # Extract prediction times and values
        pred_times = predictions_df[f'prediction_time_{hours}hr']
        pred_values = predictions_df[f'predicted_{hours}hr']
        lower_bounds = predictions_df[f'lower_bound_{hours}hr']
        upper_bounds = predictions_df[f'upper_bound_{hours}hr']

        # Find actual glucose values at (or closest to) prediction times
        actual_values = []
        for pred_time in pred_times:
            # Find closest time in actual data
            time_diffs = abs(actual_df['time'] - pred_time)
            closest_idx = time_diffs.idxmin()
            closest_time = actual_df.loc[closest_idx, 'time']

            # Only use if within 10 minutes of prediction time
            if abs((closest_time - pred_time).total_seconds()) <= 600:  # 10 minutes
                actual_values.append(actual_df.loc[closest_idx, 'glucose_level'])
            else:
                actual_values.append(np.nan)

        # Convert to array and filter out NaNs
        actual_values = np.array(actual_values)
        mask = ~np.isnan(actual_values)

        if sum(mask) > 0:
            # Calculate metrics
            actual_filtered = actual_values[mask]
            pred_filtered = pred_values.iloc[mask].values
            lower_filtered = lower_bounds.iloc[mask].values
            upper_filtered = upper_bounds.iloc[mask].values

            rmse = np.sqrt(mean_squared_error(actual_filtered, pred_filtered))
            mae = mean_absolute_error(actual_filtered, pred_filtered)
            r2 = r2_score(actual_filtered, pred_filtered)

            # CI coverage
            within_ci = np.sum((actual_filtered >= lower_filtered) &
                               (actual_filtered <= upper_filtered)) / len(actual_filtered) * 100

            # Store results
            evaluation_results[hours] = {
                'num_points': len(actual_filtered),
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'ci_coverage': within_ci
            }

            # Print results
            print(f"\n{hours}-hour Prediction Evaluation:")
            print(f"Number of evaluation points: {len(actual_filtered)}")
            print(f"RMSE: {rmse:.2f}")
            print(f"MAE: {mae:.2f}")
            print(f"R²: {r2:.3f}")
            print(f"90% CI coverage: {within_ci:.2f}%")

    return evaluation_results


# Main function to process patient data
def process_patient_data(patient_file, model_dir='models', output_file='glucose_predictions.csv'):
    """Process patient data file and make predictions"""

    # Load patient data
    print(f"Loading patient data from {patient_file}...")
    patient_df = load_patient_data(patient_file)

    # Check if we have enough data
    if len(patient_df) < 5:
        print("Error: Need at least 5 glucose readings to make predictions")
        return None

    # Load models
    print("Loading prediction models...")
    models, quantile_models = load_models(model_dir)

    if not models:
        print("Error: No models found. Please train models first.")
        return None

    # Make predictions
    print("Making predictions...")
    predictions_df = predict_for_patient(patient_df, models, quantile_models)

    if len(predictions_df) == 0:
        print("Error: Could not generate predictions")
        return None

    # Export predictions
    export_predictions(predictions_df, output_file)

    # Plot results
    print("Plotting results...")
    plot_patient_predictions(patient_df, predictions_df)

    # Evaluate predictions if future data is available
    future_end_time = predictions_df[f'prediction_time_3hr'].max()
    if patient_df['time'].max() >= future_end_time:
        print("\nEvaluating prediction accuracy...")
        evaluation_results = evaluate_predictions(predictions_df, patient_df)

        # Export evaluation results
        if evaluation_results:
            eval_df = pd.DataFrame([
                {
                    'Prediction_Horizon': f"{hours}-hour",
                    'RMSE': results['rmse'],
                    'MAE': results['mae'],
                    'R2': results['r2'],
                    'CI_Coverage': results['ci_coverage'],
                    'Num_Points': results['num_points']
                }
                for hours, results in evaluation_results.items()
            ])
            eval_df.to_csv('prediction_evaluation.csv', index=False)
            print("Evaluation results exported to prediction_evaluation.csv")

    return predictions_df


# Main function to train models and make them ready for use
def train_models(training_file, model_dir='models'):
    """Train glucose prediction models from historical data"""

    # Load and preprocess training data
    print(f"Loading training data from {training_file}...")
    df = load_and_preprocess_training_data(training_file)

    # Prepare features
    print("Creating features...")
    features_df = prepare_dataset(df)

    # Train models
    print("Training models...")
    models, quantile_models = train_glucose_prediction_models(features_df, model_dir)

    print(f"Models saved to {model_dir} directory")

    return models, quantile_models


# Main execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Glucose Prediction System')
    parser.add_argument('--train', action='store_true', help='Train new models')
    parser.add_argument('--predict', action='store_true', help='Make predictions on patient data')
    parser.add_argument('--training-file', type=str, default='glucose_insulin_smallest.csv',
                        help='CSV file with training data')
    parser.add_argument('--patient-file', type=str, help='CSV file with patient data (time and glucose_level columns)')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory for model storage')
    parser.add_argument('--output-file', type=str, default='glucose_predictions.csv',
                        help='Output file for predictions')

    args = parser.parse_args()

    # Validate arguments
    if args.train and args.predict:
        print("Error: Cannot both train and predict at the same time. Please choose one operation.")
        exit(1)

    if not args.train and not args.predict:
        print("Error: Must specify either --train or --predict")
        parser.print_help()
        exit(1)

    # Training mode
    if args.train:
        if not args.training_file:
            print("Error: Training file must be specified with --training-file")
            exit(1)

        print(f"Starting model training using data from {args.training_file}")
        train_models(args.training_file, args.model_dir)
        print("Training complete!")

    # Prediction mode
    if args.predict:
        if not args.patient_file:
            print("Error: Patient file must be specified with --patient-file")
            exit(1)

        print(f"Making predictions for patient data in {args.patient_file}")
        predictions = process_patient_data(
            args.patient_file,
            args.model_dir,
            args.output_file
        )

        if predictions is not None:
            print(f"Predictions saved to {args.output_file}")
            print("Prediction complete!")
        else:
            print("Prediction failed. Please check the error messages above.")
