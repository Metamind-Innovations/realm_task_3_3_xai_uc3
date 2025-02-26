from datetime import timedelta

import numpy as np
import pandas as pd
import xgboost_prediction as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Convert timestamps to datetime
    time_columns = ['TIMER', 'STARTTIME', 'GLCTIMER', 'ENDTIME']
    for col in time_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True)
    df = df.sort_values(by=['SUBJECT_ID', 'TIMER'])
    return df


# Create features for each patient
def create_patient_features(patient_df):
    df = patient_df.copy()
    features_df = pd.DataFrame()

    # Interested in glucose measurements only
    glucose_df = df[df['GLC'].notna()].copy()

    if len(glucose_df) <= 1:
        return None  # Not enough data for this patient

    # For each glucose reading, we want to predict future values
    for i in range(len(glucose_df) - 1):
        current_time = glucose_df.iloc[i]['TIMER']
        current_glucose = glucose_df.iloc[i]['GLC']

        # Find all glucose readings in the next 1-3 hours
        future_cutoff_1hr = current_time + timedelta(hours=1)
        future_cutoff_2hr = current_time + timedelta(hours=2)
        future_cutoff_3hr = current_time + timedelta(hours=3)

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


def prepare_dataset(df):
    all_features = pd.DataFrame()
    for subject_id, patient_data in df.groupby('SUBJECT_ID'):
        patient_features = create_patient_features(patient_data)
        if patient_features is not None:
            all_features = pd.concat([all_features, patient_features], ignore_index=True)
    all_features = all_features.fillna(method='ffill')

    # Drop rows with NaN
    for target in ['future_glucose_1hr', 'future_glucose_2hr', 'future_glucose_3hr']:
        all_features = all_features[~all_features[target].isna()]

    return all_features


def train_glucose_prediction_models(features_df):
    models = {}
    feature_cols = ['current_glucose', 'glucose_source',
                    'prev_glucose_1', 'prev_glucose_2', 'prev_glucose_3',
                    'prev_glucose_4', 'prev_glucose_5',
                    'short_insulin_dose', 'long_insulin_dose']

    for hours in [1, 2, 3]:
        target_col = f'future_glucose_{hours}hr'

        X = features_df[feature_cols]
        y = features_df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\n{hours}-hour Prediction Results:")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R²: {r2:.2f}")

        models[hours] = model

    return models


def predict_from_single_value(glucose_value, models):
    """Predict future glucose values from a single current reading."""
    features = {
        'current_glucose': glucose_value,
        'glucose_source': 0,
        'prev_glucose_1': glucose_value,
        'prev_glucose_2': glucose_value,
        'prev_glucose_3': glucose_value,
        'prev_glucose_4': glucose_value,
        'prev_glucose_5': glucose_value,
        'short_insulin_dose': 0,
        'long_insulin_dose': 0
    }

    X = pd.DataFrame([features])

    predictions = {}
    for hours, model in models.items():
        predictions[f'{hours}hr'] = model.predict(X)[0]

    return predictions


def predict_from_sequence(glucose_values, models):
    """Predict future glucose values from a sequence of readings."""
    if len(glucose_values) != 5:
        raise ValueError("Please provide exactly 5 glucose values")

    features = {
        'current_glucose': glucose_values[-1],
        'glucose_source': 0,
        'prev_glucose_1': glucose_values[-2],
        'prev_glucose_2': glucose_values[-3],
        'prev_glucose_3': glucose_values[-4],
        'prev_glucose_4': glucose_values[-5],
        'prev_glucose_5': glucose_values[-5],
        'short_insulin_dose': 0,
        'long_insulin_dose': 0
    }

    X = pd.DataFrame([features])

    predictions = {}
    for hours, model in models.items():
        predictions[f'{hours}hr'] = model.predict(X)[0]

    return predictions


def main(file_path):
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(file_path)

    print("Creating features...")
    features_df = prepare_dataset(df)

    print("Training models...")
    models = train_glucose_prediction_models(features_df)

    print("\nExample Predictions:")
    single_value = 150  # Example single value
    print(f"\nPrediction from single value {single_value}:")
    single_pred = predict_from_single_value(single_value, models)
    for time, value in single_pred.items():
        print(f"  Predicted glucose at {time}: {value:.1f}")

    sequence = [130, 145, 160, 155, 150]  # Sequence of 5 glucose values
    print(f"\nPrediction from sequence {sequence}:")
    seq_pred = predict_from_sequence(sequence, models)
    for time, value in seq_pred.items():
        print(f"  Predicted glucose at {time}: {value:.1f}")

    return models


if __name__ == "__main__":
    file_path = "dataset_small/glucose_insulin_5K.csv"
    models = main(file_path)
