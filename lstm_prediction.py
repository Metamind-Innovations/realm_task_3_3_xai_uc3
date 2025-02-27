import argparse
import hashlib
import os
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

# Constants
OUTPUT_FILE = "lstm_glucose_predictions.csv"
MODEL_DIRECTORY = "lstm_models"
MAX_SEQUENCE_LENGTH = 10
PREDICTION_HORIZONS = [1, 2, 3]  # Hours into the future
FEATURES_TO_INCLUDE = [
    'glucose_level', 'glucose_std', 'glucose_rate', 'glucose_mean',
    'glucose_min', 'glucose_max', 'glucose_range', 'glucose_acceleration',
    'hour_of_day', 'is_daytime', 'day_of_week', 'is_weekend'
]


def load_and_preprocess_training_data(file_path):
    """
    Load training data and convert time columns to datetime format.

    Performs case-insensitive column mapping to standardize column names
    and converts time-related fields to appropriate datetime format.

    :param file_path: Path to the CSV file with training data
    :type file_path: str
    :returns: Preprocessed DataFrame with mapped columns
    :rtype: pandas.DataFrame
    """
    df = pd.read_csv(file_path)

    print("Actual CSV columns:", df.columns.tolist())

    col_mapping = {}
    col_lower_to_actual = {col.lower(): col for col in df.columns}

    expected_columns = {
        'subject_id': 'SUBJECT_ID',
        'timer': 'TIMER',
        'starttime': 'STARTTIME',
        'glctimer': 'GLCTIMER',
        'endtime': 'ENDTIME',
        'input': 'INPUT',
        'input_hrs': 'INPUT_HRS',
        'glc': 'GLC',
        'infxstop': 'INFXSTOP',
        'gender': 'GENDER',
        'event': 'EVENT',
        'glcsource': 'GLCSOURCE',
        'insulintype': 'INSULINTYPE',
        'first_icu_stay': 'FIRST_ICU_STAY',
        'admission_age': 'ADMISSION_AGE',
        'los_icu_days': 'LOS_ICU_DAYS',
        'icd9_code': 'ICD9_CODE'
    }

    for exp_lower, exp_upper in expected_columns.items():
        if exp_lower in col_lower_to_actual:
            col_mapping[exp_upper] = col_lower_to_actual[exp_lower]
        else:
            if exp_upper in df.columns:
                col_mapping[exp_upper] = exp_upper

    print("Using column mapping:", col_mapping)

    time_columns = ['TIMER', 'STARTTIME', 'GLCTIMER', 'ENDTIME']
    for col in time_columns:
        if col in col_mapping and col_mapping[col] in df.columns:
            df[col_mapping[col]] = pd.to_datetime(df[col_mapping[col]], dayfirst=True, errors='coerce')

    numeric_columns = ['INPUT', 'INPUT_HRS', 'GLC', 'INFXSTOP']
    for col in numeric_columns:
        if col in col_mapping and col_mapping[col] in df.columns:
            df[col_mapping[col]] = pd.to_numeric(df[col_mapping[col]], errors='coerce')

    sort_cols = []
    if 'SUBJECT_ID' in col_mapping and col_mapping['SUBJECT_ID'] in df.columns:
        sort_cols.append(col_mapping['SUBJECT_ID'])
    if 'TIMER' in col_mapping and col_mapping['TIMER'] in df.columns:
        sort_cols.append(col_mapping['TIMER'])

    if sort_cols:
        df = df.sort_values(by=sort_cols)
    else:
        print("Warning: Could not sort by SUBJECT_ID and TIMER - columns not found")

    df.attrs['column_mapping'] = col_mapping

    return df


def load_patient_data(file_path):
    """
    Load patient data for prediction, ensuring required columns exist.

    Identifies time and glucose columns using case-insensitive matching
    and attempts to parse datetime formats using several common patterns.

    :param file_path: Path to the CSV file with patient data
    :type file_path: str
    :returns: DataFrame with time and glucose data properly formatted
    :rtype: pandas.DataFrame
    :raises ValueError: If required columns are missing or time parsing fails
    """
    df = pd.read_csv(file_path)

    print("Patient data columns:", df.columns.tolist())

    time_col = None
    for col in df.columns:
        if col.lower() == 'time':
            time_col = col
            break

    glucose_col = None
    for col in df.columns:
        if col.lower() in ['glucose_level', 'glucose', 'glc']:
            glucose_col = col
            break

    if not time_col or not glucose_col:
        raise ValueError(f"CSV file must contain time and glucose_level columns. Found: {df.columns.tolist()}")

    col_mapping = {
        'time': time_col,
        'glucose_level': glucose_col
    }

    df.attrs['column_mapping'] = col_mapping

    print(f"Using '{time_col}' as time column and '{glucose_col}' as glucose column")

    datetime_formats = ['%d/%m/%Y %H:%M:%S', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S']

    for dt_format in datetime_formats:
        try:
            df[time_col] = pd.to_datetime(df[time_col], format=dt_format)
            print(f"Parsed dates using format: {dt_format}")
            break
        except:
            continue

    if not pd.api.types.is_datetime64_dtype(df[time_col]):
        print("Using flexible datetime parser")
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

    if df[time_col].isna().all():
        raise ValueError(f"Could not parse '{time_col}' column as datetime")

    df = df.sort_values(by=time_col)

    return df


def clean_infinite_values(df):
    """
    Replace infinities and extremely large values in dataframe with NaN or capped values.

    Helps prevent numerical instability in calculations by:
    1. Replacing infinity values with NaN
    2. Capping extreme values (beyond 1e10) to reasonable maximum

    :param df: Input dataframe with potential infinite or extreme values
    :type df: pandas.DataFrame
    :returns: Cleaned dataframe with handled values
    :rtype: pandas.DataFrame
    """
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()

    if inf_count > 0:
        print(f"Warning: Found {inf_count} infinity values in the dataset. Replacing with NaN.")
        df = df.replace([np.inf, -np.inf], np.nan)

    for col in df.select_dtypes(include=[np.number]).columns:
        extreme_mask = (df[col].abs() > 1e10) & df[col].notna()
        extreme_count = extreme_mask.sum()

        if extreme_count > 0:
            print(f"Warning: Found {extreme_count} extreme values in column '{col}'. Capping to reasonable values.")
            df.loc[extreme_mask, col] = df.loc[extreme_mask, col].apply(
                lambda x: 1e10 if x > 0 else -1e10
            )

    return df


def encode_icd9_code(code):
    """
    Create a numeric representation of ICD9 code.

    Uses first 3 characters as category and hashes the rest for subcategory.
    This creates a consistent numerical representation usable for modeling.

    :param code: ICD-9 diagnosis code string
    :type code: str
    :returns: Tuple of (category, subcategory) as numerical values
    :rtype: tuple(float, float)
    """
    if pd.isna(code) or not isinstance(code, str) or len(code) < 3:
        return 0.0, 0.0

    try:
        category = float(code[:3])
    except ValueError:
        category = 0.0

    if len(code) > 3:
        hash_val = int(hashlib.md5(code[3:].encode()).hexdigest(), 16)
        subcategory = float(hash_val % 1000) / 1000  # Normalize to 0-1
    else:
        subcategory = 0.0

    return category, subcategory


def extract_patient_features(patient_df, is_training=False):
    """
    Extract glucose features and demographic info for a single patient.

    Calculates various glucose metrics including rolling statistics,
    rates of change, and time-based features. For training data,
    also creates target columns for future glucose values.

    :param patient_df: Patient dataframe with glucose readings
    :type patient_df: pandas.DataFrame
    :param is_training: Whether processing for training (True) or prediction (False)
    :type is_training: bool
    :returns: Expanded DataFrame with calculated features
    :rtype: pandas.DataFrame
    """
    col_mapping = patient_df.attrs.get('column_mapping', {})

    def get_col(expected_col):
        if expected_col in col_mapping and col_mapping[expected_col] in patient_df.columns:
            return col_mapping[expected_col]
        elif expected_col in patient_df.columns:
            return expected_col
        elif expected_col.lower() in patient_df.columns:
            return expected_col.lower()
        return None

    time_col = get_col('TIMER') or get_col('timer') or get_col('time') or get_col('TIME')
    glucose_col = get_col('GLC') or get_col('glc') or get_col('glucose_level') or get_col('GLUCOSE_LEVEL') or get_col(
        'glucose')

    if not time_col:
        print(f"Error: Time column not found in available columns: {patient_df.columns.tolist()}")
        return None
    if not glucose_col:
        print(f"Error: Glucose column not found in available columns: {patient_df.columns.tolist()}")
        return None

    df = patient_df.copy()

    df = df.sort_values(by=time_col)

    df['glucose_level'] = df[glucose_col]
    df['glucose_std'] = df['glucose_level'].rolling(window=3, min_periods=1).std().fillna(0)
    df['glucose_mean'] = df['glucose_level'].rolling(window=3, min_periods=1).mean().fillna(df['glucose_level'])
    df['glucose_min'] = df['glucose_level'].rolling(window=6, min_periods=1).min().fillna(df['glucose_level'])
    df['glucose_max'] = df['glucose_level'].rolling(window=6, min_periods=1).max().fillna(df['glucose_level'])
    df['glucose_range'] = df['glucose_max'] - df['glucose_min']

    df['time_diff'] = df[time_col].diff().dt.total_seconds()
    df.loc[df['time_diff'] <= 0, 'time_diff'] = 0.1  # Avoid division by zero
    df['glucose_diff'] = df['glucose_level'].diff()
    df['glucose_rate'] = (df['glucose_diff'] / df['time_diff'] * 60).fillna(0)  # per minute

    df['glucose_rate_diff'] = df['glucose_rate'].diff()
    df['glucose_acceleration'] = (df['glucose_rate_diff'] / df['time_diff'] * 60).fillna(0)  # per minute^2

    df.loc[df['glucose_rate'] > 10, 'glucose_rate'] = 10
    df.loc[df['glucose_rate'] < -10, 'glucose_rate'] = -10
    df.loc[df['glucose_acceleration'] > 2, 'glucose_acceleration'] = 2
    df.loc[df['glucose_acceleration'] < -2, 'glucose_acceleration'] = -2

    df['hour_of_day'] = df[time_col].dt.hour
    df['is_daytime'] = ((df['hour_of_day'] >= 7) & (df['hour_of_day'] <= 22)).astype(int)
    df['day_of_week'] = df[time_col].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    demographics = {}

    gender_col = get_col('GENDER')
    if gender_col and len(df) > 0:
        gender = df[gender_col].iloc[0]
        demographics['gender'] = 1 if gender == 'M' else 0 if gender == 'F' else 0.5
    else:
        demographics['gender'] = 0.5  # Default

    age_col = get_col('ADMISSION_AGE')
    if age_col and len(df) > 0:
        age = df[age_col].iloc[0]
        if not pd.isna(age) and isinstance(age, (int, float)):
            demographics['age'] = min(max(age, 0), 120) / 120  # Normalize 0-1
        else:
            demographics['age'] = 0.5  # Default
    else:
        demographics['age'] = 0.5  # Default

    icd9_col = get_col('ICD9_CODE')
    if icd9_col and len(df) > 0:
        icd9_code = df[icd9_col].iloc[0]
        if not pd.isna(icd9_code) and isinstance(icd9_code, str):
            demographics['icd9_category'], demographics['icd9_subcategory'] = encode_icd9_code(icd9_code)
        else:
            demographics['icd9_category'], demographics['icd9_subcategory'] = 0, 0
    else:
        demographics['icd9_category'], demographics['icd9_subcategory'] = 0, 0

    if is_training:
        for hours in PREDICTION_HORIZONS:
            avg_time_diff = df['time_diff'].median()
            if avg_time_diff <= 0 or pd.isna(avg_time_diff):
                avg_time_diff = 300  # Default to 5 minutes if can't determine

            rows_per_hour = int(3600 / avg_time_diff)
            shift_steps = hours * rows_per_hour

            target_col = f'future_glucose_{hours}hr'
            df[target_col] = df['glucose_level'].shift(-shift_steps)

        for hours in PREDICTION_HORIZONS:
            target_col = f'future_glucose_{hours}hr'
            df = df[~df[target_col].isna()]

    df = clean_infinite_values(df)

    df.attrs['demographics'] = demographics

    df.attrs['feature_mapping'] = {
        'time': time_col,
        'glucose': 'glucose_level'
    }

    return df


def prepare_lstm_dataset(patient_dfs, sequence_length=MAX_SEQUENCE_LENGTH):
    """
    Prepare sequences for LSTM model from patient dataframes.

    Creates sliding window sequences of features and matching target values
    for each prediction horizon.

    :param patient_dfs: List of patient DataFrames with features
    :type patient_dfs: list[pandas.DataFrame]
    :param sequence_length: Number of time points in each sequence
    :type sequence_length: int
    :returns: X sequences and y target values for each prediction horizon
    :rtype: tuple(numpy.ndarray, dict)
    """
    print(f"Preparing LSTM dataset with sequence length {sequence_length}...")

    feature_cols = FEATURES_TO_INCLUDE

    X_sequences = []
    y_dict = {hours: [] for hours in PREDICTION_HORIZONS}

    for patient_df in tqdm(patient_dfs, desc="Preparing patient sequences"):
        if 'glucose_level' not in patient_df.columns:
            print("Warning: glucose_level column not found in dataframe, skipping patient")
            continue

        features = []
        for col in feature_cols:
            if col in patient_df.columns:
                features.append(patient_df[col].values)
            else:
                print(f"Feature {col} not available, using zeros")
                features.append(np.zeros(len(patient_df)))

        if not features:
            print("No valid features found for patient, skipping")
            continue

        feature_array = np.column_stack(features)

        for i in range(len(feature_array) - sequence_length):
            sequence = feature_array[i:i + sequence_length]
            X_sequences.append(sequence)

            for hours in PREDICTION_HORIZONS:
                target_col = f'future_glucose_{hours}hr'
                if target_col in patient_df.columns:
                    target_value = patient_df[target_col].iloc[i + sequence_length - 1]
                    y_dict[hours].append(target_value)
                else:
                    y_dict[hours].append(patient_df['glucose_level'].iloc[i + sequence_length - 1])

    X = np.array(X_sequences)
    y = {hours: np.array(values) for hours, values in y_dict.items()}

    print(f"Created dataset with {len(X)} sequences")
    for hours, values in y.items():
        print(f"  {hours}-hour targets: {len(values)} values")

    return X, y


def create_lstm_model(sequence_length, n_features):
    """
    Create a bidirectional LSTM model for glucose prediction.

    Architecture includes:
    - Two bidirectional LSTM layers with batch normalization and dropout
    - Dense output layers for final prediction

    :param sequence_length: Number of time points in input sequences
    :type sequence_length: int
    :param n_features: Number of features per time point
    :type n_features: int
    :returns: Compiled LSTM model
    :rtype: tensorflow.keras.models.Sequential
    """
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True),
                      input_shape=(sequence_length, n_features)),
        BatchNormalization(),
        Dropout(0.2),

        Bidirectional(LSTM(32)),
        BatchNormalization(),
        Dropout(0.2),

        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )

    return model


def train_lstm_models(X, y, model_dir=MODEL_DIRECTORY,
                      test_size=0.2, validation_size=0.2, random_state=42):
    """
    Train LSTM models for different prediction horizons.

    Handles data scaling, model training with callbacks for early stopping
    and learning rate reduction, and saves models and evaluation metrics.

    :param X: Input sequences
    :type X: numpy.ndarray
    :param y: Target values for each prediction horizon
    :type y: dict
    :param model_dir: Directory to save models and results
    :type model_dir: str
    :param test_size: Fraction of data to use for testing
    :type test_size: float
    :param validation_size: Fraction of training data to use for validation
    :type validation_size: float
    :param random_state: Random seed for reproducibility
    :type random_state: int
    :returns: Trained models, scalers, and evaluation results
    :rtype: tuple(dict, dict, dict)
    """
    os.makedirs(model_dir, exist_ok=True)
    models = {}
    results = {}

    scalers = {
        'X': MinMaxScaler(),
        'y': {hours: MinMaxScaler() for hours in PREDICTION_HORIZONS}
    }

    X_reshaped = X.reshape(-1, X.shape[2])
    X_scaled_2d = scalers['X'].fit_transform(X_reshaped)
    X_scaled = X_scaled_2d.reshape(X.shape)

    for hours in PREDICTION_HORIZONS:
        print(f"\nTraining {hours}-hour prediction model")

        y_current = y[hours].reshape(-1, 1)
        y_scaled = scalers['y'][hours].fit_transform(y_current)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=test_size, random_state=random_state)

        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size_adjusted, random_state=random_state)

        print(f"Training set: {X_train.shape[0]} sequences")
        print(f"Validation set: {X_val.shape[0]} sequences")
        print(f"Test set: {X_test.shape[0]} sequences")

        model = create_lstm_model(X.shape[1], X.shape[2])
        model.summary()

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(model_dir, f'lstm_{hours}hr_checkpoint.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

        print(f"Training {hours}-hour model...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

        test_results = model.evaluate(X_test, y_test, verbose=1)
        print(f"Test loss (MSE): {test_results[0]:.4f}")
        print(f"Test MAE: {test_results[1]:.4f}")

        y_pred_scaled = model.predict(X_test)
        y_pred = scalers['y'][hours].inverse_transform(y_pred_scaled)
        y_test_orig = scalers['y'][hours].inverse_transform(y_test)

        mse = mean_squared_error(y_test_orig, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_orig, y_pred)
        r2 = r2_score(y_test_orig, y_pred)

        print(f"RMSE (original scale): {rmse:.2f}")
        print(f"MAE (original scale): {mae:.2f}")
        print(f"R² score: {r2:.3f}")

        results[hours] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'history': history.history
        }

        model_path = os.path.join(model_dir, f'lstm_model_{hours}hr.keras')
        model.save(model_path)
        print(f"Model saved to {model_path}")

        models[hours] = model

    import pickle
    with open(os.path.join(model_dir, 'scalers.pkl'), 'wb') as f:
        pickle.dump(scalers, f)

    import json
    with open(os.path.join(model_dir, 'training_results.json'), 'w') as f:
        serializable_results = {}
        for hours, res in results.items():
            serializable_results[str(hours)] = {
                'mse': float(res['mse']),
                'rmse': float(res['rmse']),
                'mae': float(res['mae']),
                'r2': float(res['r2']),
                'history': {k: [float(x) for x in v] for k, v in res['history'].items()}
            }
        json.dump(serializable_results, f, indent=2)

    return models, scalers, results


def load_lstm_models(model_dir=MODEL_DIRECTORY):
    """
    Load trained LSTM models and scalers from disk.

    Handles multiple file formats (.keras, .h5) and implements fallback
    strategies if the initial loading approach fails.

    :param model_dir: Directory containing saved models and scalers
    :type model_dir: str
    :returns: Loaded models and scalers
    :rtype: tuple(dict, dict)
    """
    models = {}

    if not os.path.exists(model_dir):
        print(f"Model directory {model_dir} not found. Please train models first.")
        return None, None

    import pickle
    scaler_path = os.path.join(model_dir, 'scalers.pkl')
    if not os.path.exists(scaler_path):
        print(f"Scalers file not found at {scaler_path}")
        return None, None

    try:
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
    except Exception as e:
        print(f"Error loading scalers: {str(e)}")
        return None, None

    found_models = False

    for hours in PREDICTION_HORIZONS:
        model_path_keras = os.path.join(model_dir, f'lstm_model_{hours}hr.keras')
        model_path_h5 = os.path.join(model_dir, f'lstm_model_{hours}hr.h5')

        model_path = None
        if os.path.exists(model_path_keras):
            model_path = model_path_keras
        elif os.path.exists(model_path_h5):
            model_path = model_path_h5

        if model_path:
            try:
                custom_objects = {
                    'Adam': Adam
                }

                try:
                    models[hours] = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                    print(f"Loaded {hours}-hour prediction model from {model_path}")
                    found_models = True
                except Exception as e1:
                    print(f"First load attempt failed: {str(e1)}")
                    try:
                        models[hours] = tf.keras.models.load_model(model_path, compile=False)
                        models[hours].compile(
                            optimizer=Adam(learning_rate=0.001),
                            loss='mean_squared_error',
                            metrics=['mae']
                        )
                        print(f"Loaded and recompiled {hours}-hour prediction model from {model_path}")
                        found_models = True
                    except Exception as e2:
                        print(f"Second load attempt failed: {str(e2)}")
                        try:
                            temp_model = create_lstm_model(MAX_SEQUENCE_LENGTH, len(FEATURES_TO_INCLUDE))
                            temp_model.load_weights(model_path)
                            models[hours] = temp_model
                            print(f"Loaded weights only for {hours}-hour prediction model")
                            found_models = True
                        except Exception as e3:
                            print(f"All loading attempts failed for {hours}-hour model: {str(e3)}")
            except Exception as e:
                print(f"Error loading model for {hours}-hour prediction: {str(e)}")
        else:
            print(f"Model file for {hours}-hour prediction not found")

    if not found_models:
        print("No models could be loaded successfully")
        return None, None

    return models, scalers


def prepare_prediction_sequences(patient_df, sequence_length=MAX_SEQUENCE_LENGTH):
    """
    Prepare sequences for prediction from patient data with variable input length support.

    Handles input data with fewer rows than required sequence length by padding
    with duplicated data while maintaining chronological order.

    :param patient_df: Patient data with glucose readings
    :type patient_df: pandas.DataFrame
    :param sequence_length: Number of time points in each sequence
    :type sequence_length: int
    :returns: Sequences and corresponding timestamps
    :rtype: tuple(numpy.ndarray, list)
    """
    df = extract_patient_features(patient_df, is_training=False)

    if df is None:
        print("Error: Failed to extract features from patient data")
        return None, None

    print(f"Input data has {len(df)} rows after preprocessing")

    if len(df) < sequence_length:
        print(f"Warning: Input has fewer than {sequence_length} readings (minimum expected).")
        print(f"Will use padding to reach required sequence length.")

        padding_needed = sequence_length - len(df)
        first_row = df.iloc[0:1]

        padding = pd.concat([first_row] * padding_needed, ignore_index=True)

        df = pd.concat([padding, df], ignore_index=True)

        time_col = df.attrs.get('feature_mapping', {}).get('time', 'time')
        if time_col in df.columns:
            time_diff = df[time_col].diff().median().total_seconds()
            if pd.isna(time_diff) or time_diff <= 0:
                time_diff = 300  # Default to 5 minutes

            first_real_time = df.iloc[padding_needed][time_col]
            for i in range(padding_needed):
                df.loc[i, time_col] = first_real_time - timedelta(seconds=time_diff * (padding_needed - i))

        print(f"Added {padding_needed} padding rows to reach required sequence length of {sequence_length}")

    time_col = df.attrs.get('feature_mapping', {}).get('time', 'time')

    feature_cols = FEATURES_TO_INCLUDE

    features = []
    for col in feature_cols:
        if col in df.columns:
            features.append(df[col].values)
        else:
            print(f"Feature {col} not available, using zeros")
            features.append(np.zeros(len(df)))

    feature_array = np.column_stack(features)

    sequences = []
    timestamps = []

    if len(df) == sequence_length:
        sequences.append(feature_array)
        timestamps.append(df[time_col].iloc[-1])
    else:
        for i in range(len(feature_array) - sequence_length + 1):
            sequence = feature_array[i:i + sequence_length]
            sequences.append(sequence)

            timestamps.append(df[time_col].iloc[i + sequence_length - 1])

    return np.array(sequences), timestamps


def predict_glucose_with_lstm(patient_df, models, scalers,
                              sequence_length=MAX_SEQUENCE_LENGTH):
    """
    Generate glucose predictions with LSTM models.

    Creates predictions for multiple time horizons with confidence
    intervals for each prediction.

    :param patient_df: Patient data with glucose readings
    :type patient_df: pandas.DataFrame
    :param models: Trained LSTM models for each prediction horizon
    :type models: dict
    :param scalers: Fitted scalers for features and targets
    :type scalers: dict
    :param sequence_length: Length of input sequences
    :type sequence_length: int
    :returns: DataFrame with predictions and timestamps
    :rtype: pandas.DataFrame
    """
    sequences, timestamps = prepare_prediction_sequences(patient_df, sequence_length)
    if sequences is None or timestamps is None:
        return pd.DataFrame()

    sequences_reshaped = sequences.reshape(-1, sequences.shape[2])
    sequences_scaled_2d = scalers['X'].transform(sequences_reshaped)
    sequences_scaled = sequences_scaled_2d.reshape(sequences.shape)

    predictions_df = pd.DataFrame({'timestamp': timestamps})

    glucose_idx = FEATURES_TO_INCLUDE.index('glucose_level')
    current_glucose = sequences[:, -1, glucose_idx]
    predictions_df['current_glucose'] = scalers['X'].inverse_transform(
        sequences_scaled[:, -1, :])[:, glucose_idx]

    for hours in models.keys():
        if hours not in scalers['y']:
            print(f"Scaler for {hours}-hour prediction not found, skipping")
            continue

        print(f"Making {hours}-hour predictions...")

        y_pred_scaled = models[hours].predict(sequences_scaled)

        y_pred = scalers['y'][hours].inverse_transform(y_pred_scaled)

        pred_times = [t + timedelta(hours=hours) for t in timestamps]

        predictions_df[f'predicted_{hours}hr'] = y_pred
        predictions_df[f'prediction_time_{hours}hr'] = pred_times

        std_dev = np.std(y_pred) * 1.96  # ~95% confidence
        predictions_df[f'lower_bound_{hours}hr'] = y_pred - std_dev
        predictions_df[f'upper_bound_{hours}hr'] = y_pred + std_dev

    return predictions_df


def plot_lstm_predictions(patient_df, predictions_df, save_png=False):
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
    hours_list = PREDICTION_HORIZONS

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

    if save_png:
        plt.savefig('lstm_glucose_predictions.png', dpi=300, bbox_inches='tight')
        print("Plot saved as lstm_glucose_predictions.png")

    plt.show()


def plot_training_history(results, model_dir=MODEL_DIRECTORY):
    """
    Plot training history for each model.

    Creates a two-panel figure showing loss and MAE during training
    for each prediction horizon model.

    :param results: Training results for each model
    :type results: dict
    :param model_dir: Directory to save the plot
    :type model_dir: str
    """
    plt.figure(figsize=(16, 8))

    for i, hours in enumerate(results.keys()):
        history = results[hours]['history']

        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label=f'{hours}-hour train')
        plt.plot(history['val_loss'], label=f'{hours}-hour validation')

        plt.subplot(1, 2, 2)
        plt.plot(history['mae'], label=f'{hours}-hour train')
        plt.plot(history['val_mae'], label=f'{hours}-hour validation')

    plt.subplot(1, 2, 1)
    plt.title('Model Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_history.png'), dpi=300)
    plt.show()


def compare_prediction_horizons(results, model_dir=MODEL_DIRECTORY):
    """
    Compare metrics across prediction horizons.

    Creates a three-panel figure comparing RMSE, MAE, and R² scores
    across different prediction horizons.

    :param results: Evaluation results for each prediction horizon
    :type results: dict
    :param model_dir: Directory to save the plot
    :type model_dir: str
    """
    hours = list(results.keys())
    rmse_values = [results[h]['rmse'] for h in hours]
    mae_values = [results[h]['mae'] for h in hours]
    r2_values = [results[h]['r2'] for h in hours]

    plt.figure(figsize=(12, 8))

    plt.subplot(1, 3, 1)
    plt.bar(hours, rmse_values)
    plt.title('RMSE by Prediction Horizon')
    plt.xlabel('Hours')
    plt.ylabel('RMSE (mg/dL)')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.bar(hours, mae_values)
    plt.title('MAE by Prediction Horizon')
    plt.xlabel('Hours')
    plt.ylabel('MAE (mg/dL)')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.bar(hours, r2_values)
    plt.title('R² Score by Prediction Horizon')
    plt.xlabel('Hours')
    plt.ylabel('R²')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'prediction_horizons.png'), dpi=300)
    plt.show()


def process_patient_data(patient_file, model_dir=MODEL_DIRECTORY, output_file=OUTPUT_FILE, sequence_length=None):
    """
    End-to-end process to load patient data, generate predictions, and export results.

    Handles adaptive sequence length selection based on input data size.

    :param patient_file: Path to CSV file with patient data
    :type patient_file: str
    :param model_dir: Directory containing trained models
    :type model_dir: str
    :param output_file: Path for saving prediction output
    :type output_file: str
    :param sequence_length: Override for sequence length, useful for very short inputs
    :type sequence_length: int
    :returns: DataFrame with predictions
    :rtype: pandas.DataFrame
    """
    print(f"Loading patient data from {patient_file}...")
    patient_df = load_patient_data(patient_file)

    row_count = len(patient_df)
    print(f"Input CSV has {row_count} rows")

    if sequence_length is None:
        if row_count < 5:
            sequence_length = max(2, row_count)
            print(f"Using reduced sequence length of {sequence_length} for short input")
        else:
            sequence_length = min(MAX_SEQUENCE_LENGTH, row_count)
            print(f"Using sequence length of {sequence_length}")
    else:
        print(f"Using user-specified sequence length of {sequence_length}")

    print("Loading LSTM models...")
    models, scalers = load_lstm_models(model_dir)

    if not models or not scalers:
        print("Error: No models or scalers found. Please train models first.")
        return None

    print("Making predictions...")
    sequences, timestamps = prepare_prediction_sequences(patient_df, sequence_length=sequence_length)

    if sequences is None or timestamps is None:
        print("Error: Could not prepare sequences for prediction")
        return None

    predictions_df = predict_from_sequences(sequences, timestamps, models, scalers)

    if len(predictions_df) == 0:
        print("Error: Could not generate any predictions")
        return None

    predictions_df.to_csv(output_file, index=False)
    print(f"Predictions exported to {output_file}")

    print("Plotting results...")
    try:
        plot_lstm_predictions(patient_df, predictions_df)
    except Exception as e:
        print(f"Error plotting predictions: {str(e)}")
        print("Continuing without visualization...")

    return predictions_df


def predict_from_sequences(sequences, timestamps, models, scalers):
    """
    Generate predictions from preprocessed sequences.

    Creates predictions and confidence intervals that adapt to the amount
    of input data and prediction horizon.

    :param sequences: Input feature sequences
    :type sequences: numpy.ndarray
    :param timestamps: Corresponding timestamps for each sequence
    :type timestamps: list
    :param models: Trained models for each prediction horizon
    :type models: dict
    :param scalers: Fitted scalers for features and targets
    :type scalers: dict
    :returns: DataFrame with predictions and confidence intervals
    :rtype: pandas.DataFrame
    """
    if len(sequences) == 0:
        print("Error: No sequences to predict from")
        return pd.DataFrame()

    sequences_reshaped = sequences.reshape(-1, sequences.shape[2])
    sequences_scaled_2d = scalers['X'].transform(sequences_reshaped)
    sequences_scaled = sequences_scaled_2d.reshape(sequences.shape)

    predictions_df = pd.DataFrame({'timestamp': timestamps})

    glucose_idx = FEATURES_TO_INCLUDE.index('glucose_level')
    current_glucose = sequences[:, -1, glucose_idx]
    predictions_df['current_glucose'] = scalers['X'].inverse_transform(
        sequences_scaled[:, -1, :])[:, glucose_idx]

    for hours in models.keys():
        if hours not in scalers['y']:
            print(f"Scaler for {hours}-hour prediction not found, skipping")
            continue

        print(f"Making {hours}-hour predictions...")

        y_pred_scaled = models[hours].predict(sequences_scaled)

        y_pred = scalers['y'][hours].inverse_transform(y_pred_scaled)

        pred_times = [t + timedelta(hours=hours) for t in timestamps]

        predictions_df[f'predicted_{hours}hr'] = y_pred
        predictions_df[f'prediction_time_{hours}hr'] = pred_times

        base_uncertainty = {
            1: 15.0,  # 1-hour prediction: ±15 mg/dL
            2: 25.0,  # 2-hour prediction: ±25 mg/dL
            3: 35.0,  # 3-hour prediction: ±35 mg/dL
        }.get(hours, 20.0)

        mean_glucose = predictions_df['current_glucose'].mean()

        percentage_factor = 0.05 + (0.01 * hours)
        glucose_based_uncertainty = mean_glucose * percentage_factor

        uncertainty = base_uncertainty + glucose_based_uncertainty

        if len(y_pred) > 1:
            std_dev = np.std(y_pred)
            if std_dev > 5.0:
                uncertainty = (uncertainty + std_dev) / 2

        predictions_df[f'lower_bound_{hours}hr'] = y_pred - uncertainty
        predictions_df[f'upper_bound_{hours}hr'] = y_pred + uncertainty

        predictions_df[f'lower_bound_{hours}hr'] = predictions_df[f'lower_bound_{hours}hr'].clip(lower=40)

        print(f"  {hours}-hour predictions complete with uncertainty of ±{uncertainty:.1f} mg/dL")

    return predictions_df


def train_models_workflow(training_file, model_dir=MODEL_DIRECTORY):
    """
    Workflow for training LSTM models.

    Loads training data, extracts features for each patient,
    prepares sequences, trains models, and evaluates results.

    :param training_file: Path to CSV file with training data
    :type training_file: str
    :param model_dir: Directory to save trained models
    :type model_dir: str
    :returns: Trained models, scalers, and evaluation results
    :rtype: tuple(dict, dict, dict)
    """
    print(f"Loading training data from {training_file}...")
    df = load_and_preprocess_training_data(training_file)

    col_mapping = df.attrs.get('column_mapping', {})

    subject_id_col = col_mapping.get('SUBJECT_ID', 'subject_id')
    if subject_id_col not in df.columns:
        subject_id_col = 'subject_id'
        if subject_id_col not in df.columns:
            print(f"Error: Could not find subject ID column. Available columns: {df.columns.tolist()}")
            return None, None, None

    print(f"Processing {df[subject_id_col].nunique()} patients...")
    patient_dfs = []

    subject_ids = df[subject_id_col].unique()
    for subject_id in tqdm(subject_ids, desc="Extracting patient features"):
        patient_data = df[df[subject_id_col] == subject_id]
        patient_features = extract_patient_features(patient_data, is_training=True)
        if patient_features is not None and len(patient_features) >= MAX_SEQUENCE_LENGTH:
            patient_dfs.append(patient_features)

    print(f"Prepared data for {len(patient_dfs)} patients")

    X, y = prepare_lstm_dataset(patient_dfs, sequence_length=MAX_SEQUENCE_LENGTH)

    models, scalers, results = train_lstm_models(X, y, model_dir=model_dir)

    plot_training_history(results, model_dir)
    compare_prediction_horizons(results, model_dir)

    print(f"Models saved to {model_dir} directory")
    return models, scalers, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTM Glucose Prediction System')
    parser.add_argument('--train', action='store_true', help='Train new models')
    parser.add_argument('--predict', action='store_true', help='Make predictions on patient data')
    parser.add_argument('--training-file', type=str, help='CSV file with training data')
    parser.add_argument('--patient-file', type=str, help='CSV file with patient data (time and glucose_level columns)')
    parser.add_argument('--model-dir', type=str, default=MODEL_DIRECTORY, help='Directory for model storage')
    parser.add_argument('--output-file', type=str, default=OUTPUT_FILE, help='Output file for predictions')
    parser.add_argument('--sequence-length', type=int,
                        help='Number of past readings to use for prediction (default: adaptive)')

    args = parser.parse_args()

    if args.train and args.predict:
        print("Error: Cannot both train and predict at the same time. Please choose one operation.")
        exit(1)

    if not args.train and not args.predict:
        print("Error: Must specify either --train or --predict")
        parser.print_help()
        exit(1)

    if args.train:
        if not args.training_file:
            print("Error: Training file must be specified with --training-file")
            exit(1)

        print(f"Starting LSTM model training using data from {args.training_file}")
        train_models_workflow(args.training_file, args.model_dir)
        print("Training complete!")

    if args.predict:
        if not args.patient_file:
            print("Error: Patient file must be specified with --patient-file")
            exit(1)

        print(f"Making predictions for patient data in {args.patient_file}")
        predictions = process_patient_data(
            args.patient_file,
            args.model_dir,
            args.output_file,
            sequence_length=args.sequence_length
        )

        if predictions is not None:
            print(f"Predictions saved to {args.output_file}")
            print("Prediction complete!")
        else:
            print("Prediction failed. Please check the error messages above.")
