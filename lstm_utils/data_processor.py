from datetime import timedelta

import numpy as np
import pandas as pd

from common.data_loader import get_column_name
from common.preprocessing import clean_infinite_values, encode_icd9_code

# Constants
MAX_SEQUENCE_LENGTH = 10
PREDICTION_HORIZONS = [1, 2, 3]  # Hours into the future
FEATURES_TO_INCLUDE = [
    'glucose_level', 'glucose_std', 'glucose_rate', 'glucose_mean',
    'glucose_min', 'glucose_max', 'glucose_range', 'glucose_acceleration',
    'hour_of_day', 'is_daytime', 'day_of_week', 'is_weekend'
]


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
    df = patient_df.copy()

    # Get essential column names
    time_col = get_column_name(df, 'TIMER') or get_column_name(df, 'timer') or get_column_name(df,
                                                                                               'time') or get_column_name(
        df, 'TIME')
    glucose_col = get_column_name(df, 'GLC') or get_column_name(df, 'glc') or get_column_name(df,
                                                                                              'glucose_level') or get_column_name(
        df, 'GLUCOSE_LEVEL') or get_column_name(df, 'glucose')

    if not time_col:
        print(f"Error: Time column not found in available columns: {df.columns.tolist()}")
        return None
    if not glucose_col:
        print(f"Error: Glucose column not found in available columns: {df.columns.tolist()}")
        return None

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

    filename_demographics = patient_df.attrs.get('demographics_from_filename', {})

    gender_col = get_column_name(df, 'GENDER')
    if gender_col and len(df) > 0:
        gender = df[gender_col].iloc[0]
        demographics['gender'] = 1 if gender == 'M' else 0 if gender == 'F' else 0.5
    elif 'gender' in filename_demographics:
        # Use gender from filename
        gender_str = filename_demographics['gender']
        demographics['gender'] = 1 if gender_str.lower() == 'male' else 0 if gender_str.lower() == 'female' else 0.5
        print(f"Using gender '{gender_str}' from filename for model input (value: {demographics['gender']})")
    else:
        demographics['gender'] = 0.5  # Default

    age_col = get_column_name(df, 'ADMISSION_AGE')
    if age_col and len(df) > 0:
        age = df[age_col].iloc[0]
        if not pd.isna(age) and isinstance(age, (int, float)):
            demographics['age'] = min(max(age, 0), 120) / 120  # Normalize 0-1
        else:
            demographics['age'] = 0.5  # Default
    elif 'age' in filename_demographics:
        # Use age from filename
        age = filename_demographics['age']
        demographics['age'] = min(max(age, 0), 120) / 120  # Normalize 0-1
        print(f"Using age {age} from filename for model input (normalized: {demographics['age']})")
    else:
        demographics['age'] = 0.5  # Default

    icd9_col = get_column_name(df, 'ICD9_CODE')
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
