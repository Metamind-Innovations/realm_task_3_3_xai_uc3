import os
import re

import pandas as pd


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
        'hadm_id': 'HADM_ID',
        'icustay_id': 'ICUSTAY_ID',
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

    numeric_columns = ['INPUT', 'INPUT_HRS', 'GLC', 'INFXSTOP', 'HADM_ID', 'ICUSTAY_ID']
    for col in numeric_columns:
        if col in col_mapping and col_mapping[col] in df.columns:
            df[col_mapping[col]] = pd.to_numeric(df[col_mapping[col]], errors='coerce')

    categorical_columns = ['GENDER', 'EVENT', 'GLCSOURCE', 'INSULINTYPE']
    for col in categorical_columns:
        if col in col_mapping and col_mapping[col] in df.columns:
            df[col_mapping[col]] = df[col_mapping[col]].astype('category')

    if 'FIRST_ICU_STAY' in col_mapping and col_mapping['FIRST_ICU_STAY'] in df.columns:
        df[col_mapping['FIRST_ICU_STAY']] = df[col_mapping['FIRST_ICU_STAY']].astype(bool)

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

    demographics = {}
    filename = os.path.basename(file_path)

    # Try to extract age and gender from filename
    age_gender_match = re.match(r'(\d+)_(male|female)', filename.lower())
    if age_gender_match:
        age_str, gender_str = age_gender_match.groups()
        try:
            age = int(age_str)
            demographics['age'] = age
            print(f"Extracted age {age} from filename")
        except ValueError:
            pass

        demographics['gender'] = gender_str
        print(f"Extracted gender '{gender_str}' from filename")

    df.attrs['demographics_from_filename'] = demographics

    datetime_formats = ['%d/%m/%Y %H:%M:%S', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S']
    for dt_format in datetime_formats:
        try:
            df[time_col] = pd.to_datetime(df[time_col], format=dt_format)
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


def get_column_name(df, expected_col):
    """
    Get the actual column name from a dataframe based on expected column name.
    Handles case insensitivity and column mapping stored in dataframe attributes.

    :param df: Dataframe to search for column
    :type df: pandas.DataFrame
    :param expected_col: Expected column name
    :type expected_col: str
    :returns: Actual column name if found, None otherwise
    :rtype: str or None
    """
    col_mapping = df.attrs.get('column_mapping', {})

    if expected_col in col_mapping and col_mapping[expected_col] in df.columns:
        return col_mapping[expected_col]
    elif expected_col in df.columns:
        return expected_col
    elif expected_col.lower() in df.columns:
        return expected_col.lower()
    return None


def export_predictions(predictions_df, output_file):
    """
    Export prediction results to CSV file

    :param predictions_df: DataFrame containing predictions
    :type predictions_df: pandas.DataFrame
    :param output_file: Path to save the predictions
    :type output_file: str
    """

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    predictions_df.to_csv(output_file, index=False)
    print(f"Predictions exported to {output_file}")
