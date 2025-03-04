import hashlib
import os
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

OUTPUT_FILE = "glucose_predictions.csv"


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
    if df is None:
        return None

    # Make a copy to avoid modifying the original
    df_cleaned = df.copy()

    inf_count = np.isinf(df_cleaned.select_dtypes(include=[np.number])).sum().sum()

    if inf_count > 0:
        print(f"Warning: Found {inf_count} infinity values in the dataset. Replacing with NaN.")
        df_cleaned = df_cleaned.replace([np.inf, -np.inf], np.nan)

    for col in df_cleaned.select_dtypes(include=[np.number]).columns:
        extreme_mask = (df_cleaned[col].abs() > 1e10) & df_cleaned[col].notna()
        extreme_count = extreme_mask.sum()

        if extreme_count > 0:
            print(f"Warning: Found {extreme_count} extreme values in column '{col}'. Capping to reasonable values.")
            df_cleaned.loc[extreme_mask, col] = df_cleaned.loc[extreme_mask, col].apply(
                lambda x: 1e10 if x > 0 else -1e10
            )

    return df_cleaned


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


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a trained model and print performance metrics.
    Works with both XGBoost and LSTM models.

    :param model: Trained model
    :param X_test: Test features
    :type X_test: pandas.DataFrame or numpy.ndarray
    :param y_test: True target values
    :type y_test: pandas.Series or numpy.ndarray
    :param model_name: Name to display in output
    :type model_name: str
    :returns: Dict with evaluation metrics
    :rtype: dict
    """
    y_pred = model.predict(X_test)

    # Reshape if needed
    if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()

    if isinstance(y_test, pd.Series):
        y_test = y_test.values

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Calculate MAPE safely (avoiding division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, np.nan))) * 100

    errors = y_test - y_pred
    mean_error = np.mean(errors)
    std_error = np.std(errors)

    print(f"\n{model_name} - Evaluation Metrics:")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R²): {r2:.3f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"Mean Error: {mean_error:.2f}")
    print(f"Standard Deviation of Error: {std_error:.2f}")

    if hasattr(model, 'feature_importances_'):
        # For tree-based models like XGBoost
        feature_importance = None

        if isinstance(X_test, pd.DataFrame):
            feature_cols = list(X_test.columns)
            importance = model.feature_importances_

            if len(feature_cols) == len(importance):
                feature_importance = dict(zip(feature_cols, importance))
                sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

                print("\nTop 10 Feature Importance:")
                for feature, imp in sorted_importance[:10]:
                    print(f"{feature}: {imp:.4f}")
            else:
                print(
                    f"Warning: Feature columns length ({len(feature_cols)}) doesn't match importance length ({len(importance)})")
        else:
            # For numpy arrays or other non-DataFrame inputs
            importance = model.feature_importances_
            print("\nFeature Importance Values (feature indices):")
            sorted_idx = np.argsort(importance)[::-1]
            for i in sorted_idx[:10]:  # Top 10
                print(f"Feature {i}: {importance[i]:.4f}")

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'mean_error': mean_error,
        'std_error': std_error
    }


def create_output_dir(directory):
    """
    Create directory if it doesn't exist

    :param directory: Directory path to create
    :type directory: str
    """
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def format_time_difference(td):
    """
    Format timedelta to a human-readable string

    :param td: Time difference
    :type td: datetime.timedelta
    :returns: Formatted time string
    :rtype: str
    """
    seconds = td.total_seconds()
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} minutes"
    else:
        return f"{seconds / 3600:.2f} hours"
