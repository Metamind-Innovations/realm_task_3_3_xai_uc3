import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib
from datetime import timedelta
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from common_utils import clean_infinite_values, encode_icd9_code, get_column_name


def create_enhanced_patient_features(patient_df):
    """
    Extract features from patient data for XGBoost model.

    :param patient_df: Patient dataframe with glucose readings
    :type patient_df: pandas.DataFrame
    :returns: Expanded DataFrame with calculated features
    :rtype: pandas.DataFrame
    """
    df = patient_df.copy()

    # Get essential column names
    glc_col = get_column_name(df, 'GLC')
    timer_col = get_column_name(df, 'TIMER')
    hadm_id_col = get_column_name(df, 'HADM_ID')
    icustay_id_col = get_column_name(df, 'ICUSTAY_ID')

    if not glc_col or not timer_col:
        print(f"Error: Required columns not found. GLC: {glc_col}, TIMER: {timer_col}")
        return None

    features_df = pd.DataFrame()
    glucose_df = df[df[glc_col].notna()].copy()

    if len(glucose_df) <= 1:
        return None

    print(f"Processing {len(glucose_df) - 1} glucose readings for feature extraction")

    # Add time of day features
    glucose_df['hour_of_day'] = glucose_df[timer_col].dt.hour
    glucose_df['is_daytime'] = ((glucose_df['hour_of_day'] >= 7) & (glucose_df['hour_of_day'] <= 22)).astype(int)
    glucose_df['day_of_week'] = glucose_df[timer_col].dt.dayofweek
    glucose_df['is_weekend'] = glucose_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # Calculate glucose variability metrics
    if len(glucose_df) > 3:
        glucose_df['glucose_std'] = glucose_df[glc_col].rolling(window=3, min_periods=1).std()
        glucose_df['glucose_mean'] = glucose_df[glc_col].rolling(window=3, min_periods=1).mean()
        glucose_df['glucose_min'] = glucose_df[glc_col].rolling(window=6, min_periods=1).min()
        glucose_df['glucose_max'] = glucose_df[glc_col].rolling(window=6, min_periods=1).max()
        glucose_df['glucose_range'] = glucose_df['glucose_max'] - glucose_df['glucose_min']

        # Calculate glucose rate of change
        glucose_df['time_diff'] = glucose_df[timer_col].diff().dt.total_seconds()
        glucose_df['time_diff'] = glucose_df['time_diff'].replace(0, np.nan)
        glucose_df.loc[glucose_df['time_diff'] < 1, 'time_diff'] = np.nan

        glucose_df['glucose_diff'] = glucose_df[glc_col].diff()
        glucose_df['glucose_rate'] = glucose_df['glucose_diff'] / glucose_df['time_diff'] * 60  # per minute

        # Add acceleration
        glucose_df['glucose_rate_diff'] = glucose_df['glucose_rate'].diff()
        glucose_df['glucose_acceleration'] = glucose_df['glucose_rate_diff'] / glucose_df['time_diff'] * 60

        # Cap extreme rates
        glucose_df.loc[glucose_df['glucose_rate'] > 10, 'glucose_rate'] = 10
        glucose_df.loc[glucose_df['glucose_rate'] < -10, 'glucose_rate'] = -10
        glucose_df.loc[glucose_df['glucose_acceleration'] > 2, 'glucose_acceleration'] = 2
        glucose_df.loc[glucose_df['glucose_acceleration'] < -2, 'glucose_acceleration'] = -2

    # Process patient demographic data
    first_row = df.iloc[0]

    # Handle gender
    gender_feature = 0  # Default
    gender_col = get_column_name(df, 'GENDER')
    if gender_col and len(df) > 0:
        gender = first_row.get(gender_col, 'Unknown')
        gender_feature = 1 if gender == 'M' else 0 if gender == 'F' else 0.5

    # Handle age
    age_feature = 50  # Default middle age
    age_col = get_column_name(df, 'ADMISSION_AGE')
    if age_col and len(df) > 0:
        age = first_row.get(age_col)
        if not pd.isna(age) and isinstance(age, (int, float)):
            age_feature = min(max(age, 0), 120) / 120  # Normalize 0-1
        else:
            age_feature = 0.5  # Default

    # Handle ICD9 code
    icd9_category = 0
    icd9_subcategory = 0
    icd9_col = get_column_name(df, 'ICD9_CODE')
    if icd9_col and len(df) > 0:
        icd9_code = first_row.get(icd9_col)
        if not pd.isna(icd9_code) and isinstance(icd9_code, str):
            icd9_category, icd9_subcategory = encode_icd9_code(icd9_code)

    # Add hospital/ICU IDs as features if available
    hadm_id_feature = 0
    icustay_id_feature = 0
    if hadm_id_col:
        hadm_id = first_row.get(hadm_id_col, 0)
        if not pd.isna(hadm_id):
            hadm_id_feature = int(hadm_id) % 1000  # Use modulo to create a manageable feature

    if icustay_id_col:
        icustay_id = first_row.get(icustay_id_col, 0)
        if not pd.isna(icustay_id):
            icustay_id_feature = int(icustay_id) % 1000

    # Get column names for insulin analysis
    event_col = get_column_name(df, 'EVENT')
    input_col = get_column_name(df, 'INPUT')
    insulintype_col = get_column_name(df, 'INSULINTYPE')

    if not event_col or not input_col or not insulintype_col:
        event_col = event_col or 'event'
        input_col = input_col or 'input'
        insulintype_col = insulintype_col or 'insulintype'

    # Process each glucose reading
    for i in tqdm(range(len(glucose_df) - 1), desc="Creating features"):
        current_time = glucose_df.iloc[i][timer_col]
        current_glucose = glucose_df.iloc[i][glc_col]

        # Define future time windows for prediction targets
        future_cutoff_1hr = current_time + timedelta(hours=1)
        future_cutoff_2hr = current_time + timedelta(hours=2)
        future_cutoff_3hr = current_time + timedelta(hours=3)

        # Calculate mean glucose values in each future window
        future_1hr = glucose_df[(glucose_df[timer_col] > current_time) &
                                (glucose_df[timer_col] <= future_cutoff_1hr)][glc_col].mean()
        future_2hr = glucose_df[(glucose_df[timer_col] > future_cutoff_1hr) &
                                (glucose_df[timer_col] <= future_cutoff_2hr)][glc_col].mean()
        future_3hr = glucose_df[(glucose_df[timer_col] > future_cutoff_2hr) &
                                (glucose_df[timer_col] <= future_cutoff_3hr)][glc_col].mean()

        # Extract historical glucose values (up to 5 previous readings)
        past_readings = glucose_df[glucose_df[timer_col] < current_time].tail(5)
        past_glucose_values = past_readings[glc_col].tolist()

        # Pad with NaN if insufficient history
        while len(past_glucose_values) < 5:
            past_glucose_values.insert(0, np.nan)

        # Get glucose variability metrics with safe defaults
        current_std = glucose_df.iloc[i].get('glucose_std', 0)
        current_rate = glucose_df.iloc[i].get('glucose_rate', 0)
        current_mean = glucose_df.iloc[i].get('glucose_mean', current_glucose)
        current_min = glucose_df.iloc[i].get('glucose_min', current_glucose)
        current_max = glucose_df.iloc[i].get('glucose_max', current_glucose)
        current_range = glucose_df.iloc[i].get('glucose_range', 0)
        current_acceleration = glucose_df.iloc[i].get('glucose_acceleration', 0)

        # Handle NaN values safely
        if pd.isna(current_std): current_std = 0
        if pd.isna(current_rate): current_rate = 0
        if pd.isna(current_mean): current_mean = current_glucose
        if pd.isna(current_min): current_min = current_glucose
        if pd.isna(current_max): current_max = current_glucose
        if pd.isna(current_range): current_range = 0
        if pd.isna(current_acceleration): current_acceleration = 0

        # Enhanced insulin analysis with time windows
        recent_insulin_events = []
        for window_hours in [1, 2, 3, 6, 12, 24]:
            window_start = current_time - timedelta(hours=window_hours)

            insulin_window = df[(df[timer_col] >= window_start) &
                                (df[timer_col] <= current_time)]

            if event_col in df.columns:
                bolus_variants = ['BOLUS_INYECTION', 'BOLUS_INJECTION', 'bolus_inyection', 'bolus_injection']
                insulin_window = insulin_window[insulin_window[event_col].isin(bolus_variants)]

            # Calculate total insulin by type for the window
            short_insulin = 0
            long_insulin = 0

            if insulintype_col in insulin_window.columns and input_col in insulin_window.columns:
                short_variants = ['Short', 'SHORT', 'short']
                long_variants = ['Long', 'LONG', 'long']

                short_insulin = insulin_window[insulin_window[insulintype_col].isin(short_variants)][input_col].sum()
                long_insulin = insulin_window[insulin_window[insulintype_col].isin(long_variants)][input_col].sum()

            total_insulin = short_insulin + long_insulin

            recent_insulin_events.append({
                'window': window_hours,
                'short_insulin': short_insulin,
                'long_insulin': long_insulin,
                'total_insulin': total_insulin,
                'insulin_rate': total_insulin / window_hours if window_hours > 0 else 0
            })

        # Calculate time since most recent insulin dose
        last_insulin = pd.DataFrame()
        if event_col in df.columns and timer_col in df.columns:
            bolus_variants = ['BOLUS_INYECTION', 'BOLUS_INJECTION', 'bolus_inyection', 'bolus_injection']
            last_insulin = df[(df[timer_col] < current_time) &
                              (df[event_col].isin(bolus_variants))].tail(1)

        time_since_insulin = 24  # Default to 24h if no insulin recorded
        last_insulin_amount = 0
        last_insulin_type_short = 0

        if len(last_insulin) > 0:
            time_diff_seconds = (current_time - last_insulin[timer_col].iloc[0]).total_seconds()
            if time_diff_seconds <= 0:
                time_since_insulin = 0.001  # Small positive value instead of 0
            else:
                time_since_insulin = time_diff_seconds / 3600  # in hours

            # Record amount and type of last insulin
            if input_col in last_insulin.columns:
                last_insulin_amount = last_insulin[input_col].iloc[0]

            if insulintype_col in last_insulin.columns:
                short_variants = ['Short', 'SHORT', 'short']
                last_insulin_type_short = 1 if last_insulin[insulintype_col].iloc[0] in short_variants else 0

        # Cap to reasonable value (maximum 48 hours)
        time_since_insulin = min(time_since_insulin, 48)

        # Get patient metadata with safety checks
        los_icu = None
        first_stay = None
        if 'LOS_ICU_days' in df.columns and 'first_ICU_stay' in df.columns:
            los_icu_raw = df['LOS_ICU_days'].iloc[0]
            if pd.isna(los_icu_raw):
                los_icu = 0
            else:
                los_icu = min(float(los_icu_raw), 365)
            first_stay = 1 if df['first_ICU_stay'].iloc[0] else 0
        else:
            los_icu = 0
            first_stay = 0

        # Get time of day information
        hour = glucose_df.iloc[i]['hour_of_day']
        is_daytime = glucose_df.iloc[i]['is_daytime']
        day_of_week = glucose_df.iloc[i].get('day_of_week', 0)
        is_weekend = glucose_df.iloc[i].get('is_weekend', 0)

        # Get additional column names
        subject_id_col = get_column_name(df, 'SUBJECT_ID')
        glcsource_col = get_column_name(df, 'GLCSOURCE')

        # Assemble enhanced feature row
        row = {
            'SUBJECT_ID': patient_df[subject_id_col].iloc[0] if subject_id_col else 0,
            'HADM_ID_FEATURE': hadm_id_feature,
            'ICUSTAY_ID_FEATURE': icustay_id_feature,
            'current_glucose': current_glucose,
            'glucose_source': 1 if glcsource_col and glucose_df.iloc[i][glcsource_col] == 'BLOOD' else 0,
            'prev_glucose_1': past_glucose_values[-1] if not np.isnan(past_glucose_values[-1]) else current_glucose,
            'prev_glucose_2': past_glucose_values[-2] if len(past_glucose_values) > 1 and not np.isnan(
                past_glucose_values[-2]) else current_glucose,
            'prev_glucose_3': past_glucose_values[-3] if len(past_glucose_values) > 2 and not np.isnan(
                past_glucose_values[-3]) else current_glucose,
            'prev_glucose_4': past_glucose_values[-4] if len(past_glucose_values) > 3 and not np.isnan(
                past_glucose_values[-4]) else current_glucose,
            'prev_glucose_5': past_glucose_values[-5] if len(past_glucose_values) > 4 and not np.isnan(
                past_glucose_values[-5]) else current_glucose,

            # Original insulin features
            'short_insulin_dose': recent_insulin_events[5]['short_insulin'],  # 24-hour window
            'long_insulin_dose': recent_insulin_events[5]['long_insulin'],  # 24-hour window

            # Enhanced insulin features with multiple time windows
            'short_insulin_1h': recent_insulin_events[0]['short_insulin'],
            'short_insulin_3h': recent_insulin_events[2]['short_insulin'],
            'short_insulin_6h': recent_insulin_events[3]['short_insulin'],
            'short_insulin_12h': recent_insulin_events[4]['short_insulin'],

            'long_insulin_1h': recent_insulin_events[0]['long_insulin'],
            'long_insulin_3h': recent_insulin_events[2]['long_insulin'],
            'long_insulin_6h': recent_insulin_events[3]['long_insulin'],
            'long_insulin_12h': recent_insulin_events[4]['long_insulin'],

            'total_insulin_1h': recent_insulin_events[0]['total_insulin'],
            'total_insulin_3h': recent_insulin_events[2]['total_insulin'],
            'total_insulin_6h': recent_insulin_events[3]['total_insulin'],
            'total_insulin_12h': recent_insulin_events[4]['total_insulin'],

            'insulin_rate_1h': recent_insulin_events[0]['insulin_rate'],
            'insulin_rate_3h': recent_insulin_events[2]['insulin_rate'],
            'insulin_rate_6h': recent_insulin_events[3]['insulin_rate'],
            'insulin_rate_12h': recent_insulin_events[4]['insulin_rate'],

            'time_since_insulin': time_since_insulin,
            'last_insulin_amount': last_insulin_amount,
            'last_insulin_type_short': last_insulin_type_short,

            # Enhanced glucose variability metrics
            'glucose_std': current_std,
            'glucose_rate': current_rate,
            'glucose_mean': current_mean,
            'glucose_min': current_min,
            'glucose_max': current_max,
            'glucose_range': current_range,
            'glucose_acceleration': current_acceleration,

            # Enhanced time of day features
            'hour_of_day': hour,
            'is_daytime': is_daytime,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,

            # Patient demographic features
            'gender': gender_feature,
            'age': age_feature,
            'icd9_category': icd9_category,
            'icd9_subcategory': icd9_subcategory,

            # Patient metadata
            'los_icu_days': los_icu,
            'first_icu_stay': first_stay,

            # Target values
            'future_glucose_1hr': future_1hr,
            'future_glucose_2hr': future_2hr,
            'future_glucose_3hr': future_3hr
        }

        # Get column for input hours
        input_hrs_col = get_column_name(df, 'INPUT_HRS')

        # If INPUT_HRS is available, add insulin duration features
        all_insulin_events = pd.DataFrame()
        if event_col in df.columns and timer_col in df.columns:
            bolus_variants = ['BOLUS_INYECTION', 'BOLUS_INJECTION', 'bolus_inyection', 'bolus_injection']
            all_insulin_events = df[(df[timer_col] < current_time) &
                                    (df[event_col].isin(bolus_variants))].tail(5)

        if input_hrs_col and len(all_insulin_events) > 0:
            # Get mean and latest insulin duration
            mean_insulin_duration = all_insulin_events[input_hrs_col].mean()
            latest_insulin_duration = all_insulin_events.iloc[-1][input_hrs_col]

            # Handle potential NaN
            if pd.isna(mean_insulin_duration): mean_insulin_duration = 0
            if pd.isna(latest_insulin_duration): latest_insulin_duration = 0

            # Cap to reasonable values
            mean_insulin_duration = min(mean_insulin_duration, 24)
            latest_insulin_duration = min(latest_insulin_duration, 24)

            row['mean_insulin_duration'] = mean_insulin_duration
            row['latest_insulin_duration'] = latest_insulin_duration

        # If INFXSTOP is available, add features related to infusion stopping
        infxstop_col = get_column_name(df, 'INFXSTOP')
        if infxstop_col:
            # Get recent infusion stop events
            recent_infxstop = df[(df[timer_col] < current_time) & (~pd.isna(df[infxstop_col]))].tail(3)

            if len(recent_infxstop) > 0:
                # Most recent event
                time_diff_seconds = (current_time - recent_infxstop.iloc[-1][timer_col]).total_seconds()
                if time_diff_seconds <= 0:
                    time_since_infxstop = 0.001
                else:
                    time_since_infxstop = time_diff_seconds / 3600

                # Cap to reasonable value
                time_since_infxstop = min(time_since_infxstop, 48)

                # Last infxstop values
                last_infxstop_value = recent_infxstop.iloc[-1][infxstop_col]
                # Mean of recent infxstop values
                mean_infxstop_value = recent_infxstop[infxstop_col].mean()

                # Handle potential extreme values
                if pd.isna(last_infxstop_value) or np.isinf(last_infxstop_value):
                    last_infxstop_value = 0
                if pd.isna(mean_infxstop_value) or np.isinf(mean_infxstop_value):
                    mean_infxstop_value = 0

                # Cap to reasonable range
                last_infxstop_value = max(min(last_infxstop_value, 1000), -1000)
                mean_infxstop_value = max(min(mean_infxstop_value, 1000), -1000)

                row['time_since_infxstop'] = time_since_infxstop
                row['last_infxstop_value'] = last_infxstop_value
                row['mean_infxstop_value'] = mean_infxstop_value
                row['infxstop_count_24h'] = len(df[(df[timer_col] >= current_time - timedelta(hours=24)) &
                                                   (df[timer_col] < current_time) &
                                                   (~pd.isna(df[infxstop_col]))])
            else:
                row['time_since_infxstop'] = 24
                row['last_infxstop_value'] = 0
                row['mean_infxstop_value'] = 0
                row['infxstop_count_24h'] = 0

        # Add new features: Time differences between various timestamps
        starttime_col = get_column_name(df, 'STARTTIME')
        endtime_col = get_column_name(df, 'ENDTIME')
        glctimer_col = get_column_name(df, 'GLCTIMER')

        if starttime_col and timer_col:
            start_time = glucose_df.iloc[i].get(starttime_col)
            if pd.notna(start_time):
                row['timer_starttime_diff'] = (current_time - start_time).total_seconds() / 60  # in minutes
            else:
                row['timer_starttime_diff'] = 0

        if endtime_col and timer_col:
            end_time = glucose_df.iloc[i].get(endtime_col)
            if pd.notna(end_time):
                row['timer_endtime_diff'] = (end_time - current_time).total_seconds() / 60  # in minutes
            else:
                row['timer_endtime_diff'] = 0

        if glctimer_col and timer_col:
            glc_time = glucose_df.iloc[i].get(glctimer_col)
            if pd.notna(glc_time):
                row['timer_glctimer_diff'] = (current_time - glc_time).total_seconds() / 60  # in minutes
            else:
                row['timer_glctimer_diff'] = 0

        features_df = pd.concat([features_df, pd.DataFrame([row])], ignore_index=True)

    features_df = clean_infinite_values(features_df)
    return features_df


def create_prediction_features(patient_df):
    """
    Create feature set for making predictions with an existing XGBoost model.

    :param patient_df: Patient dataframe with glucose readings
    :type patient_df: pandas.DataFrame
    :returns: Features dataframe ready for prediction
    :rtype: pandas.DataFrame
    """
    df = patient_df.copy()
    features_list = []

    if len(df) < 5:
        raise ValueError("Need at least 5 glucose readings to make predictions")

    col_mapping = df.attrs.get('column_mapping', {})

    time_col = col_mapping.get('time', 'time')
    glucose_col = col_mapping.get('glucose_level', 'glucose_level')

    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in dataframe")
    if glucose_col not in df.columns:
        raise ValueError(f"Glucose column '{glucose_col}' not found in dataframe")

    df = df.sort_values(by=time_col)
    df['glucose_std'] = df[glucose_col].rolling(window=3, min_periods=1).std().fillna(0)
    df['glucose_mean'] = df[glucose_col].rolling(window=3, min_periods=1).mean().fillna(df[glucose_col])
    df['glucose_min'] = df[glucose_col].rolling(window=6, min_periods=1).min().fillna(df[glucose_col])
    df['glucose_max'] = df[glucose_col].rolling(window=6, min_periods=1).max().fillna(df[glucose_col])
    df['glucose_range'] = df['glucose_max'] - df['glucose_min']

    df['time_diff'] = df[time_col].diff().dt.total_seconds()
    df.loc[df['time_diff'] <= 0, 'time_diff'] = 0.1
    df['glucose_diff'] = df[glucose_col].diff()
    df['glucose_rate'] = (df['glucose_diff'] / df['time_diff'] * 60).fillna(0)

    df['glucose_rate_diff'] = df['glucose_rate'].diff()
    df['glucose_acceleration'] = (df['glucose_rate_diff'] / df['time_diff'] * 60).fillna(0)

    df.loc[df['glucose_rate'] > 10, 'glucose_rate'] = 10
    df.loc[df['glucose_rate'] < -10, 'glucose_rate'] = -10
    df.loc[df['glucose_acceleration'] > 2, 'glucose_acceleration'] = 2
    df.loc[df['glucose_acceleration'] < -2, 'glucose_acceleration'] = -2

    df['hour_of_day'] = df[time_col].dt.hour
    df['is_daytime'] = ((df['hour_of_day'] >= 7) & (df['hour_of_day'] <= 22)).astype(int)
    df['day_of_week'] = df[time_col].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # Set default values for demographic features
    gender_feature = 0.5
    age_feature = 50
    icd9_category = 0
    icd9_subcategory = 0
    hadm_id_feature = 0
    icustay_id_feature = 0

    for i in range(len(df) - 4):
        sequence = df.iloc[i:i + 5]
        current_time = sequence.iloc[-1][time_col]
        glucose_values = sequence[glucose_col].tolist()

        # Get current glucose metrics
        current_std = sequence.iloc[-1]['glucose_std']
        current_rate = sequence.iloc[-1]['glucose_rate']
        current_mean = sequence.iloc[-1]['glucose_mean']
        current_min = sequence.iloc[-1]['glucose_min']
        current_max = sequence.iloc[-1]['glucose_max']
        current_range = sequence.iloc[-1]['glucose_range']
        current_acceleration = sequence.iloc[-1]['glucose_acceleration']

        # Get time of day features
        hour = sequence.iloc[-1]['hour_of_day']
        is_daytime = sequence.iloc[-1]['is_daytime']
        day_of_week = sequence.iloc[-1]['day_of_week']
        is_weekend = sequence.iloc[-1]['is_weekend']

        # Set reasonable default values for features
        feature_row = {
            'current_glucose': glucose_values[-1],
            'glucose_source': 0,
            'prev_glucose_1': glucose_values[-2],
            'prev_glucose_2': glucose_values[-3],
            'prev_glucose_3': glucose_values[-4],
            'prev_glucose_4': glucose_values[-5],
            'prev_glucose_5': glucose_values[-5],

            # Added hospital/ICU ID features
            'HADM_ID_FEATURE': hadm_id_feature,
            'ICUSTAY_ID_FEATURE': icustay_id_feature,

            # Original insulin features
            'short_insulin_dose': 0,
            'long_insulin_dose': 0,

            # Enhanced insulin features
            'short_insulin_1h': 0,
            'short_insulin_3h': 0,
            'short_insulin_6h': 0,
            'short_insulin_12h': 0,

            'long_insulin_1h': 0,
            'long_insulin_3h': 0,
            'long_insulin_6h': 0,
            'long_insulin_12h': 0,

            'total_insulin_1h': 0,
            'total_insulin_3h': 0,
            'total_insulin_6h': 0,
            'total_insulin_12h': 0,

            'insulin_rate_1h': 0,
            'insulin_rate_3h': 0,
            'insulin_rate_6h': 0,
            'insulin_rate_12h': 0,

            'time_since_insulin': 24,
            'last_insulin_amount': 0,
            'last_insulin_type_short': 0,

            # Enhanced glucose variability metrics
            'glucose_std': current_std,
            'glucose_rate': current_rate,
            'glucose_mean': current_mean,
            'glucose_min': current_min,
            'glucose_max': current_max,
            'glucose_range': current_range,
            'glucose_acceleration': current_acceleration,

            # Enhanced time of day features
            'hour_of_day': hour,
            'is_daytime': is_daytime,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,

            # Patient demographic features
            'gender': gender_feature,
            'age': age_feature,
            'icd9_category': icd9_category,
            'icd9_subcategory': icd9_subcategory,

            # Patient metadata
            'los_icu_days': 0,
            'first_icu_stay': 0,

            # Additional features
            'mean_insulin_duration': 0,
            'latest_insulin_duration': 0,
            'time_since_infxstop': 24,
            'last_infxstop_value': 0,
            'mean_infxstop_value': 0,
            'infxstop_count_24h': 0,

            # Added timestamp features
            'timer_starttime_diff': 0,
            'timer_endtime_diff': 0,
            'timer_glctimer_diff': 0,

            # Keep timestamp for reference
            'timestamp': current_time
        }

        features_list.append(feature_row)

    features_df = pd.DataFrame(features_list)
    features_df = clean_infinite_values(features_df)
    features_df = features_df.fillna(0)

    print(f"Created {len(features_df)} feature rows with {len(features_df.columns)} columns")
    return features_df


def prepare_dataset(df):
    """
    Prepare the complete dataset for XGBoost training by processing all patients.

    :param df: DataFrame with all patient data
    :type df: pandas.DataFrame
    :returns: Features ready for XGBoost training
    :rtype: pandas.DataFrame
    """
    all_features = pd.DataFrame()

    subject_id_col = get_column_name(df, 'SUBJECT_ID')

    if not subject_id_col:
        print("Warning: Could not find SUBJECT_ID column. Using all data as one patient.")
        patient_features = create_enhanced_patient_features(df)
        if patient_features is not None:
            all_features = pd.concat([all_features, patient_features], ignore_index=True)
    else:
        subject_ids = df[subject_id_col].unique()
        print(f"Processing {len(subject_ids)} patients...")

        for subject_id in tqdm(subject_ids, desc="Processing patients"):
            patient_data = df[df[subject_id_col] == subject_id]
            patient_features = create_enhanced_patient_features(patient_data)
            if patient_features is not None:
                all_features = pd.concat([all_features, patient_features], ignore_index=True)

    all_features = all_features.fillna(method='ffill').fillna(0)
    all_features = clean_infinite_values(all_features)

    for target in ['future_glucose_1hr', 'future_glucose_2hr', 'future_glucose_3hr']:
        all_features = all_features[~all_features[target].isna()]

    return all_features


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a trained XGBoost model and print performance metrics.

    :param model: Trained XGBoost model
    :type model: xgboost.XGBRegressor
    :param X_test: Test features
    :type X_test: pandas.DataFrame
    :param y_test: True target values
    :type y_test: pandas.Series
    :param model_name: Name to display in output
    :type model_name: str
    :returns: Dict with evaluation metrics
    :rtype: dict
    """
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

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
        feature_cols = list(X_test.columns)
        importance = model.feature_importances_

        if len(feature_cols) != len(importance):
            print(
                f"Warning: Feature columns length ({len(feature_cols)}) doesn't match importance length ({len(importance)})")
            print("\nFeature Importance Values:")
            sorted_idx = np.argsort(importance)
            for i in sorted_idx[::-1]:
                print(f"Feature {i}: {importance[i]:.4f}")
        else:
            feature_importance = dict(zip(feature_cols, importance))
            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

            print("\nFeature Importance:")
            for feature, imp in sorted_importance[:10]:  # Top 10 features
                print(f"{feature}: {imp:.4f}")

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'mean_error': mean_error,
        'std_error': std_error
    }


def plot_model_evaluation(y_test, y_pred, title="Model Evaluation", save_png=False):
    """
    Plot actual vs predicted values and error distribution.

    :param y_test: True target values
    :type y_test: numpy.ndarray or pandas.Series
    :param y_pred: Predicted values
    :type y_pred: numpy.ndarray
    :param title: Plot title
    :type title: str
    :param save_png: Whether to save plot as PNG
    :type save_png: bool
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.scatter(y_test, y_pred, alpha=0.5)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax1.set_xlabel('Actual')
    ax1.set_ylabel('Predicted')
    ax1.set_title('Actual vs Predicted Values')
    ax1.grid(True, alpha=0.3)

    errors = y_test - y_pred
    ax2.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--')
    ax2.set_xlabel('Prediction Error')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Error Distribution')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_png:
        filename = f'{title.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filename}")

    plt.show()


def train_enhanced_glucose_models(features_df, save_dir='models'):
    """
    Train XGBoost models for glucose prediction at different time horizons.

    :param features_df: DataFrame with extracted features
    :type features_df: pandas.DataFrame
    :param save_dir: Directory to save models
    :type save_dir: str
    :returns: Tuple of (models, quantile_models)
    :rtype: tuple(dict, dict)
    """
    features_df = clean_infinite_values(features_df)

    models = {}
    quantile_models = {}

    all_feature_cols = [col for col in features_df.columns
                        if not col.lower().startswith('future_')
                        and col.upper() != 'SUBJECT_ID'
                        and col.lower() != 'subject_id']

    print(f"Training with {len(all_feature_cols)} features")
    all_metrics = {}

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    metrics_dir = os.path.join(save_dir, 'metrics')
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    print("\nTRAINING ENHANCED GLUCOSE PREDICTION MODELS")
    print("=" * 50)

    for hours in tqdm([1, 2, 3], desc="Training models for different time horizons"):
        target_col = f'future_glucose_{hours}hr'
        print(f"\nTraining Enhanced {hours}-hour Prediction Model")
        print("*" * 30)

        X = features_df[all_feature_cols]
        y = features_df[target_col]

        X = clean_infinite_values(X)
        X = X.fillna(method='ffill').fillna(0)

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42,
                stratify=features_df['SUBJECT_ID'] if len(features_df['SUBJECT_ID'].unique()) > 1 else None
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"Training data size: {X_train.shape[0]} samples")
        print(f"Test data size: {X_test.shape[0]} samples")

        if np.isinf(X_train.values).any() or np.isnan(X_train.values).any():
            print("Warning: Infinities or NaNs found in training data after preparation!")
            X_train = pd.DataFrame(X_train, columns=X.columns).fillna(0)
            X_test = pd.DataFrame(X_test, columns=X.columns).fillna(0)

            X_train = X_train.replace([np.inf, -np.inf], [1e10, -1e10])
            X_test = X_test.replace([np.inf, -np.inf], [1e10, -1e10])

        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric=['rmse', 'mae'],
            early_stopping_rounds=20,
            missing=np.nan
        )

        print(f"Training enhanced {hours}-hour prediction model...")
        try:
            model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)],
                      verbose=True)
        except Exception as e:
            print(f"Error during model fitting: {e}")
            print("Trying with simpler model configuration...")

            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                missing=np.nan
            )

            model.fit(X_train, y_train)

        lower_model = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=0.05,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            missing=np.nan
        )

        print(f"Training lower bound {hours}-hour prediction model...")
        lower_model.fit(X_train, y_train, verbose=False)

        upper_model = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=0.95,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            missing=np.nan
        )

        print(f"Training upper bound {hours}-hour prediction model...")
        upper_model.fit(X_train, y_train, verbose=False)

        model_name = f"Enhanced {hours}-hour Prediction Model"
        metrics = evaluate_model(model, X_test, y_test, model_name)
        all_metrics[hours] = metrics

        y_pred = model.predict(X_test)
        plot_model_evaluation(y_test, y_pred, title=f"Enhanced {hours}-hour Prediction", save_png=False)

        lower_bound = lower_model.predict(X_test)
        upper_bound = upper_model.predict(X_test)
        within_ci = np.sum((y_test >= lower_bound) & (y_test <= upper_bound)) / len(y_test) * 100
        print(f"90% CI coverage: {within_ci:.2f}%")

        metrics_df = pd.DataFrame({
            'Metric': ['RMSE', 'MAE', 'R²', 'MAPE', 'Mean Error', 'Std Error', 'CI Coverage'],
            'Value': [metrics['rmse'], metrics['mae'], metrics['r2'],
                      metrics['mape'], metrics['mean_error'], metrics['std_error'], within_ci]
        })
        metrics_df.to_csv(os.path.join(metrics_dir, f'metrics_{hours}hr.csv'), index=False)

        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(all_feature_cols, model.feature_importances_))
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

            print("\nTop 10 Feature Importance:")
            for feature, imp in sorted_importance[:10]:
                print(f"{feature}: {imp:.4f}")

            importance_df = pd.DataFrame({
                'Feature': all_feature_cols,
                'Importance': model.feature_importances_
            })
            importance_df = importance_df.sort_values('Importance', ascending=False)
            importance_df.to_csv(os.path.join(metrics_dir, f'feature_importance_{hours}hr.csv'), index=False)

        models[hours] = model
        quantile_models[hours] = {
            'lower': lower_model,
            'upper': upper_model
        }

        joblib.dump(model, os.path.join(save_dir, f'enhanced_glucose_model_{hours}hr.pkl'))
        joblib.dump(lower_model, os.path.join(save_dir, f'enhanced_glucose_model_{hours}hr_lower.pkl'))
        joblib.dump(upper_model, os.path.join(save_dir, f'enhanced_glucose_model_{hours}hr_upper.pkl'))

    print("\nENHANCED MODEL PERFORMANCE SUMMARY")
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
    summary_df.to_csv(os.path.join(metrics_dir, 'enhanced_model_summary.csv'), index=False)

    return models, quantile_models


def load_models(model_dir='models'):
    """
    Load XGBoost models from disk.

    :param model_dir: Directory containing saved models
    :type model_dir: str
    :returns: Tuple of (models, quantile_models)
    :rtype: tuple(dict, dict)
    """
    models = {}
    quantile_models = {}

    for hours in [1, 2, 3]:
        enhanced_model_path = os.path.join(model_dir, f'enhanced_glucose_model_{hours}hr.pkl')
        enhanced_lower_path = os.path.join(model_dir, f'enhanced_glucose_model_{hours}hr_lower.pkl')
        enhanced_upper_path = os.path.join(model_dir, f'enhanced_glucose_model_{hours}hr_upper.pkl')

        if os.path.exists(enhanced_model_path) and os.path.exists(enhanced_lower_path) and os.path.exists(
                enhanced_upper_path):
            print(f"Loading enhanced models for {hours}-hour prediction")
            models[hours] = joblib.load(enhanced_model_path)
            quantile_models[hours] = {
                'lower': joblib.load(enhanced_lower_path),
                'upper': joblib.load(enhanced_upper_path)
            }
        else:
            model_path = os.path.join(model_dir, f'glucose_model_{hours}hr.pkl')
            lower_path = os.path.join(model_dir, f'glucose_model_{hours}hr_lower.pkl')
            upper_path = os.path.join(model_dir, f'glucose_model_{hours}hr_upper.pkl')

            if os.path.exists(model_path) and os.path.exists(lower_path) and os.path.exists(upper_path):
                print(f"Loading original models for {hours}-hour prediction")
                models[hours] = joblib.load(model_path)
                quantile_models[hours] = {
                    'lower': joblib.load(lower_path),
                    'upper': joblib.load(upper_path)
                }
            else:
                print(f"Warning: Models for {hours}-hour prediction not found")

    return models, quantile_models


def predict_for_patient(patient_df, models, quantile_models):
    """
    Generate predictions for a patient using trained XGBoost models.

    :param patient_df: DataFrame with patient data
    :type patient_df: pandas.DataFrame
    :param models: Dict of trained models for each horizon
    :type models: dict
    :param quantile_models: Dict of quantile models for confidence intervals
    :type quantile_models: dict
    :returns: DataFrame with predictions
    :rtype: pandas.DataFrame
    """
    features_df = create_prediction_features(patient_df)

    if len(features_df) == 0:
        return pd.DataFrame()

    feature_cols = [col for col in features_df.columns if col != 'timestamp']

    X = features_df[feature_cols]
    timestamps = features_df['timestamp']
    predictions = []

    print(f"Generating predictions for {len(X)} time points...")

    for i, row in tqdm(X.iterrows(), total=len(X), desc="Making predictions"):
        row_df = pd.DataFrame([row])
        timestamp = timestamps.iloc[i]

        pred_row = {
            'timestamp': timestamp,
            'current_glucose': row['current_glucose']
        }

        prediction_made = False

        for hours in [1, 2, 3]:
            if hours not in models or hours not in quantile_models:
                continue

            model = models[hours]
            lower_model = quantile_models[hours]['lower']
            upper_model = quantile_models[hours]['upper']

            try:
                expected_features = model.feature_names_in_

                feature_mismatch = False
                if set(row_df.columns) != set(expected_features):
                    feature_mismatch = True
                    print(f"Warning: Feature mismatch for {hours}-hour model. Aligning features...")
                elif list(row_df.columns) != list(expected_features):
                    feature_mismatch = True
                    print(f"Warning: Feature order mismatch for {hours}-hour model. Reordering features...")

                if feature_mismatch:
                    missing_features = set(expected_features) - set(row_df.columns)
                    extra_features = set(row_df.columns) - set(expected_features)

                    if missing_features:
                        print(f"Missing features: {missing_features}")
                    if extra_features:
                        print(f"Extra features: {extra_features}")

                    aligned_row = pd.DataFrame(columns=expected_features)
                    for feat in expected_features:
                        if feat in row_df.columns:
                            aligned_row[feat] = row_df[feat].values
                        else:
                            print(f"Adding missing feature: {feat}, using 0")
                            aligned_row[feat] = 0

                    row_df = aligned_row

                if not np.array_equal(row_df.columns, expected_features):
                    raise ValueError(f"Feature alignment failed. Expected: {expected_features}, Got: {row_df.columns}")

                mean_pred = model.predict(row_df)[0]
                lower_bound = lower_model.predict(row_df)[0]
                upper_bound = upper_model.predict(row_df)[0]

                pred_time = timestamp + timedelta(hours=hours)
                pred_row[f'predicted_{hours}hr'] = mean_pred
                pred_row[f'lower_bound_{hours}hr'] = lower_bound
                pred_row[f'upper_bound_{hours}hr'] = upper_bound
                pred_row[f'prediction_time_{hours}hr'] = pred_time

                prediction_made = True
                print(f"Successfully made {hours}-hour prediction: {mean_pred:.1f} mg/dL at {pred_time}")

            except Exception as e:
                print(f"Error making {hours}-hour prediction: {str(e)}")
                continue

        if prediction_made:
            predictions.append(pred_row)
        else:
            print("Warning: No successful predictions for this time point")

    return pd.DataFrame(predictions)


def plot_patient_predictions(patient_df, predictions_df, save_png=False):
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

    if save_png:
        plt.savefig('xgboost_glucose_predictions.png', dpi=300, bbox_inches='tight')
        print("Plot saved as xgboost_glucose_predictions.png")

    plt.show()


def evaluate_predictions(predictions_df, actual_df):
    """
    Evaluate prediction accuracy against actual values.

    :param predictions_df: DataFrame with predictions
    :type predictions_df: pandas.DataFrame
    :param actual_df: DataFrame with actual values
    :type actual_df: pandas.DataFrame
    :returns: Dict with evaluation results
    :rtype: dict
    """
    evaluation_results = {}

    col_mapping = actual_df.attrs.get('column_mapping', {})

    time_col = col_mapping.get('time', 'time')
    glucose_col = col_mapping.get('glucose_level', 'glucose_level')

    if time_col not in actual_df.columns:
        print(
            f"Warning: Time column '{time_col}' not found in dataframe. Available columns: {actual_df.columns.tolist()}")
        time_col = actual_df.columns[0]
    if glucose_col not in actual_df.columns:
        print(
            f"Warning: Glucose column '{glucose_col}' not found in dataframe. Available columns: {actual_df.columns.tolist()}")
        glucose_col = actual_df.columns[1]

    for hours in [1, 2, 3]:
        if f'predicted_{hours}hr' not in predictions_df.columns:
            continue

        pred_times = predictions_df[f'prediction_time_{hours}hr']
        pred_values = predictions_df[f'predicted_{hours}hr']
        lower_bounds = predictions_df[f'lower_bound_{hours}hr']
        upper_bounds = predictions_df[f'upper_bound_{hours}hr']

        actual_values = []
        for pred_time in pred_times:
            time_diffs = abs(actual_df[time_col] - pred_time)
            closest_idx = time_diffs.idxmin()
            closest_time = actual_df.loc[closest_idx, time_col]

            if abs((closest_time - pred_time).total_seconds()) <= 600:
                actual_values.append(actual_df.loc[closest_idx, glucose_col])
            else:
                actual_values.append(np.nan)

        actual_values = np.array(actual_values)
        mask = ~np.isnan(actual_values)

        if sum(mask) > 0:
            actual_filtered = actual_values[mask]
            pred_filtered = pred_values.iloc[mask].values
            lower_filtered = lower_bounds.iloc[mask].values
            upper_filtered = upper_bounds.iloc[mask].values

            rmse = np.sqrt(mean_squared_error(actual_filtered, pred_filtered))
            mae = mean_absolute_error(actual_filtered, pred_filtered)
            r2 = r2_score(actual_filtered, pred_filtered)

            within_ci = np.sum((actual_filtered >= lower_filtered) &
                               (actual_filtered <= upper_filtered)) / len(actual_filtered) * 100

            evaluation_results[hours] = {
                'num_points': len(actual_filtered),
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'ci_coverage': within_ci
            }

            print(f"\n{hours}-hour Prediction Evaluation:")
            print(f"Number of evaluation points: {len(actual_filtered)}")
            print(f"RMSE: {rmse:.2f}")
            print(f"MAE: {mae:.2f}")
            print(f"R²: {r2:.3f}")
            print(f"90% CI coverage: {within_ci:.2f}%")

    return evaluation_results


def process_patient_data(patient_file, model_dir='models', output_file='glucose_predictions.csv'):
    """
    Process patient data to generate predictions with XGBoost models.

    :param patient_file: Path to patient data CSV
    :type patient_file: str
    :param model_dir: Directory containing models
    :type model_dir: str
    :param output_file: Path to save predictions
    :type output_file: str
    :returns: DataFrame with predictions
    :rtype: pandas.DataFrame
    """
    from common_utils import load_patient_data, export_predictions

    print(f"Loading patient data from {patient_file}...")
    patient_df = load_patient_data(patient_file)

    if len(patient_df) < 5:
        print("Error: Need at least 5 glucose readings to make predictions")
        return None

    print("Loading prediction models...")
    models, quantile_models = load_models(model_dir)

    if not models:
        print("Error: No models found. Please train models first.")
        return None

    print("Making predictions...")
    predictions_df = predict_for_patient(patient_df, models, quantile_models)

    if len(predictions_df) == 0:
        print("Error: Could not generate any predictions")
        return None

    export_predictions(predictions_df, output_file)

    print("Plotting results...")
    try:
        plot_patient_predictions(patient_df, predictions_df)
    except Exception as e:
        print(f"Error plotting predictions: {str(e)}")
        print("Continuing without visualization...")

    try:
        future_end_time = None
        for hours in [3, 2, 1]:
            time_col = f'prediction_time_{hours}hr'
            if time_col in predictions_df.columns and not predictions_df[time_col].empty:
                future_end_time = predictions_df[time_col].max()
                print(f"Using {hours}-hour prediction horizon for evaluation")
                break

        if future_end_time is None:
            print("No valid prediction time horizons found for evaluation")
        else:
            col_mapping = patient_df.attrs.get('column_mapping', {})
            time_col = col_mapping.get('time', 'time')

            if patient_df[time_col].max() >= future_end_time:
                print("\nEvaluating prediction accuracy...")
                evaluation_results = evaluate_predictions(predictions_df, patient_df)

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
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        print("Continuing without evaluation...")

    return predictions_df


def train_models(training_file, model_dir='models'):
    """
    Train XGBoost models from training data.

    :param training_file: Path to training data CSV
    :type training_file: str
    :param model_dir: Directory to save models
    :type model_dir: str
    :returns: Tuple of (models, quantile_models)
    :rtype: tuple(dict, dict)
    """
    from common_utils import load_and_preprocess_training_data

    print(f"Loading training data from {training_file}...")
    df = load_and_preprocess_training_data(training_file)

    print("Creating features...")
    features_df = prepare_dataset(df)

    print("Training models...")
    models, quantile_models = train_enhanced_glucose_models(features_df, model_dir)

    print(f"Models saved to {model_dir} directory")
    return models, quantile_models