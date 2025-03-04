import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm

from common.preprocessing import clean_infinite_values, encode_icd9_code
from common.data_loader import get_column_name

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
