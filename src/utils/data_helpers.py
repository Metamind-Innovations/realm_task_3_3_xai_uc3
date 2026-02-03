from typing import Dict, Any, Tuple


def extract_prediction_info(patient_data: Dict[str, Any]) -> Tuple[int, float]:
    """
    Extract prediction time and actual blood glucose value from patient data.

    :param patient_data: Patient data JSON object.
    :return: (prediction_time, actual_value) where prediction_time is Unix epoch time in milliseconds and actual_value is the blood glucose level.
    :raises ValueError: If required fields are missing or data is invalid.
    """
    if not patient_data:
        raise ValueError("Patient data is empty")

    if "episodes" not in patient_data:
        raise ValueError("Missing 'episodes' field in patient data")

    if not patient_data["episodes"]:
        raise ValueError("Episodes list is empty")

    episode = patient_data["episodes"][0]

    if "bloodGlucose" not in episode:
        raise ValueError("Missing 'bloodGlucose' field in episode")

    blood_glucose = episode["bloodGlucose"]

    if not blood_glucose:
        raise ValueError("Blood glucose list is empty")

    last_episode_to_predict = blood_glucose[-1]

    pred_time = int(last_episode_to_predict[0])
    actual_value = last_episode_to_predict[1]

    return pred_time, actual_value


def calculate_interval_midpoint(interval: Dict[str, float]) -> float:
    """
    Calculate the midpoint (average) of a predicted blood glucose interval.

    :param interval: Prediction interval with BG5TH and BG95TH keys.
    :return: Midpoint of the interval (BG5TH + BG95TH) / 2.
    :raises ValueError: If required fields are missing.
    """
    if not interval:
        raise ValueError("Interval is empty")

    if "BG5TH" not in interval:
        raise ValueError("Missing 'BG5TH' in interval")

    if "BG95TH" not in interval:
        raise ValueError("Missing 'BG95TH' in interval")

    return (interval["BG5TH"] + interval["BG95TH"]) / 2.0
