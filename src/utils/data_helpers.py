from typing import Dict, Any, Tuple


def extract_prediction_info(patient_data: Dict[str, Any]) -> Tuple[int, float]:
    """
    Extract prediction time and actual blood glucose value from patient data.

    Args:
        patient_data (Dict[str, Any]): Patient data JSON object.

    Returns:
        Tuple[int, float]: (prediction_time, actual_value) where prediction_time
            is Unix epoch time in milliseconds and actual_value is the blood glucose level.
    """
    blood_glucose = patient_data["episodes"][0]["bloodGlucose"]
    last_episode_to_predict = blood_glucose[-1]

    pred_time = int(last_episode_to_predict[0])
    actual_value = last_episode_to_predict[1]

    return pred_time, actual_value


def calculate_interval_midpoint(interval: Dict[str, float]) -> float:
    """
    Calculate the midpoint (average) of a predicted blood glucose interval.

    Args:
        interval (Dict[str, float]): Prediction interval with BG5TH and BG95TH keys.

    Returns:
        float: Midpoint of the interval (BG5TH + BG95TH) / 2.
    """

    return (interval["BG5TH"] + interval["BG95TH"]) / 2.0
