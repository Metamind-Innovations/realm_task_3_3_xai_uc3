import argparse
import pandas as pd
from typing import Dict, Any, Literal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from utils.generic_utils import load_json_file, get_json_files, save_json
from utils.data_helpers import extract_prediction_info, calculate_interval_midpoint
from STAR_model import STARWrapper

DEMOGRAPHICS_COLUMNS = {"age": "age", "gender": "gender"}


def age_to_cat(age_col: pd.Series) -> pd.Series:
    """Convert numeric age to categorical age groups.
    Age bins: <40, 40-54, 55-69, 70+

    Args:
        age_col (pd.Series): Numeric age values

    Returns:
        pd.Series: Categorical age groups
    """
    age_bins = [0, 40, 55, 70, 120]
    age_labels = ["<40", "40-54", "55-69", "70+"]

    return pd.cut(
        age_col, bins=age_bins, labels=age_labels, include_lowest=True, right=True
    )


def compute_mean_predicted_value(predicted: pd.Series) -> float:
    """Compute mean of predicted values (demographic parity metric).

    Args:
        predicted (pd.Series): Predicted blood glucose values (interval midpoints)

    Returns:
        float: Mean predicted value
    """
    return float(predicted.mean())


def compute_miscoverage_rate(
    actual: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
) -> float:
    """Compute miscoverage rate (equalized odds metric).

    Args:
        actual (pd.Series): Ground truth blood glucose values
        lower (pd.Series): Lower bounds (BG5TH)
        upper (pd.Series): Upper bounds (BG95TH)

    Returns:
        float: Miscoverage rate (proportion of times actual is outside interval)
    """

    outside_interval = (actual < lower) | (actual > upper)
    miscoverage_rate = outside_interval.mean()

    return float(miscoverage_rate)


def calculate_fairness_metrics(
    data: pd.DataFrame, metric_type: Literal["demographic_parity", "equalized_odds"]
) -> Dict[str, Any]:
    """Calculate fairness metrics across demographic groups.

    Args:
        data (pd.DataFrame): DataFrame with demographics, predictions, and ground truth
        metric_type (Literal): Type of fairness metric to calculate:
            - "demographic_parity": mean predicted values by group
            - "equalized_odds": miscoverage rates by group

    Returns:
        Dict: Nested dictionary with metrics per demographic group
    """

    # Define metric configuration
    metric_config = {
        "demographic_parity": {
            "key": "mean_predicted_values_by_group",
            "compute_func": lambda x: compute_mean_predicted_value(
                predicted=x["interval_center"]
            ),
        },
        "equalized_odds": {
            "key": "miscoverage_rates_by_group",
            "compute_func": lambda x: compute_miscoverage_rate(
                actual=x["ground_truth"],
                lower=x["BG5TH"],
                upper=x["BG95TH"],
            ),
        },
    }

    config = metric_config[metric_type]
    metrics = {}

    for col in DEMOGRAPHICS_COLUMNS.values():
        if col not in data.columns:
            continue

        metrics[col] = {config["key"]: {}}
        unique_vals = data[col].unique().tolist()

        for group_name in unique_vals:
            if pd.isna(group_name):
                continue

            group_data = data[data[col] == group_name].copy()
            metric_value = config["compute_func"](group_data)

            metrics[col][config["key"]][f"{group_name}"] = metric_value

    return metrics


def process_single_patient(filepath: str, model: STARWrapper) -> dict:
    """Process one patient JSON file and get predictions with demographics.

    Args:
        filepath (str): Path to patient JSON file
        model (STARWrapper): STAR API wrapper instance

    Returns:
        dict: Result dictionary with predictions, demographics, and ground truth
    """
    try:
        patient_data = load_json_file(filepath)
        pred_time, actual_value = extract_prediction_info(patient_data)

        pred_interval = model.predict(
            patient_data=patient_data, prediction_time=pred_time
        )

        interval_center = calculate_interval_midpoint(pred_interval)

        # Extract demographics from patient data
        episode = patient_data["episodes"][0]
        age = episode.get("age", None)
        gender = episode.get("gender", None)  # False=Male, True=Female in json data

        return {
            "file_name": Path(filepath).name,
            "age": age,
            "gender": "Female" if gender else "Male",
            "ground_truth": actual_value,
            "BG5TH": pred_interval["BG5TH"],
            "BG95TH": pred_interval["BG95TH"],
            "interval_center": interval_center,
            "success": True,
            "error_message": None,
        }
    except Exception as e:
        return {
            "file_name": Path(filepath).name,
            "age": None,
            "gender": None,
            "ground_truth": None,
            "BG5TH": None,
            "BG95TH": None,
            "interval_center": None,
            "success": False,
            "error_message": str(e),
        }


def predict_glucose_levels(data_path: str, max_workers: int = 10) -> pd.DataFrame:
    """Process patient JSON files in parallel and return results with demographics.

    Args:
        data_path (str): Path to directory containing patient JSON files
        max_workers (int): Number of parallel workers for API calls

    Returns:
        pd.DataFrame: Results with demographics, predictions, and ground truth
    """
    patient_files = get_json_files(data_path)

    if not patient_files:
        raise ValueError(f"No JSON files found in directory: {data_path}")

    model_wrapper = STARWrapper()

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_patient, file, model_wrapper)
            for file in patient_files
        ]

        for future in tqdm(
            as_completed(futures), total=len(patient_files), desc="Processing patients"
        ):
            results.append(future.result())

    df = pd.DataFrame(results)

    return df


def fairness_bias_analysis(
    data_path: str,
    output_path: str,
) -> None:
    """Run fairness and bias analysis for STAR blood glucose predictions.
    This function loads patient JSON files, generates predictions using the STAR API,
    and computes fairness metrics across demographic groups (age and gender).

    Fairness Metrics:
    1. Demographic Parity: Checks if mean predicted values are similar across groups
    2. Equalized Odds: Checks if miscoverage rates are similar across groups

    Args:
        data_path: Path to directory with patient JSON files
        output_path: Path where results JSON will be saved
    """

    # Process all patients and get predictions with demographics
    preds_dem = predict_glucose_levels(data_path=data_path)

    # Filter only successful predictions
    preds_dem_success = preds_dem[preds_dem["success"] == True].copy()

    if preds_dem_success.empty:
        raise ValueError(
            "No successful predictions generated. "
            "Check patient data format and API connectivity."
        )

    # Convert age to categorical
    if "age" in preds_dem_success.columns:
        preds_dem_success["age"] = age_to_cat(preds_dem_success["age"])

    # Calculate equalized odds metrics
    equalized_odds_metrics = calculate_fairness_metrics(
        data=preds_dem_success, metric_type="equalized_odds"
    )

    # Calculate demographic parity metrics
    demographic_parity_metrics = calculate_fairness_metrics(
        data=preds_dem_success, metric_type="demographic_parity"
    )

    # Results
    results = {
        "equalized_odds_metrics": equalized_odds_metrics,
        "demographic_parity_metrics": demographic_parity_metrics,
    }

    # Store results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(data=results, filepath=str(output_path))
    print(f"Fairness - Bias analysis completed. Results saved to {output_path}")


def main() -> None:
    """CLI entry point for fairness and bias analysis of STAR model predictions.

    This function parses command-line arguments for patient data directory
    and output path, then runs the fairness analysis pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Analyze fairness and bias in STAR blood glucose predictions"
    )
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to directory containing patient JSON files",
    )
    parser.add_argument(
        "--output",
        default="output/fairness_analysis.json",
        help="Output JSON file path (default: output/fairness_analysis.json)",
    )

    args = parser.parse_args()

    fairness_bias_analysis(
        data_path=args.data_path,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
