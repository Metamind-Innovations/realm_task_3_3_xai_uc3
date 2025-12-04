import argparse
import copy
import numpy as np
from pathlib import Path
from typing import Dict, Any, Callable, List, Tuple, Literal, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.generic_utils import get_json_files, save_json
from utils.data_helpers import extract_prediction_info, calculate_interval_midpoint
from utils.explainer_helpers import (
    find_method_name,
    load_all_patients_data,
    ATTRIBUTES,
    MAX_WORKERS,
)
from STAR_model import STARWrapper


# ============================================================================
# ABLATION FUNCTIONS
# ============================================================================
def ablate_field(data: Dict[str, Any], category: str, field: str) -> Dict[str, Any]:
    """Remove a field from all entries in a category.

    Args:
        data (Dict[str, Any]): Patient data dictionary.
        category (str): Category name (e.g., 'insulinInfusion').
        field (str): Field name to remove.

    Returns:
        Dict[str, Any]: Modified patient data with field removed.
    """

    p = copy.deepcopy(data)
    for item in p["episodes"][0][category]:
        if field in item[1]:
            del item[1][field]
    return p


def ablate_diabetic_status(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove diabetic status field.

    Args:
        data (Dict[str, Any]): Patient data dictionary.

    Returns:
        Dict[str, Any]: Modified patient data without diabetic status.
    """

    p = copy.deepcopy(data)
    if "diabeticStatus" in p["episodes"][0]:
        del p["episodes"][0]["diabeticStatus"]
    return p


def ablate_blood_glucose(data: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only 2 BG measurements before prediction time (minimum required).

    Args:
        data (Dict[str, Any]): Patient data dictionary.

    Returns:
        Dict[str, Any]: Modified patient data with reduced blood glucose measurements.
    """

    p = copy.deepcopy(data)
    bg_data = p["episodes"][0]["bloodGlucose"]
    # Prediction time is last timestep, so keep last 2 measurements before it
    if len(bg_data) > 2:
        # Keep last 3 values
        p["episodes"][0]["bloodGlucose"] = bg_data[-3:]
    return p


def build_ablation_registry() -> Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]:
    """Build registry of ablation functions for each attribute.

    Returns:
        Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]: Dictionary mapping
            attribute names to ablation functions.
    """
    registry = {}

    # Special cases
    registry["diabeticStatus"] = ablate_diabetic_status
    registry["bloodGlucose.value"] = ablate_blood_glucose

    # Generic ablations for all other attributes
    for attr in ATTRIBUTES:
        if attr not in registry and "." in attr:
            category, field = attr.split(".")
            # Create lambda with default arguments to capture values
            registry[attr] = lambda p, c=category, f=field: ablate_field(p, c, f)

    return registry


# ============================================================================
# PERTURBATION FUNCTIONS
# ============================================================================
def build_perturbation_registry() -> Dict[str, Tuple[str, Any]]:
    """Build registry of perturbation strategies (type, magnitude).

    Returns:
        Dict[str, Tuple[str, Any]]: Dictionary mapping attribute names to
            (perturbation_type, parameter) tuples.
    """
    return {
        "diabeticStatus": ("categorical", [0, 1, 2]),
        "bloodGlucose.value": ("continuous", 0.20),
        "insulinInfusion.route": ("binary", None),
        "insulinInfusion.rate": ("continuous", 0.25),
        "insulinInfusion.fixed": ("binary", None),
        "insulinInfusion.concentration": ("continuous", 0.25),
        "insulinBolus.route": ("binary", None),
        "insulinBolus.size": ("continuous", 0.25),
        "insulinBolus.adminTime": ("continuous", 0.25),
        "insulinBolus.concentration": ("continuous", 0.25),
        "nutritionInfusion.type": ("categorical", [0, 1, 2]),
        "nutritionInfusion.carbsConcentration": ("continuous", 0.25),
        "nutritionInfusion.rate": ("continuous", 0.25),
        "nutritionInfusion.fixed": ("binary", None),
        "nutritionBolus.isMeal": ("binary", None),
        "nutritionBolus.size": ("continuous", 0.25),
        "nutritionBolus.adminTime": ("continuous", 0.25),
        "nutritionBolus.type": ("categorical", [0, 1, 2]),
        "nutritionBolus.carbsConcentration": ("continuous", 0.25),
    }


def perturb_continuous(
    data: Dict[str, Any], attr: str, magnitude: float
) -> Dict[str, Any]:
    """Perturb continuous attribute by percentage with bounds checking.

    Args:
        data (Dict[str, Any]): Patient data dictionary.
        attr (str): Attribute name (e.g., 'bloodGlucose.value').
        magnitude (float): Perturbation magnitude as decimal (e.g., 0.25 for 25%).

    Returns:
        Dict[str, Any]: Modified patient data with perturbed values.
    """
    p = copy.deepcopy(data)
    parts = attr.split(".")

    if parts[0] == "bloodGlucose":
        for i in range(len(p["episodes"][0]["bloodGlucose"])):
            original = p["episodes"][0]["bloodGlucose"][i][1]
            perturbed = original * (1 + magnitude)
            # Blood glucose should stay positive
            p["episodes"][0]["bloodGlucose"][i][1] = max(0.1, perturbed)

    elif len(parts) == 2:
        category, attribute = parts
        for item in p["episodes"][0][category]:
            if attribute in item[1]:
                original = item[1][attribute]
                perturbed = original * (1 + magnitude)
                # Rates, sizes, concentrations should stay non-negative
                item[1][attribute] = max(0, perturbed)

    return p


def perturb_binary(data: Dict[str, Any], attr: str) -> Dict[str, Any]:
    """Flip binary attribute.

    Args:
        data (Dict[str, Any]): Patient data dictionary.
        attr (str): Attribute name.

    Returns:
        Dict[str, Any]: Modified patient data with flipped binary value.
    """
    p = copy.deepcopy(data)
    parts = attr.split(".")

    if len(parts) == 2:
        category, attribute = parts
        for item in p["episodes"][0][category]:
            if attribute in item[1]:
                original = item[1][attribute]
                # Handle boolean binary
                if isinstance(original, bool):
                    item[1][attribute] = not original
                # Handle integer binary (0/1)
                elif isinstance(original, int):
                    item[1][attribute] = 1 - original
    return p


def perturb_categorical(
    data: Dict[str, Any], attr: str, values: List[int]
) -> Dict[str, Any]:
    """Change categorical attribute to different value.

    Args:
        data (Dict[str, Any]): Patient data dictionary.
        attr (str): Attribute name.
        values (List[int]): List of possible categorical values.

    Returns:
        Dict[str, Any]: Modified patient data with changed categorical value.
    """
    p = copy.deepcopy(data)

    if attr == "diabeticStatus":
        if "diabeticStatus" in p["episodes"][0]:
            current = p["episodes"][0]["diabeticStatus"]
            # Cycle to next value
            new_val = (current + 1) % len(values)
            p["episodes"][0]["diabeticStatus"] = new_val
    else:
        parts = attr.split(".")
        if len(parts) == 2:
            category, attribute = parts
            for item in p["episodes"][0][category]:
                if attribute in item[1]:
                    current = item[1][attribute]
                    new_val = (current + 1) % len(values)
                    item[1][attribute] = new_val

    return p


# ============================================================================
# MAE COMPUTATION
# ============================================================================
def process_single_patient(
    data: Dict[str, Any],
    star_api: STARWrapper,
) -> Optional[float]:
    """Process single patient and compute absolute error.

    Args:
        data (Dict[str, Any]): Patient data dictionary.
        star_api (STARWrapper): STAR API wrapper instance.

    Returns:
        Optional[float]: Absolute error between actual and predicted values,
            or None if failed.
    """
    try:
        pred_time, actual_value = extract_prediction_info(data)
        pred_interval = star_api.predict(patient_data=data, prediction_time=pred_time)

        # Calculate midpoint of prediction interval
        predicted_midpoint = calculate_interval_midpoint(pred_interval)

        # Calculate absolute error
        ae = abs(actual_value - predicted_midpoint)

        return ae

    except Exception as e:
        return None


def compute_mae(
    patients_data: List[Dict[str, Any]],
    star_api: STARWrapper,
) -> float:
    """Compute Mean Absolute Error across all patients.

    Args:
        patients_data (List[Dict[str, Any]]): List of patient data dictionaries.
        star_api (STARWrapper): STAR API wrapper instance.

    Returns:
        float: Mean absolute error across all successfully processed patients.
    """

    aes = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(process_single_patient, pat, star_api)
            for pat in patients_data
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(patients_data),
            desc="Computing MAE",
            leave=False,
        ):
            ae = future.result()

            if ae is not None:
                aes.append(ae)

    if not aes:
        return 0.0

    mae = np.mean(aes)

    return mae


# ============================================================================
# FEATURE ABLATION - FEATURE PERTURBATION ANALYSIS
# ============================================================================
def analyze_feature_importance(
    patients_data: List[Dict[str, Any]],
    star_api: STARWrapper,
    analysis_type: Literal["feature_ablation", "feature_perturbation"],
) -> Dict[str, float]:
    """Analyze feature importance using ablation or perturbation.

    Args:
        patients_data (List[Dict[str, Any]]): List of patient data dictionaries.
        star_api (STARWrapper): STAR API wrapper instance.
        analysis_type (Literal["feature_ablation", "feature_perturbation"]):
            Type of analysis.

    Returns:
        Dict[str, float]: Dictionary mapping attribute names to normalized
            importance scores (0-1).
    """

    # Compute baseline MAE
    baseline_mae = compute_mae(patients_data, star_api=star_api)

    if analysis_type == "feature_ablation":
        transform_functions = build_ablation_registry()
    elif analysis_type == "feature_perturbation":
        transform_functions = build_perturbation_registry()

    # Initialize importance dictionary
    feature_importance = {}

    # Test each attribute
    for attr_name in tqdm(ATTRIBUTES, desc="Testing attributes"):
        if analysis_type == "feature_ablation":
            current_attr_fn = transform_functions[attr_name]

        elif analysis_type == "feature_perturbation":
            perturb_type, perturb_param = transform_functions[attr_name]

            # Perturbation function with captured variables
            if perturb_type == "continuous":
                current_attr_fn = (
                    lambda p, attr=attr_name, mag=perturb_param: perturb_continuous(
                        p, attr, mag
                    )
                )
            elif perturb_type == "binary":
                current_attr_fn = lambda p, attr=attr_name: perturb_binary(p, attr)
            elif perturb_type == "categorical":
                current_attr_fn = (
                    lambda p, attr=attr_name, vals=perturb_param: perturb_categorical(
                        p, attr, vals
                    )
                )
            else:
                continue

        # Transform data
        transformed_data = []
        for d_ in tqdm(patients_data, desc="Transforming data", leave=False):
            try:
                transformed_data.append(current_attr_fn(d_))
            except Exception as e:
                continue

        # Compute MAE after transformation
        transformed_mae = compute_mae(patients_data=transformed_data, star_api=star_api)

        # Feature importance = INCREASE in MAE (higher MAE = worse predictions)
        mae_increase = max(0, transformed_mae - baseline_mae)
        feature_importance[attr_name] = mae_increase

    # Sort by importance
    feature_importance = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    )

    # Nornalize values to 0-1
    total_importance = sum(feature_importance.values())
    feature_importance = {
        key: (value / total_importance) for key, value in feature_importance.items()
    }

    return feature_importance


def feature_importance_analysis(
    data_path: str,
    output_path: str,
    sensitivity: float,
) -> None:
    """Run feature importance analysis on patient data.

    Args:
        data_path (str): Path to directory containing patient JSON files.
        output_path (str): Output directory path for results.
        sensitivity (float): Sensitivity level [0, 1]. <0.5: ablation, >=0.5: perturbation.

    Raises:
        ValueError: If sensitivity not in [0, 1] or no files found.
    """

    # Validate sensitivity
    if not 0 <= sensitivity <= 1:
        raise ValueError("Sensitivity must be between 0 and 1")

    # Determine method from sensitivity
    method = find_method_name(sensitivity)

    # Get patient files
    patient_files = get_json_files(data_path)

    if not patient_files:
        raise ValueError(f"No JSON files found in directory: {data_path}")

    # Load all patients data files
    patients_data = load_all_patients_data(patient_files)

    if not patients_data:
        raise ValueError("No patients successfully loaded")

    # Initialize model
    star_api = STARWrapper()

    # Run analysis
    results = analyze_feature_importance(
        patients_data, star_api=star_api, analysis_type=method
    )

    # Prepare output
    output_path = Path(output_path)
    output_path = output_path / f"{method}_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_json(data=results, filepath=str(output_path))
    print(f"Feature importance analysis completed. Results saved to {output_path}")


def main() -> None:
    """CLI entry point for feature importance analysis."""

    parser = argparse.ArgumentParser(
        description="Analyze feature importance in STAR blood glucose predictions using mae metric",
    )
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to directory containing patient JSON files",
    )
    parser.add_argument(
        "--output",
        default="output/",
        help="Output directory or file path (default: output/)",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=0.3,
        help="Sensitivity level [0, 1]. <0.5: feature ablation, >=0.5: feature perturbation. Default: 0.3",
    )

    args = parser.parse_args()

    feature_importance_analysis(
        data_path=args.data_path,
        output_path=args.output,
        sensitivity=args.sensitivity,
    )


if __name__ == "__main__":
    main()
