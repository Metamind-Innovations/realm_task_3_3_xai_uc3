import argparse
import copy
import json
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, Any, Callable, List, Tuple, Literal
from tqdm import tqdm

from utils.generic_utils import get_json_files, save_json
from utils.data_helpers import extract_prediction_info, calculate_interval_midpoint
from utils.explainer_helpers import (
    find_method_name,
    load_all_patients_data,
    ATTRIBUTES,
)
from STAR_model import STARDockerWrapper


# ============================================================================
# ABLATION FUNCTIONS
# ============================================================================
def ablate_field(data: Dict[str, Any], category: str, field: str) -> Dict[str, Any]:
    """
    Remove a field from all entries in a category.

    :param data: Patient data dictionary.
    :param category: Category name (e.g., 'insulinInfusion').
    :param field: Field name to remove.
    :return: Modified patient data with field removed.
    """

    p = copy.deepcopy(data)
    for item in p["episodes"][0][category]:
        if field in item[1]:
            del item[1][field]
    return p


def ablate_diabetic_status(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove diabetic status field.

    :param data: Patient data dictionary.
    :return: Modified patient data without diabetic status.
    """

    p = copy.deepcopy(data)
    if "diabeticStatus" in p["episodes"][0]:
        del p["episodes"][0]["diabeticStatus"]
    return p


def ablate_blood_glucose(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only 2 BG measurements before prediction time (minimum required).

    :param data: Patient data dictionary.
    :return: Modified patient data with reduced blood glucose measurements.
    """

    p = copy.deepcopy(data)
    bg_data = p["episodes"][0]["bloodGlucose"]
    # Prediction time is last timestep, so keep last 2 measurements before it
    if len(bg_data) > 2:
        # Keep last 3 values
        p["episodes"][0]["bloodGlucose"] = bg_data[-3:]
    return p


def build_ablation_registry() -> Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]:
    """
    Build registry of ablation functions for each attribute.

    :return: Dictionary mapping attribute names to ablation functions.
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
    """
    Build registry of perturbation strategies (type, magnitude).

    :return: Dictionary mapping attribute names to (perturbation_type, parameter) tuples.
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
    """
    Perturb continuous attribute by percentage with bounds checking.

    :param data: Patient data dictionary.
    :param attr: Attribute name (e.g., 'bloodGlucose.value').
    :param magnitude: Perturbation magnitude as decimal (e.g., 0.25 for 25%).
    :return: Modified patient data with perturbed values.
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
    """
    Flip binary attribute.

    :param data: Patient data dictionary.
    :param attr: Attribute name.
    :return: Modified patient data with flipped binary value.
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
    """
    Change categorical attribute to different value.

    :param data: Patient data dictionary.
    :param attr: Attribute name.
    :param values: List of possible categorical values.
    :return: Modified patient data with changed categorical value.
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
def compute_mae(
        patients_data: List[Dict[str, Any]],
        star_docker: STARDockerWrapper,
        temp_dir: Path,
) -> float:
    """
    Compute Mean Absolute Error across all patients using batch prediction.

    :param patients_data: List of patient data dictionaries.
    :param star_docker: STAR Docker wrapper instance.
    :param temp_dir: Temporary directory for storing patient files.
    :return: Mean absolute error across all successfully processed patients.
    """

    temp_dir.mkdir(parents=True, exist_ok=True)

    patient_files = []
    for idx, patient_data in enumerate(patients_data):
        temp_file = temp_dir / f"patient_{idx}.json"
        with open(temp_file, 'w') as f:
            json.dump(patient_data, f)
        patient_files.append(str(temp_file))

    predictions_df = star_docker.predict_batch(patient_files)

    aes = []
    for idx, patient_data in enumerate(patients_data):
        try:
            pred_time, actual_value = extract_prediction_info(patient_data)
            predicted_midpoint = calculate_interval_midpoint({
                "BG5TH": predictions_df.iloc[idx]["BG5TH"],
                "BG95TH": predictions_df.iloc[idx]["BG95TH"]
            })
            ae = abs(actual_value - predicted_midpoint)
            aes.append(ae)
        except Exception:
            continue

    for pf in patient_files:
        try:
            Path(pf).unlink(missing_ok=True)
        except Exception:
            pass

    if not aes:
        return 0.0

    mae = np.mean(aes)

    return mae


# ============================================================================
# FEATURE ABLATION - FEATURE PERTURBATION ANALYSIS
# ============================================================================
def analyze_feature_importance(
        patients_data: List[Dict[str, Any]],
        star_docker: STARDockerWrapper,
        analysis_type: Literal["feature_ablation", "feature_perturbation"],
        temp_dir: Path,
) -> Dict[str, float]:
    """
    Analyze feature importance using ablation or perturbation.

    :param patients_data: List of patient data dictionaries.
    :param star_docker: STAR Docker wrapper instance.
    :param analysis_type: Type of analysis.
    :param temp_dir: Temporary directory for storing patient files.
    :return: Dictionary mapping attribute names to normalized importance scores (0-1).
    """

    baseline_mae = compute_mae(patients_data, star_docker=star_docker, temp_dir=temp_dir / "baseline")

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
            except Exception:
                continue

        # Compute MAE after transformation
        transformed_mae = compute_mae(
            patients_data=transformed_data,
            star_docker=star_docker,
            temp_dir=temp_dir / f"transformed_{attr_name.replace('.', '_')}"
        )

        # Feature importance
        # Positive = INCREASE in MAE (higher MAE = worse predictions, feature is good for the model)
        # Negative = DECREASE in MAE (lower MAE = better predictions, feature is not good for the model)
        mae_diff = transformed_mae - baseline_mae
        feature_importance[attr_name] = mae_diff

    # Sort by importance
    feature_importance = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    )

    # Normalize by sum of absolute values
    total_abs_importance = sum(abs(value) for value in feature_importance.values())

    if total_abs_importance > 0:
        feature_importance = {
            key: (value / total_abs_importance)
            for key, value in feature_importance.items()
        }
    else:
        # All differences are zero
        feature_importance = {key: 0.0 for key in feature_importance.keys()}

    return feature_importance


def feature_importance_analysis(
        data_path: str,
        output_path: str,
        sensitivity: float,
        docker_image: str = "glucomeo",
        in_docker_run: bool = False,
) -> None:
    """
    Run feature importance analysis on patient data.

    :param data_path: Path to directory containing patient JSON files.
    :param output_path: Output directory path for results.
    :param sensitivity: Sensitivity level [0, 1]. <0.5: ablation, >=0.5: perturbation.
    :param docker_image: Docker image name for STAR model.
    :param in_docker_run: Whether running inside Docker container.
    :raises ValueError: If sensitivity not in [0, 1] or no files found.
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

    star_docker = STARDockerWrapper(docker_image=docker_image, in_docker_run=in_docker_run)

    temp_dir = Path(output_path) / "temp_explainer"
    temp_dir.mkdir(parents=True, exist_ok=True)

    results = analyze_feature_importance(
        patients_data, star_docker=star_docker, analysis_type=method, temp_dir=temp_dir
    )

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    output_path = Path(output_path)
    output_path = output_path / f"{method}_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_json(data=results, filepath=str(output_path))
    print(f"Feature importance analysis completed. Results saved to {output_path}")


def main() -> None:
    """
    CLI entry point for feature importance analysis.
    """
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
    parser.add_argument(
        "--docker_image",
        default="glucomeo",
        help="Docker image name for STAR model (default: glucomeo)",
    )
    parser.add_argument(
        "--in_docker_run",
        action="store_true",
        default=False,
        help="Whether running inside Docker container (default: False)",
    )

    args = parser.parse_args()

    feature_importance_analysis(
        data_path=args.data_path,
        output_path=args.output,
        sensitivity=args.sensitivity,
        docker_image=args.docker_image,
        in_docker_run=args.in_docker_run,
    )


if __name__ == "__main__":
    main()
