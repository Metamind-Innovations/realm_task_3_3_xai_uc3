from tqdm import tqdm
from typing import List, Dict, Any, Literal

from utils.generic_utils import load_json_file


ATTRIBUTES = [
    "diabeticStatus",
    "bloodGlucose.value",
    "insulinInfusion.route",
    "insulinInfusion.rate",
    "insulinInfusion.fixed",
    "insulinInfusion.concentration",
    "insulinBolus.route",
    "insulinBolus.size",
    "insulinBolus.adminTime",
    "insulinBolus.concentration",
    "nutritionInfusion.type",
    "nutritionInfusion.carbsConcentration",
    "nutritionInfusion.rate",
    "nutritionInfusion.fixed",
    "nutritionBolus.isMeal",
    "nutritionBolus.size",
    "nutritionBolus.adminTime",
    "nutritionBolus.type",
    "nutritionBolus.carbsConcentration",
]

MAX_WORKERS = 10


def load_all_patients_data(patient_files: List[str]) -> List[Dict[str, Any]]:
    """Load all patient data files.

    Args:
        patient_files (List[str]): List of file paths to patient JSON files.

    Returns:
        List[Dict[str, Any]]: List of successfully loaded patient data dictionaries.
    """

    patients = []

    for filepath in tqdm(patient_files, desc="Loading files"):
        try:
            patient = load_json_file(filepath)
            patients.append(patient)
        except Exception as e:
            continue

    return patients


def find_method_name(
    sensitivity: float,
) -> Literal["feature_ablation", "feature_perturbation"]:
    """Determine analysis method based on sensitivity value.

    Args:
        sensitivity (float): Sensitivity level between 0 and 1.

    Returns:
        Literal["feature_ablation", "feature_perturbation"]: Analysis method name.
            'feature_ablation' if sensitivity < 0.5, 'feature_perturbation' otherwise.
    """

    return "feature_ablation" if sensitivity < 0.5 else "feature_perturbation"
