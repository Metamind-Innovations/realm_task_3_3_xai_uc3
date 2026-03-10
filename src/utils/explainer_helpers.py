from tqdm import tqdm
from typing import List, Dict, Any, Literal

from utils.generic_utils import load_json_file


# Attributes that cause the STAR model to hang when perturbed.
# insulinInfusion.route: flipping IV (0) to SubQ (1) across all ICU patients
# triggers a physiologically impossible scenario that the model cannot process.
PROBLEMATIC_PERTURBATION_ATTRIBUTES = {"insulinInfusion.route"}

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


def load_all_patients_data(patient_files: List[str]) -> List[Dict[str, Any]]:
    """
    Load all patient data files.

    :param patient_files: List of file paths to patient JSON files.
    :return: List of successfully loaded patient data dictionaries.
    """

    patients = []

    for filepath in tqdm(patient_files, desc="Loading files"):
        try:
            patient = load_json_file(filepath)
            patients.append(patient)
        except Exception:
            continue

    return patients


def find_method_name(
        sensitivity: float,
) -> Literal["feature_ablation", "feature_perturbation"]:
    """
    Determine analysis method based on sensitivity value.

    :param sensitivity: Sensitivity level between 0 and 1.
    :return: Analysis method name. 'feature_ablation' if sensitivity < 0.5, 'feature_perturbation' otherwise.
    """

    return "feature_ablation" if sensitivity < 0.5 else "feature_perturbation"
