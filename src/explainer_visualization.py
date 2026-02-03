from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from utils.generic_utils import load_json_file
from utils.explainer_helpers import find_method_name


def plot_importances(
        features: List[str],
        importances: List[float],
        title: str,
        xlabel: str,
        ylabel: str,
        explanation_text: str,
        output_path: str,
        figsize: tuple = (8, 4),
        show: bool = False,
) -> None:
    """
    Plot horizontal bar chart of feature importances with annotations.

    :param features: Feature names.
    :param importances: Importance values.
    :param title: Plot title.
    :param xlabel: X-axis label.
    :param ylabel: Y-axis label.
    :param explanation_text: Explanatory text below plot.
    :param output_path: File path to save plot.
    :param figsize: Figure size in inches. Defaults to (8, 4).
    :param show: Whether to display plot. Defaults to False.
    """
    plt.figure(figsize=figsize)

    colors = [
        "steelblue" if v > 0 else "indianred" if v < 0 else "gray" for v in importances
    ]
    bars = plt.barh(features, importances, color=colors)

    plt.gca().invert_yaxis()

    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.tick_params(axis="both", which="major", labelsize=7)
    plt.axvline(0, color="black", linewidth=0.2, linestyle="--")

    xmin = min(importances)
    xmax = max(importances)
    x_range = xmax - xmin if xmax != xmin else 1

    xlim_min = xmin - (x_range * 0.12)
    xlim_max = xmax + (x_range * 0.12)

    plt.xlim(xlim_min, xlim_max)

    for bar, val in zip(bars, importances):
        if val >= 0:
            text_x = val + (x_range * 0.01)
            text_ha = "left"
        else:
            text_x = val - (x_range * 0.01)
            text_ha = "right"

        plt.text(
            text_x,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha=text_ha,
            fontsize=7,
        )

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.figtext(
        0.5,
        0.02,
        explanation_text,
        ha="center",
        fontsize=7,
        style="italic",
        color="#555555",
        wrap=True,
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def visualize_feature_importance(
        analysis_results: str, output_dir: str, sensitivity: float
) -> None:
    """
    Visualize feature importance analysis results and save plot.

    :param analysis_results: Path to analysis JSON file.
    :param output_dir: Output directory for plot.
    :param sensitivity: Sensitivity level [0, 1].
    :raises ValueError: If sensitivity not in [0, 1].
    """

    # Load results
    analysis_results = load_json_file(str(analysis_results))

    # Validate sensitivity
    if not 0 <= sensitivity <= 1:
        raise ValueError("Sensitivity must be between 0 and 1")

    # Determine method from sensitivity
    method = find_method_name(sensitivity)
    method_modified = method.replace("_", " ").title()

    # Output path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir.joinpath(f"{method}_importance.png"))

    # Convert dict to sorted dataframe
    df = pd.DataFrame(list(analysis_results.items()), columns=["Feature", "Importance"])
    df.sort_values(by="Importance", ascending=False, inplace=True)

    features = [f.replace(".", " ") for f in df["Feature"].to_list()]
    importances = df["Importance"].to_list()

    # Set explanation text based on method
    if method == "feature_ablation":
        explanation_text = (
            "Feature Ablation: Shows impact when features are removed. "
            "Higher positive values indicate features that are more important for accurate predictions. "
            "Lower negative values indicate features that are not important for model predictions. "
            "Values range between [-1, 1]."
        )
    elif method == "feature_perturbation":
        explanation_text = (
            "Feature Perturbation: Shows impact when features are modified. "
            "Higher positive values indicate features that are more important for accurate predictions. "
            "Lower negative values indicate features that are not important for model predictions. "
            "Values range between [-1, 1]."
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Create title
    title = f"Sensitivity [0, 1]: {sensitivity:.2f} | Method: {method_modified}"

    # Generate plot
    plot_importances(
        features=features,
        importances=importances,
        title=title,
        xlabel="Importance Score",
        ylabel="Feature",
        explanation_text=explanation_text,
        output_path=output_path,
    )

    print(f"Plot saved to {output_path}")


def main() -> None:
    """
    CLI entry point for visualizing feature importance analysis results.
    """
    parser = argparse.ArgumentParser(
        description="Visualize feature importance analysis results for STAR blood glucose predictions"
    )
    parser.add_argument(
        "--analysis_results",
        required=True,
        help="Path to analysis JSON file",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=0.3,
        help="Sensitivity level [0, 1]. <0.5: feature ablation, >=0.5: feature perturbation. Default: 0.3",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Output directory for visualizations (default: output)",
    )

    args = parser.parse_args()

    visualize_feature_importance(
        analysis_results=args.analysis_results,
        output_dir=args.output,
        sensitivity=args.sensitivity,
    )


if __name__ == "__main__":
    main()
