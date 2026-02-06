import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from utils.generic_utils import load_json_file
from typing import Dict, Tuple, List

AGE_COLORS = {
    "<40": "#66C2A5",
    "40-54": "#FC8D62",
    "55-69": "#8DA0CB",
    "70+": "#E78AC3",
}

AGE_GROUPS = ["<40", "40-54", "55-69", "70+"]

GENDER_COLORS = {"Female": "#FF69B4", "Male": "#4169E1"}


def demographic_names(data: Dict) -> List[Tuple[str, str]]:
    """
    Extract all available demographic names from a fairness JSON.

    :param data: Results JSON containing 'equalized_odds_metrics' and 'demographic_parity_metrics'.
    :return: List of demographic names.
    """

    demographic_names = list(data.get("equalized_odds_metrics").keys())

    return demographic_names


def plot_consolidated_chart(
        data: Dict, demographic_name: str, output_dir: Path
) -> None:
    """
    Create consolidated 1x2 bar chart for fairness and bias metrics.

    :param data: Full analysis results
    :param demographic_name: Demographic to visualize (e.g., 'Age', 'Gender')
    :param output_dir: Directory to save the plot
    """

    fairness_method_display = "Equalized Odds"
    bias_method_display = "Demographic Parity"
    demographic_name_display = demographic_name.capitalize()

    # Determine color palette
    color_map = AGE_COLORS if demographic_name == "age" else GENDER_COLORS

    # Prepare data structures
    miscov_categories = []
    miscov_values = []

    mpred_categories = []
    mpred_values = []

    # Extract miscoverage data (Equalized Odds)
    if demographic_name in data.get("equalized_odds_metrics", {}):
        miscoverage_rates = data["equalized_odds_metrics"][demographic_name].get(
            "miscoverage_rates_by_group", {}
        )
        for categ, val in miscoverage_rates.items():
            miscov_categories.append(categ)
            miscov_values.append(val)

    # Extract mean predicted values data (Demographic Parity)
    if demographic_name in data.get("demographic_parity_metrics", {}):
        mean_pred_values = data["demographic_parity_metrics"][demographic_name].get(
            "mean_predicted_values_by_group", {}
        )
        for categ, val in mean_pred_values.items():
            mpred_categories.append(categ)
            mpred_values.append(val)

    # Create figure with 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Fairness and Bias Analysis by {demographic_name_display}",
        fontsize=16,
        fontweight="bold",
    )

    # Left Plot: Miscoverage Rate (Fairness)
    if miscov_values:
        miscov_colors = [color_map.get(cat, "#4ECDC4") for cat in miscov_categories]
        bars = axes[0].bar(miscov_categories, miscov_values, color=miscov_colors)

        axes[0].set_title(
            f"Fairness Calculation using {fairness_method_display}",
            fontsize=12,
            fontweight="bold",
        )
        axes[0].set_xlabel(demographic_name_display)
        axes[0].set_ylabel("Miscoverage Rate")

        ymax = max(miscov_values) * 1.15 if miscov_values else 1
        axes[0].set_ylim(0, ymax)

        # Annotate bars with values
        for bar, val in zip(bars, miscov_values):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                val + (ymax * 0.01),
                f"{val:.2f}",
                ha="center",
                va="bottom",
            )

    # Right Plot: Mean Predicted Values (Bias)
    if mpred_values:
        mpred_colors = [color_map.get(cat, "#4ECDC4") for cat in mpred_categories]
        bars = axes[1].bar(mpred_categories, mpred_values, color=mpred_colors)

        axes[1].set_title(
            f"Bias Calculation using {bias_method_display}",
            fontsize=12,
            fontweight="bold",
        )
        axes[1].set_xlabel(demographic_name_display)
        axes[1].set_ylabel("Mean Predicted Value")

        ymax = max(mpred_values) * 1.15 if mpred_values else 20
        axes[1].set_ylim(0, ymax)

        # Annotate bars with values
        for bar, val in zip(bars, mpred_values):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                val + (ymax * 0.01),
                f"{val:.2f}",
                ha="center",
                va="bottom",
            )

    # Add explanatory notes
    fig.text(
        0.25,
        0.02,
        "Shows miscoverage rates across demographic groups. Lower is better.",
        ha="center",
        fontsize=10,
        style="italic",
        color="#555555",
    )

    fig.text(
        0.75,
        0.02,
        "Shows the model's mean predicted values across demographic groups. Similar values across groups indicate less bias.",
        ha="center",
        fontsize=10,
        style="italic",
        color="#555555",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.98])

    # Save plot
    filename = f"{demographic_name.lower()}_fairness_bias.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()


def visualize_fairness_bias_analysis(analysis_results: str, output_dir: str) -> None:
    """
    Generate fairness analysis visualization plots.

    :param analysis_results: Path to JSON file with fairness analysis results.
    :param output_dir: Directory path to save generated plots.
    """
    # Load data
    analysis_results = load_json_file(analysis_results)

    # Create dir to store plots
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Search for demographic names
    demogr_names = demographic_names(analysis_results)

    # Create consolidated plot for each demographic
    for dem_name in demogr_names:
        plot_consolidated_chart(
            data=analysis_results, demographic_name=dem_name, output_dir=output_dir
        )

    print(f"Plots stored in {str(output_dir)}")


def main() -> None:
    """
    Main entry point for visualizing fairness and bias analysis results.
    """

    parser = argparse.ArgumentParser(
        description="Visualize fairness and bias analysis results for STAR model predictions"
    )
    parser.add_argument(
        "--analysis_results",
        required=True,
        help="Analysis JSON file path",
    )
    parser.add_argument(
        "--output", default="output", help="Output dir for visualizations"
    )

    args = parser.parse_args()

    visualize_fairness_bias_analysis(
        analysis_results=args.analysis_results, output_dir=args.output
    )


if __name__ == "__main__":
    main()
