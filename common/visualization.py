import matplotlib.pyplot as plt


def plot_model_evaluation(y_test, y_pred, title="Model Evaluation", save_png=False):
    """
    Plot actual vs predicted values and error distribution.

    :param y_test: True target values
    :type y_test: numpy.ndarray or pandas.Series
    :param y_pred: Predicted values
    :type y_pred: numpy.ndarray
    :param title: Plot title
    :type title: str
    :param save_png: Whether to save plot as PNG
    :type save_png: bool
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.scatter(y_test, y_pred, alpha=0.5)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax1.set_xlabel('Actual')
    ax1.set_ylabel('Predicted')
    ax1.set_title('Actual vs Predicted Values')
    ax1.grid(True, alpha=0.3)

    errors = y_test - y_pred
    ax2.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--')
    ax2.set_xlabel('Prediction Error')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Error Distribution')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_png:
        filename = f'{title.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filename}")

    plt.show()
