import os

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_FILE = "results/learning_curve.csv"
PLOTS_DIRECTORY = "results/plots"


def format_training_size(value):
    """
    Convert large training sizes into readable labels.
    """

    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"

    return f"{value / 1_000:g}K"


def plot_metric(
    dataframe,
    metric,
    ylabel,
    title,
    filename,
):
    """
    Create and save one learning-curve graph.
    """

    plt.figure(figsize=(9, 6))

    training_sizes = sorted(
        dataframe["Training Size"].unique()
    )

    for model_name in dataframe["Model"].unique():

        model_data = dataframe[
            dataframe["Model"] == model_name
        ].sort_values("Training Size")

        plt.plot(
            model_data["Training Size"],
            model_data[metric],
            marker="o",
            linewidth=2,
            markersize=6,
            label=model_name,
        )

    plt.title(title)
    plt.xlabel("Training Samples")
    plt.ylabel(ylabel)

    plt.xticks(
        training_sizes,
        [
            format_training_size(size)
            for size in training_sizes
        ],
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(
        PLOTS_DIRECTORY,
        filename,
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


def plot_performance_vs_time(dataframe):
    """
    Plot F1 score against training time for the largest training size.
    """

    largest_training_size = dataframe["Training Size"].max()

    final_results = dataframe[
        dataframe["Training Size"] == largest_training_size
    ].copy()

    plt.figure(figsize=(9, 6))

    for _, row in final_results.iterrows():

        plt.scatter(
            row["Training Time (s)"],
            row["F1 Score"],
            s=120,
        )

        plt.annotate(
            row["Model"],
            (
                row["Training Time (s)"],
                row["F1 Score"],
            ),
            xytext=(8, 6),
            textcoords="offset points",
        )

    plt.title(
        "F1 Score vs Training Time "
        f"({format_training_size(largest_training_size)} Training Samples)"
    )

    plt.xlabel("Training Time (seconds)")
    plt.ylabel("F1 Score")

    plt.grid(True)
    plt.tight_layout()

    output_path = os.path.join(
        PLOTS_DIRECTORY,
        "f1_score_vs_training_time.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


def plot_final_metric_comparison(dataframe):
    """
    Compare final Accuracy, Precision, Recall, and F1 Score.
    """

    largest_training_size = dataframe["Training Size"].max()

    final_results = dataframe[
        dataframe["Training Size"] == largest_training_size
    ].copy()

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
    ]

    model_names = final_results["Model"].tolist()

    x_positions = list(
        range(len(model_names))
    )

    bar_width = 0.2

    plt.figure(figsize=(11, 6))

    for metric_number, metric in enumerate(metrics):

        positions = [
            position + metric_number * bar_width
            for position in x_positions
        ]

        plt.bar(
            positions,
            final_results[metric],
            width=bar_width,
            label=metric,
        )

    center_positions = [
        position + bar_width * 1.5
        for position in x_positions
    ]

    plt.xticks(
        center_positions,
        model_names,
        rotation=15,
    )

    plt.title(
        "Final Model Performance Comparison "
        f"({format_training_size(largest_training_size)} Training Samples)"
    )

    plt.xlabel("Machine Learning Model")
    plt.ylabel("Score")

    plt.ylim(
        final_results[metrics].min().min() - 0.001,
        final_results[metrics].max().max() + 0.001,
    )

    plt.grid(
        axis="y",
        alpha=0.4,
    )

    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(
        PLOTS_DIRECTORY,
        "final_model_metric_comparison.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


def main():
    """
    Generate all learning-curve plots.
    """

    os.makedirs(
        PLOTS_DIRECTORY,
        exist_ok=True,
    )

    dataframe = pd.read_csv(
        RESULTS_FILE
    )

    plot_metric(
        dataframe=dataframe,
        metric="F1 Score",
        ylabel="F1 Score",
        title="F1 Score vs Training Size",
        filename="f1_score_vs_training_size.png",
    )

    plot_metric(
        dataframe=dataframe,
        metric="Accuracy",
        ylabel="Accuracy",
        title="Accuracy vs Training Size",
        filename="accuracy_vs_training_size.png",
    )

    plot_metric(
        dataframe=dataframe,
        metric="Training Time (s)",
        ylabel="Training Time (seconds)",
        title="Training Time vs Training Size",
        filename="training_time_vs_training_size.png",
    )

    plot_performance_vs_time(dataframe)

    plot_final_metric_comparison(dataframe)

    print("\nAll plots were created successfully.")


if __name__ == "__main__":
    main()