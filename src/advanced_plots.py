import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


RESULTS_FILE = "results/model_comparison.csv"
PLOTS_DIRECTORY = "results/plots"

RANDOM_FOREST_MODEL = "models/random_forest.pkl"
XGBOOST_MODEL = "models/xgboost.pkl"

FEATURE_NAMES = [
    "QBER",
    "Noise Level",
    "Photon Loss Rate",
    "Detection Rate",
    "Plus Basis QBER",
    "Cross Basis QBER",
    "QBER Variation",
]


def find_column(dataframe, possible_names):
    """
    Return the first matching column name.
    """

    for column_name in possible_names:
        if column_name in dataframe.columns:
            return column_name

    raise ValueError(
        f"Could not find any of these columns: {possible_names}\n"
        f"Available columns: {dataframe.columns.tolist()}"
    )


def load_results():
    """
    Load the final model comparison results.
    """

    if not os.path.exists(RESULTS_FILE):
        raise FileNotFoundError(
            f"Could not find {RESULTS_FILE}. "
            "Run train.py before running this file."
        )

    dataframe = pd.read_csv(RESULTS_FILE)

    print("Loaded result columns:")
    print(dataframe.columns.tolist())

    return dataframe


# ==========================================================
# 1. Confusion matrices
# ==========================================================

def plot_confusion_matrices(dataframe):
    """
    Create one confusion matrix for every model.
    """

    model_column = find_column(
        dataframe,
        ["Model", "model"],
    )

    tn_column = find_column(
        dataframe,
        ["TN", "True Negatives"],
    )

    fp_column = find_column(
        dataframe,
        ["FP", "False Positives"],
    )

    fn_column = find_column(
        dataframe,
        ["FN", "False Negatives"],
    )

    tp_column = find_column(
        dataframe,
        ["TP", "True Positives"],
    )

    for _, row in dataframe.iterrows():

        model_name = str(row[model_column])

        confusion_matrix = np.array([
            [
                int(row[tn_column]),
                int(row[fp_column]),
            ],
            [
                int(row[fn_column]),
                int(row[tp_column]),
            ],
        ])

        figure, axis = plt.subplots(
            figsize=(6, 5)
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix,
            display_labels=[
                "Normal",
                "Attack",
            ],
        )

        display.plot(
            ax=axis,
            values_format=",d",
            colorbar=False,
        )

        axis.set_title(
            f"{model_name} Confusion Matrix"
        )

        plt.tight_layout()

        safe_model_name = (
            model_name
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        output_path = os.path.join(
            PLOTS_DIRECTORY,
            f"{safe_model_name}_confusion_matrix.png",
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(f"Saved: {output_path}")


# ==========================================================
# 2. False positives vs false negatives
# ==========================================================

def plot_fp_vs_fn(dataframe):
    """
    Compare false positives and false negatives.
    """

    model_column = find_column(
        dataframe,
        ["Model", "model"],
    )

    fp_column = find_column(
        dataframe,
        ["FP", "False Positives"],
    )

    fn_column = find_column(
        dataframe,
        ["FN", "False Negatives"],
    )

    model_names = dataframe[model_column].tolist()

    false_positives = dataframe[fp_column].tolist()
    false_negatives = dataframe[fn_column].tolist()

    x_positions = np.arange(
        len(model_names)
    )

    bar_width = 0.35

    plt.figure(figsize=(11, 6))

    fp_bars = plt.bar(
        x_positions - bar_width / 2,
        false_positives,
        width=bar_width,
        label="False Positives",
    )

    fn_bars = plt.bar(
        x_positions + bar_width / 2,
        false_negatives,
        width=bar_width,
        label="False Negatives",
    )

    plt.bar_label(
        fp_bars,
        padding=3,
        fmt="%d",
    )

    plt.bar_label(
        fn_bars,
        padding=3,
        fmt="%d",
    )

    plt.xticks(
        x_positions,
        model_names,
        rotation=15,
    )

    plt.title(
        "False Positive vs False Negative Comparison"
    )

    plt.xlabel("Machine Learning Model")
    plt.ylabel("Number of Errors")

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(
        PLOTS_DIRECTORY,
        "false_positive_vs_false_negative.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


# ==========================================================
# 3. Feature importance
# ==========================================================

def get_feature_importances(model):
    """
    Extract feature importance values from a saved model.
    """

    if hasattr(model, "feature_importances_"):
        return model.feature_importances_

    if hasattr(model, "named_steps"):

        final_model = list(
            model.named_steps.values()
        )[-1]

        if hasattr(
            final_model,
            "feature_importances_",
        ):
            return final_model.feature_importances_

    raise ValueError(
        "This model does not provide feature_importances_."
    )


def plot_single_feature_importance(
    model_path,
    model_name,
):
    """
    Create a feature-importance graph for one model.
    """

    if not os.path.exists(model_path):
        print(
            f"Skipped {model_name}: "
            f"{model_path} was not found."
        )
        return

    model = joblib.load(model_path)

    importances = get_feature_importances(
        model
    )

    if len(importances) != len(FEATURE_NAMES):
        raise ValueError(
            f"{model_name} has {len(importances)} features, "
            f"but FEATURE_NAMES contains "
            f"{len(FEATURE_NAMES)} names."
        )

    importance_dataframe = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Importance": importances,
    })

    importance_dataframe = (
        importance_dataframe
        .sort_values(
            "Importance",
            ascending=True,
        )
    )

    plt.figure(figsize=(9, 6))

    bars = plt.barh(
        importance_dataframe["Feature"],
        importance_dataframe["Importance"],
    )

    plt.bar_label(
        bars,
        padding=3,
        fmt="%.3f",
    )

    plt.title(
        f"{model_name} Feature Importance"
    )

    plt.xlabel("Importance Score")
    plt.ylabel("BB84 Feature")

    plt.grid(
        axis="x",
        alpha=0.3,
    )

    plt.tight_layout()

    safe_model_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    output_path = os.path.join(
        PLOTS_DIRECTORY,
        f"{safe_model_name}_feature_importance.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


def plot_feature_importances():
    """
    Create Random Forest and XGBoost feature-importance graphs.
    """

    plot_single_feature_importance(
        model_path=RANDOM_FOREST_MODEL,
        model_name="Random Forest",
    )

    plot_single_feature_importance(
        model_path=XGBOOST_MODEL,
        model_name="XGBoost",
    )


# ==========================================================
# 4. Final summary dashboard
# ==========================================================

def plot_summary_dashboard(dataframe):
    """
    Create a four-panel model performance dashboard.
    """

    model_column = find_column(
        dataframe,
        ["Model", "model"],
    )

    accuracy_column = find_column(
        dataframe,
        ["Accuracy", "accuracy"],
    )

    precision_column = find_column(
        dataframe,
        ["Precision", "precision"],
    )

    recall_column = find_column(
        dataframe,
        ["Recall", "recall"],
    )

    f1_column = find_column(
        dataframe,
        ["F1", "F1 Score", "f1"],
    )

    fp_column = find_column(
        dataframe,
        ["FP", "False Positives"],
    )

    fn_column = find_column(
        dataframe,
        ["FN", "False Negatives"],
    )

    training_time_column = find_column(
        dataframe,
        [
            "Training Time (s)",
            "Training Time",
            "Train Time",
        ],
    )

    model_names = dataframe[
        model_column
    ].tolist()

    x_positions = np.arange(
        len(model_names)
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(15, 11),
    )

    # Panel 1: Main model metrics

    metrics = [
        accuracy_column,
        precision_column,
        recall_column,
        f1_column,
    ]

    metric_labels = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
    ]

    bar_width = 0.2

    for metric_number, metric in enumerate(metrics):

        positions = (
            x_positions
            + metric_number * bar_width
        )

        axes[0, 0].bar(
            positions,
            dataframe[metric],
            width=bar_width,
            label=metric_labels[metric_number],
        )

    axes[0, 0].set_xticks(
        x_positions + bar_width * 1.5
    )

    axes[0, 0].set_xticklabels(
        model_names,
        rotation=15,
    )

    axes[0, 0].set_title(
        "Model Performance Metrics"
    )

    axes[0, 0].set_ylabel("Score")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(
        axis="y",
        alpha=0.3,
    )

    # Panel 2: FP and FN

    axes[0, 1].bar(
        x_positions - 0.18,
        dataframe[fp_column],
        width=0.36,
        label="False Positives",
    )

    axes[0, 1].bar(
        x_positions + 0.18,
        dataframe[fn_column],
        width=0.36,
        label="False Negatives",
    )

    axes[0, 1].set_xticks(
        x_positions
    )

    axes[0, 1].set_xticklabels(
        model_names,
        rotation=15,
    )

    axes[0, 1].set_title(
        "Classification Errors"
    )

    axes[0, 1].set_ylabel(
        "Number of Errors"
    )

    axes[0, 1].legend()
    axes[0, 1].grid(
        axis="y",
        alpha=0.3,
    )

    # Panel 3: Training time

    time_bars = axes[1, 0].bar(
        model_names,
        dataframe[training_time_column],
    )

    axes[1, 0].bar_label(
        time_bars,
        padding=3,
        fmt="%.1f",
    )

    axes[1, 0].set_title(
        "Model Training Time"
    )

    axes[1, 0].set_xlabel(
        "Machine Learning Model"
    )

    axes[1, 0].set_ylabel(
        "Training Time (seconds)"
    )

    axes[1, 0].tick_params(
        axis="x",
        rotation=15,
    )

    axes[1, 0].grid(
        axis="y",
        alpha=0.3,
    )

    # Panel 4: F1 score vs training time

    for _, row in dataframe.iterrows():

        axes[1, 1].scatter(
            row[training_time_column],
            row[f1_column],
            s=100,
        )

        axes[1, 1].annotate(
            row[model_column],
            (
                row[training_time_column],
                row[f1_column],
            ),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
        )

    axes[1, 1].set_title(
        "Performance vs Training Cost"
    )

    axes[1, 1].set_xlabel(
        "Training Time (seconds)"
    )

    axes[1, 1].set_ylabel(
        "F1 Score"
    )

    axes[1, 1].grid(
        alpha=0.3,
    )

    figure.suptitle(
        "BB84 AI Intrusion Detection: Final Model Summary",
        fontsize=16,
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    output_path = os.path.join(
        PLOTS_DIRECTORY,
        "final_summary_dashboard.png",
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
    Generate all advanced result visualizations.
    """

    os.makedirs(
        PLOTS_DIRECTORY,
        exist_ok=True,
    )

    dataframe = load_results()

    plot_confusion_matrices(
        dataframe
    )

    plot_fp_vs_fn(
        dataframe
    )

    plot_feature_importances()

    plot_summary_dashboard(
        dataframe
    )

    print(
        "\nAll advanced plots were created successfully."
    )


if __name__ == "__main__":
    main()