"""
Efficiently train five machine-learning models for BB84 attack detection.

Models:
1. Decision Tree
2. Random Forest
3. Logistic Regression using SGD
4. Linear SVM
5. XGBoost

The script:
- Loads only required columns.
- Uses memory-efficient data types.
- Excludes interception_probability to prevent data leakage.
- Trains one model at a time.
- Records training and prediction time.
- Saves each trained model.
- Saves partial results after every model.
"""

import gc
import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier


# ==========================================================
# Configuration
# ==========================================================

DATASET_PATH = "datasets/bb84_dataset_10M.csv"

MODELS_DIRECTORY = "models"
RESULTS_DIRECTORY = "results"

RESULTS_FILE = "results/model_comparison.csv"
METADATA_FILE = "results/training_metadata.json"

RANDOM_STATE = 42


# ----------------------------------------------------------
# Training mode
# ----------------------------------------------------------
#
# True:
#   Fast test using 500,000 training rows.
#
# False:
#   Final experiment using 8,000,000 training rows.
#
# Start with True. Change it to False only after confirming
# that every model runs successfully.
# ----------------------------------------------------------

QUICK_MODE = False

if QUICK_MODE:
    TRAIN_ROWS = 500_000
    TEST_ROWS = 100_000

    DECISION_TREE_DEPTH = 12
    RANDOM_FOREST_TREES = 25
    XGBOOST_TREES = 100

else:
    TRAIN_ROWS = 8_000_000
    TEST_ROWS = 2_000_000

    DECISION_TREE_DEPTH = 15
    RANDOM_FOREST_TREES = 100
    XGBOOST_TREES = 300


FEATURE_COLUMNS = [
    "qber",
    "noise_level",
    "photon_loss_rate",
    "detection_rate",
    "plus_basis_qber",
    "cross_basis_qber",
    "qber_variation",
]

LABEL_COLUMN = "label"


# Use smaller data types to reduce RAM usage.

DTYPES = {
    "qber": "float32",
    "noise_level": "float32",
    "photon_loss_rate": "float32",
    "detection_rate": "float32",
    "plus_basis_qber": "float32",
    "cross_basis_qber": "float32",
    "qber_variation": "float32",
    "label": "int8",
}


# ==========================================================
# Dataset loading
# ==========================================================

def load_dataset():
    """
    Load and prepare the training and testing data.

    Returns:
        x_train: Training features.
        x_test: Testing features.
        y_train: Training labels.
        y_test: Testing labels.
        load_time: Time required to load the dataset.
    """

    total_rows_needed = TRAIN_ROWS + TEST_ROWS

    print("=" * 72)
    print("LOADING BB84 DATASET")
    print("=" * 72)

    mode_name = "QUICK MODE" if QUICK_MODE else "FINAL MODE"

    print(f"Mode          : {mode_name}")
    print(f"Dataset       : {DATASET_PATH}")
    print(f"Rows requested: {total_rows_needed:,}")

    start_time = time.perf_counter()

    try:
        dataframe = pd.read_csv(
            DATASET_PATH,
            usecols=FEATURE_COLUMNS + [LABEL_COLUMN],
            dtype=DTYPES,
            nrows=total_rows_needed,
        )

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset was not found: {DATASET_PATH}"
        )

    except ValueError as error:
        raise ValueError(
            "The dataset does not contain all required columns.\n"
            f"Required columns: {FEATURE_COLUMNS + [LABEL_COLUMN]}"
        ) from error

    load_time = time.perf_counter() - start_time

    if len(dataframe) < total_rows_needed:
        raise ValueError(
            f"Only {len(dataframe):,} rows were loaded, but "
            f"{total_rows_needed:,} rows were requested."
        )

    memory_mb = (
        dataframe.memory_usage(deep=True).sum()
        / 1024 ** 2
    )

    print(f"Rows loaded   : {len(dataframe):,}")
    print(f"Load time     : {load_time:.2f} seconds")
    print(f"Memory used   : {memory_mb:.2f} MB")

    print("\nShuffling dataset...")

    dataframe = dataframe.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    training_data = dataframe.iloc[:TRAIN_ROWS]

    testing_data = dataframe.iloc[
        TRAIN_ROWS:TRAIN_ROWS + TEST_ROWS
    ]

    x_train = training_data[FEATURE_COLUMNS].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    y_train = training_data[LABEL_COLUMN].to_numpy(
        dtype=np.int8,
        copy=True,
    )

    x_test = testing_data[FEATURE_COLUMNS].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    y_test = testing_data[LABEL_COLUMN].to_numpy(
        dtype=np.int8,
        copy=True,
    )

    del dataframe
    del training_data
    del testing_data

    gc.collect()

    print(f"Training rows : {len(x_train):,}")
    print(f"Testing rows  : {len(x_test):,}")

    print("\nTraining label counts:")

    unique_labels, label_counts = np.unique(
        y_train,
        return_counts=True,
    )

    for label, count in zip(unique_labels, label_counts):
        label_name = "Attack" if label == 1 else "Normal"

        print(
            f"  {label_name}: "
            f"{count:,}"
        )

    return (
        x_train,
        x_test,
        y_train,
        y_test,
        load_time,
    )


# ==========================================================
# Model creation
# ==========================================================

def create_model(model_name):
    """
    Create and return one machine-learning model.

    Args:
        model_name: Name of the model to create.

    Returns:
        Configured machine-learning model.
    """

    if model_name == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=DECISION_TREE_DEPTH,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
        )

    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=RANDOM_FOREST_TREES,
            max_depth=DECISION_TREE_DEPTH,
            min_samples_leaf=5,
            max_samples=0.7,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=0,
        )

    if model_name == "Logistic Regression":
        return Pipeline([
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=0.0001,
                    max_iter=100,
                    tol=1e-3,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=5,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ])

    if model_name == "Linear SVM":
        return Pipeline([
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LinearSVC(
                    C=1.0,

                    # dual=False is faster because this project has
                    # many more training rows than features.
                    dual=False,

                    max_iter=3000,
                    tol=1e-4,
                    random_state=RANDOM_STATE,
                ),
            ),
        ])

    if model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=XGBOOST_TREES,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",

            # Histogram training is much faster for large datasets.
            tree_method="hist",
            max_bin=256,

            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=0,
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


# ==========================================================
# Model evaluation
# ==========================================================

def evaluate_model(
    model_name,
    model,
    x_train,
    y_train,
    x_test,
    y_test,
):
    """
    Train, test, evaluate, and save one model.

    Args:
        model_name: Display name of the model.
        model: Machine-learning model.
        x_train: Training features.
        y_train: Training labels.
        x_test: Testing features.
        y_test: Testing labels.

    Returns:
        Dictionary containing the model's metrics.
    """

    print("\n" + "=" * 72)
    print(f"TRAINING: {model_name}")
    print("=" * 72)

    training_start = time.perf_counter()

    model.fit(
        x_train,
        y_train,
    )

    training_time = (
        time.perf_counter()
        - training_start
    )

    print(
        f"Training completed in "
        f"{training_time:.2f} seconds"
    )

    print("Running predictions...")

    prediction_start = time.perf_counter()

    predictions = model.predict(x_test)

    prediction_time = (
        time.perf_counter()
        - prediction_start
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    model_path = os.path.join(
        MODELS_DIRECTORY,
        f"{safe_name}_10m.pkl",
    )

    save_start = time.perf_counter()

    # Compression level 1 provides faster saving than level 3.
    joblib.dump(
        model,
        model_path,
        compress=1,
    )

    save_time = (
        time.perf_counter()
        - save_start
    )

    print("\nResults:")
    print(f"Accuracy        : {accuracy * 100:.4f}%")
    print(f"Precision       : {precision * 100:.4f}%")
    print(f"Recall          : {recall * 100:.4f}%")
    print(f"F1 score        : {f1 * 100:.4f}%")
    print(f"True negatives  : {tn:,}")
    print(f"False positives : {fp:,}")
    print(f"False negatives : {fn:,}")
    print(f"True positives  : {tp:,}")
    print(f"Training time   : {training_time:.2f} seconds")
    print(f"Prediction time : {prediction_time:.2f} seconds")
    print(f"Model save time : {save_time:.2f} seconds")
    print(f"Model saved     : {model_path}")

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "True Negatives": int(tn),
        "False Positives": int(fp),
        "False Negatives": int(fn),
        "True Positives": int(tp),
        "Total Errors": int(fp + fn),
        "Training Time (s)": training_time,
        "Prediction Time (s)": prediction_time,
        "Model Save Time (s)": save_time,
        "Model File": model_path,
        "Status": "Completed",
    }


# ==========================================================
# Save results
# ==========================================================

def save_results(results):
    """
    Save all completed model results to a CSV file.

    Args:
        results: List of model metric dictionaries.
    """

    if len(results) == 0:
        return

    results_dataframe = pd.DataFrame(results)

    results_dataframe = results_dataframe.sort_values(
        by=[
            "F1 Score",
            "Recall",
            "Accuracy",
        ],
        ascending=False,
    ).reset_index(drop=True)

    results_dataframe.insert(
        0,
        "Rank",
        range(
            1,
            len(results_dataframe) + 1,
        ),
    )

    results_dataframe.to_csv(
        RESULTS_FILE,
        index=False,
    )


# ==========================================================
# Main
# ==========================================================

def main():
    """
    Run the complete training experiment.
    """

    os.makedirs(
        MODELS_DIRECTORY,
        exist_ok=True,
    )

    os.makedirs(
        RESULTS_DIRECTORY,
        exist_ok=True,
    )

    total_start = time.perf_counter()

    (
        x_train,
        x_test,
        y_train,
        y_test,
        dataset_load_time,
    ) = load_dataset()

    model_names = [
        "Decision Tree",
        "Random Forest",
        "Logistic Regression",
        "Linear SVM",
        "XGBoost",
    ]

    results = []
    failed_models = []

    print("\n" + "=" * 72)
    print("STARTING MODEL TRAINING")
    print("=" * 72)

    print(
        f"Models to train: "
        f"{len(model_names)}"
    )

    for model_number, model_name in enumerate(
        model_names,
        start=1,
    ):
        print(
            f"\nModel {model_number}/"
            f"{len(model_names)}"
        )

        model = None

        try:
            model = create_model(
                model_name
            )

            result = evaluate_model(
                model_name,
                model,
                x_train,
                y_train,
                x_test,
                y_test,
            )

            results.append(result)

            # Save progress after every successful model.
            save_results(results)

            print(
                f"\nFinished {model_name} successfully."
            )

        except Exception as error:
            print(
                f"\nERROR while training "
                f"{model_name}:"
            )

            print(error)

            failed_models.append({
                "model": model_name,
                "error": str(error),
            })

        finally:
            if model is not None:
                del model

            gc.collect()

    save_results(results)

    total_time = (
        time.perf_counter()
        - total_start
    )

    metadata = {
        "dataset": DATASET_PATH,
        "quick_mode": QUICK_MODE,
        "training_rows": TRAIN_ROWS,
        "testing_rows": TEST_ROWS,
        "features": FEATURE_COLUMNS,
        "excluded_feature": "interception_probability",
        "dataset_load_time_seconds": dataset_load_time,
        "total_execution_time_seconds": total_time,
        "random_state": RANDOM_STATE,
        "completed_models": [
            result["Model"]
            for result in results
        ],
        "failed_models": failed_models,
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    print("\n" + "=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)

    if results:
        results_dataframe = pd.DataFrame(
            results
        )

        results_dataframe = (
            results_dataframe.sort_values(
                by=[
                    "F1 Score",
                    "Recall",
                    "Accuracy",
                ],
                ascending=False,
            ).reset_index(drop=True)
        )

        results_dataframe.insert(
            0,
            "Rank",
            range(
                1,
                len(results_dataframe) + 1,
            ),
        )

        display_columns = [
            "Rank",
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "Total Errors",
            "Training Time (s)",
            "Prediction Time (s)",
        ]

        print(
            results_dataframe[
                display_columns
            ].to_string(
                index=False
            )
        )

        best_model = (
            results_dataframe.iloc[0]
        )

        print("\nBest model:")

        print(
            f"{best_model['Model']} — "
            f"{best_model['F1 Score'] * 100:.4f}% "
            f"F1 score"
        )

    if failed_models:
        print("\nFailed models:")

        for failed in failed_models:
            print(
                f"- {failed['model']}: "
                f"{failed['error']}"
            )

    print(
        f"\nTotal execution time: "
        f"{total_time:.2f} seconds"
    )

    print(
        f"Results saved to: "
        f"{RESULTS_FILE}"
    )

    print(
        f"Metadata saved to: "
        f"{METADATA_FILE}"
    )


if __name__ == "__main__":
    main()