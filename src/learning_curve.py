import pandas as pd
import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from xgboost import XGBClassifier
# ==========================================================
# Dataset
# ==========================================================
 
# Columns used by every model
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

print("Loading dataset...")

df = pd.read_csv(
    DATASET_PATH,
    usecols=FEATURE_COLUMNS + [LABEL_COLUMN],
    dtype=DTYPES,
)

print(f"Rows loaded: {len(df):,}")

# Shuffle once
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Dataset ready.")

# ==========================================================
# Fixed test set
# ==========================================================

test_df = df.iloc[-TEST_ROWS:]

x_test = test_df[FEATURE_COLUMNS].to_numpy(dtype="float32")

y_test = test_df[LABEL_COLUMN].to_numpy(dtype="int8")

print(f"Fixed testing samples: {len(x_test):,}")

# Remaining rows available for training
training_df = df.iloc[:-TEST_ROWS]

print(f"Maximum training samples: {len(training_df):,}")

# ==========================================================
# Model creation
# ==========================================================

def create_models():
    """
    Create fresh model objects for one training-size experiment.

    Fresh models are required because each model must start from
    the beginning for every training size.
    """

    return {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=15,
            min_samples_leaf=5,
            random_state=42,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=5,
            max_samples=0.7,
            n_jobs=-1,
            random_state=42,
        ),

        "Logistic Regression": Pipeline([
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                SGDClassifier(
                    loss="log_loss",
                    max_iter=100,
                    tol=1e-3,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=5,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]),

        "Linear SVM": Pipeline([
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LinearSVC(
                    C=1.0,
                    dual=False,
                    max_iter=3000,
                    random_state=42,
                ),
            ),
        ]),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            max_bin=256,
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        ),
    }


# ==========================================================
# Learning-curve experiment
# ==========================================================

results = []

for training_size in TRAINING_SIZES:

    print("\n" + "=" * 72)
    print(f"TRAINING SIZE: {training_size:,}")
    print("=" * 72)

    current_training_df = training_df.iloc[:training_size]

    x_train = current_training_df[FEATURE_COLUMNS].to_numpy(
        dtype="float32"
    )

    y_train = current_training_df[LABEL_COLUMN].to_numpy(
        dtype="int8"
    )

    models = create_models()

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        training_start = time.perf_counter()

        model.fit(
            x_train,
            y_train,
        )

        training_time = (
            time.perf_counter()
            - training_start
        )

        prediction_start = time.perf_counter()

        predictions = model.predict(x_test)

        prediction_time = (
            time.perf_counter()
            - prediction_start
        )

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

        results.append({
            "Training Size": training_size,
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Training Time (s)": training_time,
            "Prediction Time (s)": prediction_time,
        })

        print(f"Accuracy      : {accuracy * 100:.4f}%")
        print(f"Precision     : {precision * 100:.4f}%")
        print(f"Recall        : {recall * 100:.4f}%")
        print(f"F1 score      : {f1 * 100:.4f}%")
        print(f"Training time : {training_time:.2f} seconds")
        print(f"Prediction time: {prediction_time:.2f} seconds")

        # Save progress after every model.
        results_df = pd.DataFrame(results)

        results_df.to_csv(
            "results/learning_curve.csv",
            index=False,
        )

    del x_train
    del y_train
    del current_training_df

print("\n" + "=" * 72)
print("LEARNING-CURVE EXPERIMENT COMPLETE")
print("=" * 72)

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/learning_curve.csv",
    index=False,
)

print("Results saved to: results/learning_curve.csv")