import os
import time
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# -------------------------------------------------
# Configuration
# -------------------------------------------------

DATASET_PATH = "datasets/bb84_dataset.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CROSS_VALIDATION_FOLDS = 5


# -------------------------------------------------
# Load dataset
# -------------------------------------------------

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}\n"
        "Run dataset_generator.py first."
    )

df = pd.read_csv(DATASET_PATH)

print("=" * 65)
print("RANDOM FOREST OVERFITTING ANALYSIS")
print("=" * 65)

print(f"\nDataset shape: {df.shape}")


# -------------------------------------------------
# Prepare features and target
# -------------------------------------------------

required_columns = ["label"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' is missing from the dataset."
        )

columns_to_remove = ["label"]

# Do not allow the AI to use Eve's interception probability.
# This value would not be known in a real communication system.
if "interception_probability" in df.columns:
    columns_to_remove.append("interception_probability")

X = df.drop(columns=columns_to_remove)
y = df["label"]

print("\nFeatures used:")

for feature in X.columns:
    print(f"  - {feature}")

print(f"\nTotal features: {X.shape[1]}")


# -------------------------------------------------
# Train/test split
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# -------------------------------------------------
# Random Forest model
# -------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


# -------------------------------------------------
# Train model
# -------------------------------------------------

print("\nTraining Random Forest...")

start_time = time.time()

model.fit(X_train, y_train)

training_time = time.time() - start_time

print(f"Training completed in {training_time:.2f} seconds")


# -------------------------------------------------
# Training and testing predictions
# -------------------------------------------------

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)


# -------------------------------------------------
# Calculate training metrics
# -------------------------------------------------

train_accuracy = accuracy_score(
    y_train,
    train_predictions,
)

train_precision = precision_score(
    y_train,
    train_predictions,
    zero_division=0,
)

train_recall = recall_score(
    y_train,
    train_predictions,
    zero_division=0,
)

train_f1 = f1_score(
    y_train,
    train_predictions,
    zero_division=0,
)


# -------------------------------------------------
# Calculate testing metrics
# -------------------------------------------------

test_accuracy = accuracy_score(
    y_test,
    test_predictions,
)

test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0,
)

test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0,
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0,
)


# -------------------------------------------------
# Calculate gaps
# -------------------------------------------------

accuracy_gap = train_accuracy - test_accuracy
precision_gap = train_precision - test_precision
recall_gap = train_recall - test_recall
f1_gap = train_f1 - test_f1


# -------------------------------------------------
# Print train/test comparison
# -------------------------------------------------

print("\n" + "=" * 65)
print("TRAINING VS TESTING PERFORMANCE")
print("=" * 65)

print(
    f"\n{'Metric':<15}"
    f"{'Training':>15}"
    f"{'Testing':>15}"
    f"{'Gap':>15}"
)

print("-" * 60)

print(
    f"{'Accuracy':<15}"
    f"{train_accuracy * 100:>14.2f}%"
    f"{test_accuracy * 100:>14.2f}%"
    f"{accuracy_gap * 100:>14.2f}%"
)

print(
    f"{'Precision':<15}"
    f"{train_precision * 100:>14.2f}%"
    f"{test_precision * 100:>14.2f}%"
    f"{precision_gap * 100:>14.2f}%"
)

print(
    f"{'Recall':<15}"
    f"{train_recall * 100:>14.2f}%"
    f"{test_recall * 100:>14.2f}%"
    f"{recall_gap * 100:>14.2f}%"
)

print(
    f"{'F1 Score':<15}"
    f"{train_f1 * 100:>14.2f}%"
    f"{test_f1 * 100:>14.2f}%"
    f"{f1_gap * 100:>14.2f}%"
)


# -------------------------------------------------
# Test confusion matrix
# -------------------------------------------------

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions,
).ravel()

print("\nTest-set confusion matrix:")

print(f"True negatives : {tn}")
print(f"False positives: {fp}")
print(f"False negatives: {fn}")
print(f"True positives : {tp}")
print(f"Total errors   : {fp + fn}")


# -------------------------------------------------
# Five-fold cross-validation
# -------------------------------------------------

print("\n" + "=" * 65)
print(f"{CROSS_VALIDATION_FOLDS}-FOLD CROSS-VALIDATION")
print("=" * 65)

print(
    "\nCross-validation may take several minutes "
    "with 100,000 sessions."
)

cv_model = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

cv_start_time = time.time()

cv_scores = cross_val_score(
    cv_model,
    X_train,
    y_train,
    cv=CROSS_VALIDATION_FOLDS,
    scoring="accuracy",
    n_jobs=-1,
)

cv_time = time.time() - cv_start_time

for fold_number, score in enumerate(cv_scores, start=1):
    print(
        f"Fold {fold_number}: "
        f"{score * 100:.2f}%"
    )

cv_mean = cv_scores.mean()
cv_std = cv_scores.std()

print(f"\nAverage CV accuracy : {cv_mean * 100:.2f}%")
print(f"CV standard deviation: {cv_std * 100:.4f}%")
print(f"Cross-validation time: {cv_time:.2f} seconds")


# -------------------------------------------------
# Automatic overfitting analysis
# -------------------------------------------------

print("\n" + "=" * 65)
print("OVERFITTING VERDICT")
print("=" * 65)

gap_percentage = accuracy_gap * 100
cv_difference = abs(test_accuracy - cv_mean) * 100

if gap_percentage < 0:
    print(
        "\nThe test accuracy is slightly higher than the "
        "training accuracy."
    )
    print(
        "This is not overfitting and can occur because of "
        "random variation."
    )

elif gap_percentage <= 1:
    print("\nNo significant overfitting detected.")
    print(
        "The training and testing accuracies are very close."
    )

elif gap_percentage <= 3:
    print("\nMild overfitting may be present.")
    print(
        "The model still generalizes reasonably well, but "
        "regularization may help."
    )

elif gap_percentage <= 5:
    print("\nModerate overfitting detected.")
    print(
        "Consider limiting tree depth or increasing "
        "minimum samples per leaf."
    )

else:
    print("\nStrong overfitting detected.")
    print(
        "The model performs much better on training data "
        "than unseen testing data."
    )

if cv_std * 100 <= 0.5:
    print(
        "\nCross-validation results are stable across folds."
    )
else:
    print(
        "\nCross-validation results vary noticeably "
        "between folds."
    )

if cv_difference <= 1:
    print(
        "The test accuracy agrees with cross-validation, "
        "which indicates good generalization."
    )
else:
    print(
        "The test accuracy differs from cross-validation "
        "and should be investigated."
    )


# -------------------------------------------------
# Save results
# -------------------------------------------------

os.makedirs("results", exist_ok=True)

results = pd.DataFrame([
    {
        "Model": "Random Forest",
        "Training Accuracy": train_accuracy,
        "Testing Accuracy": test_accuracy,
        "Accuracy Gap": accuracy_gap,
        "Training Precision": train_precision,
        "Testing Precision": test_precision,
        "Training Recall": train_recall,
        "Testing Recall": test_recall,
        "Training F1": train_f1,
        "Testing F1": test_f1,
        "CV Mean Accuracy": cv_mean,
        "CV Standard Deviation": cv_std,
        "False Positives": fp,
        "False Negatives": fn,
        "Training Time (s)": training_time,
        "Cross Validation Time (s)": cv_time,
    }
])

output_path = "results/overfitting_analysis.csv"

results.to_csv(
    output_path,
    index=False,
)

print(f"\nResults saved to: {output_path}")