import os
import torch
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. SETTINGS
# ============================================================

GRAPH_FILE = "dataset/processed/smart_grid_graph.pt"

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. LOAD PROCESSED DATA
# ============================================================

print("Loading processed data...")

data = torch.load(
    GRAPH_FILE,
    weights_only=False
)

# Features
X = data.x.numpy()

# Target
y = data.y.numpy()


# ============================================================
# 3. GET SAME TRAIN / TEST SPLIT AS GNN
# ============================================================

train_mask = data.train_mask.numpy()
test_mask = data.test_mask.numpy()

X_train = X[train_mask]
y_train = y[train_mask]

X_test = X[test_mask]
y_test = y[test_mask]


print("\nDataset split:")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 4. CREATE RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 5. TRAIN
# ============================================================

model.fit(
    X_train,
    y_train
)


print("Training completed.")


# ============================================================
# 6. PREDICT
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# 7. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("RANDOM FOREST TEST RESULTS")
print("========================================")

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)


print("\nConfusion Matrix:")

print(cm)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Stable",
            "Unstable"
        ],
        zero_division=0
    )
)


# ============================================================
# 9. SAVE BASELINE RESULTS
# ============================================================

with open(
    "results/baseline_results.txt",
    "w"
) as f:

    f.write(
        "Random Forest Baseline Results\n"
    )

    f.write(
        "========================================\n\n"
    )

    f.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    f.write(
        f"Precision: {precision:.4f}\n"
    )

    f.write(
        f"Recall: {recall:.4f}\n"
    )

    f.write(
        f"F1 Score: {f1:.4f}\n"
    )

    f.write(
        "\nConfusion Matrix:\n"
    )

    f.write(
        str(cm)
    )


# ============================================================
# 10. SAVE RANDOM FOREST MODEL
# ============================================================

import joblib

joblib.dump(
    model,
    "results/models/random_forest_model.pkl"
)


# ============================================================
# 11. FINISHED
# ============================================================

print("\n========================================")
print("BASELINE TRAINING COMPLETED")
print("========================================")

print(
    "Model saved to:"
)

print(
    "results/models/random_forest_model.pkl"
)

print(
    "\nResults saved to:"
)

print(
    "results/baseline_results.txt"
)