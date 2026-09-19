import os
import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
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
MODEL_DIR = "results/models"

os.makedirs(MODEL_DIR, exist_ok=True)

EPOCHS = 100
LEARNING_RATE = 0.01
HIDDEN_CHANNELS = 32
DROPOUT = 0.30


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# 3. LOAD GRAPH
# ============================================================

print("\nLoading graph...")

data = torch.load(
    GRAPH_FILE,
    weights_only=False
)

data = data.to(device)

print("Nodes:", data.num_nodes)
print("Features:", data.num_node_features)
print("Edges:", data.num_edges)


# ============================================================
# 4. DEFINE GCN MODEL
# ============================================================

class GCN(torch.nn.Module):

    def __init__(
        self,
        input_channels,
        hidden_channels,
        output_channels
    ):
        super().__init__()

        self.conv1 = GCNConv(
            input_channels,
            hidden_channels
        )

        self.conv2 = GCNConv(
            hidden_channels,
            output_channels
        )

    def forward(self, x, edge_index):

        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=DROPOUT,
            training=self.training
        )

        x = self.conv2(
            x,
            edge_index
        )

        return x


# ============================================================
# 5. CREATE MODEL
# ============================================================

model = GCN(
    input_channels=data.num_node_features,
    hidden_channels=HIDDEN_CHANNELS,
    output_channels=2
).to(device)


# ============================================================
# 6. OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=5e-4
)


# ============================================================
# 7. TRAINING FUNCTION
# ============================================================

def train():

    model.train()

    optimizer.zero_grad()

    output = model(
        data.x,
        data.edge_index
    )

    loss = F.cross_entropy(
        output[data.train_mask],
        data.y[data.train_mask]
    )

    loss.backward()

    optimizer.step()

    return loss.item()


# ============================================================
# 8. VALIDATION FUNCTION
# ============================================================

def validate():

    model.eval()

    with torch.no_grad():

        output = model(
            data.x,
            data.edge_index
        )

        predictions = output.argmax(
            dim=1
        )

        correct = (
            predictions[data.val_mask]
            == data.y[data.val_mask]
        ).sum()

        accuracy = (
            correct.item()
            / data.val_mask.sum().item()
        )

    return accuracy


# ============================================================
# 9. TRAIN MODEL
# ============================================================

print("\n========================================")
print("STARTING GNN TRAINING")
print("========================================")

best_val_accuracy = 0.0

best_model_path = os.path.join(
    MODEL_DIR,
    "best_gnn_model.pt"
)


for epoch in range(1, EPOCHS + 1):

    loss = train()

    val_accuracy = validate()

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            best_model_path
        )

    if epoch == 1 or epoch % 10 == 0:

        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {loss:.4f} | "
            f"Validation Accuracy: "
            f"{val_accuracy:.4f}"
        )


# ============================================================
# 10. LOAD BEST MODEL
# ============================================================

print("\nBest validation accuracy:")
print(f"{best_val_accuracy:.4f}")

model.load_state_dict(
    torch.load(
        best_model_path,
        map_location=device,
        weights_only=True
    )
)


# ============================================================
# 11. TEST MODEL
# ============================================================

model.eval()

with torch.no_grad():

    output = model(
        data.x,
        data.edge_index
    )

    predictions = output.argmax(
        dim=1
    )

    probabilities = F.softmax(
        output,
        dim=1
    )


# Test predictions

y_true = data.y[
    data.test_mask
].cpu().numpy()

y_pred = predictions[
    data.test_mask
].cpu().numpy()


# ============================================================
# 12. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(
    y_true,
    y_pred
)


# ============================================================
# 13. PRINT RESULTS
# ============================================================

print("\n========================================")
print("GNN TEST RESULTS")
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
        y_true,
        y_pred,
        target_names=[
            "Stable",
            "Unstable"
        ],
        zero_division=0
    )
)


# ============================================================
# 14. SAVE FINAL MODEL INFORMATION
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_features": data.num_node_features,
        "hidden_channels": HIDDEN_CHANNELS,
        "output_classes": 2
    },
    os.path.join(
        MODEL_DIR,
        "smart_grid_gnn_model.pt"
    )
)


# ============================================================
# 15. SAVE TEST RESULTS
# ============================================================

with open(
    "results/gnn_results.txt",
    "w"
) as f:

    f.write("AI-Based Reliability Prediction Framework\n")
    f.write("Smart Grid GNN Results\n")
    f.write("========================================\n\n")

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

    f.write("\nConfusion Matrix:\n")

    f.write(
        str(cm)
    )


# ============================================================
# 16. FINAL MESSAGE
# ============================================================

print("\n========================================")
print("GNN TRAINING COMPLETED")
print("========================================")

print(
    "Model saved to:"
)

print(
    "results/models/smart_grid_gnn_model.pt"
)

print(
    "\nResults saved to:"
)

print(
    "results/gnn_results.txt"
)