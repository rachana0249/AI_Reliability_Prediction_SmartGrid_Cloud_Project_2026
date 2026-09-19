import os
import sys

import joblib
import numpy as np
import torch

from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


# ============================================================
# 1. FLASK SETUP
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# 2. PATHS
# ============================================================

GRAPH_FILE = "dataset/processed/smart_grid_graph.pt"
SCALER_FILE = "dataset/processed/scaler.pkl"

GNN_MODEL_FILE = "results/models/smart_grid_gnn_model.pt"
RF_MODEL_FILE = "results/models/random_forest_model.pkl"


# ============================================================
# 3. FEATURE NAMES
# ============================================================

FEATURES = [
    "tau1", "tau2", "tau3", "tau4",
    "p1", "p2", "p3", "p4",
    "g1", "g2", "g3", "g4"
]


# ============================================================
# 4. DEVICE
# ============================================================

device = torch.device("cpu")


# ============================================================
# 5. LOAD SCALER
# ============================================================

print("Loading scaler...")

scaler = joblib.load(
    SCALER_FILE
)


# ============================================================
# 6. LOAD RANDOM FOREST
# ============================================================

print("Loading Random Forest...")

rf_model = joblib.load(
    RF_MODEL_FILE
)


# ============================================================
# 7. GNN MODEL DEFINITION
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

    def forward(
        self,
        x,
        edge_index
    ):

        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = self.conv2(
            x,
            edge_index
        )

        return x


# ============================================================
# 8. LOAD ORIGINAL GRAPH
# ============================================================

print("Loading graph...")

graph = torch.load(
    GRAPH_FILE,
    weights_only=False
)

graph = graph.to(device)


# ============================================================
# 9. LOAD GNN
# ============================================================

print("Loading GNN model...")

gnn_model = GCN(
    input_channels=12,
    hidden_channels=32,
    output_channels=2
).to(device)


checkpoint = torch.load(
    GNN_MODEL_FILE,
    map_location=device,
    weights_only=True
)

# Our saved model contains model_state_dict

if "model_state_dict" in checkpoint:

    gnn_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    gnn_model.load_state_dict(
        checkpoint
    )


gnn_model.eval()


print("All models loaded successfully.")


# ============================================================
# 10. GNN PREDICTION
# ============================================================

def predict_gnn(features):

    # --------------------------------------------------------
    # Scale the new input
    # --------------------------------------------------------

    scaled = scaler.transform(
        np.array(features).reshape(1, -1)
    )

    scaled = scaled.astype(
        np.float32
    )


    # --------------------------------------------------------
    # Existing graph features
    # --------------------------------------------------------

    existing_x = graph.x.cpu().numpy()


    # --------------------------------------------------------
    # Add new node
    # --------------------------------------------------------

    new_x = np.vstack([
        existing_x,
        scaled
    ])


    new_node_index = len(existing_x)


    # --------------------------------------------------------
    # Find 8 nearest existing nodes
    # --------------------------------------------------------

    neighbors = NearestNeighbors(
        n_neighbors=8
    )

    neighbors.fit(
        existing_x
    )

    _, indices = neighbors.kneighbors(
        scaled
    )


    # --------------------------------------------------------
    # Create edges for new node
    # --------------------------------------------------------

    new_edges = []

    for neighbor in indices[0]:

        # New node -> existing node
        new_edges.append([
            new_node_index,
            int(neighbor)
        ])

        # Existing node -> new node
        new_edges.append([
            int(neighbor),
            new_node_index
        ])


    # --------------------------------------------------------
    # Combine original + new edges
    # --------------------------------------------------------

    original_edges = graph.edge_index.cpu().numpy()

    new_edges = np.array(
        new_edges
    ).T

    combined_edges = np.hstack([
        original_edges,
        new_edges
    ])


    # --------------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------------

    x_tensor = torch.tensor(
        new_x,
        dtype=torch.float
    )

    edge_tensor = torch.tensor(
        combined_edges,
        dtype=torch.long
    )


    # --------------------------------------------------------
    # GNN prediction
    # --------------------------------------------------------

    with torch.no_grad():

        output = gnn_model(
            x_tensor,
            edge_tensor
        )

        probabilities = F.softmax(
            output,
            dim=1
        )

        prediction = output[
            new_node_index
        ].argmax().item()

        stable_probability = probabilities[
            new_node_index
        ][0].item()

        unstable_probability = probabilities[
            new_node_index
        ][1].item()


    if prediction == 0:

        status = "Stable"

    else:

        status = "Unstable"


    # Reliability score is the model's
    # predicted probability of stability.

    reliability_score = (
        stable_probability * 100
    )


    return {
        "prediction": status,
        "stable_probability": round(
            stable_probability * 100,
            2
        ),
        "unstable_probability": round(
            unstable_probability * 100,
            2
        ),
        "reliability_score": round(
            reliability_score,
            2
        )
    }


# ============================================================
# 11. RANDOM FOREST PREDICTION
# ============================================================

def predict_random_forest(features):

    scaled = scaler.transform(
        np.array(features).reshape(1, -1)
    )

    prediction = rf_model.predict(
        scaled
    )[0]

    probabilities = rf_model.predict_proba(
        scaled
    )[0]


    if prediction == 0:

        status = "Stable"

    else:

        status = "Unstable"


    stable_probability = probabilities[0]

    unstable_probability = probabilities[1]


    return {
        "prediction": status,
        "stable_probability": round(
            stable_probability * 100,
            2
        ),
        "unstable_probability": round(
            unstable_probability * 100,
            2
        )
    }


# ============================================================
# 12. HOME ROUTE
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Smart Grid AI Reliability Prediction API",
        "status": "running"
    })


# ============================================================
# 13. HEALTH CHECK
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({
        "status": "healthy",
        "models": {
            "gnn": "loaded",
            "random_forest": "loaded"
        }
    })


# ============================================================
# 14. PREDICTION API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json()


        if data is None:

            return jsonify({
                "error": "No JSON data received"
            }), 400


        # ----------------------------------------------------
        # Get 12 features
        # ----------------------------------------------------

        features = []

        for feature in FEATURES:

            if feature not in data:

                return jsonify({
                    "error":
                    f"Missing feature: {feature}"
                }), 400

            features.append(
                float(data[feature])
            )


        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        gnn_result = predict_gnn(
            features
        )

        rf_result = predict_random_forest(
            features
        )


        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return jsonify({

            "input": {
                feature: features[i]
                for i, feature
                in enumerate(FEATURES)
            },

            "gnn": gnn_result,

            "random_forest": rf_result

        })


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# 15. RUN SERVER
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("SMART GRID AI API")
    print("========================================")

    print("Server: http://127.0.0.1:9000")
    print("Health: http://127.0.0.1:9000/health")
    print("Prediction: POST /predict")

    app.run(
        host="0.0.0.0",
        port=9000,
        debug=True
    )