import os
import numpy as np
import torch
import joblib

from flask import Flask, request, jsonify
from flask_cors import CORS

from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import GCNConv


app = Flask(__name__)
CORS(app)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

GRAPH_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "smart_grid_graph.pt"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "processed",
    "scaler.pkl"
)

GNN_MODEL_PATH = os.path.join(
    BASE_DIR,
    "results",
    "models",
    "smart_grid_gnn_model.pt"
)

RF_MODEL_PATH = os.path.join(
    BASE_DIR,
    "results",
    "models",
    "random_forest_model.pkl"
)


# ============================================================
# FEATURES
# ============================================================

FEATURE_NAMES = [
    "tau1",
    "tau2",
    "tau3",
    "tau4",
    "p1",
    "p2",
    "p3",
    "p4",
    "g1",
    "g2",
    "g3",
    "g4"
]


# ============================================================
# GNN
# ============================================================

class SmartGridGCN(torch.nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=32,
        output_dim=2
    ):

        super().__init__()

        self.conv1 = GCNConv(
            input_dim,
            hidden_dim
        )

        self.conv2 = GCNConv(
            hidden_dim,
            output_dim
        )

        self.dropout = torch.nn.Dropout(
            0.30
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

        x = torch.relu(x)

        x = self.dropout(x)

        x = self.conv2(
            x,
            edge_index
        )

        return x


# ============================================================
# LOAD DATA / MODELS
# ============================================================

print("Loading Smart Grid AI models...")


graph_data = torch.load(
    GRAPH_PATH,
    map_location="cpu",
    weights_only=False
)

scaler = joblib.load(
    SCALER_PATH
)

rf_model = joblib.load(
    RF_MODEL_PATH
)


INPUT_DIM = graph_data.x.shape[1]


gnn_model = SmartGridGCN(
    input_dim=INPUT_DIM,
    hidden_dim=32,
    output_dim=2
)


checkpoint = torch.load(
    GNN_MODEL_PATH,
    map_location="cpu",
    weights_only=False
)


if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    gnn_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    gnn_model.load_state_dict(
        checkpoint
    )


gnn_model.eval()


print("GNN model loaded.")
print("Random Forest model loaded.")
print("Graph loaded.")
print("All models ready.")


# ============================================================
# RISK
# ============================================================

def get_risk_level(
    reliability
):

    reliability = float(
        reliability
    )

    if reliability >= 70:

        return "LOW"

    elif reliability >= 40:

        return "MEDIUM"

    elif reliability >= 20:

        return "HIGH"

    else:

        return "CRITICAL"


# ============================================================
# MAINTENANCE
# ============================================================

def get_maintenance_recommendation(
    risk_level
):

    recommendations = {

        "LOW":
            "Continue normal grid monitoring.",

        "MEDIUM":
            "Increase monitoring frequency and inspect unusual parameters.",

        "HIGH":
            "Inspect grid parameters and consider preventive maintenance.",

        "CRITICAL":
            "Immediate inspection recommended. High instability risk detected."
    }

    return recommendations.get(
        risk_level,
        "Continue monitoring."
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_features(
    features
):

    missing = []

    for feature in FEATURE_NAMES:

        if feature not in features:

            missing.append(
                feature
            )

    if missing:

        return False, {

            "error":
                "Missing features",

            "missing":
                missing
        }


    try:

        for feature in FEATURE_NAMES:

            value = float(
                features[feature]
            )

            if not np.isfinite(
                value
            ):

                return False, {

                    "error":
                        f"Invalid value for {feature}"
                }

    except Exception:

        return False, {

            "error":
                "All feature values must be numeric."
        }


    return True, None


# ============================================================
# GNN PREDICTION
# ============================================================

def predict_gnn(
    features
):

    values = np.array(

        [
            float(
                features[name]
            )

            for name in FEATURE_NAMES

        ],

        dtype=np.float32

    ).reshape(
        1,
        -1
    )


    scaled_values = scaler.transform(
        values
    )


    x_existing = graph_data.x.clone()


    new_x = torch.tensor(
        scaled_values,
        dtype=torch.float32
    )


    # Find nearest neighbours

    neighbors_model = NearestNeighbors(
        n_neighbors=8
    )

    neighbors_model.fit(
        x_existing.numpy()
    )


    distances, indices = (
        neighbors_model.kneighbors(
            new_x.numpy()
        )
    )


    nearest_nodes = indices[0]


    # New node

    new_node_index = (
        x_existing.shape[0]
    )


    combined_x = torch.cat(

        [
            x_existing,
            new_x
        ],

        dim=0
    )


    # New edges

    existing_edges = (
        graph_data.edge_index.clone()
    )


    new_edges = []


    for node in nearest_nodes:

        new_edges.append(
            [
                new_node_index,
                int(node)
            ]
        )

        new_edges.append(
            [
                int(node),
                new_node_index
            ]
        )


    new_edges = torch.tensor(
        new_edges,
        dtype=torch.long
    ).t()


    combined_edges = torch.cat(

        [
            existing_edges,
            new_edges
        ],

        dim=1
    )


    # Prediction

    with torch.no_grad():

        output = gnn_model(
            combined_x,
            combined_edges
        )


        probabilities = torch.softmax(

            output[
                new_node_index
            ],

            dim=0

        )


        predicted_class = int(

            torch.argmax(
                probabilities
            ).item()

        )


    stable_probability = (
        float(
            probabilities[0].item()
        )
        * 100
    )


    unstable_probability = (
        float(
            probabilities[1].item()
        )
        * 100
    )


    prediction = (

        "Stable"

        if predicted_class == 0

        else "Unstable"
    )


    reliability_score = (
        stable_probability
    )


    risk_level = get_risk_level(
        reliability_score
    )


    recommendation = (
        get_maintenance_recommendation(
            risk_level
        )
    )


    return {

        "prediction":
            prediction,

        "reliability_score":
            round(
                reliability_score,
                2
            ),

        "stable_probability":
            round(
                stable_probability,
                2
            ),

        "unstable_probability":
            round(
                unstable_probability,
                2
            ),

        "risk_level":
            risk_level,

        "maintenance_recommendation":
            recommendation
    }


# ============================================================
# RANDOM FOREST
# ============================================================

def predict_random_forest(
    features
):

    values = np.array(

        [
            float(
                features[name]
            )

            for name in FEATURE_NAMES
        ],

        dtype=np.float32

    ).reshape(
        1,
        -1
    )


    scaled_values = scaler.transform(
        values
    )


    predicted_class = int(

        rf_model.predict(
            scaled_values
        )[0]

    )


    probabilities = (
        rf_model.predict_proba(
            scaled_values
        )[0]
    )


    stable_probability = (
        float(
            probabilities[0]
        )
        * 100
    )


    unstable_probability = (
        float(
            probabilities[1]
        )
        * 100
    )


    prediction = (

        "Stable"

        if predicted_class == 0

        else "Unstable"
    )


    # Feature importance

    importances = (
        rf_model.feature_importances_
    )


    feature_importance = []


    for name, importance in zip(

        FEATURE_NAMES,
        importances

    ):

        feature_importance.append({

            "feature":
                name,

            "importance":
                round(
                    float(
                        importance
                    ),
                    4
                )
        })


    feature_importance.sort(

        key=lambda item:
            item["importance"],

        reverse=True
    )


    return {

        "prediction":
            prediction,

        "stable_probability":
            round(
                stable_probability,
                2
            ),

        "unstable_probability":
            round(
                unstable_probability,
                2
            ),

        "feature_importance":
            feature_importance
    }


# ============================================================
# HOME
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)

def home():

    return jsonify({

        "project":
            "AI-Based Reliability Prediction Framework for Smart Grid Cloud Applications",

        "status":
            "running",

        "models": [

            "Graph Neural Network",

            "Random Forest"
        ],

        "features": [

            "Reliability Prediction",

            "Risk Classification",

            "Model Agreement",

            "Feature Importance",

            "Predictive Maintenance",

            "Fault Scenario Simulation",

            "What-If Analysis",

            "Real-Time Simulation"
        ]
    })


# ============================================================
# HEALTH
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)

def health():

    return jsonify({

        "status":
            "healthy",

        "models": {

            "gnn":
                "loaded",

            "random_forest":
                "loaded"
        },

        "graph": {

            "nodes":
                int(
                    graph_data.x.shape[0]
                ),

            "edges":
                int(
                    graph_data.edge_index.shape[1]
                )
        }
    })


# ============================================================
# PREDICT
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)

def predict():

    try:

        features = request.get_json()


        if not features:

            return jsonify({

                "error":
                    "No JSON input received."

            }), 400


        valid, error = (
            validate_features(
                features
            )
        )


        if not valid:

            return jsonify(
                error
            ), 400


        gnn_result = (
            predict_gnn(
                features
            )
        )


        rf_result = (
            predict_random_forest(
                features
            )
        )


        models_agree = (

            gnn_result[
                "prediction"
            ]

            ==

            rf_result[
                "prediction"
            ]
        )


        if models_agree:

            model_agreement = "HIGH"

        else:

            model_agreement = (
                "DISAGREEMENT"
            )


        return jsonify({

            "success":
                True,

            "gnn":
                gnn_result,

            "random_forest":
                rf_result,

            "model_agreement":
                model_agreement,

            "input_features":
                features
        })


    except Exception as e:

        print(
            "Prediction error:",
            str(e)
        )


        return jsonify({

            "success":
                False,

            "error":
                str(e)

        }), 500


# ============================================================
# SERVER
# ============================================================

if __name__ == "__main__":

    print(
        "\nSmart Grid AI API running on port 9000..."
    )


    app.run(

        host="0.0.0.0",

        port=9000,

        debug=False
    )
