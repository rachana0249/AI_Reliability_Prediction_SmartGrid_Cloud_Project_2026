import os
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data


# ============================================================
# 1. PATHS
# ============================================================

INPUT_FILE = "dataset/raw/Data_for_UCI_named.csv"
OUTPUT_DIR = "dataset/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)


# ============================================================
# 3. REMOVE DUPLICATES
# ============================================================

before = len(df)

df = df.drop_duplicates()

after = len(df)

print("Duplicates removed:", before - after)


# ============================================================
# 4. CHECK MISSING VALUES
# ============================================================

print("\nMissing values:")

print(df.isnull().sum().sum())

if df.isnull().sum().sum() > 0:
    df = df.dropna()


# ============================================================
# 5. SELECT INPUT FEATURES
# ============================================================

features = [
    "tau1", "tau2", "tau3", "tau4",
    "p1", "p2", "p3", "p4",
    "g1", "g2", "g3", "g4"
]

X = df[features].values

# Target:
# stable   -> 0
# unstable -> 1

y = df["stabf"].map({
    "stable": 0,
    "unstable": 1
}).values


# ============================================================
# 6. SCALE FEATURES
# ============================================================

print("\nScaling features...")

scaler = StandardScaler()

X = scaler.fit_transform(X)

X = X.astype(np.float32)

y = y.astype(np.int64)


# ============================================================
# 7. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

indices = np.arange(len(X))

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=42,
    stratify=y
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=42,
    stratify=y[temp_idx]
)

print("\nDataset split:")

print("Training samples:", len(train_idx))
print("Validation samples:", len(val_idx))
print("Testing samples:", len(test_idx))


# ============================================================
# 8. CREATE KNN GRAPH
# ============================================================

print("\nCreating KNN graph...")

# Each row is treated as a node.
# Similar samples are connected.

K = 8

adjacency = kneighbors_graph(
    X,
    n_neighbors=K,
    mode="connectivity",
    include_self=False
)

# Convert sparse matrix to edge list

coo = adjacency.tocoo()

edge_index = torch.tensor(
    np.vstack((coo.row, coo.col)),
    dtype=torch.long
)


# ============================================================
# 9. CREATE PYTORCH GEOMETRIC DATA
# ============================================================

graph = Data(
    x=torch.tensor(X, dtype=torch.float),
    edge_index=edge_index,
    y=torch.tensor(y, dtype=torch.long)
)

# Masks for train / validation / test

graph.train_mask = torch.zeros(len(X), dtype=torch.bool)
graph.val_mask = torch.zeros(len(X), dtype=torch.bool)
graph.test_mask = torch.zeros(len(X), dtype=torch.bool)

graph.train_mask[train_idx] = True
graph.val_mask[val_idx] = True
graph.test_mask[test_idx] = True


# ============================================================
# 10. SAVE GRAPH
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "smart_grid_graph.pt"
)

torch.save(graph, output_file)


# ============================================================
# 11. SAVE SCALER
# ============================================================

import joblib

joblib.dump(
    scaler,
    os.path.join(OUTPUT_DIR, "scaler.pkl")
)


# ============================================================
# 12. SAVE FEATURE INFORMATION
# ============================================================

with open(
    os.path.join(OUTPUT_DIR, "features.txt"),
    "w"
) as f:

    for feature in features:
        f.write(feature + "\n")


# ============================================================
# 13. SUMMARY
# ============================================================

print("\n========================================")
print("PREPROCESSING COMPLETED")
print("========================================")

print("Nodes:", graph.num_nodes)
print("Features per node:", graph.num_node_features)
print("Edges:", graph.num_edges)

print("Training nodes:", graph.train_mask.sum().item())
print("Validation nodes:", graph.val_mask.sum().item())
print("Testing nodes:", graph.test_mask.sum().item())

print("\nSaved files:")

print("dataset/processed/smart_grid_graph.pt")
print("dataset/processed/scaler.pkl")
print("dataset/processed/features.txt")