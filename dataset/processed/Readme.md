# Processed Dataset

This folder contains the processed smart-grid data used by the training and inference pipeline.

## Purpose

The processed dataset is generated from the raw CSV file and stores everything needed for the model pipeline:

- feature list
- scaled training data representation
- saved scaler
- KNN graph used by the GCN

## Files

- features.txt: list of input features used by the model
- scaler.pkl: StandardScaler fitted on the training data
- smart_grid_graph.pt: PyTorch Geometric graph object used by the GCN

## Important note

The graph in this folder is a KNN similarity graph, not an actual physical smart-grid topology graph.

## Source

This data is derived from:

```text
dataset/raw/Data_for_UCI_named.csv
```

