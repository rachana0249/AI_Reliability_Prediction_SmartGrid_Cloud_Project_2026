# Results

This folder stores the trained model artifacts and evaluation outputs for the project.

## Purpose

The results directory contains the generated outputs from model training and evaluation, including:

- baseline accuracy metrics
- GCN performance report
- saved trained models

## Files

- baseline_results.txt: evaluation metrics for the Random Forest baseline
- gnn_results.txt: evaluation metrics for the GCN model
- models/best_gnn_model.pt: best validation GCN checkpoint
- models/smart_grid_gnn_model.pt: GCN model file used by the backend
- models/random_forest_model.pkl: saved Random Forest model used by the backend

## Actual metrics currently stored

### Random Forest

- Accuracy: 0.9267
- Precision: 0.9265
- Recall: 0.9613
- F1 Score: 0.9436

### GNN

- Accuracy: 0.8513
- Precision: 0.8563
- Recall: 0.9216
- F1 Score: 0.8878

## Notes

These are the performance values currently present in the repository. They should be cited directly when reporting model comparison results.

