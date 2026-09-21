# AI Model Pipeline

This folder contains the scripts used to prepare the dataset, train the machine learning models, and evaluate them.

## Purpose

The AI model pipeline handles the core machine-learning workflow:

- load the dataset
- clean and preprocess the data
- scale the features
- split data for training and evaluation
- construct the KNN-based similarity graph
- train the GCN model
- train the Random Forest baseline
- save evaluation results and model artifacts

## Files

- preprocess.py: dataset loading, scaling, split, and KNN graph generation
- train_baseline.py: Random Forest training and baseline metric calculation
- train_gnn.py: GCN model definition, training, validation, and final model saving

## Important implementation details

### Data preprocessing

The dataset is loaded from:

```text
dataset/raw/Data_for_UCI_named.csv
```

The model uses these 12 features:

- tau1, tau2, tau3, tau4
- p1, p2, p3, p4
- g1, g2, g3, g4

The target is `stabf`, mapped to:

- stable -> 0
- unstable -> 1

### Graph construction

The graph is not built from physical smart-grid topology. It is created as a KNN similarity graph using `kneighbors_graph` with 8 neighbors per node.

### Model outputs

The train scripts save outputs into:

- dataset/processed/
- results/models/
- results/

## Run the model workflow

```bash
python src/ai_model/preprocess.py
python src/ai_model/train_gnn.py
python src/ai_model/train_baseline.py
```

## Notes

The repository currently contains the local ML prototype and the saved model/result files. It does not include a full MLOps or cloud deployment pipeline for model serving.

