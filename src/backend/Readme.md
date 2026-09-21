# Backend

This folder contains the Flask application for the smart-grid reliability prediction system.

## Purpose

The backend is responsible for:

- loading the processed graph, scaler, and trained models
- validating user input
- running GNN and Random Forest prediction
- calculating risk level and maintenance recommendation
- returning JSON responses for the frontend

## Files

- app.py: main Flask API server
- app_backup.py: backup copy of the earlier backend version
- requirements.txt: Python dependencies for running the API

## API endpoints

The implemented endpoints are:

- GET /: project metadata
- GET /health: health status of the application and loaded models
- POST /predict: receive the 12 grid features and return prediction results

## Model loading

The app loads:

- dataset/processed/smart_grid_graph.pt
- dataset/processed/scaler.pkl
- results/models/smart_grid_gnn_model.pt
- results/models/random_forest_model.pkl

## Run locally

```bash
cd /path/to/project
python -m venv .venv
source .venv/bin/activate
pip install -r src/backend/requirements.txt
python src/backend/app.py
```

The service runs on port 9000.

## Output format

The prediction API returns JSON with:

- gnn result
- random_forest result
- model_agreement
- input_features

## Important note

This repository contains the local Flask implementation, but it does not include the full AWS deployment package, Lambda code, or SNS alert implementation.

