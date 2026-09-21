# Frontend

This folder contains the web-based user interface for the smart-grid reliability prediction system.

## Purpose

The frontend allows users to:

- enter smart-grid operating values
- run prediction using the trained AI models
- view GNN and Random Forest results
- inspect reliability, risk, and maintenance guidance
- test fault scenarios
- compare model agreement
- analyze feature importance
- perform what-if parameter changes
- run a simulated real-time monitoring view

## Files

- index.html: main dashboard page
- script.js: all interactive logic, API calls, scenario simulation, what-if analysis, and live monitoring
- style.css: dashboard styling
- vercel.json: API rewrite configuration for frontend hosting
- index_backup*.html and script_backup*.js: older backup files kept in the repository
- style_backup.css: older backup stylesheet

## How it works

- The user enters 12 smart-grid parameters.
- The browser sends a JSON payload to the Flask backend.
- The backend returns GNN and Random Forest outputs.
- The UI displays prediction results, model agreement, risk, and recommendation.
- Scenario and live-simulation tools modify values and trigger prediction calls again.

## Local usage

Start the backend first, then open the frontend locally:

```bash
python src/backend/app.py
```

```bash
cd src/frontend
python -m http.server 8000
```

Open:

```text
http://localhost:8000/index.html
```

## Important note

This frontend is a prototype interface. Its cloud integration references an external Lambda endpoint in the browser code, but the Lambda source and deployment configuration are not included in this repository.

