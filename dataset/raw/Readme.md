# Raw Dataset

This folder contains the original dataset used in the project.

## File

- Data_for_UCI_named.csv

## Dataset information

This is the Smart Grid Stability Prediction dataset used by the project. It contains the raw smart-grid operating measurements before preprocessing.

## What the project uses

The raw file is loaded by the preprocessing script:

```python
src/ai_model/preprocess.py
```

The script performs:

- duplicate removal
- missing value cleanup
- feature selection
- standard scaling
- train/validation/test split
- KNN graph construction

## Actual dataset size

From the current repository data:

- 10,000 rows
- 14 columns
- 12 feature columns used for modeling
- target column: stabf

