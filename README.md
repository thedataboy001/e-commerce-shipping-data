# E-Commerce Shipping Data — Classification Model

Predict whether a shipment reached on time using tabular customer and shipment data.

Kaggle dataset: https://www.kaggle.com/datasets/prachi13/customer-analytics/data

## Project Overview

This repository contains a pipeline to preprocess e-commerce shipment data, train a Random Forest classifier, log experiments with MLflow, and expose inference via a FastAPI + Gradio interface.

Key components:

- `src/feature_eng_pipeline.py` — data loading and preprocessing for training and inference.
- `src/inference_pipeline.py` — model training (Random Forest) and saving artifacts.
- `src/training_pipeline.py` — orchestration script that trains the model and saves results.
- `src/api_app.py` — FastAPI app and Gradio UI for model inference.
- `config/config.yaml` — project configuration (data paths, model hyperparameters, MLflow settings).

## Quickstart

Prerequisites:

- Docker (optional but recommended)
- Python 3.11 (if running locally)

Install dependencies locally:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements_api.txt
```

Run training locally:

```bash
python -m src.training_pipeline
```

This will load data from `data/02-preprocessed/processed_data.csv`, train the model, log metrics to MLflow, and write the inference model to the `models/` folder.

## Docker

Build the image:

```bash
docker build -t e-commerce-shipping .
```

Run the container (exposes FastAPI on port 8000 and Gradio on 7860):

```bash
docker run -p 8000:8000 -p 7860:7860 e-commerce-shipping
```

## Testing

Run unit tests with `pytest`:

```bash
pytest -q
# or run the file
pytest tests/test_training.py -q
```

## Configuration

Edit `config/config.yaml` to change data paths, model hyperparameters, MLflow tracking URI, and API settings. Sensitive values (webhooks, credentials) should be provided via environment variables or GitHub Secrets and not committed to the repo.

## File structure

Top-level files and folders:

- `data/` — raw and processed CSVs
- `models/` — model artifacts saved for inference
- `mlruns/` — MLflow local runs (if used)
- `src/` — application source code
- `tests/` — unit and integration tests
- `config/` — YAML configuration

## Notes

- The repository uses scikit-learn's `RandomForestClassifier` and saves artifacts (`scaler.joblib`, `feature_columns.joblib`, `model.pkl`) under the `models/` directory.
- MLflow is configured in `config/config.yaml`; run an MLflow server separately if you want a centralized tracking server.

## Contact

For questions or contributions, open an issue or submit a PR.