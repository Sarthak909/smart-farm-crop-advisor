# Smart Farm Crop & Resource Advisor

A small end-to-end machine-learning project that recommends a crop from soil, weather, and farm-management conditions. It uses the Kaggle **Smart Farming Data 2024 (SF24)** dataset, with 2,200 samples, 22 input features, and 22 crop labels.

## Project flow

1. Explore and split the dataset, train, and evaluate the classifier in the Jupyter notebook. The notebook is the training source of truth and was executed outside VS Code.
2. Save the preprocessing and fitted Random Forest pipeline to `model.joblib`. Save split metrics, dataset details, and estimator parameters to `model_metadata.json`.
3. Serve the model through FastAPI (`/predict`, `/health`, and `/model-info`).
4. Build and run the API as a Docker container.
5. Run GitHub Actions on pushes and pull requests to execute API checks and build the Docker image.

## Dataset

- Source: [Smart Farming Data 2024 (SF24) on Kaggle](https://www.kaggle.com/datasets/datasetengineer/smart-farming-data-2024-sf24)
- File: `data/Crop_recommendationV2.csv`
- License shown on Kaggle: CC0: Public Domain
- Target: `label` (recommended crop)
- Inputs: nutrient indicators (N, P, K), weather and soil conditions, and derived farm-management indicators

This is a curated practice dataset. Some attributes are derived or generated, so the score below describes the dataset's own distribution and is not evidence of real-world agricultural performance.

## Model and evaluation

The notebook performs a stratified 80/20 train/test split with `random_state=42`. It fits a scikit-learn pipeline with one-hot encoding for discrete soil/farm categories and a 450-tree Random Forest classifier. The saved pipeline contains the fitted preprocessing and estimator parameters; `model_metadata.json` records the hyperparameters and evaluation details.

On the held-out 440-row test set, the executed notebook measured:

- Accuracy: **0.9977** (rounded to 99.8%)
- Macro F1: **0.9977** (rounded to 0.998)

The model's top-three scores are relative class scores and are not calibrated agronomic probabilities.

## Train or reproduce the model in Jupyter / Colab

The assignment specifies Jupyter or Colab for model training. From the repository root:

```bash
python -m pip install -r requirements-notebook.txt
jupyter lab notebooks/smart_farm_crop_advisor.ipynb
```

Run every notebook cell. It reads the bundled CSV and regenerates `model.joblib` and `model_metadata.json` in the repository root. For Colab, upload or clone the repository so the notebook and `data/` directory are both available.

## Run the FastAPI service in VS Code / terminal

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API requests. `POST /predict` expects all 22 input features; `GET /health` checks service readiness and `GET /model-info` returns model metadata.

## Run with Docker

```bash
docker compose up --build
```

Then open `http://localhost:8000/docs`. The container health check calls `/health`.

## Checks and CI

Run the same API checks locally with:

```bash
python -m pip install -r requirements.txt
pytest -q
```

`.github/workflows/ci.yml` runs the checks and builds the Docker image on pushes to `main`/`master`, pull requests, and manual dispatch.

## Repository layout

```text
app/main.py                         FastAPI app and request schema
data/Crop_recommendationV2.csv      Kaggle source dataset
model.joblib                        Fitted preprocessing + classifier pipeline
model_metadata.json                 Dataset, split metrics, and hyperparameters
notebooks/smart_farm_crop_advisor.ipynb  Training and evaluation notebook
tests/test_api.py                   API and artifact checks
Dockerfile, docker-compose.yml      Container setup
.github/workflows/ci.yml            GitHub Actions CI
```
