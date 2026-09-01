# Churn Prediction — Production Project

A production-ready rebuild of the original 3-notebook churn project
(`01_EDA_Feature_Engineering`, `02_Model_Training_Evaluation`,
`03_Model_Interpretation_Business_Insights`).

## What changed vs. the original notebooks

| Issue in the notebooks | Fix here |
|---|---|
| Scaler/feature-selection fit on the **full** dataset before splitting | `train_test_split` happens first; every stateful transform lives inside a `Pipeline` fit only on `X_train` |
| SMOTE applied outside cross-validation | SMOTE is a pipeline step (`imblearn.Pipeline`), so it's refit per-fold and never touches val/test |
| Test set reused repeatedly for model choice + threshold tuning | Three-way split — **validation** is used for tuning, **test** is scored exactly once |
| `pd.to_datetime("today")` in recency features (non-reproducible) | `as_of_date` is an explicit parameter everywhere |
| Feature engineering duplicated/re-typed across notebooks 1 & 3 | One function, `build_user_features()`, used by training *and* inference |
| `dropna()` silently discarded ~8% of users | Cold-start users are explicitly flagged and routed to an "Insufficient-data" segment instead of the model |
| Hardcoded `../data/...` paths, thresholds, hyperparameters | Everything lives in `config/config.yaml` |
| Model saved with `joblib.dump(model, ...)`, notebook 3 had to guess which CSV (scaled/unscaled) matched it | One artifact: `{pipeline, threshold, feature_columns}` — impossible to mismatch |
| No tests, no serving layer, no way to score a new batch of users | `pytest` suite, FastAPI service, Dockerfile |

## Project layout

```
config/config.yaml          # every path, threshold, hyperparameter — nothing hardcoded in code
src/
  data/                      # raw data loading + synthetic data generator for demos/CI
  features/build_features.py # single source of truth for feature engineering (train + inference)
  pipeline/preprocessing.py  # ColumnTransformer + SMOTE + model, one fittable object
  models/train.py            # split -> tune -> threshold -> evaluate -> save ONE artifact
  inference/predict.py       # batch scoring, cold-start handling, risk banding
  utils/config.py            # config loader + logger
api/main.py                  # FastAPI serving layer
tests/                       # pytest unit tests
notebooks/04_Advanced_Modeling_and_Interpretation.ipynb   # the exploratory/advanced-ML notebook
Dockerfile
requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt
export PYTHONPATH=$(pwd)

# 1. No real data on hand? Generate a synthetic dataset with the same schema:
python -m src.data.generate_synthetic_data

# 2. Run tests
pytest tests/ -v

# 3. Train (fixes leakage, tunes hyperparameters, saves saved_models/churn_pipeline.joblib)
python -m src.models.train

# 4. Serve
uvicorn api.main:app --reload --port 8000
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"user_id": 1, ...}'
```

To use your **real** data, drop `fam_users.csv` / `fam_transactions.csv` into
`data/raw/` (paths are configurable in `config/config.yaml`) and skip step 1.

## Docker

```bash
docker build -t churn-service .
docker run -p 8000:8000 churn-service
```

## What's still needed for a full production rollout

This project fixes the modeling/engineering issues and gives you a
deployable service — it does **not** include, and a real rollout should add:

- **CI/CD**: a GitHub Actions (or similar) workflow running `pytest` and a
  minimum-AUC gate on every PR, auto-building/pushing the Docker image.
- **Experiment tracking / model registry**: log each training run (params,
  metrics, data snapshot) to MLflow or similar, and promote to production
  only when a challenger beats the incumbent on the untouched test set.
- **Feature store / scheduled feature refresh**: right now `build_user_features`
  is called on-demand; at real scale you'd materialize it on a schedule.
- **Monitoring**: track input feature drift and prediction-distribution
  drift over time, alert when they move, and set a retraining trigger.
- **A/B or shadow evaluation**: validate the model against real business
  outcomes before it drives retention campaigns.
