"""
Minimal FastAPI service that exposes the trained churn pipeline over HTTP.

Run locally:    uvicorn api.main:app --reload --port 8000
Run in Docker:  see Dockerfile

Design notes:
- The model artifact is loaded ONCE at process startup (not per-request) --
  loading a joblib pipeline on every call would add real latency and load
  under traffic.
- Input validation is handled by Pydantic models, so malformed requests
  fail fast with a clear 422 error instead of throwing an obscure pandas
  KeyError deep inside the pipeline.
- /health is separate from /predict so a load balancer / k8s liveness
  probe can check the service is up without paying for a model call.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference.predict import assign_risk_band, load_artifact
from src.utils.config import get_logger, load_config

logger = get_logger(__name__)

_state: dict = {}  # holds the loaded pipeline/config for the lifetime of the process


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model artifact...")
    cfg = load_config()
    _state["cfg"] = cfg
    _state["artifact"] = load_artifact(cfg)
    logger.info("Model artifact loaded. Service ready.")
    yield
    _state.clear()


app = FastAPI(title="Churn Prediction Service", version="1.0.0", lifespan=lifespan)


class UserFeatures(BaseModel):
    """
    A single user's ALREADY-ENGINEERED feature row (i.e. the output of
    build_user_features for one user_id). Keeping the API contract at the
    feature level -- rather than accepting raw transaction lists per
    request -- keeps latency predictable; a batch job upstream (or a
    feature store) is responsible for keeping these features fresh.
    """
    user_id: int
    days_since_registration: Optional[float] = None
    app_opens_per_week: Optional[float] = None
    avg_session_duration: Optional[float] = None
    support_tickets: Optional[float] = None
    referrals_made: Optional[float] = None
    has_customized_card: Optional[int] = None
    has_set_savings_goal: Optional[int] = None
    has_used_offers: Optional[int] = None
    total_transactions: Optional[float] = None
    total_amount: Optional[float] = None
    avg_amount: Optional[float] = None
    max_amount: Optional[float] = None
    min_amount: Optional[float] = None
    active_days: Optional[float] = None
    unique_merchants: Optional[float] = None
    transaction_failure_rate: Optional[float] = None
    days_between_first_and_last_txn: Optional[float] = None
    age_group: Optional[str] = None
    device_type: Optional[str] = None
    city: Optional[str] = None
    most_common_payment_method: Optional[str] = None
    most_common_transaction_type: Optional[str] = None


class PredictionResponse(BaseModel):
    user_id: int
    churn_probability: float = Field(..., ge=0, le=1)
    churn_prediction: int
    risk_segment: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "artifact" in _state}


@app.post("/predict", response_model=PredictionResponse)
def predict(user: UserFeatures):
    if "artifact" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    cfg = _state["cfg"]
    artifact = _state["artifact"]
    pipeline, threshold, feature_columns = artifact["pipeline"], artifact["threshold"], artifact["feature_columns"]
    pos_label = cfg["target"]["positive_label"]

    row = pd.DataFrame([user.model_dump(exclude={"user_id"})])
    missing_cols = [c for c in feature_columns if c not in row.columns]
    if missing_cols:
        raise HTTPException(status_code=422, detail=f"Missing required features: {missing_cols}")

    try:
        proba_churn = pipeline.predict_proba(row[feature_columns])[:, pos_label][0]
    except Exception as exc:  # noqa: BLE001 -- surface pipeline errors as a clean 500, log the real cause
        logger.exception("Prediction failed for user_id=%s", user.user_id)
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    return PredictionResponse(
        user_id=user.user_id,
        churn_probability=float(proba_churn),
        churn_prediction=int(proba_churn >= threshold),
        risk_segment=assign_risk_band(proba_churn, cfg),
    )
