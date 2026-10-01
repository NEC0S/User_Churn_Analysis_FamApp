"""
ChurnLens backend — serves real predictions from the trained LightGBM pipeline,
with a per-prediction SHAP explanation, for the connected frontend
(frontend/churn_lens_connected.html) to call.

Run locally:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Then open frontend/churn_lens_connected.html and point its
"Backend API URL" field at http://localhost:8000/predict
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Load the model + metadata
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(HERE, "model_artifact")
PIPELINE_PATH = os.path.join(ARTIFACT_DIR, "churn_pipeline.joblib")

# Prefer model_card.json (written by the production notebook). Fall back to the
# older metadata.json name/key format if that's all that's present, so a stale
# folder state degrades gracefully instead of crashing on startup.
MODEL_CARD_PATH = os.path.join(ARTIFACT_DIR, "model_card.json")
LEGACY_METADATA_PATH = os.path.join(ARTIFACT_DIR, "metadata.json")

if os.path.exists(MODEL_CARD_PATH):
    metadata = json.load(open(MODEL_CARD_PATH))
    MODEL_NAME = metadata["champion_model"]
    THRESHOLD = metadata["deployment_threshold"]
elif os.path.exists(LEGACY_METADATA_PATH):
    print("WARNING: model_card.json not found, falling back to legacy metadata.json. "
          "Re-run the production notebook to regenerate the current artifact.")
    metadata = json.load(open(LEGACY_METADATA_PATH))
    MODEL_NAME = metadata["model_name"]
    THRESHOLD = metadata["decision_threshold"]
else:
    raise FileNotFoundError(
        f"No model_card.json or metadata.json found in {ARTIFACT_DIR}. "
        "Run the production notebook first to generate the deployment artifact."
    )

FEATURE_COLUMNS = metadata["feature_columns"]
pipeline = joblib.load(PIPELINE_PATH)

_clf = pipeline.named_steps["clf"]
_pre = pipeline.named_steps["prep"]
_ohe = _pre.named_transformers_["cat"]
_ENCODED_FEATURE_NAMES = _pre.transformers_[0][2] + list(_ohe.get_feature_names_out(_pre.transformers_[1][2]))
_explainer = shap.TreeExplainer(_clf)

print(f"Loaded model '{MODEL_NAME}' | threshold={THRESHOLD:.3f} | "
      f"{len(FEATURE_COLUMNS)} features from {ARTIFACT_DIR}")

# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------
app = FastAPI(title="ChurnLens API")

# Add every origin the frontend will actually be served from. localhost:8080 covers
# `python -m http.server 8080` during local testing; add your deployed frontend's
# real URL once you host it (Netlify/Vercel/GitHub Pages), e.g. "https://churnlens.netlify.app".
ALLOWED_ORIGINS = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:5173",   # Vite dev server default
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Feature defaults for fields the simplified UI doesn't collect
# ---------------------------------------------------------------------------
DEFAULTS = {
    "age_group": "23-30",
    "city": "Mumbai",
    "device_type": "Android",
    "has_customized_card": 0,
    "has_set_savings_goal": 0,
    "app_opens_per_week": 10,
    "avg_session_duration": 5.0,
    "referrals_made": 0,
    "top_txn_type": "Bills",
}


class PredictRequest(BaseModel):
    txn_count: int = Field(..., ge=0, description="Transactions in the feature window")
    total_amount: float = Field(..., ge=0, description="Total spend in the feature window")
    days_since_last_txn: int = Field(..., ge=0)
    txn_trend: float = Field(0, description="late-half count minus early-half count")
    failure_rate: float = Field(0, ge=0, le=1, description="fraction of failed transactions, e.g. 0.04")
    support_tickets: int = Field(0, ge=0)
    tenure_days_at_cutoff: int = Field(0, ge=0)
    has_used_offers: bool = False
    overrides: dict = Field(default_factory=dict, description="Explicit values for any field not collected by the UI")


def build_feature_row(req: PredictRequest) -> pd.DataFrame:
    row = dict(DEFAULTS)
    row.update({
        "txn_count": req.txn_count,
        "total_amount": req.total_amount,
        "days_since_last_txn": req.days_since_last_txn if req.txn_count > 0 else 999,
        "txn_trend": req.txn_trend,
        "failure_rate": req.failure_rate,
        "support_tickets": req.support_tickets,
        "tenure_days_at_cutoff": req.tenure_days_at_cutoff,
        "has_used_offers": int(req.has_used_offers),
    })

    # Derive fields the simplified UI doesn't ask for, from what it does ask for.
    row["avg_amount"] = req.total_amount / req.txn_count if req.txn_count > 0 else 0
    row["std_amount"] = row["avg_amount"] * 0.4
    row["n_merchants"] = min(max(req.txn_count, 0), 6)
    row["n_payment_methods"] = 2 if req.txn_count > 3 else (1 if req.txn_count > 0 else 0)
    row["n_txn_types"] = min(max(req.txn_count, 0), 4)
    row["active_span_days"] = 20 if req.txn_count > 0 else 0
    late = max(0, req.txn_count / 2 + req.txn_trend / 2)
    row["txn_count_late"] = round(late)
    row["txn_count_early"] = max(0, req.txn_count - round(late))

    # Explicit overrides win last, for callers who have real data for these fields
    row.update(req.overrides)

    missing = [c for c in FEATURE_COLUMNS if c not in row]
    if missing:
        raise KeyError(f"features not derivable and not overridden: {missing}")

    return pd.DataFrame([row])[FEATURE_COLUMNS]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api")
def root():
    return {"service": "ChurnLens API", "docs": "/docs", "health": "/health", "predict": "POST /predict"}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "threshold": THRESHOLD, "n_features": len(FEATURE_COLUMNS)}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        X = build_feature_row(req)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature: {e}")

    try:
        X_pre = _pre.transform(X)
        proba = float(pipeline.predict_proba(X)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    flagged = proba >= THRESHOLD

    # Real per-prediction explanation: SHAP values for THIS row, not a fixed global ranking.
    sv = _explainer.shap_values(X_pre)
    if isinstance(sv, list):
        sv = sv[1]  # positive (churn) class
    sv_row = np.asarray(sv)[0]
    order = np.argsort(-np.abs(sv_row))[:5]
    top_factors = [
        {"feature": _ENCODED_FEATURE_NAMES[i], "contribution": float(sv_row[i])}
        for i in order
    ]

    return {
        "probability": proba,
        "threshold": THRESHOLD,
        "flagged": flagged,
        "top_factors": top_factors,  # signed: positive -> pushes toward churn, negative -> toward retention
        "features_used": X.iloc[0].to_dict(),
    }

# ---------------------------------------------------------------------------
# Serve the built React frontend (single-container deployment)
# ---------------------------------------------------------------------------
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

STATIC_DIR = os.path.join(HERE, "static")

if os.path.isdir(STATIC_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(STATIC_DIR, "assets")), name="assets")

    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        candidate = os.path.realpath(os.path.join(STATIC_DIR, full_path))
        # serve real files (favicon etc.), staying inside STATIC_DIR
        if full_path and candidate.startswith(os.path.realpath(STATIC_DIR) + os.sep) and os.path.isfile(candidate):
            return FileResponse(candidate)
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))