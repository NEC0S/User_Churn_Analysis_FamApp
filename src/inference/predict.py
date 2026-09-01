"""
Scoring / inference module. Used by both the FastAPI service (real-time)
and a batch scoring job (replaces notebook 3).

The critical property here: this file calls the SAME `build_user_features`
function training used, and loads the SAME pipeline artifact that was fit
during training -- there is no separate, hand-maintained copy of the
scaling/encoding logic that could quietly drift out of sync (that drift is
exactly what happened between notebooks 1/2/3, where notebook 3 had to
guess whether to load the scaled or unscaled CSV).
"""
from __future__ import annotations

import joblib
import pandas as pd

from src.features.build_features import build_user_features, get_missing_transaction_mask
from src.utils.config import get_logger, load_config, resolve_path

logger = get_logger(__name__)


def load_artifact(cfg: dict | None = None):
    cfg = cfg or load_config()
    path = resolve_path(cfg["artifacts"]["pipeline_path"])
    if not path.exists():
        raise FileNotFoundError(f"No trained pipeline found at {path}. Run `python -m src.models.train` first.")
    return joblib.load(path)


def assign_risk_band(prob_churn: float, cfg: dict) -> str:
    bands = cfg["risk_bands"]
    if prob_churn >= bands["high"]:
        return "High-risk"
    if prob_churn >= bands["medium"]:
        return "Medium-risk"
    return "Low-risk"


def score_users(users: pd.DataFrame, transactions: pd.DataFrame, as_of_date: str | None = None) -> pd.DataFrame:
    """
    End-to-end scoring for a batch of users: build features -> handle
    cold-start users explicitly -> run the trained pipeline -> attach a
    business-readable risk band.

    Cold-start handling: notebook 01 just `dropna()`-ed users with no
    transaction history. A live scoring job can't drop 8% of its users --
    it has to return *something*. Here, users with no transaction history
    are flagged and given a neutral "Insufficient-data" segment instead of
    a fabricated model score, which is safer than either crashing or
    silently guessing.
    """
    cfg = load_config()
    artifact = load_artifact(cfg)
    pipeline, threshold, feature_columns = artifact["pipeline"], artifact["threshold"], artifact["feature_columns"]
    pos_label = cfg["target"]["positive_label"]

    df = build_user_features(users, transactions, as_of_date=as_of_date)
    cold_start_mask = get_missing_transaction_mask(df)

    scoreable = df.loc[~cold_start_mask].copy()
    cold_start = df.loc[cold_start_mask, ["user_id"]].copy()

    results = []

    if len(scoreable) > 0:
        X = scoreable[[c for c in feature_columns if c in scoreable.columns]]
        proba_churn = pipeline.predict_proba(X)[:, pos_label]
        scored = pd.DataFrame({
            "user_id": scoreable["user_id"].values,
            "churn_probability": proba_churn,
            "churn_prediction": (proba_churn >= threshold).astype(int),
            "risk_segment": [assign_risk_band(p, cfg) for p in proba_churn],
        })
        results.append(scored)

    if len(cold_start) > 0:
        logger.info("%d users had no transaction history and were routed to Insufficient-data segment.", len(cold_start))
        cold_start["churn_probability"] = None
        cold_start["churn_prediction"] = None
        cold_start["risk_segment"] = "Insufficient-data"
        results.append(cold_start)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(
        columns=["user_id", "churn_probability", "churn_prediction", "risk_segment"]
    )
