"""
Training entrypoint. This is notebooks 01 + 02 rewritten as a script that
fixes the leakage issues and produces ONE deployable artifact.

Run: python -m src.models.train

KEY DIFFERENCES FROM THE ORIGINAL NOTEBOOKS
---------------------------------------------
1. Split happens immediately after building features -- before any
   scaling, encoding, or feature-selection statistic is computed. Nothing
   downstream ever sees the test set until final evaluation.
2. Three-way split (train/val/test), not two-way. Model comparison and
   threshold tuning use the VALIDATION set. The TEST set is scored exactly
   once, at the very end, to report the number you'd actually expect in
   production.
3. Preprocessing + SMOTE + model are one `Pipeline` object (see
   src/pipeline/preprocessing.py) fit only on X_train. RandomizedSearchCV
   cross-validates that whole pipeline, so every fold refits its own
   scaler/encoder -- no leakage is possible even inside CV.
4. The known leakage columns (recency-from-today, raw date columns) are
   dropped by name, sourced from config.yaml, so the fix is documented
   and can't silently regress.
5. One artifact (`churn_pipeline.joblib`) is saved -- not a bare model
   plus two separately-scaled CSVs the way notebook 3 had to guess between.
"""
from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    make_scorer, recall_score, roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from src.data.load_data import load_raw_tables
from src.features.build_features import build_user_features, get_missing_transaction_mask
from src.pipeline.preprocessing import build_full_pipeline
from src.utils.config import get_logger, load_config, resolve_path

logger = get_logger(__name__)


def prepare_dataset(cfg: dict) -> pd.DataFrame:
    """Load raw tables, engineer features, and apply row-level cleaning decisions explicitly."""
    users, transactions = load_raw_tables(cfg)

    # Use a fixed as_of_date for training so the resulting feature values
    # (and therefore the trained model) are reproducible on re-run --
    # unlike the notebooks' pd.to_datetime("today").
    df = build_user_features(users, transactions, as_of_date="2025-12-15")

    missing_mask = get_missing_transaction_mask(df)
    logger.info(
        "%d / %d users (%.1f%%) have no transaction history and will be dropped from TRAINING "
        "(the inference service handles these separately -- see src/inference/predict.py).",
        missing_mask.sum(), len(df), 100 * missing_mask.mean(),
    )
    df = df.loc[~missing_mask].copy()

    # Drop leakage / non-feature columns by name (from config, not ad hoc).
    drop_cols = [c for c in cfg["leakage_columns"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    return df


def split_data(df: pd.DataFrame, cfg: dict):
    target_col = cfg["target"]["column"]
    feature_cols = cfg["numeric_features"] + cfg["categorical_features"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df[target_col]

    # First split off the TEST set. It will not be touched again until the
    # final `evaluate on test` step at the bottom of this file.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
        stratify=y,
    )
    # Then split train/val out of what's left. Val is used for model
    # comparison and threshold tuning -- the role the notebooks
    # mistakenly gave to the test set.
    val_fraction_of_trainval = cfg["split"]["val_size"] / (1 - cfg["split"]["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_fraction_of_trainval,
        random_state=cfg["split"]["random_state"],
        stratify=y_trainval,
    )
    logger.info(
        "Split sizes -> train: %d, val: %d, test: %d", len(X_train), len(X_val), len(X_test)
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def tune_pipeline(pipeline, X_train, y_train, cfg: dict):
    """
    Hyperparameter search over the WHOLE pipeline (preprocessing + SMOTE +
    model), so every candidate is evaluated with fold-appropriate
    preprocessing -- never with statistics borrowed from other folds.
    """
    search_cfg = cfg["hyperparameter_search"]
    if not search_cfg.get("enabled", False):
        pipeline.fit(X_train, y_train)
        return pipeline

    # Positive label for churn is 0 in this dataset -- make that explicit
    # in the scorer rather than relying on sklearn's default pos_label=1.
    recall_churn = make_scorer(recall_score, pos_label=cfg["target"]["positive_label"])

    param_distributions = {f"model__{k}": [v] if not isinstance(v, list) else v
                            for k, v in cfg["model"]["xgboost_params"].items()
                            if k not in ("random_state", "n_jobs", "eval_metric")}
    # Widen a couple of key params into real search ranges for a meaningful search
    param_distributions.update({
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    })

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=search_cfg["n_iter"],
        scoring=recall_churn,
        cv=search_cfg["cv_folds"],
        random_state=search_cfg["random_state"],
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    logger.info("Best CV recall(churn): %.4f | best params: %s", search.best_score_, search.best_params_)
    return search.best_estimator_


def pick_threshold(pipeline, X_val, y_val, cfg: dict) -> float:
    """
    Sweep thresholds on the VALIDATION set (never test) and pick the one
    that maximizes F1 on the churn class -- this is where the notebooks'
    "let's try 0.4 and see" becomes a principled, data-driven choice that
    doesn't consume the test set.
    """
    pos_label = cfg["target"]["positive_label"]
    # proba_churn is already P(churn) because we sliced predict_proba at
    # pos_label's column index -- no further remapping needed. (An earlier
    # version of this function inverted `preds` here, which silently
    # picked a nonsensical threshold; caught by comparing against the
    # evaluate() function below, which computes this correctly.)
    proba_churn = pipeline.predict_proba(X_val)[:, pos_label]
    y_churn = (y_val == pos_label).astype(int)

    best_threshold, best_f1 = 0.5, -1
    for t in [i / 100 for i in range(10, 91, 2)]:
        preds = (proba_churn >= t).astype(int)
        f1 = f1_score(y_churn, preds)
        if f1 > best_f1:
            best_f1, best_threshold = f1, t
    logger.info("Selected decision threshold=%.2f on validation set (F1=%.4f)", best_threshold, best_f1)
    return best_threshold


def evaluate(pipeline, X, y, threshold: float, cfg: dict, split_name: str) -> dict:
    pos_label = cfg["target"]["positive_label"]
    proba_churn = pipeline.predict_proba(X)[:, pos_label]
    preds_churn = (proba_churn >= threshold).astype(int)
    y_churn = (y == pos_label).astype(int)

    report = classification_report(y_churn, preds_churn, output_dict=True)
    auc = roc_auc_score(y_churn, proba_churn)
    cm = confusion_matrix(y_churn, preds_churn).tolist()

    logger.info("[%s] ROC-AUC=%.4f | churn recall=%.4f | churn f1=%.4f",
                split_name, auc, report["1"]["recall"], report["1"]["f1-score"])

    return {"split": split_name, "roc_auc": auc, "classification_report": report, "confusion_matrix": cm}


def main():
    cfg = load_config()
    df = prepare_dataset(cfg)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, cfg)

    pipeline = build_full_pipeline(
        numeric_features=[c for c in cfg["numeric_features"] if c in X_train.columns],
        categorical_features=[c for c in cfg["categorical_features"] if c in X_train.columns],
        algorithm=cfg["model"]["algorithm"],
        model_params={k: v for k, v in cfg["model"]["xgboost_params"].items()},
        use_smote=cfg["model"]["use_smote"],
        random_state=cfg["split"]["random_state"],
    )

    pipeline = tune_pipeline(pipeline, X_train, y_train, cfg)
    threshold = pick_threshold(pipeline, X_val, y_val, cfg)

    val_metrics = evaluate(pipeline, X_val, y_val, threshold, cfg, "validation")
    test_metrics = evaluate(pipeline, X_test, y_test, threshold, cfg, "test")  # touched exactly once

    # Persist the ONE artifact that inference will load: the full pipeline
    # plus the threshold it was tuned with. Bundling the threshold here
    # (instead of hardcoding it again in the serving code) is what stops
    # training and serving from drifting apart.
    artifact = {"pipeline": pipeline, "threshold": threshold, "feature_columns": list(X_train.columns)}
    pipeline_path = resolve_path(cfg["artifacts"]["pipeline_path"])
    pipeline_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, pipeline_path)
    logger.info("Saved pipeline artifact -> %s", pipeline_path)

    metrics_path = resolve_path(cfg["artifacts"]["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({"validation": val_metrics, "test": test_metrics, "threshold": threshold}, f, indent=2, default=str)
    logger.info("Saved metrics -> %s", metrics_path)


if __name__ == "__main__":
    main()
