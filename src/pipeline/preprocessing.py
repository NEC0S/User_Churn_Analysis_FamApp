"""
Builds ONE sklearn-compatible pipeline object that owns every stateful
transform: imputation -> scaling -> one-hot encoding -> SMOTE -> model.

WHY THIS IS THE MOST IMPORTANT FILE IN THE PROJECT
----------------------------------------------------
The original notebooks fit `StandardScaler` and did correlation/chi2-based
feature selection on the FULL dataset before ever calling
`train_test_split`. That means the scaler's mean/std, and the decision of
which columns to keep, were both informed by rows that were later labeled
"test data" -- a leak that inflates every reported metric.

Wrapping everything in a single `imblearn.pipeline.Pipeline` fixes this
structurally, not just by "remembering to split first":
  - `pipeline.fit(X_train, y_train)` only ever sees training rows, so the
    scaler/encoder statistics can never see test data.
  - When this same pipeline is used inside `RandomizedSearchCV`, sklearn
    refits the imputer/scaler/encoder from scratch on each CV fold's
    training slice -- so even cross-validation can't leak.
  - SMOTE lives INSIDE the pipeline (via imblearn, not sklearn's Pipeline)
    so it's only ever applied to a training fold, never to validation/test
    data. Oversampling the validation or test set (even indirectly, via a
    global SMOTE call before the split) fabricates synthetic examples that
    make evaluation numbers meaningless.
  - At inference time, you load and call ONE object. There's no way to
    accidentally apply the transforms in the wrong order or skip a step --
    which is exactly the train/serve-skew risk notebook 3 had (it silently
    assumed which of the scaled/non-scaled CSVs matched the saved model).
"""
from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """
    Numeric branch: median-impute (robust to outliers, unlike mean) then
    standard-scale.
    Categorical branch: most-frequent-impute then one-hot encode, with
    `handle_unknown="ignore"` so a category never seen in training (a new
    city, a new payment method the business adds next quarter) doesn't
    crash the service in production -- it just gets an all-zero encoding
    instead of raising.
    """
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_features),
            ("categorical", categorical_pipe, categorical_features),
        ],
        remainder="drop",  # explicit: any column not listed is intentionally excluded
    )
    return preprocessor


def build_model(algorithm: str, params: dict):
    """Factory so config.yaml can pick the algorithm without touching code."""
    if algorithm == "xgboost":
        return XGBClassifier(**params)
    if algorithm == "random_forest":
        return RandomForestClassifier(**params)
    if algorithm == "logistic_regression":
        return LogisticRegression(**params)
    raise ValueError(f"Unknown algorithm '{algorithm}' in config.yaml")


def build_full_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    algorithm: str,
    model_params: dict,
    use_smote: bool = True,
    random_state: int = 42,
) -> ImbPipeline:
    """
    Returns the single artifact that gets fit on training data and saved
    with joblib. Everything downstream (RandomizedSearchCV, evaluation,
    the FastAPI service) interacts with this one object -- never with the
    scaler or model directly -- so preprocessing can never drift out of
    sync between training and serving.
    """
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    model = build_model(algorithm, model_params)

    steps = [("preprocessor", preprocessor)]
    if use_smote:
        # SMOTE must come AFTER preprocessing (it needs numeric input) and
        # will only ever be invoked on the training fold that reaches
        # .fit() -- imblearn's Pipeline is smote-aware and skips resampling
        # during .transform()/.predict(), which is exactly what you want.
        steps.append(("smote", SMOTE(random_state=random_state)))
    steps.append(("model", model))

    return ImbPipeline(steps=steps)
