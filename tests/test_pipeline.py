"""
Tests for the preprocessing/model pipeline. Focused on the properties that
actually matter for a production ML system: the pipeline must be fittable
end-to-end, must survive unseen categories at inference, and must not leak
test-set statistics into training.
"""
import numpy as np
import pandas as pd
import pytest

from src.pipeline.preprocessing import build_full_pipeline


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({
        "total_transactions": rng.integers(0, 50, n),
        "active_days": rng.integers(0, 30, n),
        "age_group": rng.choice(["teen", "adult", "senior"], n),
        "city": rng.choice(["Mumbai", "Delhi"], n),
    })
    y = pd.Series(rng.choice([0, 1], n, p=[0.3, 0.7]))
    return X, y


def test_pipeline_fits_and_predicts(toy_data):
    X, y = toy_data
    pipeline = build_full_pipeline(
        numeric_features=["total_transactions", "active_days"],
        categorical_features=["age_group", "city"],
        algorithm="logistic_regression",
        model_params={"max_iter": 200, "random_state": 42},
        use_smote=True,
    )
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    proba = pipeline.predict_proba(X)
    assert len(preds) == len(X)
    assert proba.shape == (len(X), 2)


def test_pipeline_handles_unseen_category_at_inference(toy_data):
    """A category not present during training (e.g. a brand-new city) must not crash scoring."""
    X, y = toy_data
    pipeline = build_full_pipeline(
        numeric_features=["total_transactions", "active_days"],
        categorical_features=["age_group", "city"],
        algorithm="logistic_regression",
        model_params={"max_iter": 200, "random_state": 42},
        use_smote=True,
    )
    pipeline.fit(X, y)

    new_row = X.iloc[[0]].copy()
    new_row["city"] = "Kolkata"  # unseen during training
    # Should not raise -- handle_unknown="ignore" in the OneHotEncoder is the reason.
    proba = pipeline.predict_proba(new_row)
    assert proba.shape == (1, 2)


def test_pipeline_preprocessor_is_refit_per_call_no_state_leaks_across_fits(toy_data):
    """
    Fitting the pipeline twice on two different subsets should give
    different learned statistics -- proof the scaler isn't silently
    reusing global statistics computed once on a larger/different dataset
    (the exact bug this project's pipeline design prevents).
    """
    X, y = toy_data
    pipeline = build_full_pipeline(
        numeric_features=["total_transactions", "active_days"],
        categorical_features=["age_group", "city"],
        algorithm="logistic_regression",
        model_params={"max_iter": 200, "random_state": 42},
        use_smote=False,
    )
    pipeline.fit(X.iloc[:100], y.iloc[:100])
    mean_a = pipeline.named_steps["preprocessor"].named_transformers_["numeric"].named_steps["scaler"].mean_.copy()

    pipeline.fit(X.iloc[100:], y.iloc[100:])
    mean_b = pipeline.named_steps["preprocessor"].named_transformers_["numeric"].named_steps["scaler"].mean_.copy()

    assert not np.allclose(mean_a, mean_b), "Scaler statistics should differ between two different training subsets"
