"""
Unit tests for the feature-engineering module. These would run in CI on
every PR -- catching, for example, a future edit that reintroduces
`pd.to_datetime("today")` and breaks reproducibility.
"""
import pandas as pd
import pytest

from src.features.build_features import build_user_features, get_missing_transaction_mask


@pytest.fixture
def sample_users():
    return pd.DataFrame({
        "user_id": [1, 2, 3],
        "age_group": ["adult", "teen", "senior"],
        "is_active": [1, 0, 1],
    })


@pytest.fixture
def sample_transactions():
    return pd.DataFrame({
        "transaction_id": [101, 102, 103, 104],
        "user_id": [1, 1, 2, 2],
        "transaction_date": ["2025-01-01", "2025-01-10", "2025-02-01", "2025-02-05"],
        "amount": [100.0, 200.0, 50.0, 75.0],
        "status": ["success", "success", "success", "failed"],
        "transaction_type": ["p2p", "bill_payment", "p2p", "merchant"],
        "payment_method": ["upi", "card", "upi", "upi"],
        "merchant": ["m1", "m2", "m1", "m3"],
    })


def test_build_user_features_row_count_matches_users(sample_users, sample_transactions):
    """Every user should get exactly one output row, transactions or not (user 3 has none)."""
    result = build_user_features(sample_users, sample_transactions, as_of_date="2025-03-01")
    assert len(result) == len(sample_users)
    assert set(result["user_id"]) == {1, 2, 3}


def test_recency_features_are_reproducible_given_fixed_as_of_date(sample_users, sample_transactions):
    """
    The core bug this project fixes: recomputing features on two different
    days must give IDENTICAL values when as_of_date is fixed explicitly.
    """
    run_1 = build_user_features(sample_users, sample_transactions, as_of_date="2025-03-01")
    run_2 = build_user_features(sample_users, sample_transactions, as_of_date="2025-03-01")
    pd.testing.assert_series_equal(
        run_1["days_since_last_transaction"], run_2["days_since_last_transaction"]
    )


def test_cold_start_users_are_flagged_not_dropped(sample_users, sample_transactions):
    """User 3 has no transactions -- must appear with NaNs, not be silently removed."""
    result = build_user_features(sample_users, sample_transactions, as_of_date="2025-03-01")
    mask = get_missing_transaction_mask(result)
    assert mask.sum() == 1
    assert result.loc[mask, "user_id"].iloc[0] == 3


def test_aggregation_values_are_correct(sample_users, sample_transactions):
    result = build_user_features(sample_users, sample_transactions, as_of_date="2025-03-01")
    user_1 = result[result["user_id"] == 1].iloc[0]
    assert user_1["total_transactions"] == 2
    assert user_1["total_amount"] == 300.0
    assert user_1["unique_merchants"] == 2
