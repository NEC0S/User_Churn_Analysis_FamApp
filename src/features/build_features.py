"""
Feature engineering, extracted from notebook 01 into a pure, testable
function.

WHY THIS MATTERS FOR PRODUCTION
--------------------------------
In the original notebooks, feature engineering lived in one-off cells that
only ran once, in order, inside a Jupyter kernel. That's fine for
exploration but breaks the moment you need to:
  1. Score a *new* batch of users tomorrow (you'd have to re-run/re-copy
     notebook cells).
  2. Guarantee training and inference compute features identically
     (train/serve skew is one of the most common causes of a model that
     works in the notebook but degrades in production).
  3. Reproduce a training run from six months ago.

`build_user_features()` fixes all three: it's a plain function with no
notebook state, it's the single source of truth called by both train.py and
predict.py, and it takes `as_of_date` explicitly instead of the notebooks'
`pd.to_datetime("today")` (which silently changes every feature value
depending on which day you happen to run the code).
"""
from __future__ import annotations

import pandas as pd


def _aggregate_transactions(transactions: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """
    Roll the raw, one-row-per-transaction table up to one row per user.

    Mirrors notebook 01's aggregation, with two fixes:
      - `as_of_date` is a parameter, not "today" -- so re-running this for
        a training snapshot from last month gives the same numbers it gave
        last month.
      - transaction_failure_rate is computed defensively (won't blow up if
        a user has zero transactions after a future filter is added).
    """
    transactions = transactions.copy()
    transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])

    agg = transactions.groupby("user_id").agg(
        total_transactions=("transaction_id", "count"),
        total_amount=("amount", "sum"),
        avg_amount=("amount", "mean"),
        max_amount=("amount", "max"),
        min_amount=("amount", "min"),
        active_days=("transaction_date", lambda x: x.nunique()),
        unique_merchants=("merchant", "nunique"),
        transaction_failure_rate=("status", lambda x: (x == "failed").mean()),
        most_common_payment_method=(
            "payment_method", lambda x: x.mode().iat[0] if not x.mode().empty else "unknown"
        ),
        most_common_transaction_type=(
            "transaction_type", lambda x: x.mode().iat[0] if not x.mode().empty else "unknown"
        ),
        last_transaction_date=("transaction_date", "max"),
        first_transaction_date=("transaction_date", "min"),
    ).reset_index()

    agg["days_between_first_and_last_txn"] = (
        agg["last_transaction_date"] - agg["first_transaction_date"]
    ).dt.days

    # Recency relative to an explicit, reproducible reference date -- NOT
    # datetime.now(). This is what the notebooks got wrong: recomputing
    # this feature a week later would silently shift every user's value by
    # 7 days even though nothing about the user changed.
    agg["days_since_last_transaction"] = (as_of_date - agg["last_transaction_date"]).dt.days

    return agg


def build_user_features(
    users: pd.DataFrame,
    transactions: pd.DataFrame,
    as_of_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Build the full user-level feature table from raw `users` and
    `transactions` tables.

    Parameters
    ----------
    users : raw users table (one row per user)
    transactions : raw transactions table (one row per transaction)
    as_of_date : the reference "now" for recency features. Pass an explicit
        date for training-set snapshots / backfills; omit it in a live
        scoring job to default to the current timestamp.

    Returns
    -------
    DataFrame, one row per user_id, with engineered transaction features
    merged onto the user attributes. Users with no transactions at all get
    NaNs in the transaction-derived columns rather than being silently
    dropped -- the caller (training script or inference service) decides
    how to handle that, instead of it being buried in a notebook cell.
    """
    if as_of_date is None:
        as_of_date = pd.Timestamp.utcnow().tz_localize(None)
    else:
        as_of_date = pd.to_datetime(as_of_date)

    txn_features = _aggregate_transactions(transactions, as_of_date)

    df = pd.merge(users, txn_features, on="user_id", how="left")
    return df


def get_missing_transaction_mask(df: pd.DataFrame) -> pd.Series:
    """
    Users with zero transaction history (new sign-ups, typically) will have
    NaN in every transaction-derived column after the left join above.

    Notebook 01 just called `.dropna()` and threw these ~8% of rows away.
    That's a defensible choice for an offline analysis, but in a live
    scoring service you cannot skip 8% of your users -- you have to decide
    what a "new user" churn score even means (e.g. route them to a
    heuristic / cold-start rule instead of the ML model). This helper makes
    that population explicit so the caller has to make a deliberate choice.
    """
    return df["total_transactions"].isna()
