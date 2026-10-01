
"""
Production scoring module for the FAM churn model.
Import `assemble_modeling_table` and reuse it for BOTH retraining and live scoring, so feature
logic never diverges between training and serving (the most common source of production churn-
model bugs). See model_card.json in this folder for the trained threshold, feature list, and
known limitations before using this in a live pipeline.
"""
import pandas as pd
import joblib
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def build_features(txns, cutoff, feature_start):
    feat_txns = txns[(txns['transaction_date'] >= feature_start) & (txns['transaction_date'] <= cutoff)].copy()
    agg = feat_txns.groupby('user_id').agg(
        txn_count=('transaction_id', 'count'),
        total_amount=('amount', 'sum'),
        avg_amount=('amount', 'mean'),
        std_amount=('amount', 'std'),
        n_merchants=('merchant', 'nunique'),
        n_payment_methods=('payment_method', 'nunique'),
        n_txn_types=('transaction_type', 'nunique'),
        last_txn_date=('transaction_date', 'max'),
        first_txn_date=('transaction_date', 'min'),
    ).reset_index()
    agg['days_since_last_txn'] = (cutoff - agg['last_txn_date']).dt.days
    agg['active_span_days'] = (agg['last_txn_date'] - agg['first_txn_date']).dt.days
    fail = feat_txns.groupby('user_id')['status'].apply(lambda s: (s == 'Failed').mean()).rename('failure_rate')
    agg = agg.merge(fail, on='user_id', how='left')
    mid = feature_start + (cutoff - feature_start) / 2
    early = feat_txns[feat_txns['transaction_date'] <= mid].groupby('user_id').size().rename('txn_count_early')
    late = feat_txns[feat_txns['transaction_date'] > mid].groupby('user_id').size().rename('txn_count_late')
    agg = agg.merge(early, on='user_id', how='left').merge(late, on='user_id', how='left')
    agg[['txn_count_early', 'txn_count_late']] = agg[['txn_count_early', 'txn_count_late']].fillna(0)
    agg['txn_trend'] = agg['txn_count_late'] - agg['txn_count_early']
    top_type = (feat_txns.groupby(['user_id', 'transaction_type']).size()
                .reset_index(name='n').sort_values('n', ascending=False)
                .drop_duplicates('user_id')[['user_id', 'transaction_type']]
                .rename(columns={'transaction_type': 'top_txn_type'}))
    agg = agg.merge(top_type, on='user_id', how='left')
    agg = agg.drop(columns=['last_txn_date', 'first_txn_date'])
    return agg


def assemble_modeling_table(users, txns, cutoff, feature_start, label_end=None):
    agg = build_features(txns, cutoff, feature_start)
    users = users.copy()
    users['tenure_days_at_cutoff'] = (cutoff - users['registration_date']).dt.days
    SAFE_USER_COLS = ['user_id', 'age_group', 'city', 'device_type', 'tenure_days_at_cutoff',
                       'has_customized_card', 'has_set_savings_goal', 'has_used_offers']
    MEDIUM_RISK_COLS = ['app_opens_per_week', 'avg_session_duration', 'support_tickets', 'referrals_made']
    table = users[SAFE_USER_COLS + MEDIUM_RISK_COLS].merge(agg, on='user_id', how='left')
    txn_feature_cols = [c for c in agg.columns if c not in ('user_id', 'top_txn_type')]
    table[txn_feature_cols] = table[txn_feature_cols].fillna(0)
    table['days_since_last_txn'] = table['days_since_last_txn'].replace(0, 999).where(table['txn_count'] > 0, 999)
    table['top_txn_type'] = table['top_txn_type'].fillna('None')
    if label_end is not None:
        label_txns = txns[(txns['transaction_date'] > cutoff) & (txns['transaction_date'] <= label_end)]
        active_in_label_window = set(label_txns['user_id'].unique())
        table['churn'] = (~table['user_id'].isin(active_in_label_window)).astype(int)
    return table


def score_users(users, txns, cutoff, feature_start, pipeline_path=None, meta_path=None):
    """Score current users as of `cutoff`, using [feature_start, cutoff] transaction history.
    No label is built (the future outcome doesn't exist yet at scoring time)."""
    pipeline_path = pipeline_path or os.path.join(_HERE, 'churn_pipeline.joblib')
    meta_path = meta_path or os.path.join(_HERE, 'model_card.json')
    pipe = joblib.load(pipeline_path)
    meta = json.load(open(meta_path))

    table = assemble_modeling_table(users, txns, cutoff, feature_start, label_end=None)
    X = table[meta['feature_columns']]
    proba = pipe.predict_proba(X)[:, 1]
    flag = (proba >= meta['deployment_threshold']).astype(int)
    return pd.DataFrame({'user_id': table['user_id'], 'churn_probability': proba, 'churn_flag': flag})
