"""
churn_features.py -- SINGLE SOURCE OF TRUTH for feature + label logic.
Imported by the training notebook AND by churn_scoring.py, so train/serve skew is impossible
by construction.

Concept: a *snapshot* = (cutoff day D).
    features  <- transactions with day <= D          (never anything after D)
    label     <- transactions with D < day <= D + horizon
One row per (user, cutoff). Many cutoffs are stacked to form the training table.
"""
import numpy as np
import pandas as pd

WINDOWS = (3, 7, 14)              # look-back windows in days (day 0 = cutoff day)
LOOKBACK = max(WINDOWS)
FAILED = 'Failed'

STATIC_CAT = ['age_group', 'city', 'device_type']
STATIC_NUM = ['has_customized_card', 'has_set_savings_goal', 'has_used_offers']
ENGAGEMENT = ['app_opens_per_week', 'avg_session_duration', 'support_tickets', 'referrals_made']
CAT_FEATURES = STATIC_CAT + ['top_txn_type']
META_COLS = ['user_id', 'cutoff', 'churn', 'eligible']

FEATURE_GROUPS = {
    'user_static': STATIC_CAT + STATIC_NUM + ['tenure_days', 'is_new_user'],
    'recency': ['days_since_last_txn', 'days_since_last_ok_txn', 'active_span_days'],
    'windowed_activity': [f'{p}_{w}d' for p in ('cnt', 'amt', 'fail', 'active_days', 'merchants')
                          for w in WINDOWS],
    'rhythm_gaps': ['gap_mean', 'gap_max', 'gap_std', 'overdue_ratio', 'overdue_vs_max'],
    'momentum': ['share_last3', 'cnt_7d_vs_prev7', 'amt_7d_vs_prev7', 'velocity_ratio', 'rate_ratio_7d'],
    'history_baseline': ['hist_cnt', 'hist_active_days', 'hist_active_share', 'hist_rate'],
    'mix_quality': ['amt_mean', 'amt_std', 'amt_max', 'n_merchants', 'n_pay_methods', 'n_txn_types',
                    'top_merchant_share', 'top_txn_type', 'top_txn_type_share', 'weekend_share',
                    'failure_rate'],
    'engagement_cols': ENGAGEMENT,
}


def prep_txns(txns):
    """One-off vectorised prep (call once, reuse for every cutoff)."""
    t = txns.copy()
    t['day'] = t['transaction_date'].dt.normalize()
    t['is_failed'] = (t['status'] == FAILED).astype('int8')
    t['ok_amount'] = t['amount'].where(t['is_failed'] == 0, 0.0)
    return t


def snapshot_features(t, users, cutoff, data_start, use_engagement=False):
    """Feature table for ONE cutoff day. `t` must come from prep_txns().
    Uses only rows with day <= cutoff. Returns one row per eligible-registered user."""
    cutoff = pd.Timestamp(cutoff).normalize()
    data_start = pd.Timestamp(data_start).normalize()
    L = f'{LOOKBACK}d'

    hist = t[t['day'] <= cutoff]                                   # everything known at cutoff
    win = hist[hist['day'] > cutoff - pd.Timedelta(days=LOOKBACK)].copy()
    win['days_ago'] = (cutoff - win['day']).dt.days                # 0 = cutoff day

    # ---- windowed activity (3d / 7d / 14d) ---------------------------------------------
    f = pd.DataFrame(index=pd.Index(win['user_id'].unique(), name='user_id'))
    for w in WINDOWS:
        g = win[win['days_ago'] < w].groupby('user_id')
        f[f'cnt_{w}d'] = g.size()
        f[f'amt_{w}d'] = g['ok_amount'].sum()                      # successful money only
        f[f'fail_{w}d'] = g['is_failed'].sum()
        f[f'active_days_{w}d'] = g['day'].nunique()
        f[f'merchants_{w}d'] = g['merchant'].nunique()
    zero_cols = list(f.columns)
    f[zero_cols] = f[zero_cols].fillna(0)

    # ---- lookback-wide amount / diversity / span ----------------------------------------
    g = win.groupby('user_id')
    f['amt_mean'] = g['amount'].mean()
    f['amt_std'] = g['amount'].std()
    f['amt_max'] = g['amount'].max()
    f['n_merchants'] = g['merchant'].nunique()
    f['n_pay_methods'] = g['payment_method'].nunique()
    f['n_txn_types'] = g['transaction_type'].nunique()
    f['active_span_days'] = (g['day'].max() - g['day'].min()).dt.days
    f['failure_rate'] = f[f'fail_{L}'] / f[f'cnt_{L}']
    f['weekend_share'] = win.assign(_w=(win['day'].dt.dayofweek >= 5)).groupby('user_id')['_w'].mean()

    # ---- concentration -------------------------------------------------------------------
    mc = win.groupby(['user_id', 'merchant']).size().groupby(level='user_id').max()
    f['top_merchant_share'] = mc / f[f'cnt_{L}']
    tc = (win.groupby(['user_id', 'transaction_type']).size().rename('n').reset_index()
          .sort_values(['user_id', 'n', 'transaction_type'], ascending=[True, False, True])   # deterministic ties
          .drop_duplicates('user_id').set_index('user_id'))
    f['top_txn_type'] = tc['transaction_type']
    f['top_txn_type_share'] = tc['n'] / f[f'cnt_{L}']

    # ---- purchase rhythm: gaps between ACTIVE days, and "overdue" vs own pattern -----------
    days = win[['user_id', 'day']].drop_duplicates().sort_values(['user_id', 'day'])
    days['gap'] = days.groupby('user_id')['day'].diff().dt.days
    gg = days.groupby('user_id')['gap']
    f['gap_mean'], f['gap_max'], f['gap_std'] = gg.mean(), gg.max(), gg.std()

    # ---- full-history recency + baseline (still only day <= cutoff) ------------------------
    h = hist.groupby('user_id').agg(hist_cnt=('amount', 'size'), hist_active_days=('day', 'nunique'),
                                    hist_last=('day', 'max'))
    h['hist_last_ok'] = hist[hist['is_failed'] == 0].groupby('user_id')['day'].max()
    observed_days = (cutoff - data_start).days + 1
    h['days_since_last_txn'] = (cutoff - h['hist_last']).dt.days
    h['days_since_last_ok_txn'] = (cutoff - h['hist_last_ok']).dt.days
    h['hist_active_share'] = h['hist_active_days'] / observed_days
    h['hist_rate'] = h['hist_cnt'] / observed_days
    h = h.drop(columns=['hist_last', 'hist_last_ok'])

    # ---- users: registered on/before cutoff only ------------------------------------------
    u = users.copy()
    u['_reg'] = u['registration_date'].dt.normalize()
    u = u[u['_reg'] <= cutoff]
    u['tenure_days'] = (cutoff - u['_reg']).dt.days
    u['is_new_user'] = (u['tenure_days'] <= LOOKBACK).astype('int8')
    keep = ['user_id'] + STATIC_CAT + STATIC_NUM + ['tenure_days', 'is_new_user']
    if use_engagement:
        keep += ENGAGEMENT
    table = u[keep].set_index('user_id').join(f).join(h)

    fill0 = zero_cols + ['n_merchants', 'n_pay_methods', 'n_txn_types', 'hist_cnt', 'hist_active_days',
                         'hist_active_share', 'hist_rate']
    table[fill0] = table[fill0].fillna(0)
    table['top_txn_type'] = table['top_txn_type'].fillna('None')

    # ---- momentum / ratios (after zero-fill so dormant users are well defined) -------------
    table['share_last3'] = table['cnt_3d'] / table[f'cnt_{L}'].replace(0, np.nan)
    table['cnt_7d_vs_prev7'] = table['cnt_7d'] - (table[f'cnt_{L}'] - table['cnt_7d'])
    table['amt_7d_vs_prev7'] = table['amt_7d'] - (table[f'amt_{L}'] - table['amt_7d'])
    table['velocity_ratio'] = (table['cnt_3d'] / 3) / (table[f'cnt_{L}'] / LOOKBACK + 1e-6)
    table['rate_ratio_7d'] = (table['cnt_7d'] / 7) / (table['hist_rate'] + 1e-6)
    table['overdue_ratio'] = table['days_since_last_txn'] / (table['gap_mean'] + 1.0)
    table['overdue_vs_max'] = table['days_since_last_txn'] - table['gap_max']

    table = table.replace([np.inf, -np.inf], np.nan).reset_index()
    table['cutoff'] = cutoff
    table['eligible'] = (table[f'cnt_{L}'] > 0).astype('int8')    # active in look-back => meaningful to score
    return table


def make_label(t, user_ids, cutoff, horizon):
    """churn=1 if the user has ZERO transactions in (cutoff, cutoff+horizon]. Label-only; never a feature."""
    cutoff = pd.Timestamp(cutoff).normalize()
    fut = t[(t['day'] > cutoff) & (t['day'] <= cutoff + pd.Timedelta(days=horizon))]
    active = set(fut['user_id'].unique())
    return (~pd.Series(user_ids).isin(active)).astype('int8').values


def make_cutoffs(txns, horizon, step=1):
    """Every cutoff with a full look-back behind it and a full label horizon ahead of it."""
    d0, d1 = txns['transaction_date'].min().normalize(), txns['transaction_date'].max().normalize()
    first = d0 + pd.Timedelta(days=LOOKBACK - 1)
    last = d1 - pd.Timedelta(days=horizon)
    return list(pd.date_range(first, last, freq=f'{step}D'))


def build_training_table(txns, users, cutoffs, horizon, use_engagement=False):
    t = prep_txns(txns)
    data_start = txns['transaction_date'].min()
    parts = []
    for c in cutoffs:
        tab = snapshot_features(t, users, c, data_start, use_engagement)
        tab['churn'] = make_label(t, tab['user_id'].values, c, horizon)
        parts.append(tab)
    return pd.concat(parts, ignore_index=True)


def build_live_table(txns, users, cutoff, data_start, use_engagement=False):
    """Scoring mode: same features, NO label (the future does not exist yet)."""
    return snapshot_features(prep_txns(txns), users, cutoff, data_start, use_engagement)


def feature_columns(table):
    return [c for c in table.columns if c not in META_COLS]


def to_model_frame(df, features, cat_features, categories, as_string=False):
    """Model-ready frame. Categoricals get a FIXED category list learned at training time, so
    scoring can never shift category codes. as_string=True gives plain strings (CatBoost / sklearn OHE)."""
    X = df[features].copy()
    for c in cat_features:
        s = X[c].astype(str)
        X[c] = s.astype(object) if as_string else pd.Categorical(s, categories=categories[c])
    return X
