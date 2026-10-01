"""
churn_scoring.py -- batch scoring. Uses the SAME churn_features module as training.
Usage:  from churn_scoring import score_users
        scores = score_users(users_df, txns_df, cutoff='2025-06-01')   # txns up to cutoff (later rows are ignored)
"""
import os
import joblib
import numpy as np
import pandas as pd
import churn_features as cf

_HERE = os.path.dirname(os.path.abspath(__file__))


def load_bundle(path=None):
    return joblib.load(path or os.path.join(_HERE, 'churn_bundle.joblib'))


def score_users(users, txns, cutoff, bundle=None, explain=True, top_k=3):
    b = bundle or load_bundle()
    table = cf.build_live_table(txns, users, cutoff, b['data_start'], b['use_engagement'])
    X_tree = cf.to_model_frame(table, b['features'], b['cat_features'], b['categories'])
    X_cb = cf.to_model_frame(table, b['features'], b['cat_features'], b['categories'], as_string=True)
    w = b['weights']
    raw = (w['lgb'] * b['lgb'].predict_proba(X_tree)[:, 1]
           + w['xgb'] * b['xgb'].predict_proba(X_tree)[:, 1]
           + w['cb'] * b['cb'].predict_proba(X_cb)[:, 1])
    p = b['calibrator'].predict(raw)
    out = pd.DataFrame({'user_id': table['user_id'].values, 'eligible': table['eligible'].values,
                        'churn_probability': p, 'rank_score': raw})   # probability for EV math, raw blend for ranking
    out['churn_flag'] = ((p >= b['threshold']) & (table['eligible'].values == 1)).astype(int)
    if explain:
        contrib = b['lgb'].predict(X_tree, pred_contrib=True)[:, :-1]
        top = np.argsort(-contrib, axis=1)[:, :top_k]
        out['top_reasons'] = [', '.join(b['features'][j] for j in row) for row in top]
    return out.sort_values('rank_score', ascending=False).reset_index(drop=True)
