# This script hardcodes the entire churn prediction pipeline in one place.
# Reason: Earlier, the process was split across 3 different notebooks.
# To avoid confusion and reduce manual steps, we're consolidating everything here.
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder

# Load the trained model
model = joblib.load(r"saved_models/final_model.pkl")

# Final selected features from training
categorical_cols = [
    'age_group_16-18', 'age_group_19-22', 'age_group_23-30', 'age_group_31+',
    'most_common_payment_method_UPI',
    'most_common_transaction_type_Education',
    'most_common_transaction_type_Entertainment',
    'most_common_transaction_type_Food',
    'most_common_transaction_type_Other',
    'most_common_transaction_type_Shopping',
    'most_common_transaction_type_Travel'
]

numerical_cols = [
    'total_transactions', 'active_days', 'unique_merchants', 'days_between_first_and_last_txn'
]

all_required_features = numerical_cols + categorical_cols


def load_new_data(user_data_path, txn_data_path):
    users = pd.read_csv(user_data_path)
    txns = pd.read_csv(txn_data_path)
    return users, txns


def preprocess(users, txns):
    txns['transaction_date'] = pd.to_datetime(txns['transaction_date'])

    # Aggregate transaction data
    agg_txn = txns.groupby('user_id').agg(
        total_transactions=('transaction_id', 'count'),
        total_amount=('amount', 'sum'),
        avg_amount=('amount', 'mean'),
        max_amount=('amount', 'max'),
        min_amount=('amount', 'min'),
        active_days=('transaction_date', lambda x: x.nunique()),
        unique_merchants=('merchant', 'nunique'),
        transaction_failure_rate=('status', lambda x: (x == 'failed').mean()),
        most_common_payment_method=('payment_method', lambda x: x.mode()[0] if not x.mode().empty else 'unknown'),
        most_common_transaction_type=('transaction_type', lambda x: x.mode()[0] if not x.mode().empty else 'unknown'),
        last_transaction_date=('transaction_date', 'max'),
        first_transaction_date=('transaction_date', 'min')
    )

    # Date-based features
    agg_txn['days_between_first_and_last_txn'] = (
        agg_txn['last_transaction_date'] - agg_txn['first_transaction_date']
    ).dt.days

    agg_txn['days_since_last_transaction'] = (
        pd.to_datetime("today") - agg_txn['last_transaction_date']
    ).dt.days

    agg_txn = agg_txn.reset_index()

    # Merge with user data
    df = pd.merge(users, agg_txn, on='user_id', how='left')
    df = df.dropna()

    # Drop weak features
    weak_corr_features = [
        'days_since_registration', 'app_opens_per_week', 'avg_session_duration',
        'support_tickets', 'referrals_made', 'has_customized_card',
        'has_set_savings_goal', 'has_used_offers',
        'total_amount', 'avg_amount', 'max_amount', 'min_amount'
    ]
    df = df.drop(columns=weak_corr_features)

    # Drop irrelevant or date fields
    df = df.drop(columns=['registration_date', 'last_transaction_date', 'first_transaction_date', 'city', 'device_type','transaction_failure_rate'])

    # One-hot encode categorical
    cat_cols = ['age_group', 'most_common_payment_method', 'most_common_transaction_type']
    ohe = OneHotEncoder(sparse_output=False, drop=None)
    X_cat_encoded = ohe.fit_transform(df[cat_cols])
    encoded_cols = ohe.get_feature_names_out(cat_cols)
    X_cat_df = pd.DataFrame(X_cat_encoded, columns=encoded_cols, index=df.index)

    # Drop original categorical
    df_numeric = df.drop(columns=cat_cols + ['is_active'])

    # Combine numerical + categorical
    df_final = pd.concat([df_numeric, X_cat_df], axis=1)

    # Ensure all expected features exist
    for col in all_required_features:
        if col not in df_final.columns:
            df_final[col] = 0

    df_final = df_final[all_required_features]

    # Keep user ID for mapping
    user_ids_df = df[['user_id']].copy()
    return user_ids_df, df_final


def predict_churn(user_ids_df, X):
    churn_probs = model.predict_proba(X)[:, 1]  # Class 1 = active, so class 0 = churn
    user_ids_df['churn_probability'] = churn_probs

    # Add churn risk label
    def label_risk(prob):
        if prob >= 0.8:
            return "High"
        elif prob >= 0.5:
            return "Medium"
        else:
            return "Low"

    user_ids_df['churn_risk_segment'] = user_ids_df['churn_probability'].apply(label_risk)

    return user_ids_df.sort_values(by='churn_probability', ascending=False)


def get_at_risk_users(user_data_path, txn_data_path, threshold=0.5):
    users, txns = load_new_data(user_data_path, txn_data_path)
    user_ids_df, X = preprocess(users, txns)
    results = predict_churn(user_ids_df, X)
    at_risk = results[results['churn_probability'] >= threshold]
    return at_risk
