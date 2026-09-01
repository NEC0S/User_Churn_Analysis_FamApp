from User_Churn_Analysis_FamApp.legacy_v1.churn_predictor import get_at_risk_users

at_risk = get_at_risk_users("data/fam_users_sample.csv", "data/fam_transactions_sample.csv", threshold=0.4)
print(at_risk.head(50)) 
