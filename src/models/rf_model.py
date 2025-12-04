from sklearn.ensemble import RandomForestClassifier

def get_rf_model():
    return RandomForestClassifier(n_estimators=200)