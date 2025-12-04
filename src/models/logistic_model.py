from sklearn.linear_model import LogisticRegression

def get_logistic_model():
    return LogisticRegression(max_iter=500)