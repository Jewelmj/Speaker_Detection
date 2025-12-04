import numpy as np
import os
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config.settings import MODEL_TO_USE, PROCESSED_DIR, MODEL_DIR

from src.models.svm_model import get_svm_model
from src.models.logistic_model import get_logistic_model
from src.models.rf_model import get_rf_model
from src.models.save_evaluation import (
    save_confusion_matrix,
    save_classification_report,
    save_roc_curve
)

def load_data():
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    return X_train, y_train, X_test, y_test


def get_model(model_name):
    if model_name == "svm":
        return get_svm_model()
    elif model_name == "logistic":
        return get_logistic_model()
    elif model_name == "random_forest":
        return get_rf_model()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_selected_model():
    print(f"Training model: {MODEL_TO_USE}")

    X_train, y_train, X_test, y_test = load_data()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = get_model(MODEL_TO_USE)

    print("Training...")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print("\nEvaluation on Test Set:")
    print(f"Accuracy: {acc:.4f}")

    save_confusion_matrix(y_test, y_pred, MODEL_TO_USE)
    save_classification_report(y_test, y_pred, MODEL_TO_USE)
    save_roc_curve(model, X_test_scaled, y_test, MODEL_TO_USE)

    print(f"\nEvaluation saved under: experiments/evaluation/")

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, os.path.join(MODEL_DIR, f"{MODEL_TO_USE}_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

    print(f"Saved {MODEL_TO_USE} model and scaler.")
    print(f"Training completed.")
