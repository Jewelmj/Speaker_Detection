import os
import argparse
import joblib
import numpy as np
import torch

from src.config.settings import INFERENCE_MODEL, MODEL_DIR
from src.features.audio_feature_extractor import extract_audio_features

from src.models.mlp_model import MLPModel


def load_scaler():
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("Scaler not found! Run training first.")
    return joblib.load(scaler_path)


def load_model(model_name, input_dim=None):
    """
    Loads either classical ML model or the MLP neural network.
    """
    model_path = os.path.join(MODEL_DIR, f"{model_name}_model.joblib")
    nn_path = os.path.join(MODEL_DIR, "mlp_model.pt")

    if model_name == "mlp":
        if input_dim is None:
            raise ValueError("input_dim is required for MLP model.")

        model = MLPModel(input_dim)
        model.load_state_dict(torch.load(nn_path, map_location="cpu"))
        model.eval()
        return model

    if os.path.exists(model_path):
        return joblib.load(model_path)

    raise FileNotFoundError(f"Model '{model_name}' not found. Train it first.")


def predict(model, scaler, feature_vector, model_name):
    """
    Runs prediction depending on which model is used.
    """
    feature_vector = np.array(feature_vector).reshape(1, -1)
    feature_vector = scaler.transform(feature_vector)

    if model_name == "mlp":
        x_tensor = torch.tensor(feature_vector, dtype=torch.float32)
        prob = model(x_tensor).item()
        label = 1 if prob >= 0.5 else 0
        return label, prob

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(feature_vector)[0][1]
        label = int(prob >= 0.5)
        return label, prob

    pred = model.predict(feature_vector)[0]
    return int(pred), float(pred)


def run_inference(audio_path):
    print(f"\nRunning inference using model: {INFERENCE_MODEL}")
    print(f"Audio: {audio_path}")

    features = extract_audio_features(audio_path)
    scaler = load_scaler()

    input_dim = len(features)
    model = load_model(INFERENCE_MODEL, input_dim=input_dim)

    label, prob = predict(model, scaler, features, INFERENCE_MODEL)

    print("\nPrediction Result:")
    if label == 1:
        print(f"Target Speaker (Confidence: {prob:.3f})")
    else:
        print(f"NOT Target Speaker (Confidence: {prob:.3f})")

    return label, prob