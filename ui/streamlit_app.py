import streamlit as st
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.models.infer import run_inference
from src.models.trainer import train_selected_model
from src.data.mertadata_prep import prepare_metadata
from src.features.audio_feature_extractor import batch_extract_features
from src.main import clean_all
from src.config import settings

st.set_page_config(page_title="Speaker Recognition", layout="wide")
st.title("Speaker Recognition Control Panel")

st.sidebar.header("Navigation")

mode = st.sidebar.radio("Choose Mode:", ["Train Model", "Inference"])

feature_options = ["mfcc", "delta", "deltadelta", "spectral", "melspec"]

selected_features = st.sidebar.multiselect(
    "Select Feature Types:",
    feature_options,
    default=settings.FEATURE_TYPES
)

settings.FEATURE_TYPES = selected_features

model_choice = st.sidebar.selectbox(
    "Model to Use:",
    ["svm", "logistic", "random_forest", "mlp"],
    index=["svm", "logistic", "random_forest", "mlp"].index(settings.MODEL_TO_USE)
)

if st.sidebar.button("Clear Cache"):
    clean_all()
    st.sidebar.success("Cache cleared!")

settings.MODEL_TO_USE = model_choice
settings.INFERENCE_MODEL = model_choice

if mode == "Train Model":
    st.header("Train Model")

    st.write(f"Selected Model: **{model_choice}**")
    st.write(f"Selected Features: {', '.join(settings.FEATURE_TYPES)}")

    if len(settings.FEATURE_TYPES) == 0:
        st.error("Please select at least one feature type in the sidebar.")
        st.stop()

    if st.button("Start Training"):
        st.write("Preparing metadata & features...")
        prepare_metadata()
        batch_extract_features()

        st.write("Training model...")
        try:
            train_selected_model()
            st.success(f"Model '{model_choice}' trained successfully!")
        except Exception as e:
            st.error(f"Training Failed: {e}")

        st.subheader("Evaluation Results")

        cm_path = os.path.join(settings.EVAL_DIR, f"{model_choice}_confusion_matrix.png")
        roc_path = os.path.join(settings.EVAL_DIR, f"{model_choice}_roc_curve.png")
        report_path = os.path.join(settings.EVAL_DIR, f"{model_choice}_classification_report.txt")

        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix")

        if os.path.exists(roc_path):
            st.image(roc_path, caption="ROC Curve")

        if os.path.exists(report_path):
            st.subheader("Classification Report")
            with open(report_path, "r") as f:
                st.text(f.read())

elif mode == "Inference":
    st.header("Run Inference")
    st.write(f"Using Features: {', '.join(settings.FEATURE_TYPES)}")

    uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

    if len(settings.FEATURE_TYPES) == 0:
        st.error("Please select at least one feature type in the sidebar.")
        st.stop()

    if uploaded_file is not None:
        temp_path = "temp_uploaded.wav"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        st.audio(temp_path)

        st.write("Running inference...")
        try:
            label, prob = run_inference(temp_path)
            if label == 1:
                st.success(f"Target Speaker Detected (Confidence: {prob:.3f})")
            else:
                st.error(f"NOT Target Speaker (Confidence: {prob:.3f})")
        except Exception as e:
            st.error(f"Inference Error: {e}")

        os.remove(temp_path)
