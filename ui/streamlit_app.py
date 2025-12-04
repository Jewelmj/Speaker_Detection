import streamlit as st
import os
import sys
import json
import pandas as pd

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

mode = st.sidebar.radio(
    "Choose Mode:",
    ["Train Model", "Inference", "Model Comparison", "Training History"]
)

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

elif mode == "Model Comparison":
    st.header("Multi-Model Comparison")

    st.write("This will train **all models** using the selected features:")
    st.write(f"**{', '.join(settings.FEATURE_TYPES)}**")

    if st.button("Start Full Comparison"):
        model_list = ["svm", "logistic", "random_forest"]
        results = []

        prepare_metadata()
        batch_extract_features()

        for m in model_list:
            st.write(f"Training {m}...")
            settings.MODEL_TO_USE = m
            settings.INFERENCE_MODEL = m

            train_selected_model(m)

            json_path = os.path.join(settings.EVAL_DIR, f"{m}_results.json")
            acc = None

            if os.path.exists(json_path):
                with open(json_path, "r") as jf:
                    data = json.load(jf)
                    acc = data.get("accuracy", None)

            results.append({
                "Model": m,
                "Accuracy": data.get("accuracy"),
                "Precision": data.get("precision"),
                "Recall": data.get("recall"),
                "F1 Score": data.get("f1"),
                "ROC-AUC": data.get("roc_auc"),
            })

        st.success("Comparison completed!")

        df = pd.DataFrame(results)
        st.dataframe(df)

        st.bar_chart(df.set_index("Model"))

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

elif mode == "Training History":
    st.header("Training History")

    log_path = "experiments/logs/train_history.jsonl"

    if not os.path.exists(log_path):
        st.info("No training history found.")
    else:
        import json
        import pandas as pd

        rows = []
        with open(log_path, "r") as f:
            for line in f:
                rows.append(json.loads(line))

        df = pd.DataFrame(rows)

        st.subheader("Training Table")
        st.dataframe(df)

        st.subheader("Accuracy Over Time")
        if "accuracy" in df.columns:
            chart_df = df[["timestamp", "accuracy"]].set_index("timestamp")
            st.line_chart(chart_df)

        best_idx = df["accuracy"].idxmax()
        best_row = df.loc[best_idx]

        st.subheader("Best Model")
        st.write(f"**Model:** {best_row['model']}")
        st.write(f"**Features:** {best_row['features']}")
        st.write(f"**Accuracy:** {best_row['accuracy']:.3f}")
        st.write(f"**Timestamp:** {best_row['timestamp']}")