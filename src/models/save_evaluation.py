import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc
)

from config.settings import EVAL_DIR


def save_confusion_matrix(y_true, y_pred, model_name):
    os.makedirs(EVAL_DIR, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    fig.savefig(os.path.join(EVAL_DIR, f"{model_name}_confusion_matrix.png"))
    plt.close(fig)


def save_classification_report(y_true, y_pred, model_name):
    report = classification_report(y_true, y_pred)
    os.makedirs(EVAL_DIR, exist_ok=True)

    with open(os.path.join(EVAL_DIR, f"{model_name}_classification_report.txt"), "w") as f:
        f.write(report)


def save_roc_curve(model, X_test_scaled, y_test, model_name):
    if not hasattr(model, "predict_proba"):
        return  

    probs = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")

    ax.set_title(f"ROC Curve - {model_name}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()

    fig.savefig(os.path.join(EVAL_DIR, f"{model_name}_roc_curve.png"))
    plt.close(fig)
