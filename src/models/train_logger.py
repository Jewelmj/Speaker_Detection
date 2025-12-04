import json
import os
from datetime import datetime
from src.config import settings

LOG_PATH = f"{settings.LOG_DIR}/train_history.jsonl"


def log_training_run(model_name, features, metrics, extra_info=None):
    """
    Append a training record to train_history.jsonl.
    Each line is a JSON object.
    """
    os.makedirs("experiments/logs", exist_ok=True)

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "features": features,
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "roc_auc": metrics.get("roc_auc"),
    }

    if extra_info:
        record.update(extra_info)

    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
