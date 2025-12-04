import numpy as np
import pandas as pd
import os
from src.features.mfcc import extract_mfcc

from src.config.settings import DATASET_METADATA_PATH, FEATURES_TO_USE

def extract_audio_features(file_path: str):
    """
    Extract features from a single audio file.
    Currently supports MFCC only.
    """
    feature_list = []

    if "mfcc" in FEATURES_TO_USE:
        feature_list.append(extract_mfcc(file_path))
    else:
        raise ValueError(f"Unsupported feature type: {FEATURES_TO_USE}")

    return np.concatenate(feature_list)


def batch_extract_features():
    """
    Reads dataset_info.csv
    Extracts MFCC for each file
    Saves numpy arrays in data/processed/
    """

    df = pd.read_csv(DATASET_METADATA_PATH)

    X_train, y_train = [], []
    X_test, y_test = [], []

    print("Extracting MFCC features...")

    for i, row in df.iterrows():
        fp = row["filepath"]
        label = row["label"]
        split = row["split"]

        try:
            features = extract_audio_features(fp)
        except Exception as e:
            print(f"Error reading {fp}: {e}")
            continue

        if split == "train":
            X_train.append(features)
            y_train.append(label)
        else:
            X_test.append(features)
            y_test.append(label)

    os.makedirs("data/processed", exist_ok=True)

    np.save("data/processed/X_train.npy", np.array(X_train))
    np.save("data/processed/y_train.npy", np.array(y_train))
    np.save("data/processed/X_test.npy", np.array(X_test))
    np.save("data/processed/y_test.npy", np.array(y_test))

    print("Feature extraction completed successfully!")
    print("Saved files under: data/processed/")