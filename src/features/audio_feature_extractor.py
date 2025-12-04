import os
import numpy as np
import pandas as pd

from src.config.settings import DATASET_METADATA_PATH, FEATURE_TYPES
from src.features.mfcc import extract_mfcc
from src.features.delta import extract_delta, extract_deltadelta
from src.features.spectral import extract_spectral_features
from src.features.mel_spectrogram import extract_melspec


def extract_audio_features(file_path: str) -> np.ndarray:
    """
    Extract a combined feature vector for a single audio file,
    based on FEATURE_TYPES defined in config.settings.

    FEATURE_TYPES can include:
        "mfcc", "delta", "deltadelta", "spectral", "melspec"
    """
    feature_list = []

    if "mfcc" in FEATURE_TYPES:
        feature_list.append(extract_mfcc(file_path))

    if "delta" in FEATURE_TYPES:
        feature_list.append(extract_delta(file_path))

    if "deltadelta" in FEATURE_TYPES:
        feature_list.append(extract_deltadelta(file_path))

    if "spectral" in FEATURE_TYPES:
        feature_list.append(extract_spectral_features(file_path))

    if "melspec" in FEATURE_TYPES:
        feature_list.append(extract_melspec(file_path))

    if not feature_list:
        raise ValueError("FEATURE_TYPES is empty – no features selected in settings.py")

    return np.concatenate(feature_list)


def batch_extract_features():
    """
    Reads dataset_info.csv, extracts features for each file,
    and saves numpy arrays in data/processed.
    """
    df = pd.read_csv(DATASET_METADATA_PATH)

    X_train, y_train = [], []
    X_test, y_test = [], []

    print("Extracting features for all audio files...")

    for _, row in df.iterrows():
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

    print("Feature extraction completed. Saved to data/processed/")
