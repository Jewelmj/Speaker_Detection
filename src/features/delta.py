import librosa
import numpy as np

def extract_delta(file_path: str, sr: int = 16000, n_mfcc: int = 13) -> np.ndarray:
    """
    Extract Δ (delta) MFCC features.
    Returns [mean(delta), std(delta)] → 2 * n_mfcc dims.
    """
    y, sr = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc, order=1)

    d_mean = np.mean(delta, axis=1)
    d_std = np.std(delta, axis=1)

    return np.concatenate([d_mean, d_std])

def extract_deltadelta(file_path: str, sr: int = 16000, n_mfcc: int = 13) -> np.ndarray:
    """
    Extract ΔΔ (delta-delta) MFCC features.
    Returns [mean(delta2), std(delta2)] → 2 * n_mfcc dims.
    """
    y, sr = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    dd_mean = np.mean(delta2, axis=1)
    dd_std = np.std(delta2, axis=1)

    return np.concatenate([dd_mean, dd_std])
