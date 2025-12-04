import librosa
import numpy as np


def extract_spectral_features(file_path: str, sr: int = 16000) -> np.ndarray:
    """
    Extract a set of spectral features:
    - spectral centroid
    - spectral bandwidth
    - spectral rolloff
    - spectral contrast
    - zero-crossing rate

    For each, we take mean and std over time and concatenate into one vector.
    """
    y, sr = librosa.load(file_path, sr=sr)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)         
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)       
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)           
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)                      

    def agg(feat: np.ndarray) -> np.ndarray:
        return np.concatenate([np.mean(feat, axis=1), np.std(feat, axis=1)])

    features = [
        agg(centroid),
        agg(bandwidth),
        agg(rolloff),
        agg(contrast),
        agg(zcr),
    ]

    return np.concatenate(features)
