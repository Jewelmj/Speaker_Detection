import librosa
import numpy as np


def extract_melspec(file_path: str, sr: int = 16000, n_mels: int = 40) -> np.ndarray:
    """
    Extract log-Mel spectrogram features.
    We compute Mel-spectrogram → log scale → take mean & std over time.
    Returns a 1D vector of length 2 * n_mels.
    """
    y, sr = librosa.load(file_path, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)

    m_mean = np.mean(S_db, axis=1)
    m_std = np.std(S_db, axis=1)

    return np.concatenate([m_mean, m_std])
