import numpy as np
import librosa
import parselmouth
import warnings
warnings.filterwarnings("ignore")


def extract_audio_features(y, sr=22050):
    """
    Extract audio features (53D) robustly for deepfake detection.
    
    Features include:
    - MFCC mean/std
    - Delta MFCC mean/std
    - Spectral centroid / rolloff
    - Zero-crossing rate
    - Chroma features
    - Spectral contrast
    - Jitter & HNR
    Handles short audio without crashing.
    """
    if y is None or len(y) < 1000:
        return np.zeros(53)

    # Convert to mono
    if hasattr(y, "ndim") and y.ndim > 1:
        y = librosa.to_mono(y)

    # Pad very short audio to 1 sec
    if len(y) < sr * 1.0:
        y = np.pad(y, (0, int(sr*1.0) - len(y)), mode='constant')

    # -------------------
    # MFCC
    # -------------------
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # Delta MFCC, safely
    if mfcc.shape[1] > 2:
        delta_mfcc = librosa.feature.delta(mfcc)
    else:
        delta_mfcc = np.zeros_like(mfcc)

    mfcc_feat = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(delta_mfcc, axis=1),
        np.std(delta_mfcc, axis=1)
    ])  # 52D so far

    # -------------------
    # Spectral features
    # -------------------
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048, hop_length=512))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=2048, hop_length=512))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    # -------------------
    # Chroma
    # -------------------
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    # -------------------
    # Spectral contrast
    # -------------------
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_contrast_mean = np.mean(spec_contrast, axis=1)
    spec_contrast_std = np.std(spec_contrast, axis=1)

    # -------------------
    # Jitter / HNR
    # -------------------
    try:
        snd = parselmouth.Sound(y, sr)
        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
        jitter_local = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0.02, 1.3)
        hnr = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr_value = parselmouth.praat.call(hnr, "Get mean", 0, 0)
    except Exception:
        jitter_local = 0.0
        hnr_value = 20.0

    # -------------------
    # Combine all features
    # -------------------
    features = np.concatenate([
        mfcc_feat,
        [spectral_centroid, spectral_rolloff, zcr, jitter_local, hnr_value]
    ])

    return features
