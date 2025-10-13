import io
import numpy as np
import soundfile as sf
import base64
import wave
import librosa

def load_audio_bytes(audio_bytes, sr=22050):
    """
    Load audio bytes into (y, sr) numpy array.
    Supports WAV/FLAC via soundfile, MP3/M4A via librosa.
    Always returns mono.
    """
    try:
        # Try soundfile first (WAV/FLAC)
        y, sr_file = sf.read(io.BytesIO(audio_bytes), dtype='float32')
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        return y, sr_file
    except Exception:
        # Fallback to librosa (MP3, M4A)
        y, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr, mono=True)
        return y, sr

def chunk_audio(y, sr, chunk_seconds=15):
    """
    Split audio array into chunks of chunk_seconds.
    Returns list of numpy arrays.
    """
    if y is None or len(y) == 0:
        return []
    samples_per_chunk = int(sr * chunk_seconds)
    return [y[i:i+samples_per_chunk] for i in range(0, len(y), samples_per_chunk)]



def recorder_bytes_to_wav(blob_b64: str) -> bytes:
    """Convert data from the `streamlit-audio-recorder` component (base64 blob) to WAV bytes.

    The recorder typically returns a data URL or base64 audio blob. This helper handles common formats.
    """
    # If the input is a data URL like 'data:audio/wav;base64,...'
    if blob_b64.startswith('data:'):
        header, b64 = blob_b64.split(',', 1)
        return base64.b64decode(b64)

    # If it's plain base64
    try:
        return base64.b64decode(blob_b64)
    except Exception:
        # Fallback: return empty wav
        with io.BytesIO() as buf:
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(22050)
                wf.writeframes(b'')
            return buf.getvalue()
