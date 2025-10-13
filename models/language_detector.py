import whisper
import numpy as np
import torch
import librosa

class LanguageDetector:
    """
    Real multilingual language detector using OpenAI Whisper.
    Detects spoken language and provides transcription.
    """

    def __init__(self, model_size="small"):
        self.model_size = model_size
        self.model = whisper.load_model(model_size)

    def predict_from_audio(self, y, sr):
        """
        Detect language and transcribe speech using Whisper.
        Args:
            y: np.ndarray, audio waveform
            sr: int, sample rate
        Returns:
            (language, transcription)
        """
        # 1️⃣ Resample to 16kHz if needed
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000

        # 2️⃣ Convert to torch tensor (float32)
        audio_tensor = torch.tensor(y, dtype=torch.float32)

        # 3️⃣ Pad or trim to 30 seconds (Whisper requirement)
        audio_tensor = whisper.pad_or_trim(audio_tensor)

        # 4️⃣ Compute log-Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio_tensor).to(self.model.device)

        # 5️⃣ Detect language
        _, probs = self.model.detect_language(mel)
        language = max(probs, key=probs.get)

        # 6️⃣ Transcribe
        result = self.model.transcribe(audio_tensor)
        transcription = result['text'].strip()

        return language, transcription
