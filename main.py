import streamlit as st
from streamlit import session_state as ss
import numpy as np
from models.deepfake_detector import DeepfakeDetector
from models.language_detector import LanguageDetector
from models.feature_extractor import extract_audio_features
from utils.audio_utils import load_audio_bytes, chunk_audio
from utils.visualization import create_waveform_plot, create_confidence_meter
from utils.model_loader import load_models
import time
from audio_recorder_streamlit import audio_recorder

st.set_page_config(page_title="Deepfake Voice Detection", page_icon="🎤", layout="wide")

# recommit
# -------------------------
# Session Initialization
# -------------------------
def initialize_session_state():
    if 'detection_history' not in ss:
        ss.detection_history = []
    if 'model_loaded' not in ss:
        ss.model_loaded = False
    if 'current_audio' not in ss:
        ss.current_audio = None


# -------------------------
# Microphone Recording
# -------------------------
def handle_microphone_recording(model, lang_detector, language_mode="Auto-detect"):
    """Handle microphone recording using streamlit-audio-recorder"""
    
    st.subheader("🎙️ Record from Microphone")
    
    # Audio recorder component
    audio_bytes = audio_recorder(
        text="Click the icon to record",
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        icon_name="microphone-lines",
        icon_size="4x",
    )
    
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        
        with st.spinner("Processing recorded audio..."):
            try:
                # Convert audio bytes to numpy array
                y, sr = load_audio_bytes(audio_bytes)
                
                if y is not None and len(y) > 0:
                    # Process the audio
                    result, _chunks = process_audio_in_chunks(y, sr, model, lang_detector, language_mode)
                    display_results(result, y, sr)
                else:
                    st.error("No valid audio data recorded. Please try again.")
                    
            except Exception as e:
                st.error(f"Error processing recorded audio: {str(e)}")
    else:
        st.info("👆 Click the microphone button above to start recording your voice")


# -------------------------
# Process Audio
# -------------------------
def process_audio_in_chunks(y, sr, model, lang_detector, language_mode="Auto-detect", chunk_seconds=15):
    """
    Detect language once on full audio, then chunk audio for deepfake detection.
    """
    start_time = time.time()

    # Language detection or override
    if language_mode == "Auto-detect":
        language, transcription = lang_detector.predict_from_audio(y, sr)
    else:
        language = language_mode
        transcription = "(Transcription skipped — manual language mode)"

    # Deepfake detection
    chunks = chunk_audio(y, sr, chunk_seconds)
    chunk_preds = []

    for chunk in chunks:
        features = extract_audio_features(chunk, sr)
        pred = model.predict_from_features(features)
        chunk_preds.append({
            'confidence': float(pred['confidence']),
            'is_deepfake': bool(pred['is_deepfake'])
        })

    # Aggregate predictions
    avg_conf = float(np.mean([c['confidence'] for c in chunk_preds])) * 100
    votes = sum(1 if c['is_deepfake'] else 0 for c in chunk_preds)
    is_deepfake = votes > (len(chunk_preds) / 2)

    result = {
        'confidence': avg_conf,
        'is_deepfake': is_deepfake,
        'language': language,
        'transcription': transcription,
        'time': time.time() - start_time
    }

    return result, chunks


# -------------------------
# Display Results
# -------------------------
def display_results(result, y=None, sr=22050):
    confidence = result['confidence']
    is_deepfake = result['is_deepfake']
    language = result['language']
    transcription = result['transcription']

    if is_deepfake:
        st.error(f"⚠️ DEEPFAKE DETECTED (Confidence: {confidence:.1f}%)")
    else:
        st.success(f"✅ AUTHENTIC VOICE (Confidence: {confidence:.1f}%)")

    col1, col2, col3 = st.columns(3)
    col1.metric("Confidence", f"{confidence:.1f}%")
    col2.metric("Language", language)
    col3.metric("Processing Time", f"{result['time']:.2f}s")

    if transcription:
        st.info(f"📝 Transcription: {transcription}")

    st.plotly_chart(create_confidence_meter(confidence))

    if y is not None:
        max_points = 20000
        y_plot = y[::len(y)//max_points] if len(y) > max_points else y
        st.plotly_chart(create_waveform_plot(y_plot, sr))

    ss.detection_history.append(result)


# -------------------------
# File Upload Handler
# -------------------------
def handle_file_upload(model, lang_detector, language_mode="Auto-detect", key="default"):
    uploaded_file = st.file_uploader("Upload audio file", type=['wav', 'mp3', 'flac', 'm4a'], key=key)
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        with st.spinner("Processing audio..."):
            y, sr = load_audio_bytes(audio_bytes)
            result, _chunks = process_audio_in_chunks(y, sr, model, lang_detector, language_mode)
            display_results(result, y, sr)


# -------------------------
# Main App
# -------------------------
def main():
    initialize_session_state()
    st.title("🎤 Real-Time Deepfake Voice Detection")
    st.markdown("Detect AI-generated or deepfaked voices in real-time (demo/stub models).")

    with st.sidebar:
        st.header("Settings")
        sensitivity = st.slider("Detection Sensitivity", 0.1, 1.0, 0.7)
        language_mode = st.selectbox("Language Mode", ["Auto-detect", "English", "Hindi", "Urdu"])
        show_advanced = st.checkbox("Show Advanced Analysis")

    # Load models (with spinner)
    if not ss.model_loaded:
        with st.spinner("Loading models (Whisper + Deepfake Detector)..."):
            model, _, lang_detector = load_models()
            ss.model_loaded = True
            ss.model = model
            ss.lang_detector = lang_detector
        st.success("✅ Models loaded successfully!")
    else:
        model = ss.model
        lang_detector = ss.lang_detector

    tab1, tab2, tab3 = st.tabs(["Live Detection", "Batch File Analysis", "History"])

    with tab1:
        st.subheader("Live / Recorded Audio")
        
        # Create two columns for different input methods
        input_method = st.radio(
            "Choose input method:",
            ["🎙️ Record from Microphone", "📁 Upload Audio File"],
            horizontal=True
        )
        
        if input_method == "🎙️ Record from Microphone":
            handle_microphone_recording(model, lang_detector, language_mode)
        else:
            st.info("Upload an audio file for analysis.")
            handle_file_upload(model, lang_detector, language_mode, key="live_upload")

    with tab2:
        st.subheader("Batch File Analysis")
        st.info("Upload files for analysis.")
        handle_file_upload(model, lang_detector, language_mode, key="batch_upload")

    with tab3:
        st.subheader("Detection History")
        if not ss.detection_history:
            st.info("No detections yet.")
        else:
            for i, rec in enumerate(ss.detection_history[::-1]):
                st.write(
                    f"{i+1}. 🌐 Language: **{rec['language']}** — "
                    f"Confidence: **{rec['confidence']:.1f}%** — "
                    f"Deepfake: **{rec['is_deepfake']}**"
                )


if __name__ == '__main__':
    main()
