# Deepfake Voice Detector (Streamlit Demo)

This repository contains a Streamlit-based demo for a real-time deepfake voice detection system. The current implementation uses lightweight stubs for models and is intended as a scaffold and development starting point.

Quick start:

```powershell
# create virtual env
python -m venv .venv
.\\.venv\\Scripts\\activate

# install requirements
pip install -r requirements.txt

# run the app
streamlit run main.py
```

Notes:
- The `models/` directory contains placeholder detectors. Replace with trained models in `trained_models/`.
- The demo accepts uploaded audio files and runs a simple heuristic-based detection.
