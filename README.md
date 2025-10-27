# Audio Deepfake Detection System

An advanced machine learning system for detecting audio deepfakes using acoustic feature analysis and random forest classification.

## Overview

This project implements a robust audio deepfake detection system that analyzes 53-dimensional acoustic features to identify artificially generated or manipulated audio content. The system uses a Random Forest classifier trained on a comprehensive dataset of real and fake audio samples.

## Features

- 53-dimensional acoustic feature extraction
- Advanced Random Forest classification model
- Real-time audio analysis capabilities
- Support for multiple audio formats
- Interactive visualization of detection results
- Language detection integration
- High accuracy and low false-positive rate

## Tech Stack

- **Core ML Framework**: scikit-learn, XGBoost
- **Audio Processing**: librosa, soundfile, praat-parselmouth
- **Deep Learning**: PyTorch
- **Speech Processing**: OpenAI Whisper
- **Data Analysis**: NumPy, Pandas
- **Visualization**: Plotly, Matplotlib, Seaborn
- **UI**: Streamlit

## Installation

1. Clone the repository:
```bash
git clone https://github.com/n-eha0/deepfake-release.git
cd deepfake-release
```

2. Create and activate a virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

3. Install dependencies:
```powershell
pip install -r requirements.txt
```

## Usage

### Running the Application

To start the Streamlit application:

```powershell
streamlit run main.py
```

### Training the Model

To train the model on your dataset:

```powershell
python train_real_dataset.py
```

### Running Predictions

Use the trained model for predictions:

```python
from models.deepfake_detector import DeepfakeDetector

# Initialize detector
detector = DeepfakeDetector()

# Make predictions
result = detector.predict_from_features(features)
print(f"Confidence: {result['confidence']}%")
print(f"Is Deepfake: {result['is_deepfake']}")
```

## Project Structure

```
├── main.py                 # Main application entry point
├── train_real_dataset.py   # Model training script
├── models/
│   ├── deepfake_detector.py    # Core detector implementation
│   ├── feature_extractor.py    # Feature extraction
│   └── language_detector.py    # Language detection
├── utils/
│   ├── audio_utils.py      # Audio processing utilities
│   ├── model_loader.py     # Model loading utilities
│   └── visualization.py    # Visualization tools
└── trained_models/         # Saved model artifacts
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Developed by Neha Gupta

## Acknowledgments

- Thanks to all contributors who have helped with the development


---

For more information or to report issues, please visit the [GitHub repository](https://github.com/n-eha0/deepfake-release).
