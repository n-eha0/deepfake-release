import joblib
import os
from models.deepfake_detector import DeepfakeDetector
from models.language_detector import LanguageDetector

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'trained_models'))

def load_models():
    try:
        deepfake_path = os.path.join(MODEL_DIR, 'deepfake_model.pkl')
        scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
        if os.path.exists(deepfake_path):
            model = joblib.load(deepfake_path)
        else:
            model = DeepfakeDetector()

        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            scaler = None
    except Exception:
        model = DeepfakeDetector()
        scaler = None

    lang_detector = LanguageDetector()
    # If model is a scikit-learn estimator, wrap to provide predict_from_features
    if hasattr(model, 'predict') and not hasattr(model, 'predict_from_features'):
        class SKLearnWrapper:
            def __init__(self, estimator, scaler=None):
                self.estimator = estimator
                self.scaler = scaler

            def predict_from_features(self, features):
                import numpy as _np
                feats = _np.asarray(features).reshape(1, -1)
                if self.scaler is not None:
                    feats = self.scaler.transform(feats)
                prob = self.estimator.predict_proba(feats)[0]
                # Assume class 1 is deepfake
                deepfake_prob = float(prob[1]) if len(prob) > 1 else float(prob[0])
                is_deepfake = deepfake_prob > 0.5
                return {'confidence': deepfake_prob, 'is_deepfake': is_deepfake}

        model = SKLearnWrapper(model, scaler)

    return model, scaler, lang_detector
