# import numpy as np
# import joblib
# import os

# class DeepfakeDetector:
#     """RandomForest-based deepfake detector using 29D audio features."""

#     def __init__(self, model_dir="trained_models"):
#         model_path = os.path.join(model_dir, "deepfake_model.pkl")
#         scaler_path = os.path.join(model_dir, "scaler.pkl")

#         if not os.path.exists(model_path) or not os.path.exists(scaler_path):
#             raise FileNotFoundError(
#                 f"Model not found in {model_dir}. "
#                 "Run train_demo_model.py first."
#             )

#         self.model = joblib.load(model_path)
#         self.scaler = joblib.load(scaler_path)

#     def predict_from_features(self, features):
#         X = np.array(features).reshape(1, -1)
#         Xs = self.scaler.transform(X)
#         prob_fake = self.model.predict_proba(Xs)[0][1]

#         if prob_fake > 0.5:
#             # More likely fake
#             return {"confidence": float(prob_fake * 100), "is_deepfake": True}
#         else:
#             # More likely real
#             return {"confidence": float((1 - prob_fake) * 100), "is_deepfake": False}
import numpy as np
import joblib
import os

class DeepfakeDetector:
    """RandomForest-based deepfake detector using advanced 53D audio features."""

    def __init__(self, model_dir="trained_models"):
        model_path = os.path.join(model_dir, "deepfake_model.pkl")
        scaler_path = os.path.join(model_dir, "scaler.pkl")

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Model not found in {model_dir}. Run train_real_dataset.py first."
            )

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict_from_features(self, features):
        """
        Input: features (array-like, 53D)
        Output: dict with confidence and is_deepfake flag
        """
        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        prob_fake = self.model.predict_proba(X_scaled)[0][1]

        is_deepfake = prob_fake > 0.5
        confidence = prob_fake if is_deepfake else 1 - prob_fake

        return {
            "confidence": float(confidence * 100),  # scale to 0-100%
            "is_deepfake": bool(is_deepfake)
        }
