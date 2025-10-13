import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from models.feature_extractor import extract_audio_features
import librosa
from tqdm import tqdm

DATA_DIR = "audio_dataset"
MODEL_DIR = "trained_models"
MAX_DURATION = 15  # seconds, process only first 15s for speed

def load_dataset():
    X, y = [], []
    for label, subfolder in enumerate(["real", "fake"]):
        folder_path = os.path.join(DATA_DIR, subfolder)
        if not os.path.exists(folder_path):
            continue

        files = [f for f in os.listdir(folder_path) if f.lower().endswith((".wav", ".mp3", ".flac", ".m4a"))]

        for file in tqdm(files, desc=f"Loading {subfolder} audio"):
            file_path = os.path.join(folder_path, file)
            try:
                y_audio, sr = librosa.load(file_path, sr=22050, duration=15)
                feat = extract_audio_features(y_audio, sr)
                X.append(feat)
                y.append(label)
            except Exception as e:
                print(f"Error loading {file}: {e}")

    return np.array(X), np.array(y)

def train_and_save():
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, y = load_dataset()
    print(f"Loaded {len(y)} audio files. Features shape: {X.shape}")

    # Train/test split for proper evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    )
    clf.fit(X_train_scaled, y_train)

    # Evaluate
    train_acc = (clf.predict(X_train_scaled) == y_train).mean()
    test_acc = (clf.predict(X_test_scaled) == y_test).mean()
    print(f"Train accuracy: {train_acc:.1%}")
    print(f"Test accuracy: {test_acc:.1%}")

    # Save model + scaler
    joblib.dump(clf, os.path.join(MODEL_DIR, "deepfake_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    print(f"✅ Model + scaler saved to {MODEL_DIR}")

if __name__ == "__main__":
    train_and_save()
