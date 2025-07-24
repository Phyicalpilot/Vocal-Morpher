import os
import librosa
import numpy as np

def extract_feats(file):
    y, sr = librosa.load(file, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    f0, _, _ = librosa.pyin(y, fmin=80, fmax=300)
    f0 = np.nan_to_num(f0)
    f0 = np.interp(np.arange(mfccs.shape[1]), np.linspace(0, len(f0)-1, len(f0)), f0)
    return np.vstack([mfccs, f0[:mfccs.shape[1]]])

source_dir = "data/train/source"
target_dir = "data/train/target"

X = []  # Source
Y = []  # Target

for fname in os.listdir(source_dir):
    if not fname.endswith(".wav"):
        continue
    source_path = os.path.join(source_dir, fname)
    target_path = os.path.join(target_dir, fname)
    if os.path.exists(target_path):
        x = extract_feats(source_path)
        y = extract_feats(target_path)
        min_len = min(x.shape[1], y.shape[1])
        X.append(x[:, :min_len].T)
        Y.append(y[:, :min_len].T)

X = np.vstack(X)
Y = np.vstack(Y)

np.save("X_train.npy", X)
np.save("Y_train.npy", Y)
