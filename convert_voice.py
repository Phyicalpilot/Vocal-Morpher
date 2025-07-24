import torch
import librosa
import pyworld as pw
import numpy as np
import soundfile as sf

from train_model import VCNet

model = VCNet()
model.load_state_dict(torch.load("vc_model.pth"))
model.eval()

def convert(input_wav, output_wav):
    y, sr = librosa.load(input_wav, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    f0, _, _ = librosa.pyin(y, fmin=80, fmax=300)
    f0 = np.nan_to_num(f0)
    f0_interp = np.interp(np.arange(mfccs.shape[1]), np.linspace(0, len(f0)-1, len(f0)), f0)
    feats = np.vstack([mfccs, f0_interp[:mfccs.shape[1]]]).T

    with torch.no_grad():
        feats = torch.tensor(feats, dtype=torch.float32)
        out_feats = model(feats).numpy().T

    # Extract vocoder features from original
    _f0, t = pw.harvest(y, sr)
    sp = pw.cheaptrick(y, _f0, t, sr)
    ap = pw.d4c(y, _f0, t, sr)

    # Replace F0 with predicted F0 (last row)
    f0_pred = out_feats[-1]
    f0_pred = np.interp(np.arange(len(_f0)), np.linspace(0, len(f0_pred)-1, len(f0_pred)), f0_pred)
    y_out = pw.synthesize(f0_pred, sp, ap, sr)
    sf.write(output_wav, y_out, sr)

convert("example.wav", "converted.wav")
