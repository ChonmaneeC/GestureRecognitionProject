# src/make_norm_from_dataset.py
# สร้าง models/gesture_norm.npz จาก dataset/gestures.npz

import os, json
import numpy as np

DATASET = "dataset/gestures.npz"
OUTDIR  = "models"
OUTPATH = os.path.join(OUTDIR, "gesture_norm.npz")

def main():
    if not os.path.exists(DATASET):
        raise FileNotFoundError(f"Dataset not found: {DATASET} (ให้รัน prepare_dataset.py ก่อน)")

    os.makedirs(OUTDIR, exist_ok=True)

    data = np.load(DATASET, allow_pickle=True)
    X = data["X"]          # (N, T, F)
    classes = list(data["classes"])
    T = int(X.shape[1])
    F = int(X.shape[2])

    # mean/std แบบ global ต่อ feature
    Xmean = X.mean(axis=(0,1), keepdims=True)  # (1,1,F)
    Xstd  = X.std(axis=(0,1), keepdims=True) + 1e-6

    np.savez(OUTPATH,
             mean=Xmean.astype(np.float32),
             std=Xstd.astype(np.float32),
             classes=np.array(classes, dtype=object),
             T=T, F=F)
    print(f"[OK] wrote {OUTPATH}  with keys: mean, std, classes, T, F")
    print(f"     classes={classes}, T={T}, F={F}")

if __name__ == "__main__":
    main()
