import os, json
import numpy as np

DATASET = "dataset/gestures.npz"
OUTDIR  = "models"
OUTPATH = os.path.join(OUTDIR, "gesture_norm.npz")

def main():
    if not os.path.exists(DATASET):
        raise FileNotFoundError(f"Dataset not found: {DATASET} (Run prepare_dataset.py first)")

    os.makedirs(OUTDIR, exist_ok=True)

    data = np.load(DATASET, allow_pickle=True)
    X = data["X"]
    classes = list(data["classes"])
    T = int(X.shape[1])
    F = int(X.shape[2])

    Xmean = X.mean(axis=(0,1), keepdims=True)
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
