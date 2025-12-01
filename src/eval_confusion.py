# src/eval_confusion.py
import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib.patches as patches

DATA_PATH = "dataset/gestures.npz"
MODEL_DIR = "models/gesture_lstm"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best.keras")
NORM_PATH = "models/gesture_norm.npz"
META_PATH = os.path.join(MODEL_DIR, "train_meta.json")

CM_PNG_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")
CR_PNG_PATH = os.path.join(MODEL_DIR, "classification_report.png")


def save_classification_report_png(report_text, output_path):
    """Convert text classification report to a PNG image."""
    fig = plt.figure(figsize=(8, 10))
    plt.text(0.01, 0.99, report_text, fontsize=10, family="monospace", va='top')
    plt.axis('off')

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Saved classification report image to: {output_path}")

def main():
    # Load Data & Model
    if not os.path.exists(DATA_PATH): raise FileNotFoundError(DATA_PATH)
    if not os.path.exists(BEST_MODEL_PATH): raise FileNotFoundError(BEST_MODEL_PATH)
    if not os.path.exists(NORM_PATH): raise FileNotFoundError(NORM_PATH)

    data = np.load(DATA_PATH, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    classes = list(data["classes"])
    print(f"[INFO] Loaded dataset: X={X.shape}, y={y.shape}, classes={classes}")

    # load train metadata
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        val_split = float(meta.get("val_split", 0.15))
        seed = int(meta.get("seed", 42))
    else:
        val_split = 0.15
        seed = 42

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    (tr_idx, val_idx), = sss.split(np.zeros(len(y)), y)

    Xval_raw, yval = X[val_idx], y[val_idx]
    print(f"[INFO] Using validation set: {Xval_raw.shape}")

    # normalize
    norm = np.load(NORM_PATH, allow_pickle=True)
    Xmean, Xstd = norm["mean"], norm["std"]
    Xval = (Xval_raw - Xmean) / Xstd

    # load model
    model = tf.keras.models.load_model(BEST_MODEL_PATH)
    print(f"[INFO] Model loaded from {BEST_MODEL_PATH}")

    prob = model.predict(Xval, verbose=0)
    y_pred = prob.argmax(axis=1)

    # Confusion Matrix
    cm = confusion_matrix(yval, y_pred)

    print("\n=== Confusion Matrix ===")
    print(cm)

    fig_cm, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title("Confusion Matrix (Validation Set)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(CM_PNG_PATH, dpi=300)
    print(f"[OK] Saved confusion matrix image to: {CM_PNG_PATH}")
    plt.close(fig_cm)

    # Classification Report
    report = classification_report(yval, y_pred, target_names=classes, digits=3)
    print("\n=== Classification Report ===")
    print(report)

    save_classification_report_png(report, CR_PNG_PATH)


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
