#confusion matrix and classification report on validation set

import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

DATA_PATH = "dataset/gestures.npz"
MODEL_DIR = "models/gesture_lstm"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best.keras")
NORM_PATH = "models/gesture_norm.npz"
META_PATH = os.path.join(MODEL_DIR, "train_meta.json")

CM_PNG_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")
CR_PNG_PATH = os.path.join(MODEL_DIR, "classification_report.png")


def save_classification_report_png(report_dict, classes, out_path):
    columns = ["class", "precision", "recall", "f1-score", "support"]
    cell_text = []

    total_support = 0
    for label in classes:
        r = report_dict[label]
        total_support += int(r["support"])
        cell_text.append(
            [
                label,
                f"{r['precision']:.3f}",
                f"{r['recall']:.3f}",
                f"{r['f1-score']:.3f}",
                f"{int(r['support'])}",
            ]
        )

    acc = float(report_dict["accuracy"])
    cell_text.append(
        [
            "accuracy",
            "",
            "",
            f"{acc:.3f}",
            f"{total_support}",
        ]
    )

    # macro avg
    r_macro = report_dict["macro avg"]
    cell_text.append(
        [
            "macro avg",
            f"{r_macro['precision']:.3f}",
            f"{r_macro['recall']:.3f}",
            f"{r_macro['f1-score']:.3f}",
            f"{int(r_macro['support'])}",
        ]
    )

    # weighted avg
    r_weighted = report_dict["weighted avg"]
    cell_text.append(
        [
            "weighted avg",
            f"{r_weighted['precision']:.3f}",
            f"{r_weighted['recall']:.3f}",
            f"{r_weighted['f1-score']:.3f}",
            f"{int(r_weighted['support'])}",
        ]
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#f7f7f7")
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        cellLoc="center",
        loc="center",
        bbox=[0.02, 0.05, 0.96, 0.75],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.4)

    for (row, col), cell in table.get_celld().items():
        if row == 0: 
            cell.set_facecolor("#3568b0")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            if col == 0:
                cell.get_text().set_weight("bold")

    fig.suptitle(
        "Classification Report (Validation Set)",
        fontsize=18,
        fontweight="bold",
        y=0.96,
    )

    plt.subplots_adjust(top=0.82, bottom=0.08)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"[OK] Saved classification report image to: {out_path}")
    plt.close(fig)


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(DATA_PATH)
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(BEST_MODEL_PATH)
    if not os.path.exists(NORM_PATH):
        raise FileNotFoundError(NORM_PATH)

    data = np.load(DATA_PATH, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    classes = list(data["classes"])
    N, T, F = X.shape
    print(f"[INFO] Loaded: X={X.shape}, y={y.shape}, classes={classes}")

    # ----- read meta: val_split + seed -----
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        val_split = float(meta.get("val_split", 0.15))
        seed = int(meta.get("seed", 42))
    else:
        val_split = 0.15
        seed = 42
        print("[WARN] train_meta.json not found, using default val_split=0.15, seed=42")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    (tr_idx, val_idx), = sss.split(np.zeros(len(y)), y)

    Xval_raw, yval = X[val_idx], y[val_idx]
    print(f"[INFO] Eval on val set: {Xval_raw.shape}, classes={len(classes)}")

    # ----- normalize -----
    norm = np.load(NORM_PATH, allow_pickle=True)
    Xmean = norm["mean"]
    Xstd = norm["std"]
    Xval = (Xval_raw - Xmean) / Xstd

    # ----- Load model -----
    model = tf.keras.models.load_model(BEST_MODEL_PATH)
    print("[INFO] Model loaded:", BEST_MODEL_PATH)

    prob = model.predict(Xval, verbose=0)
    y_pred = prob.argmax(axis=1)

    # ----- Confusion matrix -----
    cm = confusion_matrix(yval, y_pred)
    print("\n=== Confusion Matrix (rows=true, cols=pred) ===")
    print("   " + "  ".join([f"{i:2d}" for i in range(len(classes))]))
    for i, row in enumerate(cm):
        print(f"{i:2d} " + "  ".join([f"{x:2d}" for x in row]) + f"   | {classes[i]}")

    # ----- Classification report (text) -----
    print("\n=== Classification Report ===")
    report_str = classification_report(
        yval, y_pred, target_names=classes, digits=3
    )
    print(report_str)

    # ----- Confusion matrix PNG -----
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, ax=ax_cm, values_format="d")
    plt.title("Confusion Matrix (Validation Set)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    os.makedirs(MODEL_DIR, exist_ok=True)
    plt.savefig(CM_PNG_PATH, dpi=300)
    print(f"[OK] Saved confusion matrix image to: {CM_PNG_PATH}")
    plt.close(fig_cm)

    # ----- Classification report PNG -----
    report_dict = classification_report(
        yval, y_pred, target_names=classes, digits=3, output_dict=True
    )
    save_classification_report_png(report_dict, classes, CR_PNG_PATH)


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
