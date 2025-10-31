# Train an LSTM to classify gesture sequences (X: N x T x F, y: N)
# - Load from dataset/gestures.npz
# - Perform normalization
# - Train with callbacks
# - Save the model and statistics to models/gesture_lstm/

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from collections import Counter

def parse_args():
    p = argparse.ArgumentParser(description="Train LSTM gesture classifier")
    p.add_argument("--data", type=str, default="dataset/gestures.npz",
                   help="path to gestures.npz (must contain X,y,classes)")
    p.add_argument("--outdir", type=str, default="models/gesture_lstm",
                   help="output directory for model and artifacts")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_split", type=float, default=0.15,
                   help="validation fraction (0-0.5)")
    p.add_argument("--lstm_units", type=int, default=128)
    p.add_argument("--dense_units", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--use_class_weights", action="store_true",
                   help="use inverse-frequency class weights")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def build_model(T, F, num_classes, lstm_units=128, dense_units=128, dropout=0.3, lr=1e-3):
    inp = layers.Input(shape=(T, F))
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.LSTM(lstm_units, return_sequences=True)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(lstm_units)(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inp, out)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def make_class_weights(y, num_classes):
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    w = inv / inv.sum() * num_classes
    return {i: float(w[i]) for i in range(num_classes)}, counts.tolist()

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    os.makedirs(args.outdir, exist_ok=True)

    # ---------- Load ----------
    data = np.load(args.data, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    classes = list(data["classes"])
    num_classes = len(classes)
    N, T, F = X.shape
    print(f"[INFO] Loaded: X={X.shape}, y={y.shape}, num_classes={num_classes}")
    print(f"[INFO] Classes: {classes}")

    # ---------- Split train/val (stratified) ----------
    from sklearn.model_selection import StratifiedShuffleSplit

    assert 0.0 < args.val_split < 0.5, "val_split should be in (0, 0.5)"
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_split, random_state=args.seed)
    (tr_idx, val_idx), = sss.split(np.zeros(len(y)), y)

    Xtr_raw, ytr = X[tr_idx], y[tr_idx]
    Xval_raw, yval = X[val_idx], y[val_idx]
    print(f"[INFO] Split: train={len(Xtr_raw)}  val={len(Xval_raw)}")

    # ---------- Normalize (fit on train only) ----------
    Xmean = Xtr_raw.mean(axis=(0, 1), keepdims=True)
    Xstd  = Xtr_raw.std(axis=(0, 1), keepdims=True) + 1e-6

    Xtr = (Xtr_raw - Xmean) / Xstd
    Xval = (Xval_raw - Xmean) / Xstd


    # ---------- Class weights (optional) ----------
    class_weight = None
    counts = np.bincount(y, minlength=num_classes)
    if args.use_class_weights:
        class_weight, counts_list = make_class_weights(ytr, num_classes)
        print(f"[INFO] Using class weights: {class_weight}")
    else:
        counts_list = counts.tolist()

    # ---------- Build ----------
    model = build_model(T, F, num_classes,
                        lstm_units=args.lstm_units,
                        dense_units=args.dense_units,
                        dropout=args.dropout,
                        lr=args.lr)
    model.summary()

    # ---------- Callbacks ----------
    ckpt_path = os.path.join(args.outdir, "best.keras")
    cbs = [
        callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy",
                                  save_best_only=True, mode="max", verbose=1),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=6,
                                restore_best_weights=True, mode="max", verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                                    min_lr=1e-6, verbose=1)
    ]

    # ---------- Train ----------
    hist = model.fit(
        Xtr, ytr,
        validation_data=(Xval, yval),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=cbs,
        class_weight=class_weight
    )

    # ---------- Evaluate ----------
    val_loss, val_acc = model.evaluate(Xval, yval, verbose=0)
    print(f"[OK] Val: loss={val_loss:.4f}  acc={val_acc:.4f}")

    # ---------- Save model & artifacts ----------
    # 1) SavedModel
    model.save(os.path.join(args.outdir, "final.keras"))
    # 2) Normalization + classes
    norm_path = os.path.join("models", "gesture_norm.npz")
    os.makedirs(os.path.dirname(norm_path), exist_ok=True)
    np.savez(norm_path, mean=Xmean, std=Xstd, classes=np.array(classes), T=X.shape[1], F=X.shape[2])
    print(f"[OK] Saved final model to {os.path.join(args.outdir, 'final.keras')}")
    print(f"[OK] Saved norm: {norm_path}")
    # 3) Metadata
    meta = {
        "data": os.path.abspath(args.data),
        "outdir": os.path.abspath(args.outdir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "val_split": args.val_split,
        "lstm_units": args.lstm_units,
        "dense_units": args.dense_units,
        "dropout": args.dropout,
        "use_class_weights": args.use_class_weights,
        "seed": args.seed,
        "classes": classes,
        "num_classes": num_classes,
        "counts": counts_list,
        "val_metrics": {"loss": float(val_loss), "accuracy": float(val_acc)}
    }

    counts_train = Counter(ytr.tolist())
    counts_val   = Counter(yval.tolist())

    meta.update({
    "T": int(X.shape[1]),
    "F": int(X.shape[2]),
    "class_counts_train": {classes[k]: int(v) for k, v in counts_train.items()},
    "class_counts_val":   {classes[k]: int(v) for k, v in counts_val.items()},
    })
    
    with open(os.path.join(args.outdir, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote train_meta.json")

if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
