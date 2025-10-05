# src/prepare_dataset.py
import os
import sys
import csv
import json
import argparse
import numpy as np
from collections import Counter

def parse_args():
    p = argparse.ArgumentParser(description="Pack sequence dataset into a single .npz")
    p.add_argument("--in_dir",  type=str, default="dataset/sequences",
                   help="root folder that contains class subfolders with .npy clips (any nesting inside is OK)")
    p.add_argument("--out_npz", type=str, default="dataset/gestures.npz",
                   help="output .npz path")
    p.add_argument("--summary_csv", type=str, default="dataset/class_counts.csv",
                   help="output CSV with per-class counts")
    p.add_argument("--min_per_class", type=int, default=1,
                   help="drop classes having fewer than this many valid clips")
    p.add_argument("--limit_per_class", type=int, default=0,
                   help="cap number of clips per class (0 = unlimited)")
    p.add_argument("--seed", type=int, default=42, help="shuffle seed")
    p.add_argument("--verbose", action="store_true", help="print scanned files and reasons for skipping")
    return p.parse_args()

def read_meta(in_dir):
    meta_path = os.path.join(in_dir, "_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def recursive_list_label_files(root):
    """
    Walk under root and yield (label, filepath) where `label` is the
    first-level directory name right under `root`.
      Accepts layouts like:
        root/<label>/*.npy
        root/<label>/<user>/*.npy
        root/<label>/<user>/<hand>/*.npy
        root/<label>/<anything>/**/<anything>.npy
    """
    root = os.path.abspath(root)
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if not fn.lower().endswith(".npy"):
                continue
            fpath = os.path.join(dirpath, fn)
            rel = os.path.relpath(fpath, root)
            parts = rel.split(os.sep)
            if len(parts) < 2:
                continue
            label = parts[0]
            yield label, fpath

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if not os.path.isdir(args.in_dir):
        print(f"[ERROR] input folder not found: {args.in_dir}")
        sys.exit(1)

    # discover labels as first-level dirs
    labels = sorted([d for d in os.listdir(args.in_dir)
                     if os.path.isdir(os.path.join(args.in_dir, d)) and not d.startswith("_")])
    if not labels:
        print(f"[ERROR] no class folders under: {args.in_dir}")
        sys.exit(1)

    # collect files recursively per label
    per_class_files = {c: [] for c in labels}
    for label, fpath in recursive_list_label_files(args.in_dir):
        per_class_files.setdefault(label, []).append(fpath)

    # drop labels with no files at all
    labels = [c for c in labels if len(per_class_files.get(c, [])) > 0]
    if not labels:
        print("[ERROR] found no .npy files under any class folders.")
        sys.exit(1)

    print("[INFO] Classes:", labels)
    meta = read_meta(args.in_dir)
    expected_T = meta.get("T") if isinstance(meta, dict) else None

    # filter by min_per_class
    kept = [c for c in labels if len(per_class_files[c]) >= args.min_per_class]
    dropped = sorted(set(labels) - set(kept))
    if dropped:
        print(f"[WARN] drop classes due to min_per_class={args.min_per_class}: {dropped}")
    labels = kept
    label2id = {c: i for i, c in enumerate(labels)}

    X_list, y_list = [], []
    per_class_counts = Counter()
    bad_clips = 0
    T_ref, F_ref = None, None

    for c in labels:
        files = sorted(per_class_files[c])
        if args.limit_per_class > 0 and len(files) > args.limit_per_class:
            idx = rng.permutation(len(files))[:args.limit_per_class]
            files = [files[i] for i in idx]

        for fp in files:
            try:
                arr = np.load(fp)  # expect (T, 63)
            except Exception as e:
                bad_clips += 1
                if args.verbose:
                    print(f"[SKIP] {fp} (np.load error: {e})")
                continue

            if arr.ndim != 2:
                bad_clips += 1
                if args.verbose:
                    print(f"[SKIP] {fp} (ndim={arr.ndim}, expect 2)")
                continue

            T, F = arr.shape
            if T_ref is None:
                T_ref, F_ref = T, F
                if expected_T is not None and expected_T != T_ref:
                    print(f"[WARN] _meta.T={expected_T} but first clip has T={T_ref} (using T={T_ref})")

            if F != F_ref:
                bad_clips += 1
                if args.verbose:
                    print(f"[SKIP] {fp} (F={F}, expect {F_ref})")
                continue

            if T != T_ref:
                # safer to skip than silently pad/trim
                bad_clips += 1
                if args.verbose:
                    print(f"[SKIP] {fp} (T={T}, expect {T_ref})")
                continue

            X_list.append(arr.astype(np.float32))
            y_list.append(label2id[c])
            per_class_counts[c] += 1

    if not X_list:
        print("[ERROR] no valid clips to pack. Check directory layout, shapes, or min_per_class/limit_per_class.")
        sys.exit(1)

    # stack & shuffle
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    # save dataset
    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    np.savez_compressed(args.out_npz, X=X, y=y, classes=np.array(labels))
    print(f"[OK] saved: {args.out_npz}")
    print(f"     shape: X={X.shape} y={y.shape}  (T={X.shape[1]}, F={X.shape[2]})")

    # summary
    print("\n[SUMMARY] per-class counts (after filtering):")
    for c in labels:
        print(f"  {c:16s} : {per_class_counts[c]}")
    if bad_clips:
        print(f"[WARN] skipped bad clips: {bad_clips}")

    # CSV
    os.makedirs(os.path.dirname(args.summary_csv), exist_ok=True)
    with open(args.summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "count"])
        for c in labels:
            w.writerow([c, per_class_counts[c]])
        w.writerow(["_bad_clips_skipped", bad_clips])
    print(f"[OK] wrote summary CSV: {args.summary_csv}")

    # manifest JSON next to NPZ
    manifest = {
        "npz": os.path.abspath(args.out_npz),
        "num_samples": int(len(X)),
        "T": int(X.shape[1]),
        "F": int(X.shape[2]),
        "classes": labels,
        "per_class_counts": dict(per_class_counts),
        "bad_clips_skipped": int(bad_clips),
        "seed": args.seed
    }
    with open(os.path.splitext(args.out_npz)[0] + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote manifest: {os.path.splitext(args.out_npz)[0] + '.manifest.json'}")

if __name__ == "__main__":
    main()
