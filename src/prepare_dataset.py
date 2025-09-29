# src/prepare_dataset.py
# รวมคลิป .npy (shape: T x 63) ใน dataset/sequences/*/*.npy
# -> บันทึกเป็น dataset/gestures.npz (X, y, classes)
#    พร้อมสรุปสถิติ class_counts.csv และพิมพ์รายงานบน console

import os
import sys
import csv
import json
import argparse
import numpy as np
from glob import glob
from collections import defaultdict, Counter

def parse_args():
    p = argparse.ArgumentParser(description="Pack sequence dataset into a single .npz")
    p.add_argument("--in_dir",  type=str, default="dataset/sequences",
                   help="input directory that contains class subfolders with .npy clips")
    p.add_argument("--out_npz", type=str, default="dataset/gestures.npz",
                   help="output .npz path")
    p.add_argument("--summary_csv", type=str, default="dataset/class_counts.csv",
                   help="output CSV with per-class counts")
    p.add_argument("--min_per_class", type=int, default=1,
                   help="skip class if it has fewer than this many clips")
    p.add_argument("--limit_per_class", type=int, default=0,
                   help="(optional) cap number of clips per class (0=unlimited)")
    p.add_argument("--seed", type=int, default=42, help="shuffle seed")
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

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if not os.path.isdir(args.in_dir):
        print(f"[ERROR] input folder not found: {args.in_dir}")
        sys.exit(1)

    # discover classes = subfolders
    classes = sorted([d for d in os.listdir(args.in_dir)
                      if os.path.isdir(os.path.join(args.in_dir, d)) and not d.startswith("_")])
    if not classes:
        print(f"[ERROR] no class folders under: {args.in_dir}")
        sys.exit(1)

    label2id = {c: i for i, c in enumerate(classes)}
    print("[INFO] Classes:", classes)

    # Try to read expected T from meta (optional)
    meta = read_meta(args.in_dir)
    expected_T = None
    if isinstance(meta, dict):
        expected_T = meta.get("T", None)

    X_list, y_list = [], []
    bad_clips = 0
    per_class_files = {c: sorted(glob(os.path.join(args.in_dir, c, "*.npy"))) for c in classes}

    # filter classes by min_per_class (filenames first)
    classes_kept = [c for c in classes if len(per_class_files[c]) >= args.min_per_class]
    dropped = set(classes) - set(classes_kept)
    if dropped:
        print(f"[WARN] drop classes due to min_per_class={args.min_per_class}: {sorted(list(dropped))}")
    classes = classes_kept
    label2id = {c: i for i, c in enumerate(classes)}

    # load clips
    per_class_counts = Counter()
    T_ref, F_ref = None, None

    for c in classes:
        fpaths = per_class_files[c]
        if args.limit_per_class > 0 and len(fpaths) > args.limit_per_class:
            # random sample limit_per_class
            idx = rng.permutation(len(fpaths))[:args.limit_per_class]
            fpaths = [fpaths[i] for i in idx]

        for fp in fpaths:
            try:
                arr = np.load(fp)  # expect (T, 63)
                if arr.ndim != 2:
                    bad_clips += 1
                    continue

                T, F = arr.shape
                # set reference shape on first valid clip
                if T_ref is None:
                    T_ref, F_ref = T, F
                    if expected_T is not None and expected_T != T_ref:
                        print(f"[WARN] _meta.T={expected_T} but found clip T={T_ref}; will enforce T={T_ref}")

                # enforce same feature length
                if F_ref is not None and F != F_ref:
                    bad_clips += 1
                    continue

                # enforce same T (trim or skip)
                if T != T_ref:
                    # simple policy: skip mismatched T (safer than pad/trim silently)
                    bad_clips += 1
                    continue

                X_list.append(arr.astype(np.float32))
                y_list.append(label2id[c])
                per_class_counts[c] += 1

            except Exception:
                bad_clips += 1
                continue

    if not X_list:
        print("[ERROR] no valid clips to pack. Check shapes or min_per_class/limit_per_class.")
        sys.exit(1)

    # stack
    X = np.stack(X_list, axis=0)  # (N, T, F)
    y = np.array(y_list, dtype=np.int64)

    # shuffle
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    # save
    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    np.savez_compressed(args.out_npz, X=X, y=y, classes=np.array(classes))
    print(f"[OK] saved: {args.out_npz}")
    print(f"     shape: X={X.shape} y={y.shape}  (T={X.shape[1]}, F={X.shape[2]})")

    # print & save summary
    print("\n[SUMMARY] per-class counts (after filtering):")
    for c in classes:
        print(f"  {c:16s} : {per_class_counts[c]}")
    if bad_clips:
        print(f"[WARN] skipped bad clips: {bad_clips}")

    # write CSV
    os.makedirs(os.path.dirname(args.summary_csv), exist_ok=True)
    with open(args.summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "count"])
        for c in classes:
            w.writerow([c, per_class_counts[c]])
        w.writerow(["_bad_clips_skipped", bad_clips])
    print(f"[OK] wrote summary CSV: {args.summary_csv}")

    # also save a tiny JSON manifest next to NPZ
    manifest = {
        "npz": os.path.abspath(args.out_npz),
        "num_samples": int(len(X)),
        "T": int(X.shape[1]),
        "F": int(X.shape[2]),
        "classes": classes,
        "per_class_counts": dict(per_class_counts),
        "bad_clips_skipped": int(bad_clips),
        "seed": args.seed
    }
    with open(os.path.splitext(args.out_npz)[0] + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote manifest: {os.path.splitext(args.out_npz)[0] + '.manifest.json'}")

if __name__ == "__main__":
    main()
