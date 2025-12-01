import os
import cv2
import time
import json
import argparse
import numpy as np
import mediapipe as mp
from collections import deque
from datetime import datetime
from pathlib import Path

# ================ CONFIG =================
CLASSES = [
    "desktop_left", "desktop_right",
    "tab_left", "tab_right",
    "scroll_left", "scroll_right",
    "scroll_up", "scroll_down",
    "screenshot",
    "idle"
]

KEYMAP = {
    'a': "desktop_left",   'd': "desktop_right",
    'j': "tab_left",       'l': "tab_right",
    'h': "scroll_left",    'n': "scroll_right",
    'u': "scroll_up",      'k': "scroll_down",
    'o': "screenshot",
    'z': "idle"
}

EXPECTED_FINGERS = {
    "desktop_left": 5,  "desktop_right": 5,  "screenshot": None,
    "tab_left": 3,      "tab_right": 3,
    "scroll_left": 2,   "scroll_right": 2,   "scroll_up": 2,   "scroll_down": 2,
    "idle": None
}

T = 30
OUT_ROOT = Path("dataset/sequences")
MIN_DET, MIN_TRK = 0.6, 0.6
SHOW_HINT = True

# ========================================

def parse_args():
    ap = argparse.ArgumentParser(description="Collect hand-landmark sequences to .npy")
    ap.add_argument("--user", required=True, help="collector name, e.g., alice")
    ap.add_argument("--hand", choices=["left", "right"], default="right", help="which hand you use")
    ap.add_argument("--camera", type=int, default=0, help="cv2 camera index")
    ap.add_argument("--label", default="idle", help="initial label (use hotkeys to switch)")
    ap.add_argument("--frames", type=int, default=T, help="frames per sequence (T)")
    ap.add_argument("--countdown", type=int, default=2, help="seconds before recording starts")
    ap.add_argument("--min_det", type=float, default=MIN_DET)
    ap.add_argument("--min_trk", type=float, default=MIN_TRK)
    return ap.parse_args()

# ---------- geometry helpers ----------
def angle_between(v1, v2):
    num = np.dot(v1, v2)
    den = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
    c = np.clip(num / den, -1.0, 1.0)
    return np.degrees(np.arccos(c))

def is_extended(lm2d, tip, pip_, mcp):
    v1 = np.array([lm2d[tip][0]-lm2d[pip_][0], lm2d[tip][1]-lm2d[pip_][1]])
    v2 = np.array([lm2d[mcp][0]-lm2d[pip_][0], lm2d[mcp][1]-lm2d[pip_][1]])
    ang = angle_between(v1, v2)
    return ang > 160
def count_fingers(lm2d):
    return {
        'thumb':  is_extended(lm2d, 4, 3, 2),
        'index':  is_extended(lm2d, 8, 6, 5),
        'middle': is_extended(lm2d,12,10,9),
        'ring':   is_extended(lm2d,16,14,13),
        'pinky':  is_extended(lm2d,20,18,17),
    }

def expected_finger_hint(label):
    if label in ("desktop_left", "desktop_right"):
        return "Use 5 fingers to swipe left/right."
    if label in ("tab_left", "tab_right"):
        return "Use 3 fingers (index + middle + ring) to swipe left/right."
    if label in ("scroll_left", "scroll_right", "scroll_up", "scroll_down"):
        return "Use 2 fingers (pointer + middle) to sweep in the direction"
    if label == "screenshot":
        return "Open your hand with 5 fingers → make a fist in the same clip."
    if label == "idle":
        return "Hands remain still/no gestures"
    return ""

def save_meta(root: Path, T_value: int):
    root.mkdir(parents=True, exist_ok=True)
    meta_path = root / "_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"T": T_value, "classes": CLASSES, "expected_fingers": EXPECTED_FINGERS},
            f, ensure_ascii=False, indent=2
        )

def ensure_dirs(label: str, user: str, hand: str) -> Path:
    d = OUT_ROOT / label / user / hand
    d.mkdir(parents=True, exist_ok=True)
    return d

def save_clip(label: str, user: str, hand: str, clip_np: np.ndarray) -> str:
    """
    Save as .npy: shape = (T, 63)
    path: dataset/sequences/<label>/<user>/<hand>/clip_YYYYmmdd_HHMMSS_xxxxxx.npy
    """
    out_dir = ensure_dirs(label, user, hand)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = out_dir / f"clip_{ts}.npy"
    np.save(out_path.as_posix(), clip_np.astype(np.float32))
    return out_path.as_posix()

# --------------- Main ---------------
def main():
    args = parse_args()
    T_local = int(args.frames)

    save_meta(OUT_ROOT, T_local)
    for c in CLASSES:
        (OUT_ROOT / c).mkdir(parents=True, exist_ok=True)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    print(f"[INFO] Collector: {args.user}  Hand: {args.hand}  T={T_local}")
    print("[INFO] Hotkeys: A/D desktop  J/L tab  H/N scroll←/→  U/K scroll↑/↓  O screenshot  Z idle  |  S start/stop  Q quit")

    buf = deque(maxlen=T_local)
    current = args.label if args.label in CLASSES else "idle"
    recording = False
    saved = {c: 0 for c in CLASSES}
    prev_tip = None

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=args.min_det,
        min_tracking_confidence=args.min_trk
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            status = f"[{current}] user:{args.user}/{args.hand}  REC:{'ON' if recording else 'OFF'}  saved:{saved[current]}"
            cv2.putText(frame, status, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 220, 255) if recording else (255, 255, 255), 2)

            if SHOW_HINT:
                hint = expected_finger_hint(current)
                if hint:
                    cv2.putText(frame, hint, (10, 56),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 200), 2)

            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                handed_label = None
                if res.multi_handedness:
                    handed_label = res.multi_handedness[0].classification[0].label
                pts_2d = []
                vec63 = []
                for i in range(21):
                    x = hand.landmark[i].x
                    y = hand.landmark[i].y
                    z = hand.landmark[i].z
                    #if handed_label == "Left":
                       # x = 1.0 - x  # canonicalize left->right
                    pts_2d.append((x, y))
                    vec63.extend([x, y, z])
                vec63 = np.array(vec63, dtype=np.float32)

                mp_draw.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

                fingers = count_fingers(pts_2d)
                num_up = int(sum(fingers.values()))
                cv2.putText(frame, f"Fingers up: {num_up}", (10, 86),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                if recording:
                    need = EXPECTED_FINGERS.get(current)
                    ok_to_record = True if need is None else (num_up == need)

                    if ok_to_record:
                        buf.append(vec63)

                    cv2.rectangle(frame, (10, h-30), (10 + int((len(buf)/T_local)*(w-20)), h-10), (0, 200, 0), -1)
                    cv2.rectangle(frame, (10, h-30), (w-10, h-10), (255, 255, 255), 2)

                    if len(buf) == T_local:
                        clip_np = np.stack(list(buf), axis=0)  # (T,63)
                        out_path = save_clip(current, args.user, args.hand, clip_np)
                        saved[current] += 1
                        buf.clear()
                        cv2.putText(frame, f"Saved: {Path(out_path).name}", (10, h-46),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                tip = np.array(pts_2d[8], dtype=np.float32)
                if prev_tip is not None and SHOW_HINT:
                    delta = tip - prev_tip
                    cxy = (int(tip[0]*w), int(tip[1]*h))
                    exy = (int((tip[0]+delta[0]*3)*w), int((tip[1]+delta[1]*3)*h))
                    cv2.arrowedLine(frame, cxy, exy, (100, 255, 100), 2, tipLength=0.4)
                prev_tip = tip.copy()

            else:
                prev_tip = None

            cv2.putText(frame,
                        "A/D desktop  J/L tab  H/N scroll<- ->  U/K scroll^ v  O screenshot  Z idle  |  S start/stop  Q quit",
                        (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)

            cv2.imshow(f"Collect Sequences (T={T_local})", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('s'):
                buf.clear()
                if not recording and args.countdown > 0:
                    t0 = time.time()
                    while time.time() - t0 < args.countdown:
                        ok2, frame2 = cap.read()
                        if not ok2: break
                        frame2 = cv2.flip(frame2, 1)
                        remain = args.countdown - int(time.time() - t0)
                        cv2.putText(frame2, f"Recording in {remain}", (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,200,255), 3)
                        cv2.imshow(f"Collect Sequences (T={T_local})", frame2)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            cap.release(); cv2.destroyAllWindows(); return
                recording = not recording
            elif chr(k).lower() in KEYMAP:
                current = KEYMAP[chr(k).lower()]
                buf.clear()

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Saved counts:", {k: int(v) for k, v in saved.items()})

if __name__ == "__main__":
    main()
