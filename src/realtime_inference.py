# Real-time gesture inference (MediaPipe → LSTM → pyautogui actions)

import os
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pyautogui
import platform

# ===================== Config =====================
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

MODEL_DIR = "models/gesture_lstm"
MODEL_PATH = "models/gesture_lstm/best.keras"
NORM_PATH = "models/gesture_norm.npz"

T_DEFAULT = 30
F_DEFAULT = 63

# probability per class
THRESH = {
    "desktop_left": 0.75,
    "desktop_right": 0.75,
    "tab_left": 0.75,
    "tab_right": 0.75,
    "scroll_up": 0.70,
    "scroll_down": 0.70,
    "scroll_left": 0.70,
    "scroll_right": 0.70,
    "screenshot": 0.80,
    "idle": 1.10,
}

COOLDOWN = {
    "desktop_left": 0.9,
    "desktop_right": 0.9,
    "tab_left": 0.7,
    "tab_right": 0.7,
    "screenshot": 1.2,
    "scroll_up": 0.0,
    "scroll_down": 0.0,
    "scroll_left": 0.0,
    "scroll_right": 0.0,
    "idle": 0.0,
}

SCROLL_STEP_Y = 600
HSCROLL_STEP_X = 80
DRAW_PROB_BAR = True

MIN_DET, MIN_TRK = 0.6, 0.6  # MediaPipe thresholds

OS = platform.system().lower()

# ================ Actions Mapping ==================
def do_action(label: str):
    if label == "desktop_left":
        if OS == "windows": pyautogui.hotkey("ctrl", "winleft", "left")
        elif OS == "darwin": pyautogui.hotkey("ctrl", "left")
        else: pyautogui.hotkey("ctrl", "alt", "left")

    elif label == "desktop_right":
        if OS == "windows": pyautogui.hotkey("ctrl", "winleft", "right")
        elif OS == "darwin": pyautogui.hotkey("ctrl", "right")
        else: pyautogui.hotkey("ctrl", "alt", "right")

    elif label == "tab_left":
        pyautogui.hotkey("ctrl", "shift", "tab")

    elif label == "tab_right":
        pyautogui.hotkey("ctrl", "tab")

    elif label == "scroll_up":
        pyautogui.scroll(+SCROLL_STEP_Y)

    elif label == "scroll_down":
        pyautogui.scroll(-SCROLL_STEP_Y)

    elif label == "scroll_left":
        try:
            pyautogui.hscroll(-HSCROLL_STEP_X)
        except Exception:
            pyautogui.press("left")

    elif label == "scroll_right":
        try:
            pyautogui.hscroll(+HSCROLL_STEP_X)
        except Exception:
            pyautogui.press("right")

    elif label == "screenshot":
        if OS == "windows":
            pyautogui.hotkey("winleft", "shift", "s")
        elif OS == "darwin":
            pyautogui.hotkey("command", "shift", "4")
        else:
            pass

# ================== Utilities =====================
def load_model_and_norm():
    """
    Load models in order of priority:
      1) MODEL_PATH (.keras)
      2) models/gesture_lstm/final.keras
      3) SavedModel folder (fallback with keras.layers.TFSMLayer)
    """
    candidates = [
        MODEL_PATH,
        os.path.join(MODEL_DIR, "final.keras"),
        MODEL_DIR,
    ]
    model = None
    last_err = None
    for path in candidates:
        try:
            if os.path.isdir(path):
                # Keras 3: SavedModel must be loaded as a layer.
                from keras.layers import TFSMLayer
                model = TFSMLayer(path, call_endpoint="serving_default")
            else:
                model = tf.keras.models.load_model(path)
            print(f"[INFO] Loaded model from: {path}")
            break
        except Exception as e:
            last_err = e

    if model is None:
        raise RuntimeError(
            f"Cannot load model from any of: {candidates}\nLast error: {last_err}"
        )

    if not os.path.exists(NORM_PATH):
        raise FileNotFoundError(f"Normalization file not found: {NORM_PATH}")

    norm = np.load(NORM_PATH, allow_pickle=True)
    Xmean = norm["mean"]
    Xstd = norm["std"]
    classes = list(norm["classes"])
    T = int(norm["T"]) if "T" in norm.files else T_DEFAULT
    F = int(norm["F"]) if "F" in norm.files else F_DEFAULT
    return model, Xmean, Xstd, classes, T, F


def topk(prob, classes, k=3):
    idx = np.argsort(prob)[::-1][:k]
    return [(classes[i], float(prob[i])) for i in idx]

# ================ Main Inference ===================
def main():
    model, Xmean, Xstd, classes, T, F = load_model_and_norm()
    print("[INFO] classes:", classes)

   # map threshold / cooldown only for classes that actually exist
    thr = {c: THRESH.get(c, 0.8) for c in classes}
    cd  = {c: COOLDOWN.get(c, 0.6) for c in classes}
    last_time = {c: 0.0 for c in classes}
    buf = deque(maxlen=T)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    mp_hands = mp.solutions.hands
    drawer = mp.solutions.drawing_utils
    styles = mp.solutions.drawing_styles

    print("[INFO] Press 'q' to quit.")

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=MIN_DET,
        min_tracking_confidence=MIN_TRK,
        model_complexity=0
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            label_show = "…"
            conf_show = 0.0
            top3 = []

            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                handed = None
                if res.multi_handedness:
                    handed = res.multi_handedness[0].classification[0].label  # 'Left' or 'Right'

                # landmarks → vec63 (mirror x if Left)
                pts = []
                for i in range(21):
                    x = hand.landmark[i].x
                    y = hand.landmark[i].y
                    z = hand.landmark[i].z
                    if handed == "Left":
                        x = 1.0 - x
                    pts.extend([x, y, z])
                vec = np.asarray(pts, dtype=np.float32)
                buf.append(vec)

                drawer.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS,
                                      styles.get_default_hand_landmarks_style(),
                                      styles.get_default_hand_connections_style())

                # infer when buffer is full
                if len(buf) == T:
                    X = np.stack(list(buf))[None, ...]
                    Xn = (X - Xmean) / Xstd

                    # Supports both regular Keras models and TFSMLayer (SavedModel).
                    if hasattr(model, "predict"):
                        prob = model.predict(Xn, verbose=0)[0]
                    else:
                        # TFSMLayer returns a dict of endpoints → uses the first value
                        out = model(Xn)
                        if isinstance(out, dict):
                            out = list(out.values())[0]
                        prob = np.array(out)[0]

                    k = int(prob.argmax())
                    label = classes[k]
                    conf = float(prob[k])
                    label_show, conf_show = label, conf
                    top3 = topk(prob, classes, k=3)

                    # Decide with threshold + cooldown
                    now = time.time()
                    if label != "idle" and conf >= thr.get(label, 0.8):
                        if now - last_time[label] >= cd.get(label, 0.6):
                            do_action(label)
                            last_time[label] = now

            # ================= Overlay =================
            cv2.putText(frame, f"{label_show} : {conf_show:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0) if label_show != "…" else (200, 200, 200), 2)

            if DRAW_PROB_BAR and top3:
                base_y = 60
                for i, (lab, p) in enumerate(top3):
                    bar_w = int(300 * p)
                    cv2.putText(frame, f"{lab:<14s} {p:0.2f}", (10, base_y + i * 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.rectangle(frame, (170, base_y - 12 + i * 28),
                                  (170 + bar_w, base_y - 12 + i * 28 + 18), (0, 180, 255), -1)
                    cv2.rectangle(frame, (170, base_y - 12 + i * 28),
                                  (170 + 300, base_y - 12 + i * 28 + 18), (255, 255, 255), 1)

            cv2.putText(frame, "q: quit", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)

            cv2.imshow("Realtime Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
