# src/realtime_inference.py
# Real-time gesture inference (MediaPipe Hands → LSTM → pyautogui actions)
# - Full-screen screenshot (no drag) per OS
# - Gate screenshot with "open-hand -> fist" sequence to avoid false triggers

import os
import time
import platform
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pyautogui

# ===================== Config =====================
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# --- Model paths ---
MODEL_DIR  = "models/gesture_lstm"
MODEL_PATH = "models/gesture_lstm/best.keras"   # preferred
NORM_PATH  = "models/gesture_norm.npz"          # mean/std/classes/T/F saved at training

# Defaults (overridden by norm file if present)
T_DEFAULT = 30
F_DEFAULT = 63

# Per-class probability threshold
THRESH = {
    "desktop_left": 0.75,
    "desktop_right": 0.75,
    "tab_left": 0.60,
    "tab_right": 0.60,
    "scroll_up": 0.70,
    "scroll_down": 0.70,
    "scroll_left": 0.70,
    "scroll_right": 0.70,
    "screenshot": 0.80,
    "idle": 1.10,   # intentionally unreachable
}

# Cooldowns (seconds) to prevent repeated one-shot actions
COOLDOWN = {
    "desktop_left": 0.4,
    "desktop_right": 0.4,
    "tab_left": 0.7,
    "tab_right": 0.7,
    "screenshot": 1.2,
    # continuous actions: allow 0
    "scroll_up": 0.0,
    "scroll_down": 0.0,
    "scroll_left": 0.0,
    "scroll_right": 0.0,
    "idle": 0.0,
}

SCROLL_STEP_Y   = 600   # vertical scroll per trigger
HSCROLL_STEP_X  = 80    # horizontal scroll (fallback arrow keys if needed)
DRAW_PROB_BAR   = True

# ---- Horizontal scroll mode (บางแอปไม่รองรับ hscroll) ----
USE_SHIFT_FOR_HSCROLL = True   # True = ใช้ Shift+wheel ทำแนวนอน, False = ใช้ pyautogui.hscroll()
HSCROLL_WHEEL_FACTOR  = 10     # ขนาดการเลื่อนแนวนอนเมื่อใช้ wheel (ปรับได้)

def hscroll_signed(amount):
    """เลื่อนแนวนอนด้วยวิธีที่เลือกไว้"""
    if USE_SHIFT_FOR_HSCROLL:
        # amount > 0 = ขวา, amount < 0 = ซ้าย
        pyautogui.keyDown('shift')
        pyautogui.scroll(int(amount) * HSCROLL_WHEEL_FACTOR)
        pyautogui.keyUp('shift')
    else:
        try:
            pyautogui.hscroll(int(amount))
        except Exception:
            # fallback เป็นปุ่มซ้าย/ขวา
            pyautogui.press('right' if amount > 0 else 'left')

# -------- Active-app helper (prefer tab switch inside browsers) --------
BROWSER_APP_KEYWORDS = ["chrome", "microsoft edge", "edge", "firefox", "brave", "opera", "arc"]

def is_browser_active():
    try:
        win = pyautogui.getActiveWindow()
        if not win:
            return False
        title = (win.title or "")
        return any(k in title.lower() for k in BROWSER_APP_KEYWORDS)
    except Exception:
        return False

# MediaPipe thresholds
MIN_DET, MIN_TRK = 0.6, 0.6

OS = platform.system().lower()

# --- Screenshot behavior ---
SCREENSHOT_REQUIRE_SEQUENCE = True   # must be "open -> fist" to allow screenshot
SCREENSHOT_WINDOW          = 1.0     # seconds allowed between open and fist
OPEN_MIN_FINGERS           = 4       # consider "open" if >= this many extended
FIST_MAX_FINGERS           = 1       # consider "fist" if <= this many extended

# ================ Actions Mapping ==================
def do_action(label: str):
    if label == "desktop_left":
        if OS == "windows":
            # Alt+Shift+Tab = ย้อนกลับโปรแกรมก่อนหน้า
            pyautogui.keyDown("alt")
            pyautogui.keyDown("shift")
            pyautogui.press("tab")
            pyautogui.keyUp("shift")
            pyautogui.keyUp("alt")
        elif OS == "darwin":
            pyautogui.hotkey("command", "shift", "tab")
        else:
            pyautogui.hotkey("alt", "shift", "tab")

    elif label == "desktop_right":
        if OS == "windows":
            # Alt+Tab = ไปโปรแกรมถัดไป
            pyautogui.keyDown("alt")
            pyautogui.press("tab")
            pyautogui.keyUp("alt")
        elif OS == "darwin":
            pyautogui.hotkey("command", "tab")
        else:
            pyautogui.hotkey("alt", "tab")

    elif label == "tab_left":
        pyautogui.hotkey("ctrl", "shift", "tab")

    elif label == "tab_right":
        pyautogui.hotkey("ctrl", "tab")

    elif label == "scroll_up":
        pyautogui.scroll(+SCROLL_STEP_Y)

    elif label == "scroll_down":
        pyautogui.scroll(-SCROLL_STEP_Y)

    elif label == "scroll_left":
        hscroll_signed(-1)   # ซ้าย = ค่าติดลบ

    elif label == "scroll_right":
        hscroll_signed(+1)   # ขวา = ค่าบวก

    elif label == "screenshot":
        # Full-screen screenshot by OS (no selection drag)
        if OS == "windows":
            # Saves to Pictures\Screenshots automatically
            pyautogui.hotkey("winleft", "printscreen")
        elif OS == "darwin":
            # Saves a file on Desktop
            pyautogui.hotkey("command", "shift", "3")
        else:
            # Most Linux DEs bind PrintScreen to full screen
            pyautogui.press("printscreen")

# ================== Utilities =====================
def _angle_between(v1, v2):
    num = float(np.dot(v1, v2))
    den = float(np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
    c = np.clip(num / den, -1.0, 1.0)
    return np.degrees(np.arccos(c))

def _is_extended(lm2d, tip, pip_, mcp):
    # Determine if finger is extended using PIP joint angle (2D)
    v1 = np.array([lm2d[tip][0]-lm2d[pip_][0], lm2d[tip][1]-lm2d[pip_][1]], dtype=np.float32)
    v2 = np.array([lm2d[mcp][0]-lm2d[pip_][0], lm2d[mcp][1]-lm2d[pip_][1]], dtype=np.float32)
    return _angle_between(v1, v2) > 160.0

def count_fingers_from_pts2d(lm2d):
    # lm2d: list[(x,y)] index per MediaPipe Hands 0..20
    return int(sum([
        _is_extended(lm2d, 4, 3, 2),    # thumb
        _is_extended(lm2d, 8, 6, 5),    # index
        _is_extended(lm2d,12,10,9),     # middle
        _is_extended(lm2d,16,14,13),    # ring
        _is_extended(lm2d,20,18,17),    # pinky
    ]))

def load_model_and_norm():
    """
    Load model with preference:
      1) models/gesture_lstm/best.keras
      2) models/gesture_lstm/final.keras
      3) SavedModel dir via keras.layers.TFSMLayer (Keras 3)
    """
    candidates = [
        os.path.join(MODEL_DIR, "best.keras"),
        os.path.join(MODEL_DIR, "final.keras"),
        MODEL_DIR,  # fallback: SavedModel directory
    ]
    model = None
    last_err = None
    for path in candidates:
        try:
            if os.path.isdir(path):
                # Keras 3 SavedModel as inference-only layer
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
    Xmean   = norm["mean"]
    Xstd    = norm["std"]
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
    buf = deque(maxlen=T)

    # thresholds & cooldown maps for existing classes
    thr = {c: THRESH.get(c, 0.8) for c in classes}
    cd  = {c: COOLDOWN.get(c, 0.6) for c in classes}
    last_time = {c: 0.0 for c in classes}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    mp_hands = mp.solutions.hands
    drawer   = mp.solutions.drawing_utils
    styles   = mp.solutions.drawing_styles

    print("[INFO] Press 'q' to quit.")

    # screenshot sequence gating state
    last_open_time = 0.0
    had_open_recently = False

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

                # infer handedness
                handed = None
                if res.multi_handedness:
                    handed = res.multi_handedness[0].classification[0].label  # 'Left' or 'Right'

                # landmarks → vec63 (mirror x if Left to canonicalize)
                pts_2d = []
                vec = []
                for i in range(21):
                    x = hand.landmark[i].x
                    y = hand.landmark[i].y
                    z = hand.landmark[i].z
                    if handed == "Left":
                        x = 1.0 - x
                    pts_2d.append((x, y))
                    vec.extend([x, y, z])
                vec = np.asarray(vec, dtype=np.float32)  # (63,)
                buf.append(vec)

                # draw hand
                drawer.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS,
                                      styles.get_default_hand_landmarks_style(),
                                      styles.get_default_hand_connections_style())

                # ------------ screenshot sequence gating (open -> fist) ------------
                num_up = count_fingers_from_pts2d(pts_2d)
                now = time.time()
                # detect open
                if num_up >= OPEN_MIN_FINGERS:
                    had_open_recently = True
                    last_open_time = now
                # detect fist
                is_fist_now = (num_up <= FIST_MAX_FINGERS)
                open_to_fist_ok = had_open_recently and is_fist_now and (now - last_open_time <= SCREENSHOT_WINDOW)
                # expire window
                if had_open_recently and (now - last_open_time > SCREENSHOT_WINDOW):
                    had_open_recently = False

                # -------------------- inference when buffer full --------------------
                if len(buf) == T:
                    X = np.stack(list(buf))[None, ...]  # (1, T, 63)
                    Xn = (X - Xmean) / Xstd
                    prob = model.predict(Xn, verbose=0)[0]  # (C,)

                    # --- ทำนาย gesture ---
                    k = int(prob.argmax())
                    label = classes[k]
                    conf = float(prob[k])
                    label_show, conf_show = label, conf
                    top3 = topk(prob, classes, k=3)

                    # --- ยืนยันทิศทางจริง (เฉพาะแนวนอน) ---
                    if label in ("scroll_left", "scroll_right", "tab_left", "tab_right", "desktop_left", "desktop_right"):
                        def infer_lr_direction(buf, take_frac=0.35, eps=0.02):
                            # ใช้ "ศูนย์กลางมือ" แทนปลายนิ้วชี้อย่างเดียว → ทนกว่า
                            idx_candidates = [0, 5, 9, 13, 17, 1, 2, 3, 4]  # wrist + MCPs + โหนดนิ้วโป้ง
                            xs = []
                            for f in buf:
                                xs.append(np.mean([f[i*3 + 0] for i in idx_candidates]))
                            if not xs:
                                return None
                            k = max(1, int(len(xs)*take_frac))
                            x0 = sum(xs[:k])/k
                            x1 = sum(xs[-k:])/k
                            dx = x1 - x0
                            if dx > eps:  return "right"
                            if dx < -eps: return "left"
                            return None

                        dir_fix = infer_lr_direction(buf)
                        if dir_fix is not None:
                            base = "tab" if "tab" in label else ("desktop" if "desktop" in label else "scroll")
                            label = f"{base}_{dir_fix}"

                    # --- Context-aware remap (ให้ tab_* มี priority เมื่ออยู่ในเบราว์เซอร์) ---
                    if conf >= thr.get(label, 0.8):
                        if label in ("desktop_left", "desktop_right") and is_browser_active():
                            # กำลังอยู่ใน browser → แทนที่จะ Alt+Tab ให้เปลี่ยนเป็นเปลี่ยนแท็บ
                            label = "tab_right" if label.endswith("right") else "tab_left"
                        elif label in ("tab_left", "tab_right") and not is_browser_active():
                            # ไม่ได้อยู่ใน browser → สลับโปรแกรมแทน
                            label = "desktop_right" if label.endswith("right") else "desktop_left"

                    # --- ทำ action ---
                    if label != "idle" and conf >= thr.get(label, 0.8):
                        if label == "screenshot" and SCREENSHOT_REQUIRE_SEQUENCE:
                            if open_to_fist_ok:
                                if now - last_time[label] >= cd.get(label, 0.6):
                                    do_action(label)
                                    last_time[label] = now
                                    had_open_recently = False
                        else:
                            if now - last_time[label] >= cd.get(label, 0.6):
                                do_action(label)
                                last_time[label] = now


            # ================= Overlay =================
            cv2.putText(frame, f"{label_show} : {conf_show:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if label_show!="…" else (200,200,200), 2)

            if DRAW_PROB_BAR and top3:
                base_y = 60
                for i, (lab, p) in enumerate(top3):
                    bar_w = int(300 * p)
                    cv2.putText(frame, f"{lab:<14s} {p:0.2f}", (10, base_y + i*28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                    cv2.rectangle(frame, (170, base_y - 12 + i*28),
                                  (170 + bar_w, base_y - 12 + i*28 + 18), (0,180,255), -1)
                    cv2.rectangle(frame, (170, base_y - 12 + i*28),
                                  (170 + 300, base_y - 12 + i*28 + 18), (255,255,255), 1)

            cv2.putText(frame, "q: quit", (10, h-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 1)

            cv2.imshow("Realtime Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
