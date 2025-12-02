# Real-time gesture inference (MediaPipe Hands → LSTM → pyautogui actions)

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

MODEL_DIR  = "models/gesture_lstm"
MODEL_PATH = "models/gesture_lstm/best.keras"
NORM_PATH  = "models/gesture_norm.npz"

T_DEFAULT = 30
F_DEFAULT = 63

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
    "idle": 1.10,
}

COOLDOWN = {
    "desktop_left": 0.4,
    "desktop_right": 0.4,
    "tab_left": 0.7,
    "tab_right": 0.7,
    "screenshot": 1.2,
    "scroll_up": 0.0,
    "scroll_down": 0.0,
    "scroll_left": 0.0,
    "scroll_right": 0.0,
    "idle": 0.0,
}

SCROLL_STEP_Y   = 600
HSCROLL_STEP_X  = 80
DRAW_PROB_BAR   = True

USE_SHIFT_FOR_HSCROLL = True
HSCROLL_WHEEL_FACTOR  = 10

def hscroll_signed(amount):
    if USE_SHIFT_FOR_HSCROLL:
        delta = int(amount) * HSCROLL_WHEEL_FACTOR
        print(f"[HSCROLL] amount={amount}, delta={delta}")
        pyautogui.keyDown('shift')
        pyautogui.scroll(delta)
        pyautogui.keyUp('shift')
    else:
        try:
            print(f"[HSCROLL] amount={amount} via hscroll()")
            pyautogui.hscroll(int(amount))
        except Exception:
            direction_key = 'right' if amount > 0 else 'left'
            print(f"[HSCROLL] fallback key: {direction_key}")
            pyautogui.press(direction_key)

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

MIN_DET, MIN_TRK = 0.6, 0.6

OS = platform.system().lower()

SCREENSHOT_REQUIRE_SEQUENCE = True 
SCREENSHOT_WINDOW          = 1.0 
OPEN_MIN_FINGERS           = 4
FIST_MAX_FINGERS           = 1 

DIR_TAIL_FRAC = 0.28 
DIR_MIN_DX    = 0.06
DIR_DEBOUNCE_SEC = 0.45

OPEN_GATE_CLASSES = {"desktop_left","desktop_right","tab_left","tab_right"}

# ================ Actions Mapping ==================
def do_action(label: str):
    print(f"[ACTION] {label} at {time.strftime('%H:%M:%S')}")

    if label == "desktop_left":
        if OS == "windows":
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
        hscroll_signed(+1)

    elif label == "scroll_right":
        hscroll_signed(-1)

    elif label == "screenshot":
        if OS == "windows":
            pyautogui.hotkey("winleft", "printscreen")
        elif OS == "darwin":
            pyautogui.hotkey("command", "shift", "3")
        else:
            pyautogui.press("printscreen")

# ================== Utilities =====================
def _angle_between(v1, v2):
    num = float(np.dot(v1, v2))
    den = float(np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
    c = np.clip(num / den, -1.0, 1.0)
    return np.degrees(np.arccos(c))

def _is_extended(lm2d, tip, pip_, mcp):
    v1 = np.array([lm2d[tip][0]-lm2d[pip_][0], lm2d[tip][1]-lm2d[pip_][1]], dtype=np.float32)
    v2 = np.array([lm2d[mcp][0]-lm2d[pip_][0], lm2d[mcp][1]-lm2d[pip_][1]], dtype=np.float32)
    return _angle_between(v1, v2) > 175.0

def count_fingers_from_pts2d(lm2d):
    return int(sum([
        _is_extended(lm2d, 4, 3, 2),
        _is_extended(lm2d, 8, 6, 5),
        _is_extended(lm2d,12,10,9),
        _is_extended(lm2d,16,14,13),
        _is_extended(lm2d,20,18,17),
    ]))

def load_model_and_norm():
    candidates = [
        os.path.join(MODEL_DIR, "best.keras"),
        os.path.join(MODEL_DIR, "final.keras"),
        MODEL_DIR,
    ]
    model = None
    last_err = None
    for path in candidates:
        try:
            if os.path.isdir(path):
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

    last_open_time = 0.0
    had_open_recently = False

    last_lr = None
    last_lr_ts = 0.0

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

            final_label_for_display = label_show

            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]

                handed = None
                if res.multi_handedness:
                    handed = res.multi_handedness[0].classification[0].label

                pts_2d = []
                vec = []
                for i in range(21):
                    x = hand.landmark[i].x
                    y = hand.landmark[i].y
                    z = hand.landmark[i].z
                    pts_2d.append((x, y))
                    vec.extend([x, y, z])
                vec = np.asarray(vec, dtype=np.float32)
                buf.append(vec)

                drawer.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS,
                                      styles.get_default_hand_landmarks_style(),
                                      styles.get_default_hand_connections_style())

                num_up = count_fingers_from_pts2d(pts_2d)
                cv2.putText(frame, f"num_up: {num_up}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

                now = time.time()
                if num_up >= OPEN_MIN_FINGERS:
                    had_open_recently = True
                    last_open_time = now
                is_fist_now = (num_up <= FIST_MAX_FINGERS)
                open_to_fist_ok = had_open_recently and is_fist_now and (now - last_open_time <= SCREENSHOT_WINDOW)
                if had_open_recently and (now - last_open_time > SCREENSHOT_WINDOW):
                    had_open_recently = False

                if len(buf) == T:
                    X = np.stack(list(buf))[None, ...]
                    Xn = (X - Xmean) / Xstd
                    prob = model.predict(Xn, verbose=0)[0]

                    k = int(prob.argmax())
                    label = classes[k]
                    conf = float(prob[k])
                    label_show, conf_show = label, conf
                    top3 = topk(prob, classes, k=3)

                    if conf >= 0.40:
                        print(
                            f"[PRED] raw_label={label_show} conf={conf_show:.2f} "
                            f"top3={[(l, round(p,2)) for (l,p) in top3]} "
                            f"handed={handed} num_up={num_up}"
                        )

                    # ===== Direction fix for LR gestures =====
                    if label in ("scroll_left", "scroll_right", "tab_left", "tab_right", "desktop_left", "desktop_right"):
                        def infer_lr_direction(buf_, take_frac=DIR_TAIL_FRAC, eps=DIR_MIN_DX):
                            idx_candidates = [0, 5, 9, 13, 17, 1, 2, 3, 4]
                            xs = []
                            for f in buf_:
                                xs.append(np.mean([f[i*3 + 0] for i in idx_candidates]))
                            if not xs:
                                return None, 0.0, 0.0
                            k_tail = max(1, int(len(xs) * take_frac))
                            x0 = sum(xs[:k_tail]) / k_tail
                            x1 = sum(xs[-k_tail:]) / k_tail
                            dx = x1 - x0
                            if dx > eps:
                                return "right", x0, x1
                            if dx < -eps:
                                return "left", x0, x1
                            return None, x0, x1

                        dir_fix, x0_debug, x1_debug = infer_lr_direction(buf)
                        print(
                            f"[DIR_DEBUG] handed={handed} raw_label={label} "
                            f"x_start={x0_debug:.3f} x_end={x1_debug:.3f}"
                        )

                        if dir_fix is not None:
                            base = "tab" if "tab" in label else ("desktop" if "desktop" in label else "scroll")
                            now_dir = dir_fix
                            accept_dir = True
                            if last_lr is not None and now_dir != last_lr and (now - last_lr_ts) < DIR_DEBOUNCE_SEC:
                                accept_dir = False
                            if accept_dir:
                                label = f"{base}_{now_dir}"
                                if now_dir != last_lr:
                                    last_lr = now_dir
                                    last_lr_ts = now
                            else:
                                label = f"{base}_{last_lr if last_lr is not None else now_dir}"
                            print(f"[DIR_APPLY] handed={handed} dir_fix={dir_fix} final_label={label}")
                        else:
                            print(f"[DIR_DEBUG] handed={handed} dx too small → keep label={label}")

                    # Gate open-palm
                    if num_up >= OPEN_MIN_FINGERS and label in OPEN_GATE_CLASSES:
                        label = "idle"
                        cv2.putText(frame, "[GATED: OPEN HAND]", (10, 52),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
                        print(f"[GATE] open hand → force idle (handed={handed}, num_up={num_up})")

                    final_label_for_display = label

                    if label != "idle":
                        print(f"[FINAL] label={label} conf={conf:.2f} handed={handed} num_up={num_up}")

                    if label != "idle" and conf >= thr.get(label, 0.8):
                        if label == "screenshot" and SCREENSHOT_REQUIRE_SEQUENCE:
                            if open_to_fist_ok and (now - last_time[label] >= cd.get(label, 0.6)):
                                do_action(label)
                                last_time[label] = now
                                had_open_recently = False
                            else:
                                print(
                                    f"[SKIP] screenshot: open_to_fist_ok={open_to_fist_ok}, "
                                    f"cooldown={now-last_time[label]:.2f}/{cd.get(label,0.6):.2f}"
                                )
                        else:
                            if now - last_time[label] >= cd.get(label, 0.6):
                                do_action(label)
                                last_time[label] = now
                            else:
                                print(
                                    f"[SKIP] {label}: cooldown {now-last_time[label]:.2f} < "
                                    f"{cd.get(label,0.6):.2f}"
                                )

            # ================= Overlay =================
            cv2.putText(frame, f"{final_label_for_display} : {conf_show:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0,255,0) if final_label_for_display!="…" else (200,200,200),
                        2)

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
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
