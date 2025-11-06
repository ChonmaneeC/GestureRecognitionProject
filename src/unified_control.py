# src/unified_control.py
# Run Rule-based mouse control + LSTM actions together from ONE webcam/MediaPipe stream

import os, time, platform
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pyautogui

# ===================== Common Config =====================
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
OS = platform.system().lower()

# ====== Model paths (same as realtime_inference.py) ======
MODEL_DIR  = "models/gesture_lstm"
MODEL_PATH = "models/gesture_lstm/best.keras"
NORM_PATH  = "models/gesture_norm.npz"
T_DEFAULT, F_DEFAULT = 30, 63

# ====== LSTM thresholds & cooldowns ======
THRESH = {
    "desktop_left": 0.75, "desktop_right": 0.75,
    "tab_left": 0.60, "tab_right": 0.60,
    "scroll_up": 0.70, "scroll_down": 0.70,
    "scroll_left": 0.70, "scroll_right": 0.70,
    "screenshot": 0.80, "idle": 1.10,
}
COOLDOWN = {
    "desktop_left": 0.4, "desktop_right": 0.4,
    "tab_left": 0.7, "tab_right": 0.7,
    "screenshot": 1.2,
    "scroll_up": 0.0, "scroll_down": 0.0,
    "scroll_left": 0.0, "scroll_right": 0.0, "idle": 0.0,
}
SCROLL_STEP_Y = 600
USE_SHIFT_FOR_HSCROLL = True
HSCROLL_WHEEL_FACTOR = 10

# ---- Direction stabilization (debounce) ----
DIR_TAIL_FRAC    = 0.28
DIR_MIN_DX       = 0.040
DIR_DEBOUNCE_SEC = 0.25

# ---- Screenshot gating (open -> fist) ----
SCREENSHOT_REQUIRE_SEQUENCE = True
SCREENSHOT_WINDOW = 1.0
OPEN_MIN_FINGERS  = 4
FIST_MAX_FINGERS  = 1

# ====== Rule-based mouse control config ======
SHOW_OVERLAY = True
CURSOR_SMOOTHING = 0.35
CURSOR_CLAMP_MARGIN = 0.02
PINCH_DOWN = 0.040
PINCH_UP   = 0.055
DOUBLE_CLICK_WINDOW = 0.35
CLICK_COOLDOWN = 0.20
DRAG_HOLD_SEC = 0.25
MIN_DET, MIN_TRK = 0.6, 0.6

OPEN_PALM_EXTENDED_MIN = 4
FIST_EXTENDED_MAX = 0

# Anti-jitter add-on (freeze + deadzone)
FREEZE_AFTER_ACTION_SEC = 0.12
DEADZONE_PX = 10

# ===================== Helpers =====================
def _angle_between(v1, v2):
    num = float(np.dot(v1, v2))
    den = float(np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
    c = np.clip(num / den, -1.0, 1.0)
    return np.degrees(np.arccos(c))

def _is_extended(lm2d, tip, pip_, mcp):
    v1 = np.array([lm2d[tip][0]-lm2d[pip_][0], lm2d[tip][1]-lm2d[pip_][1]], dtype=np.float32)
    v2 = np.array([lm2d[mcp][0]-lm2d[pip_][0], lm2d[mcp][1]-lm2d[pip_][1]], dtype=np.float32)
    return _angle_between(v1, v2) > 160.0

def count_fingers_from_pts2d(lm2d):
    return int(sum([
        _is_extended(lm2d, 4, 3, 2),
        _is_extended(lm2d, 8, 6, 5),
        _is_extended(lm2d,12,10, 9),
        _is_extended(lm2d,16,14,13),
        _is_extended(lm2d,20,18,17),
    ]))

def hscroll_signed(amount):
    if USE_SHIFT_FOR_HSCROLL:
        pyautogui.keyDown('shift')
        pyautogui.scroll(int(amount) * HSCROLL_WHEEL_FACTOR)
        pyautogui.keyUp('shift')
    else:
        try:
            pyautogui.hscroll(int(amount))
        except Exception:
            pyautogui.press('right' if amount > 0 else 'left')

BROWSER_APP_KEYWORDS = ["chrome", "microsoft edge", "edge", "firefox", "brave", "opera", "arc"]
def is_browser_active():
    try:
        win = pyautogui.getActiveWindow()
        if not win: return False
        title = (win.title or "")
        return any(k in title.lower() for k in BROWSER_APP_KEYWORDS)
    except Exception:
        return False

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
        raise RuntimeError(f"Cannot load model from any of: {candidates}\nLast error: {last_err}")
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

# ===================== Actions =====================
def do_action(label: str):
    if label == "desktop_left":
        if OS == "windows":
            pyautogui.keyDown("alt"); pyautogui.keyDown("shift"); pyautogui.press("tab"); pyautogui.keyUp("shift"); pyautogui.keyUp("alt")
        elif OS == "darwin":
            pyautogui.hotkey("command", "shift", "tab")
        else:
            pyautogui.hotkey("alt", "shift", "tab")
    elif label == "desktop_right":
        if OS == "windows":
            pyautogui.keyDown("alt"); pyautogui.press("tab"); pyautogui.keyUp("alt")
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
        hscroll_signed(-1)
    elif label == "scroll_right":
        hscroll_signed(+1)
    elif label == "screenshot":
        if OS == "windows":
            pyautogui.hotkey("winleft", "printscreen")
        elif OS == "darwin":
            pyautogui.hotkey("command", "shift", "3")
        else:
            pyautogui.press("printscreen")

# ===================== Main =====================
def main():
    # ---- Load model/norm for LSTM ----
    model, Xmean, Xstd, classes, T, F = load_model_and_norm()
    thr = {c: THRESH.get(c, 0.8) for c in classes}
    cd  = {c: COOLDOWN.get(c, 0.6) for c in classes}
    last_time = {c: 0.0 for c in classes}

    # ---- Shared camera/mediapipe ----
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    mp_hands = mp.solutions.hands
    drawer   = mp.solutions.drawing_utils
    styles   = mp.solutions.drawing_styles

    # ---- Buffers/state for LSTM ----
    buf = deque(maxlen=T)
    last_lr_dir, last_lr_ts = None, 0.0
    last_open_time, had_open_recently = 0.0, False

    # ---- States for Rule-based mouse ----
    screen_w, screen_h = pyautogui.size()
    smooth_cursor = None
    last_action_text = ""
    left_pinch_down = False
    left_pinch_start_ts = 0.0
    dragging = False
    last_click_ts = 0.0
    last_right_click_ts = 0.0
    cursor_frozen_until = 0.0
    last_screen_pos = None

    # ---- Mode toggle ----
    use_rule = True
    use_lstm = True
    print("[INFO] Unified control running. Keys: 1=toggle Rule, 2=toggle LSTM, p=pause/resume, q=quit")

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=MIN_DET,
        min_tracking_confidence=MIN_TRK,
        model_complexity=0
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            now = time.time()
            label_show, conf_show = "…", 0.0
            top3 = []
            num_up = 0
            handed = None
            pts_2d = None
            vec63 = None

            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                if res.multi_handedness:
                    handed = res.multi_handedness[0].classification[0].label

                pts_2d, vec = [], []
                for i in range(21):
                    x = hand.landmark[i].x
                    y = hand.landmark[i].y
                    z = hand.landmark[i].z
                    if handed == "Left":
                        x = 1.0 - x
                    pts_2d.append((x, y))
                    vec.extend([x, y, z])
                vec63 = np.asarray(vec, dtype=np.float32)
                drawer.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS,
                                      styles.get_default_hand_landmarks_style(),
                                      styles.get_default_hand_connections_style())

                # --------- Count fingers (shared) ---------
                num_up = count_fingers_from_pts2d(pts_2d)

                # ================= Rule-based (mouse) =================
                if use_rule:
                    # move (with freeze/deadzone)
                    def tip_point(idx=8):
                        return np.array([pts_2d[idx][0], pts_2d[idx][1]], dtype=np.float32)

                    cur = tip_point(8)
                    if smooth_cursor is None:
                        smooth_cursor = cur.copy()
                    else:
                        smooth_cursor = CURSOR_SMOOTHING*smooth_cursor + (1-CURSOR_SMOOTHING)*cur

                    cx = float(np.clip(smooth_cursor[0], CURSOR_CLAMP_MARGIN, 1-CURSOR_CLAMP_MARGIN))
                    cy = float(np.clip(smooth_cursor[1], CURSOR_CLAMP_MARGIN, 1-CURSOR_CLAMP_MARGIN))
                    sx, sy = int(cx*screen_w), int(cy*screen_h)

                    allow_clicks = not (num_up >= OPEN_PALM_EXTENDED_MIN or num_up <= FIST_EXTENDED_MAX)
                    dist_left  = float(np.linalg.norm(np.array(pts_2d[8]) - np.array(pts_2d[4])))
                    dist_right = float(np.linalg.norm(np.array(pts_2d[16]) - np.array(pts_2d[4])))

                    cursor_is_frozen = (now < cursor_frozen_until) or (left_pinch_down and not dragging)
                    if not cursor_is_frozen:
                        if last_screen_pos is None:
                            last_screen_pos = (sx, sy); pyautogui.moveTo(sx, sy)
                        else:
                            dx = sx - last_screen_pos[0]; dy = sy - last_screen_pos[1]
                            if (dx*dx + dy*dy) ** 0.5 >= DEADZONE_PX:
                                pyautogui.moveTo(sx, sy); last_screen_pos = (sx, sy)

                    # Left click / double click / drag
                    if allow_clicks:
                        if not left_pinch_down and dist_left < PINCH_DOWN:
                            left_pinch_down = True; left_pinch_start_ts = now
                        elif left_pinch_down and dist_left >= PINCH_UP:
                            held = now - left_pinch_start_ts
                            if dragging:
                                pyautogui.mouseUp(button='left')
                                dragging = False; last_action_text = "[DRAG END]"
                            else:
                                if (now - last_click_ts) <= DOUBLE_CLICK_WINDOW:
                                    pyautogui.doubleClick()
                                    last_action_text = "[DOUBLE LEFT CLICK]"
                                    last_click_ts = 0.0
                                else:
                                    pyautogui.click()
                                    last_action_text = "[LEFT CLICK]"
                                    last_click_ts = now
                                cursor_frozen_until = now + FREEZE_AFTER_ACTION_SEC
                            left_pinch_down = False

                        if left_pinch_down and not dragging:
                            if (now - left_pinch_start_ts) >= DRAG_HOLD_SEC:
                                pyautogui.mouseDown(button='left')
                                dragging = True; last_action_text = "[DRAG START]"
                                cursor_frozen_until = 0.0

                    # Right click
                    if allow_clicks and (now - last_right_click_ts) > CLICK_COOLDOWN:
                        if dist_right < PINCH_DOWN:
                            pyautogui.rightClick()
                            last_action_text = "[RIGHT CLICK]"
                            last_right_click_ts = now
                            cursor_frozen_until = now + FREEZE_AFTER_ACTION_SEC

                # ================= LSTM Inference =================
                # Pause LSTM when rule-based is likely interacting (pinch/drag) to avoid conflicts
                block_lstm = dragging or left_pinch_down
                if use_lstm and not block_lstm:
                    buf.append(vec63)
                    # open->fist gate
                    if num_up >= OPEN_MIN_FINGERS:
                        had_open_recently = True; last_open_time = now
                    is_fist_now = (num_up <= FIST_MAX_FINGERS)
                    open_to_fist_ok = had_open_recently and is_fist_now and (now - last_open_time <= SCREENSHOT_WINDOW)
                    if had_open_recently and (now - last_open_time > SCREENSHOT_WINDOW):
                        had_open_recently = False

                    if len(buf) == T:
                        X = np.stack(list(buf))[None, ...]
                        # norm
                        Xn = (X - Xmean) / Xstd
                        prob = model.predict(Xn, verbose=0)[0]
                        k = int(prob.argmax()); classes_list = classes
                        label = classes_list[k]; conf = float(prob[k])
                        label_show, conf_show = label, conf
                        # fix direction (tail + debounce) only for LR labels
                        if label in ("scroll_left","scroll_right","tab_left","tab_right","desktop_left","desktop_right"):
                            def stable_lr_direction(buf_, tail_frac=DIR_TAIL_FRAC, min_dx=DIR_MIN_DX):
                                idx_candidates = [0,5,9,13,17,1,2,3,4]
                                B = len(buf_)
                                if B < 3: return None, 0.0
                                k2 = max(3, int(B*tail_frac))
                                xs = []
                                for f in list(buf_)[-k2:]:
                                    xs.append(np.mean([f[i*3 + 0] for i in idx_candidates]))
                                if len(xs) < 3: return None, 0.0
                                mid = len(xs)//2
                                x0 = float(sum(xs[:mid]) / max(1, mid))
                                x1 = float(sum(xs[mid:]) / max(1, len(xs)-mid))
                                dx = x1 - x0
                                if dx >  min_dx: return "right", dx
                                if dx < -min_dx: return "left",  dx
                                return None, dx
                            dir_now, dx_val = stable_lr_direction(buf)
                            if dir_now is not None:
                                if (last_lr_dir is None) or (dir_now == last_lr_dir) or (now - last_lr_ts >= DIR_DEBOUNCE_SEC):
                                    if dir_now != last_lr_dir:
                                        last_lr_dir, last_lr_ts = dir_now, now
                                else:
                                    dir_now = last_lr_dir
                            if dir_now is not None:
                                base = "tab" if "tab" in label else ("desktop" if "desktop" in label else "scroll")
                                label = f"{base}_{dir_now}"
                        # context remap between desktop/tab based on active window
                        if conf >= thr.get(label, 0.8):
                            if label in ("desktop_left","desktop_right") and is_browser_active():
                                label = "tab_right" if label.endswith("right") else "tab_left"
                            elif label in ("tab_left","tab_right") and not is_browser_active():
                                label = "desktop_right" if label.endswith("right") else "desktop_left"
                        # fire action with cooldown
                        if label != "idle" and conf >= thr.get(label, 0.8):
                            if label == "screenshot" and SCREENSHOT_REQUIRE_SEQUENCE:
                                if open_to_fist_ok and (now - last_time[label] >= cd.get(label, 0.6)):
                                    do_action(label); last_time[label] = now; had_open_recently = False
                            else:
                                if now - last_time[label] >= cd.get(label, 0.6):
                                    do_action(label); last_time[label] = now

            else:
                # No hand: reset rule-based states
                smooth_cursor = None
                last_screen_pos = None
                if dragging:
                    pyautogui.mouseUp(button='left'); dragging = False
                left_pinch_down = False

            # ================= Overlay =================
            if SHOW_OVERLAY:
                cv2.putText(frame, f"LSTM: {label_show} {conf_show:.2f}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if label_show!="…" else (200,200,200), 2)
                cv2.putText(frame, f"Rule={'ON' if use_rule else 'OFF'}  LSTM={'ON' if use_lstm else 'OFF'}", (10, 54),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1)
                if last_action_text:
                    cv2.putText(frame, last_action_text, (10, h-12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow("Unified Control (Rule + LSTM) - 1:Rule 2:LSTM p:pause q:quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('1'): use_rule = not use_rule
            elif key == ord('2'): use_lstm = not use_lstm
            elif key == ord('p'):
                use_rule = False; use_lstm = False

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
