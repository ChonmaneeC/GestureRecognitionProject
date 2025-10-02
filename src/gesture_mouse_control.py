# Gesture-based Mouse & System Control (Rule-based, stable)

import cv2
import time
import platform
import numpy as np
import mediapipe as mp
import pyautogui
from collections import defaultdict

# ========================= Config =========================
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

CURSOR_SMOOTHING = 0.35
CURSOR_CLAMP_MARGIN = 0.02

DOUBLE_CLICK_WINDOW = 0.35
PINCH_HYSTERESIS = 0.015

SCROLL_SCALE = 120
SCROLL_ACCUM_TRIGGER = 50
SCROLL_Y_THRESH = 0.006
PAN_X_THRESH = 0.018

SWIPE_X_THRESH = 0.055

DESKTOP_SWIPE_X = 0.06

COOLDOWN_SEC = 0.75

MIN_DET = 0.55
MIN_TRK = 0.55

SHOW_OVERLAY = True

# ====================== OS Hotkeys ========================
OS = platform.system().lower()

def switch_tab_right():
    pyautogui.hotkey('ctrl', 'tab')

def switch_tab_left():
    pyautogui.hotkey('ctrl', 'shift', 'tab')

def switch_desktop_right():
    if OS == 'windows': pyautogui.hotkey('ctrl', 'winleft', 'right')
    elif OS == 'darwin': pyautogui.hotkey('ctrl', 'right')
    else: pyautogui.hotkey('ctrl', 'alt', 'right')

def switch_desktop_left():
    if OS == 'windows': pyautogui.hotkey('ctrl', 'winleft', 'left')
    elif OS == 'darwin': pyautogui.hotkey('ctrl', 'left')
    else: pyautogui.hotkey('ctrl', 'alt', 'left')

# ==================== Geometry Helpers ===================
def angle_between(v1, v2):
    num = np.dot(v1, v2)
    den = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
    c = np.clip(num / den, -1.0, 1.0)
    return np.degrees(np.arccos(c))

def is_extended(lm2d, tip, pip_, mcp):
    """Finger extensor angle (2D)"""
    v1 = np.array([lm2d[tip][0]-lm2d[pip_][0], lm2d[tip][1]-lm2d[pip_][1]])
    v2 = np.array([lm2d[mcp][0]-lm2d[pip_][0], lm2d[mcp][1]-lm2d[pip_][1]])
    ang = angle_between(v1, v2)
    return ang > 160

def count_fingers(lm2d):
    fingers = {
        'thumb':  is_extended(lm2d, 4, 3, 2),
        'index':  is_extended(lm2d, 8, 6, 5),
        'middle': is_extended(lm2d, 12,10,9),
        'ring':   is_extended(lm2d, 16,14,13),
        'pinky':  is_extended(lm2d, 20,18,17),
    }
    
    return fingers

def tip_point(lm2d, idx=8):
    return np.array([lm2d[idx][0], lm2d[idx][1]], dtype=np.float32)

def pinch_distance(lm2d, a=8, b=4):
    """Index finger-thumb distance (normalized 0..1)"""
    p1 = np.array(lm2d[a]); p2 = np.array(lm2d[b])
    return float(np.linalg.norm(p1 - p2))

# ======================= Main ============================
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

screen_w, screen_h = pyautogui.size()

# States
last_action_time = 0.0
last_click_time = 0.0
dragging = False
smooth_cursor = None
prev_tip = None

# wrist swipe
prev_wrist_x = None
smooth_wrist_x = None

# accumulators
scroll_accum = defaultdict(float)

with mp_hands.Hands(model_complexity=0,
                    max_num_hands=1,
                    min_detection_confidence=MIN_DET,
                    min_tracking_confidence=MIN_TRK) as hands:

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        h, w = frame.shape[:2]
        action_text = ""

        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            lm2d = [(lm.x, lm.y) for lm in hand.landmark]
            mp.solutions.drawing_utils.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            fingers = count_fingers(lm2d)
            num_up = sum(fingers.values())

            # ====== 1 finger = Mouse (move / click / right / drag) ======
            if num_up == 1 and fingers['index'] and not any([fingers[k] for k in ['middle','ring','pinky']]):
                cur = tip_point(lm2d, 8)
                if smooth_cursor is None:
                    smooth_cursor = cur.copy()
                else:
                    smooth_cursor = CURSOR_SMOOTHING*smooth_cursor + (1-CURSOR_SMOOTHING)*cur

                # clamp + map to screen
                cx = float(np.clip(smooth_cursor[0], CURSOR_CLAMP_MARGIN, 1-CURSOR_CLAMP_MARGIN))
                cy = float(np.clip(smooth_cursor[1], CURSOR_CLAMP_MARGIN, 1-CURSOR_CLAMP_MARGIN))
                sx, sy = int(cx*screen_w), int(cy*screen_h)
                pyautogui.moveTo(sx, sy)
                action_text = f"Mouse move ({sx},{sy})"

                # Click detection:
                # Left click: pinch index+thumb
                dist = pinch_distance(lm2d, 8, 4)
                pinch_pressed = dist < PINCH_HYSTERESIS
                if pinch_pressed and not dragging:
                    now = time.time()
                    if now - last_click_time < DOUBLE_CLICK_WINDOW:
                        pyautogui.doubleClick(); action_text = "Double Click"
                    else:
                        pyautogui.click(); action_text = "Left Click"
                    last_click_time = now

                # Right click: index + ring
                if fingers['index'] and fingers['ring']:
                    pyautogui.rightClick(); action_text = "Right Click"
                    time.sleep(0.15)  # กันรัวเกินไป

            # ====== 2 finger = Scroll / Pan ======
            elif fingers['index'] and fingers['middle'] and num_up == 2:
                cur = tip_point(lm2d, 8)
                if prev_tip is None:
                    prev_tip = cur.copy()
                delta = cur - prev_tip
                prev_tip = cur.copy()

                # accumulate scroll (Y)
                scroll_accum[0] += -delta[1] * SCROLL_SCALE * 100
                if abs(scroll_accum[0]) > SCROLL_ACCUM_TRIGGER:
                    amt = int(scroll_accum[0])
                    pyautogui.scroll(amt)
                    action_text = f"Scroll {amt}"
                    scroll_accum[0] = 0

                if abs(delta[0]) > PAN_X_THRESH and (time.time()-last_action_time) > COOLDOWN_SEC:
                    if delta[0] > 0:
                        pyautogui.press('right'); action_text = "Pan Right"
                    else:
                        pyautogui.press('left'); action_text = "Pan Left"
                    last_action_time = time.time()

            # ====== 3 finger = Switch Virtual Desktop ======
            elif fingers['index'] and fingers['middle'] and fingers['ring'] and num_up == 3:
                dx = lm2d[8][0] - lm2d[5][0]
                if abs(dx) > DESKTOP_SWIPE_X and (time.time()-last_action_time) > COOLDOWN_SEC:
                    if dx > 0:
                        switch_desktop_right(); action_text = "Desktop Right"
                    else:
                        switch_desktop_left(); action_text = "Desktop Left"
                    last_action_time = time.time()

            # ====== Swipe with both hands = Switch Tab ======
            else:
                wx = lm2d[0][0]
                if smooth_wrist_x is None:
                    smooth_wrist_x = wx; prev_wrist_x = wx
                else:
                    smooth_wrist_x = 0.5*smooth_wrist_x + 0.5*wx

                delta_x = smooth_wrist_x - (prev_wrist_x if prev_wrist_x is not None else smooth_wrist_x)
                prev_wrist_x = smooth_wrist_x

                if abs(delta_x) > SWIPE_X_THRESH and (time.time()-last_action_time) > COOLDOWN_SEC:
                    if delta_x > 0:
                        switch_tab_right(); action_text = "Tab Right"
                    else:
                        switch_tab_left(); action_text = "Tab Left"
                    last_action_time = time.time()

            if SHOW_OVERLAY:
                cv2.putText(frame, action_text, (10, h-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                cv2.putText(frame, f"Fingers: {num_up}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        else:
            prev_tip = None
            smooth_cursor = None
            smooth_wrist_x = None
            prev_wrist_x = None

        cv2.imshow("Gesture Mouse Control (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
