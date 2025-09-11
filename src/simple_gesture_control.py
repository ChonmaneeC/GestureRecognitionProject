# Gesture-based Mouse + Control System

import cv2, time, platform
import numpy as np
import mediapipe as mp
import pyautogui

# ---------- Config ----------
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

SCROLL_SCALE = 120
SCROLL_THRESH_Y = 0.008
PAN_THRESH_X = 0.02
SWIPE_THRESH_X = 0.06
COOLDOWN_SEC = 0.8
SMOOTHING = 0.4

OS = platform.system().lower()

# ---------- OS-specific hotkeys ----------
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

# ---------- Helpers ----------
def angle_between(v1, v2):
    num = np.dot(v1, v2)
    den = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
    c = np.clip(num / den, -1.0, 1.0)
    return np.degrees(np.arccos(c))

def is_extended(lm, tip, pip_, mcp):
    v1 = np.array([lm[tip][0]-lm[pip_][0], lm[tip][1]-lm[pip_][1]])
    v2 = np.array([lm[mcp][0]-lm[pip_][0], lm[mcp][1]-lm[pip_][1]])
    ang = angle_between(v1, v2)
    return ang > 160

def count_fingers(lms):
    fingers = {
        'thumb':  False,
        'index':  is_extended(lms, 8, 6, 5),
        'middle': is_extended(lms, 12,10,9),
        'ring':   is_extended(lms, 16,14,13),
        'pinky':  is_extended(lms, 20,18,17),
    }
    if lms[4][0] < lms[3][0]:
        fingers['thumb'] = True
    return fingers

def tip_point(lms, idx=8):
    return np.array([lms[idx][0], lms[idx][1]], dtype=np.float32)

# ---------- Main loop ----------
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

last_action_time = 0.0
last_click_time = 0.0
dragging = False

screen_w, screen_h = pyautogui.size()

with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            lms = [(lm.x, lm.y) for lm in hand.landmark]
            mp.solutions.drawing_utils.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            fingers = count_fingers(lms)
            num_up = sum(fingers.values())
            cv2.putText(frame, f"Fingers: {num_up}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            now = time.time()
            ready = (now - last_action_time) > COOLDOWN_SEC

            # ---------- 1 นิ้ว = Mouse ----------
            if num_up == 1 and fingers['index']:
                x = int(lms[8][0] * screen_w)
                y = int(lms[8][1] * screen_h)
                pyautogui.moveTo(x, y)

                # Left click (index + thumb)
                if fingers['index'] and fingers['thumb']:
                    if now - last_click_time < 0.4:
                        pyautogui.doubleClick(); print("[MOUSE] Double Click")
                    else:
                        pyautogui.click(); print("[MOUSE] Click")
                    last_click_time = now

                # Right click (index + middle)
                if fingers['index'] and fingers['middle']:
                    pyautogui.rightClick(); print("[MOUSE] Right Click")

                # Drag (index + thumb ค้าง)
                if fingers['index'] and fingers['thumb'] and not dragging:
                    pyautogui.mouseDown(); dragging = True
                    print("[MOUSE] Drag Start")
                elif not (fingers['index'] and fingers['thumb']) and dragging:
                    pyautogui.mouseUp(); dragging = False
                    print("[MOUSE] Drag End")

                continue

            # ---------- 2 นิ้ว = Scroll/Pan ----------
            if fingers['index'] and fingers['middle'] and num_up == 2:
                tip = tip_point(lms, 8)
                # ใช้ delta Y เลื่อนจอ
                delta_y = tip[1] - lms[6][1]
                if abs(delta_y) > SCROLL_THRESH_Y:
                    amount = int(-delta_y * SCROLL_SCALE * 100)
                    pyautogui.scroll(amount); print("[ACTION] Scroll", amount)

                continue

            # ---------- 3 นิ้ว = Switch Desktop ----------
            if fingers['index'] and fingers['middle'] and fingers['ring'] and num_up == 3 and ready:
                delta_x = lms[8][0] - lms[5][0]
                if delta_x > SWIPE_THRESH_X:
                    switch_desktop_right(); print("[ACTION] Desktop Right")
                    last_action_time = now
                elif delta_x < -SWIPE_THRESH_X:
                    switch_desktop_left(); print("[ACTION] Desktop Left")
                    last_action_time = now
                continue

            # ---------- Swipe whole hand = Switch Tab ----------
            if ready:
                delta_x = lms[0][0] - lms[9][0]
                if delta_x > SWIPE_THRESH_X:
                    switch_tab_right(); print("[ACTION] Tab Right")
                    last_action_time = now
                elif delta_x < -SWIPE_THRESH_X:
                    switch_tab_left(); print("[ACTION] Tab Left")
                    last_action_time = now

        cv2.imshow("Gesture Control (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
