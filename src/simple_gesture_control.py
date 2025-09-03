# Prototype ควบคุมคอมจริง

import cv2, time, platform
import numpy as np
import mediapipe as mp
import pyautogui

# ---------- Config ----------
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

SCROLL_SCALE = 120          # ขนาดการเลื่อนแนวตั้งต่อการเคลื่อนนิ้ว
SWIPE_THRESH = 0.10         # ระยะสัดส่วนเฟรม (0-1 ของความกว้างภาพ) ที่นับเป็น "ปัด"
COOLDOWN_SEC = 0.7          # กันสั่งซ้ำเร็วเกิน
SMOOTHING = 0.6             # ค่าถ่วงให้ตำแหน่งนิ้วนุ่มนวลขึ้น [0..1]

# ---------- OS-specific hotkeys ----------
OS = platform.system().lower()

def switch_tab_right():
    if OS == 'darwin': pyautogui.hotkey('ctrl', 'tab')         # macOS บางแอปก็ใช้ได้
    else: pyautogui.hotkey('ctrl', 'tab')

def switch_tab_left():
    if OS == 'darwin': pyautogui.hotkey('ctrl', 'shift', 'tab')
    else: pyautogui.hotkey('ctrl', 'shift', 'tab')

def switch_desktop_right():
    if OS == 'windows': pyautogui.hotkey('ctrl', 'winleft', 'right')
    elif OS == 'darwin': pyautogui.hotkey('ctrl', 'right')
    else: pyautogui.hotkey('ctrl', 'alt', 'right')  # อาจต้องแมพคีย์ลัดบน Linux เอง

def switch_desktop_left():
    if OS == 'windows': pyautogui.hotkey('ctrl', 'winleft', 'left')
    elif OS == 'darwin': pyautogui.hotkey('ctrl', 'left')
    else: pyautogui.hotkey('ctrl', 'alt', 'left')

# ---------- Geometry helpers ----------
def angle_between(v1, v2):
    num = np.dot(v1, v2)
    den = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
    c = np.clip(num / den, -1.0, 1.0)
    return np.degrees(np.arccos(c))

def is_extended(lm, tip, pip_, mcp):
    v1 = np.array([lm[tip][0]-lm[pip_][0], lm[tip][1]-lm[pip_][1]])
    v2 = np.array([lm[mcp][0]-lm[pip_][0], lm[mcp][1]-lm[pip_][1]])
    ang = angle_between(v1, v2)
    return ang > 160  # นิ้วยืดจะเข้าใกล้ 180°

def count_fingers(lms):
    # lms: list[(x,y)] normalized 0..1
    # นับเฉพาะ index(8), middle(12), ring(16) เป็นฐาน 1-3 นิ้ว
    fingers = {
        'index':  is_extended(lms, 8, 6, 5),
        'middle': is_extended(lms, 12,10,9),
        'ring':   is_extended(lms, 16,14,13),
        # pinky/ thumb ไว้เพิ่มทีหลังตามโจทย์
    }
    return fingers

def tip_point(lms, idx=8):
    return np.array([lms[idx][0], lms[idx][1]], dtype=np.float32)

# ---------- Main loop ----------
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

last_action_time = 0
prev_pos = None
smooth_pos = None

with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    while True:
        ok, frame = cap.read()
        if not ok: break
        h, w = frame.shape[:2]
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            lms = [(lm.x, lm.y) for lm in hand.landmark]

            fingers = count_fingers(lms)
            num_up = sum(fingers.values())

            # ตำแหน่งนิ้วชี้ (index tip = 8)
            pos = tip_point(lms, 8)
            if smooth_pos is None:
                smooth_pos = pos.copy()
            else:
                smooth_pos = SMOOTHING*smooth_pos + (1-SMOOTHING)*pos

            if prev_pos is None:
                prev_pos = smooth_pos.copy()

            delta = smooth_pos - prev_pos
            prev_pos = smooth_pos.copy()

            # วาด UI
            import mediapipe as mp
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, f"Fingers up: {num_up}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            now = time.time()
            ready = (now - last_action_time) > COOLDOWN_SEC

            # ----- Mapping -----
            # 1 นิ้ว: scroll / pan
            if num_up == 1:
                # เลื่อนตามแกน Y (ลงบวก)
                if abs(delta[1]) > 0.01:  # ยกนิ้วขึ้นลงชัดเจน
                    amount = int(-delta[1] * SCROLL_SCALE * 100)  # ปรับความไว
                    if amount != 0:
                        pyautogui.scroll(amount)

                # ซ้าย/ขวา: pan ด้วยปุ่มลูกศร
                if abs(delta[0]) > 0.03 and ready:
                    if delta[0] > 0:
                        pyautogui.press('right')
                    else:
                        pyautogui.press('left')
                    last_action_time = now

            # 2 นิ้ว (index+middle): สลับแท็บ
            elif num_up == 2 and ready:
                if delta[0] > SWIPE_THRESH:
                    switch_tab_right(); last_action_time = now
                elif delta[0] < -SWIPE_THRESH:
                    switch_tab_left(); last_action_time = now

            # 3 นิ้ว (index+middle+ring): สลับ Virtual Desktop
            elif num_up == 3 and ready:
                if delta[0] > SWIPE_THRESH:
                    switch_desktop_right(); last_action_time = now
                elif delta[0] < -SWIPE_THRESH:
                    switch_desktop_left(); last_action_time = now

        cv2.imshow("Simple Gesture Control (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
