# src/gesture_mouse_control.py
# Rule-based mouse & click control (with colored overlay + debug)
# Left click:  Thumb + Index
# Right click: Thumb + Ring

import time
import platform
from collections import defaultdict

import cv2
import numpy as np
import mediapipe as mp
import pyautogui

# ============== Config ==============
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

SHOW_OVERLAY = True

CURSOR_SMOOTHING = 0.35
CURSOR_CLAMP_MARGIN = 0.02

PINCH_DOWN = 0.040   # easier detection
PINCH_UP   = 0.055

DOUBLE_CLICK_WINDOW = 0.35
CLICK_COOLDOWN = 0.20
DRAG_HOLD_SEC = 0.25

MIN_DET = 0.55
MIN_TRK = 0.55

OPEN_PALM_EXTENDED_MIN = 4
FIST_EXTENDED_MAX = 0

OS = platform.system().lower()

FREEZE_AFTER_ACTION_SEC = 0.12   # แช่เคอร์เซอร์หลังคลิกสั้น ๆ
DEADZONE_PX = 10                 # ขยับเมาส์เฉพาะเมื่อเกินระยะนี้ (พิกเซล)


# ============== Helpers ==============
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

def tip_point(lm2d, idx=8):
    return np.array([lm2d[idx][0], lm2d[idx][1]], dtype=np.float32)

def pinch_distance(lm2d, a, b):
    p1 = np.array(lm2d[a]); p2 = np.array(lm2d[b])
    return float(np.linalg.norm(p1 - p2))


# ============== Main ==============
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    screen_w, screen_h = pyautogui.size()
    smooth_cursor = None
    last_action = ""

    left_pinch_down = False
    left_pinch_start_ts = 0.0
    dragging = False
    last_click_ts = 0.0
    last_right_click_ts = 0.0

    cursor_frozen_until = 0.0
    last_screen_pos = None


    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils

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
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                fingers = count_fingers(lm2d)
                num_up = int(sum(fingers.values()))

                # ===== Mouse move (with freeze + deadzone) =====
                cur = tip_point(lm2d, 8)
                if smooth_cursor is None:
                    smooth_cursor = cur.copy()
                else:
                    smooth_cursor = CURSOR_SMOOTHING*smooth_cursor + (1-CURSOR_SMOOTHING)*cur

                cx = float(np.clip(smooth_cursor[0], CURSOR_CLAMP_MARGIN, 1-CURSOR_CLAMP_MARGIN))
                cy = float(np.clip(smooth_cursor[1], CURSOR_CLAMP_MARGIN, 1-CURSOR_CLAMP_MARGIN))
                sx, sy = int(cx*screen_w), int(cy*screen_h)

                now = time.time()
                cursor_is_frozen = (now < cursor_frozen_until) or (left_pinch_down and not dragging)

                if not cursor_is_frozen:
                    if last_screen_pos is None:
                        last_screen_pos = (sx, sy)
                        pyautogui.moveTo(sx, sy)
                    else:
                        dx = sx - last_screen_pos[0]
                        dy = sy - last_screen_pos[1]
                        if (dx*dx + dy*dy) ** 0.5 >= DEADZONE_PX:
                            pyautogui.moveTo(sx, sy)
                            last_screen_pos = (sx, sy)

                # ===== Pinch detection =====
                dist_left  = pinch_distance(lm2d, 8, 4)   # index + thumb
                dist_right = pinch_distance(lm2d,16, 4)   # ring + thumb
                now = time.time()
                allow_clicks = not (num_up >= OPEN_PALM_EXTENDED_MIN or num_up <= FIST_EXTENDED_MAX)

                # Overlay lines (debug colors)
                x_thumb, y_thumb = int(lm2d[4][0]*w), int(lm2d[4][1]*h)
                x_index, y_index = int(lm2d[8][0]*w), int(lm2d[8][1]*h)
                x_ring, y_ring   = int(lm2d[16][0]*w), int(lm2d[16][1]*h)

                if dist_left < PINCH_UP:
                    cv2.line(frame, (x_thumb, y_thumb), (x_index, y_index), (0,255,0), 5)
                if dist_right < PINCH_UP:
                    cv2.line(frame, (x_thumb, y_thumb), (x_ring, y_ring), (255,200,0), 5)

                # ===== Left click / double click / drag =====
                if allow_clicks:
                    if not left_pinch_down and dist_left < PINCH_DOWN:
                        print(f"[DEBUG] Left pinch detected: {dist_left:.3f}")
                        left_pinch_down = True
                        left_pinch_start_ts = now

                    elif left_pinch_down and dist_left >= PINCH_UP:
                        held = now - left_pinch_start_ts
                        if dragging:
                            pyautogui.mouseUp(button='left')
                            dragging = False
                            action_text = "[DRAG END]"
                        else:
                            if (now - last_click_ts) <= DOUBLE_CLICK_WINDOW:
                                pyautogui.doubleClick()
                                action_text = "[DOUBLE LEFT CLICK]"
                                print("[DEBUG] Double click triggered")
                                last_click_ts = 0.0
                                cursor_frozen_until = now + FREEZE_AFTER_ACTION_SEC
                            else:
                                pyautogui.click()
                                action_text = "[LEFT CLICK DETECTED]"
                                print("[DEBUG] Single left click triggered")
                                last_click_ts = now
                                cursor_frozen_until = now + FREEZE_AFTER_ACTION_SEC
                        left_pinch_down = False

                    if left_pinch_down and not dragging:
                        if (now - left_pinch_start_ts) >= DRAG_HOLD_SEC:
                            pyautogui.mouseDown(button='left')
                            dragging = True
                            action_text = "[DRAG START]"
                            print("[DEBUG] Drag started")
                            cursor_frozen_until = 0.0

                # ===== Right click =====
                if allow_clicks and (now - last_right_click_ts) > CLICK_COOLDOWN:
                    if dist_right < PINCH_DOWN:
                        pyautogui.rightClick()
                        action_text = "[RIGHT CLICK DETECTED]"
                        print(f"[DEBUG] Right pinch detected: {dist_right:.3f}")
                        last_right_click_ts = now
                        cursor_frozen_until = now + FREEZE_AFTER_ACTION_SEC


                # ===== Overlay Info =====
                if SHOW_OVERLAY:
                    cv2.putText(frame, f"Fingers: {num_up}", (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    cv2.putText(frame, f"LeftPinch: {dist_left:.3f} | RightPinch: {dist_right:.3f}",
                                (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1)

            else:
                smooth_cursor = None
                last_screen_pos = None
                if dragging:
                    pyautogui.mouseUp(button='left')
                    dragging = False
                left_pinch_down = False

            if SHOW_OVERLAY:
                if action_text:
                    last_action = action_text
                cv2.putText(frame, last_action, (10, h-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow("Gesture Mouse (Overlay Debug) - q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
