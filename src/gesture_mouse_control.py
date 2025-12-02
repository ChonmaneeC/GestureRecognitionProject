# Rule-based mouse & click control

import time
import platform
import cv2
import numpy as np
import mediapipe as mp
import pyautogui

# ============== Config ==============
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

SHOW_OVERLAY = True

CURSOR_SMOOTHING = 0.6
CURSOR_CLAMP_MARGIN = 0.02

PINCH_DOWN  = 0.045
PINCH_UP    = 0.052

CLICK_HOLD_MAX = 0.28
CLICK_COOLDOWN = 0.25
DRAG_HOLD_SEC  = 0.45

MIN_DET = 0.55
MIN_TRK = 0.55

OPEN_PALM_EXTENDED_MIN = 4
FIST_EXTENDED_MAX = 0

FREEZE_AFTER_ACTION_SEC = 0.18
DEADZONE_PX = 14

RIGHT_HOLD_SEC = 0.20 

OS = platform.system().lower()

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
    return ang > 155

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
    left_pinch_start_ts = 0
    dragging = False
    last_click_ts = 0

    right_start_ts = 0
    right_active = False
    last_right_click_ts = 0

    cursor_frozen_until = 0
    last_screen_pos = None

    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=MIN_DET,
        min_tracking_confidence=MIN_TRK
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res  = hands.process(rgb)
            h, w = frame.shape[:2]
            action_text = ""
            now = time.time()

            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                lm2d = [(lm.x, lm.y) for lm in hand.landmark]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                fingers = count_fingers(lm2d)
                num_up = int(sum(fingers.values()))

                # ===== Cursor Move =====
                cur = tip_point(lm2d, 8)
                smooth_cursor = cur if smooth_cursor is None else (
                    CURSOR_SMOOTHING*smooth_cursor + (1-CURSOR_SMOOTHING)*cur
                )

                cx = float(np.clip(smooth_cursor[0], CURSOR_CLAMP_MARGIN, 1-CURSOR_CLAMP_MARGIN))
                cy = float(np.clip(smooth_cursor[1], CURSOR_CLAMP_MARGIN, 1-CURSOR_CLAMP_MARGIN))
                sx, sy = int(cx*screen_w), int(cy*screen_h)

                cursor_is_frozen = (now < cursor_frozen_until) or (left_pinch_down and not dragging)

                if not cursor_is_frozen:
                    if last_screen_pos is None:
                        last_screen_pos = (sx, sy)
                        pyautogui.moveTo(sx, sy)
                    else:
                        dx = sx - last_screen_pos[0]
                        dy = sy - last_screen_pos[1]
                        if (dx*dx + dy*dy)**0.5 >= DEADZONE_PX:
                            pyautogui.moveTo(sx, sy)
                            last_screen_pos = (sx, sy)

                # ===== LEFT CLICK (thumb + index pinch) =====
                dist_left = pinch_distance(lm2d, 8, 4)
                allow_clicks = not (num_up >= OPEN_PALM_EXTENDED_MIN or num_up <= FIST_EXTENDED_MAX)

                if allow_clicks and not left_pinch_down and dist_left < PINCH_DOWN:
                    left_pinch_down = True
                    left_pinch_start_ts = now

                if left_pinch_down and dist_left >= PINCH_UP:
                    held = now - left_pinch_start_ts

                    if dragging:
                        pyautogui.mouseUp(button="left")
                        dragging = False
                        action_text = "[DRAG END]"
                        print(action_text)
                    else:
                        if held <= CLICK_HOLD_MAX:
                            pyautogui.click()
                            action_text = "[LEFT CLICK]"
                            print(action_text)
                            cursor_frozen_until = now + FREEZE_AFTER_ACTION_SEC
                    left_pinch_down = False

                # ===== DRAG START =====
                if left_pinch_down and not dragging:
                    if (now - left_pinch_start_ts) >= DRAG_HOLD_SEC:
                        pyautogui.mouseDown(button="left")
                        dragging = True
                        action_text = "[DRAG START]"
                        print(action_text)
                        cursor_frozen_until = 0.0

                # ===== RIGHT CLICK (4-finger gesture, thumb folded, + must hold for RIGHT_HOLD_SEC) =====
                is_right_shape = (num_up == 4 and not fingers['thumb'])

                if is_right_shape and not right_active:
                    right_active = True
                    right_start_ts = now

                if right_active:
                    if not is_right_shape:
                        right_active = False
                    else:
                        if (now - right_start_ts) >= RIGHT_HOLD_SEC:
                            if (now - last_right_click_ts) > CLICK_COOLDOWN:
                                pyautogui.rightClick()
                                action_text = "[RIGHT CLICK]"
                                print(action_text)
                                last_right_click_ts = now
                                cursor_frozen_until = now + FREEZE_AFTER_ACTION_SEC
                                right_active = False   # reset

                # ===== Overlay =====
                if SHOW_OVERLAY:
                    cv2.putText(frame, f"Fingers:{num_up}", (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            else:
                smooth_cursor = None
                last_screen_pos = None
                left_pinch_down = False
                right_active = False

                if dragging:
                    pyautogui.mouseUp(button="left")
                    dragging = False

            if SHOW_OVERLAY:
                if action_text:
                    last_action = action_text
                cv2.putText(frame, last_action, (10, h-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow("Gesture Mouse Control", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
