import cv2
import mediapipe as mp
import numpy as np
import os
from collections import deque
from datetime import datetime

# ========== Config ==========
GESTURE = "one_finger"
DIRECTIONS = ["up", "down", "left", "right"]
OUT_DIR = "dataset"
FRAMES_PER_CLIP = 30      # T เฟรมต่อคลิป
MIN_DET, MIN_TRK = 0.6, 0.6

# ========== Prepare =========
for d in DIRECTIONS:
    os.makedirs(os.path.join(OUT_DIR, GESTURE, d), exist_ok=True)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

current_dir = "up"       # ทิศทางเริ่มต้น
recording = False
buf = deque(maxlen=FRAMES_PER_CLIP)
saved = {d: 0 for d in DIRECTIONS}

def save_clip(direction, clip_np):
    # clip_np shape: (T, 63)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = os.path.join(OUT_DIR, GESTURE, direction, f"clip_{ts}.npy")
    np.save(out_path, clip_np)
    return out_path

with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=MIN_DET,
    min_tracking_confidence=MIN_TRK
) as hands:
    print("[INFO] Keys: W/A/S/D change direction | S start/stop | Q quit")
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        h, w = frame.shape[:2]
        status = f"DIR:{current_dir.upper()} | REC:{'ON' if recording else 'OFF'}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,0) if recording else (0,165,255), 2)

        # โชว์สถิติบันทึก
        y0 = 60
        for d in DIRECTIONS:
            cv2.putText(frame, f"{d}: {saved[d]}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255,255,255), 1)
            y0 += 22

        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            # นับนิ้วแบบง่าย ๆ: นิ้วชี้ยืด = one_finger (คร่าว ๆ)
            # (คุณสามารถใช้ฟังก์ชันนับนิ้วจากไฟล์ control ของคุณมาแทนได้)
            lms = [(lm.x, lm.y, lm.z) for lm in hand.landmark]
            # เตรียมเวคเตอร์ 63 ค่า
            vec = []
            for (x,y,z) in lms:
                vec += [x,y,z]
            vec = np.array(vec, dtype=np.float32)

            if recording:
                buf.append(vec)
                # ถ้าเต็ม T เฟรมแล้ว — บันทึกเป็นคลิป
                if len(buf) == FRAMES_PER_CLIP:
                    clip_np = np.stack(list(buf), axis=0)  # (T, 63)
                    path = save_clip(current_dir, clip_np)
                    saved[current_dir] += 1
                    buf.clear()
                    # แสดงบนจอ
                    cv2.putText(frame, f"Saved: {path}", (10, h-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        cv2.imshow("Collect One-Finger Sequences", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in (ord('w'), ord('a'), ord('s'), ord('d')):
            mapping = {'w': 'up', 'a': 'left', 's': 'down', 'd': 'right'}
            current_dir = mapping[chr(key)]
        elif key == ord('s'):
            recording = not recording
            buf.clear()

cap.release()
cv2.destroyAllWindows()
print("[INFO] Done. Saved:", saved)
