import cv2
import mediapipe as mp
import csv
import os

# ตั้งค่า MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# gesture ที่จะเก็บ
gesture_name = "one_finger"
save_path = f"dataset/{gesture_name}.csv"
os.makedirs("dataset", exist_ok=True)

# ถ้ายังไม่มีไฟล์ csv ให้สร้างพร้อม header
if not os.path.exists(save_path):
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = []
        for i in range(21):  # 21 landmark
            header += [f"x{i}", f"y{i}", f"z{i}"]
        header.append("label")
        writer.writerow(header)

cap = cv2.VideoCapture(0)
print(f"Collecting gesture: {gesture_name}. Press 'q' to stop.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            coords.append(gesture_name)  # label

            # append ลงไฟล์ csv
            with open(save_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(coords)
                print("Saved frame:", coords[:6], "...")  # แสดง x,y,z ของ landmark 2 จุดแรก

            # วาด landmark บนภาพ
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Collecting Data", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
