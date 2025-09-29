# src/preview_clip.py
import numpy as np
import cv2
import mediapipe as mp
import sys

mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def preview(path):
    data = np.load(path)  # (T, 63)
    T = data.shape[0]
    for t in range(T):
        vec = data[t]  # (63,)
        pts = [(vec[i], vec[i+1]) for i in range(0, 63, 3)]
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 20
        for (x, y) in pts:
            cx, cy = int(x*640), int(y*480)
            cv2.circle(frame, (cx, cy), 5, (0,255,0), -1)
        cv2.putText(frame, f"{t+1}/{T}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.imshow("Preview", frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    path = sys.argv[1]
    preview(path)
