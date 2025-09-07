# Gesture Recognition Using Skeletal Data for Real-Time Human-Computer Interaction

This project is developed as part of the Project Development (DES400&CSS400), SIIT, Thammasat University. {CN2-2025}

This project implements a real-time gesture recognition system using skeletal data extracted from a live webcam feed. 
It allows natural, contactless interaction with a computer by mapping gestures (e.g., finger movements) to predefined commands.

---

## 📌 Features
- Real-time hand and body landmark detection using **MediaPipe Pose/Hands**  
- Deep learning-based gesture classification with **LSTM**  
- Supports gestures such as:  
  - Single index finger → Scroll / pan screen  
  - Two fingers → Switch browser tabs  
  - Three fingers → Change workspace  
  - Closed fist → Capture screenshot  
  - Thumbs-up → Lock screen  
- Hands-free alternative to keyboard shortcuts and mouse clicks  

---

## 🛠️ Installation

```bash
### 1. Clone the repository
git clone https://github.com/your-username/gesture-hci.git
cd gesture-hci

### 2. Create and activate virtual environment (Python 3.11 recommended)
py -3.11 -m venv .venv
.venv\Scripts\activate   # Windows

###3. Install dependencies
pip install -r requirements.txt

---

## ▶️ Usage

### Test camera
python src/test_camera.py

### Show hand landmarks
python src/hand_landmarks_demo.py

### Run gesture control prototype
python src/simple_gesture_control.py

---