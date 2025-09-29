# Gesture Recognition Using Skeletal Data for Real-Time Human-Computer Interaction

## 1. Project Overview
This project is a **Senior Project** {CN2-2025} at SIIT (Thammasat University).  
It focuses on **Real-Time Hand Gesture Recognition using Skeletal Data** (MediaPipe Hands + LSTM).  
The goal is to use a **webcam** to detect hand gestures → train with **LSTM** → control the computer in real-time (desktop switching, tab switching, scrolling, mouse actions, etc.).

---

## 2. Features
Supported gestures:

1. **5 fingers (open palm)** → swipe left/right = Switch Desktop  
2. **3 fingers** → swipe left/right = Switch Browser Tab  
3. **2 fingers** → swipe up/down/left/right = Scroll / Pan  
4. **Open palm ➝ Fist (5 → 0)** = Capture Screenshot  
5. **1 finger (index)** → Mouse control (movement, drag)  
6. **Thumb + Index pinch** → Left Click (single)  
   - Double pinch quickly = Double Left Click  
7. **Thumb + Ring pinch** → Right Click  

Supports **both left and right hand** (mirror correction applied automatically).

---

## 🛠️ Installation

```bash
### 1. Clone the repository
git clone https://github.com/ChonmaneeC/GestureRecognitionProject.git
cd ProjectGesture

### 2. Create and activate virtual environment (Python 3.11 recommended)
py -3.11 -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux / Mac

###3. Install dependencies
pip install -r requirements.txt


```

## ▶️ Usage

```bash
### Collect dataset
#Record gesture samples (30 frames = 1 sequence):
python src/collect_sequences.py --label desktop_left

### Prepare dataset
#Combine raw .npy into gestures.npz:
python src/prepare_dataset.py

### Train LSTM model
#Train with class weights, validation split, early stopping:
python src/train_lstm.py --epochs 12 --batch_size 32 --use_class_weights

### Run real-time inference
#Control your PC using hand gestures:
python src/realtime_inference.py

```

## Notes
- Both best.keras (model) and gesture_norm.npz (normalization info) must exist inside models/.
- Gestures are recognized with cooldown timers to avoid repeated triggers (e.g., desktop switch, tab switch).
- Scrolling and mouse actions are continuous (no cooldown).
---