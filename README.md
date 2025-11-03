# Gesture Recognition Using Skeletal Data for Real-Time Human-Computer Interaction

## 📌 Project Overview
This project is a **Senior Project (CN2-2025)** at SIIT, Thammasat University.  
It focuses on **real-time hand gesture recognition using skeletal data** (MediaPipe Hands + LSTM).  
The main goal is to use a **webcam** to detect hand gestures, train with **LSTM**, and control the computer in real time for actions such as desktop switching, tab switching, scrolling, and mouse interactions.

---

## ✨ Features
The system supports both **left** and **right** hands (mirror correction applied automatically).  
Supported gestures are divided into two groups:

### 🔹 LSTM-based gestures (temporal sequences)
1. **5 fingers swipe (left/right)** → Switch Desktop  
2. **3 fingers swipe (left/right)** → Switch Browser Tab  
3. **2 fingers swipe (up/down/left/right)** → Scroll / Pan  
4. **Open palm ➝ Fist (5 → 0)** → Screenshot  
5. **Idle (no gesture)** → Prevents false triggers  

### 🔹 Rule-based gestures (low-latency controls)
1. **1 finger (index)** → Mouse movement / drag  
2. **Thumb + Index pinch** → Left Click (single)  
   - Double pinch quickly = Double Left Click  
3. **Thumb + Ring pinch** → Right Click  

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/ChonmaneeC/GestureRecognitionProject.git
cd GestureRecognitionProject

# 2. Create and activate virtual environment (Python 3.11 recommended)
py -3.11 -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate   # Linux / Mac

# 3. Install dependencies
pip install -r requirements.txt

```

## ▶️ Usage

```bash
### Collect dataset
#Record gesture samples (30 frames = 1 sequence):
python src/collect_sequences.py --user <name> --hand right --frames 30

### Prepare dataset
#Combine collected .npy clips into a single dataset:
python src/prepare_dataset.py

### Train LSTM model
#Train with class weights, validation split, early stopping:
python src/train_lstm.py --epochs 12 --batch_size 32 --use_class_weights

### Run real-time inference
#Control your PC using hand gestures:
python src/realtime_inference.py

```

## 📂 Dataset and Models

```bash
Due to large file sizes, dataset sequences and trained models are stored on Google Drive:  
[👉 Download here](https://drive.google.com/drive/folders/1LDwHEnwSbyWNUQFL7FyXwD3U3u5Gfear?usp=sharing)

```

## Notes
- Both best.keras (trained model) and gesture_norm.npz (normalization info) must exist inside the models/ directory before running inference.
- Gestures such as desktop switch, tab switch use cooldown timers to avoid repeated triggers.
---