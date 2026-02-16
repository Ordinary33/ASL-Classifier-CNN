# Real-Time ASL Alphabet Translator

A hybrid computer vision application that translates American Sign Language (ASL) finger spelling into text in real-time. This system combines MediaPipe for hand localization with a ResNet18 deep learning model for classification, allowing for dynamic, high-accuracy inference even in complex backgrounds.

It features a "Hold-to-Type" interface with visual feedback, enabling users to construct full sentences naturally.

## Demo


https://github.com/user-attachments/assets/dfa4bdab-d43f-4879-82ef-afa894daba3c



## 🚀 Key Features

* **⚡ Hybrid Architecture:** Uses MediaPipe for precise hand tracking and ROI cropping, feeding only the clean hand image to a ResNet18 classifier for maximum accuracy.
* **📷 Dynamic Auto-zoom:** The system automatically crops, centers, and resizes the hand image (224x224) regardless of distance from the camera, acting as a "smart scope" for the AI.
* **🧠 Smart Typing System:** Implements a "Hold-to-Type" logic with a visual progress bar. The system requires sign stability before "locking in" a letter, preventing jittery outputs.
* **📝 Sentence Management** Includes logic for **Space**, **Delete**, and **Clear** operations, allowing users to write and edit full sentences on screen.
---

## 🛠️ Tech Stack

* **Computer Vision:** OpenCV (cv2)
* **Machine Learning:** PyTorch, Scikit-Learn (Metrics)
* **Environment:** Python 3.11+, Poetry
* **Localization**: MediaPipe Hands

---

## 📂 Project Structure

```text
├── data/                       # Dataset storage
├── models/                     # Saved Model Files (.pth)
│   └── checkpoints/
├── src/
│   ├── __init__.py
│   ├── model.py                # PyTorch CNN Architecture (ASLCNNModel)
│   ├── predict.py              # Inference Logic (ASLPredictor Class)
│   ├── transforms.py           # Image Preprocessing & Augmentation Pipelines
│   ├── loaders.py              # Torch Dataset & DataLoader Logic
│   └── run_webcam.py           # Main Application Entry Point
├── train.py                    # Training Loop & Validation
├── pyproject.toml              # Dependencies & Configuration
└── README.md                   # Documentation
```

## ⚡ Installation & Setup

### 1. Prerequisites
* Python 3.11+ installed.

* Poetry installed for dependency management.

* A working Webcam.

### 2. Clone and Install
```bash
git clone [https://github.com/yourusername/ASL-Classifier-CNN.git](https://github.com/yourusername/ASL-Classifier-CNN.git)
cd ASL-Classifier-CNN
poetry install
```

### 3. Setup Models
Place your trained model weights in the checkpoints directory:

models/checkpoints/asl_cnn_model.pth

(If you haven't trained yet, run poetry run python train.py first)

## 🏃‍♂️ How to Run
**Start the Translator**
This launches the webcam interface with the overlay.
```bash
poetry run python -m src.run_webcam
```

### Controls:
- Green Box: Place your hand inside the green ROI (Region of Interest) rectangle.
- 'q': Press q to quit the application.

### 🧠 Model Performance
The model utilizes a custom CNN architecture optimized for 224x224 RGB images.
* It achieved perfect accuracy (1.00) on the validation test set.

* It achieved perfect recall (1.00) on the validation test set.

### 🔮 Future Improvements
* Dynamic Backgrounds: Add a toggle to switch between camera view and a solid black background for privacy.
* Text-to-Speech (TTS): Integrate pyttsx3 to read the generated sentences aloud.
