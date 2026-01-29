# Real-Time ASL Alphabet Recognizer

A deep learning computer vision application that translates American Sign Language (ASL) finger spelling into text in real-time. This system transfer learning using the pretrained weights of the resnet18 model to perform image classification on 29 classes (A-Z, space, del, nothing) and visualizes predictions via an **OpenCV** interface with prediction smoothing.

## Demo

---

## 🚀 Key Features

* **⚡ Deep Learning Engine:** Custom **PyTorch** CNN architecture featuring multiple convolutional blocks, Batch Normalization, and Dropout for robust feature extraction.
* **📷 Real-Time Inference:** High-performance video processing pipeline using **OpenCV** that handles frame capture, preprocessing, and prediction at 30+ FPS.
* **🧠 Smart Smoothing:** Implements a statistical buffer (Deque + Mode) to stabilize predictions and eliminate label flickering.
* **🛡️ Production Grade:** Built with strict type checking (**Mypy**), formatting (**Black/Flake8**), and dependency management (**Poetry**).

---

## 🛠️ Tech Stack

* **Computer Vision:** OpenCV (cv2)
* **Machine Learning:** PyTorch, Scikit-Learn (Metrics)
* **Environment:** Python 3.13+, Poetry

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
* Python 3.13+ installed.

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
* Background Removal: Integrate MediaPipe Hands to segment the hand from the background for higher accuracy in messy rooms.

* Sentence Formation: Add logic to string letters together into words and sentences.
