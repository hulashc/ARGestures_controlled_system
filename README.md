# ARGestures Controlled System

A real-time **hand gesture controlled system** built with Python, MediaPipe, and OpenCV. Uses a webcam to track hand landmarks and translate gestures into system controls — enabling touchless interaction with your computer through AR-style overlays.

---

## ✋ Features

- Real-time hand tracking via MediaPipe Hand Landmarker
- Gesture recognition engine for classifying distinct hand poses
- AR overlay UI rendered directly onto the camera feed
- Touch simulation engine for simulating pointer/click interactions
- Menu state management for navigating gesture-driven menus
- Configurable settings via `config.py`
- Utility for discovering available camera devices

---

## 📁 Project Structure

```
ARGestures_controlled_system/
├── main.py                # Entry point — starts the gesture control loop
├── gesture_engine.py      # Gesture recognition and classification logic
├── hand_tracking.py       # MediaPipe hand landmark detection
├── ui_renderer.py         # AR overlay and UI rendering (OpenCV)
├── menu_state.py          # Gesture-driven menu state machine
├── touch_engine.py        # Touch/click simulation
├── utils.py               # Shared utility functions
├── config.py              # Configuration constants and settings
├── find_camera.py         # Camera device discovery utility
├── hand_landmarker.task   # MediaPipe Hand Landmarker model file
└── requirements.txt       # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- A webcam

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/hulashc/ARGestures_controlled_system.git
   cd ARGestures_controlled_system
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

### Running the System

```bash
python main.py
```

If you're unsure which camera index to use, run the camera finder first:

```bash
python find_camera.py
```

Then update `config.py` with the correct camera index.

---

## ⚙️ Configuration

Edit `config.py` to customise behaviour:

| Setting | Description |
|---|---|
| `CAMERA_INDEX` | Index of the webcam to use |
| `MIN_DETECTION_CONFIDENCE` | Hand detection confidence threshold |
| `MIN_TRACKING_CONFIDENCE` | Hand tracking confidence threshold |
| Other gesture thresholds | Tune sensitivity per gesture |

---

## 🧰 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `opencv-python` | ≥ 4.10.0 | Camera capture and AR rendering |
| `mediapipe` | ≥ 0.10.30 | Hand landmark detection model |
| `numpy` | ≥ 1.26.0 | Numerical operations for gesture maths |

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
