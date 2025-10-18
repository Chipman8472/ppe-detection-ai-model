# PPE Violation Detector (YOLOv8 + Roboflow)

Real-time desktop app (Tkinter) that flags **missing PPE** (e.g., **NO-Hardhat, NO-Mask, NO-Safety Vest**) in video.  
Plays an audible alert, overlays boxes, logs each violation, and saves snapshot images.

---

## ✨ Features

- **YOLOv8** model loaded from a local `best.pt` (exported from **Roboflow** or Ultralytics training)
- **GUI** preview with bounding boxes (Tkinter Canvas)
- **Audible alert** via Text-to-Speech (pyttsx3)
- **Auto-log** to `violation_log.csv` with timestamps
- **Auto-snapshot** of each violation to `/snapshots`
- **One click** Start/Stop

---

## 🧠 Model & Classes

The app treats the following classes as **violations** and triggers alerts:

- `NO-Hardhat`
- `NO-Mask`
- `NO-Safety Vest`

> Any other detected classes (e.g., `person`) will be drawn in **green** and won’t trigger alerts by default.  
> Update the list in code if your dataset uses different/extra labels.

---

## 📦 Requirements

- Python **3.9+** (3.10/3.11 recommended)
- OS: Windows, macOS, or Linux (GUI + audio supported)
- GPU optional (CPU works; GPU speeds up inference)

### Install dependencies

```bash
pip install ultralytics opencv-python pillow pyttsx3
