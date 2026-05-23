# YOLO Object Detection — Real-Time Webcam Deployment with Streamlit

Deploy a custom **YOLO World** model for **real-time object detection** in the browser using **Python**, **OpenCV**, and **Streamlit**. This repository provides a minimal, production-style setup for running live webcam inference with an Ultralytics-trained weights file (`best.pt`).

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLO%20World-00FFFF?logo=ultralytics)](https://docs.ultralytics.com/models/yolo-world/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Video%20Capture-5C3EE8?logo=opencv)](https://opencv.org/)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Run with GitHub Codespaces](#run-with-github-codespaces)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Related Resources](#related-resources)
- [License](#license)

---

## Overview

**Yolo-Deployment** is a lightweight **computer vision** demo that streams your webcam through a **YOLO** detector and displays annotated frames in a **Streamlit** web app. It is ideal for:

- Prototyping **custom YOLO model** deployment after training
- Teaching **real-time object detection** with minimal boilerplate
- Sharing a reproducible **YOLO + Streamlit** stack via Dev Containers or local install

The app loads `best.pt` (your trained weights), captures video with **OpenCV**, runs inference with **Ultralytics YOLOWorld**, and renders bounding boxes in the browser.

---

## Features

| Feature | Description |
|--------|-------------|
| **Real-time webcam inference** | Live video from the default camera (`cv2.VideoCapture(0)`) |
| **YOLO World model** | Uses `ultralytics.YOLOWorld` with custom `best.pt` weights |
| **Streamlit UI** | Simple checkbox to start/stop the webcam feed |
| **Annotated output** | Detections drawn with `results[0].plot()` |
| **Dev Container ready** | One-click setup in GitHub Codespaces / VS Code Dev Containers |
| **Minimal dependencies** | Small `requirements.txt` for fast installs |

---

## Tech Stack

- **[Ultralytics YOLO World](https://docs.ultralytics.com/models/yolo-world/)** — open-vocabulary / world-aware object detection
- **[Streamlit](https://streamlit.io/)** — Python web UI for ML demos
- **[OpenCV](https://opencv.org/)** (`opencv-python`) — webcam capture and image processing
- **[Pillow](https://python-pillow.org/)** — frame conversion for Streamlit display
- **[NumPy](https://numpy.org/)** — array operations

---

## Prerequisites

- **Python 3.11+** (matches the Dev Container image)
- A **webcam** (built-in or USB)
- Trained weights file: **`best.pt`** in the project root (included in this repo or replace with your own checkpoint)
- (Optional) **Docker** or **VS Code Dev Containers** for containerized development

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/danishjavedcodes/Yolo-Deployment.git
cd Yolo-Deployment
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place your model weights

Ensure `best.pt` exists in the project root (same directory as `interface.py`). The default app loads:

```python
model = YOLOWorld('./best.pt')
```

To use a different checkpoint, update the path in `interface.py`.

### 5. Run the Streamlit app

```bash
streamlit run interface.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`), check **Run Webcam**, and allow camera access when prompted.

---

## Run with GitHub Codespaces

This repo includes a [Dev Container](https://containers.dev/) configuration (`.devcontainer/devcontainer.json`):

1. Open the repository on GitHub.
2. Click **Code** → **Codespaces** → **Create codespace on main**.
3. Wait for `requirements.txt` to install and Streamlit to start (port **8501**).
4. Open the forwarded port preview and enable **Run Webcam**.

> **Note:** Webcam access in cloud environments may be limited. For the best experience, run locally on a machine with a physical camera.

---

## Project Structure

```
Yolo-Deployment/
├── interface.py          # Streamlit app: webcam + YOLO inference loop
├── best.pt               # Trained YOLO World weights (Ultralytics checkpoint)
├── requirements.txt      # Python dependencies
├── .devcontainer/        # Dev Container / Codespaces configuration
│   └── devcontainer.json
├── .vscode/              # VS Code tasks, launch, and editor settings
└── public/               # Static assets (fonts; optional for UI extensions)
```

---

## How It Works

1. **Model load** — `YOLOWorld('./best.pt')` loads your custom weights at startup.
2. **Webcam capture** — When **Run Webcam** is enabled, OpenCV reads frames from device `0`.
3. **Inference** — Each frame is passed to `model(frame)`; results include boxes and labels.
4. **Visualization** — `results[0].plot()` draws annotations; BGR is converted to RGB for Streamlit.
5. **Display** — `st.empty()` updates the image placeholder each frame for a live feed.

Core application entry point: [`interface.py`](interface.py).

---

## Configuration

| Setting | Location | Default |
|--------|----------|---------|
| Model path | `interface.py` | `./best.pt` |
| Camera index | `interface.py` | `0` |
| Streamlit port | CLI / Dev Container | `8501` |

Example: use an external USB camera:

```python
cap = cv2.VideoCapture(1)  # change index from 0 to 1
```

---

## Troubleshooting

| Issue | Possible fix |
|-------|----------------|
| **Could not open webcam** | Close other apps using the camera; try a different `VideoCapture` index |
| **Model file not found** | Confirm `best.pt` is in the repo root or update the path in `interface.py` |
| **Slow inference** | Use a GPU-enabled environment; reduce input resolution before inference |
| **No detections** | Retrain or verify `best.pt` matches your classes; check lighting and framing |
| **Codespaces: no camera** | Run locally with `streamlit run interface.py` |

---

## FAQ

### What is YOLO World?

**YOLO World** is an Ultralytics model variant that supports flexible, open-vocabulary detection workflows. See the [official YOLO World documentation](https://docs.ultralytics.com/models/yolo-world/).

### Can I use YOLOv8 or YOLOv11 instead of YOLO World?

Yes. Replace `YOLOWorld` with the appropriate Ultralytics class (e.g. `YOLO`) and point to compatible `.pt` weights. Update imports and model initialization in `interface.py`.

### How do I deploy beyond Streamlit?

For production APIs, consider exporting to **ONNX** or **TensorRT** and serving with **FastAPI**, **Flask**, or a cloud GPU service. This repo focuses on rapid **Streamlit deployment** for demos and prototyping.

### Is `best.pt` required?

Yes. The app expects trained weights at `./best.pt`. Train with [Ultralytics](https://docs.ultralytics.com/modes/train/) or download your checkpoint into the project root.

---

## Related Resources

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [YOLO World Model Guide](https://docs.ultralytics.com/models/yolo-world/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

---

## License

Add a license file (e.g. `LICENSE` with MIT or Apache-2.0) if you plan to open-source or distribute this project. Model weights and third-party assets may have separate terms from Ultralytics and your training data.

---

## Author

**Danish Javed** — [github.com/danishjavedcodes](https://github.com/danishjavedcodes)

Repository: [github.com/danishjavedcodes/Yolo-Deployment](https://github.com/danishjavedcodes/Yolo-Deployment)

---

### Keywords

`yolo object detection` · `yolo world` · `streamlit deployment` · `real-time webcam detection` · `ultralytics yolo` · `opencv python` · `computer vision` · `custom yolo model` · `machine learning demo` · `python object detection`
