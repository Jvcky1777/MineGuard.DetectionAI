# MineGuard AI — Live PPE Detection Dashboard

An industrial-grade real-time Personal Protective Equipment (PPE) detection system powered by **Ultralytics YOLO** (`best.pt`), **FastAPI**, and a modern **React 19** frontend.

---

## Features

- **Live Camera Feed with Boundary Boxes**:
  - Real-time video with detected objects highlighted using crisp bounding boxes, class labels, and confidence percentages.
  - Dual Camera Modes:
    1. **Local Camera (OpenCV)**: Connects directly to system webcam hardware via OpenCV at high FPS.
    2. **Browser Webcam**: Uses HTML5 `getUserMedia` directly in the browser and sends frames to the detection engine.
- **One-Click Start / Stop Controls**:
  - High-visibility **Start Detection** and **Stop Detection** buttons.
  - Safe camera resource acquisition and release.
- **Real-Time PPE Compliance Matrix**:
  - Live compliance monitoring for:
    - **Hard Hat**: *Hard hat on* vs *bare head* / *no helmet*
    - **Safety Suit**: *Safety suit on* vs *no suit on*
    - **Safety Boots**: *Boots on* vs *no boots*
    - **Gloves**: *gloves on* vs *no gloves*
- **Safety Violation Alerts**:
  - Visual flashing alert banner and optional Web Audio chime when safety violations are detected.
- **Interactive AI Controls**:
  - Real-time **Confidence Threshold** slider (10% to 95%).
  - Frame Snapshot capture button (saves detected frame with bounding boxes as image).
  - Live FPS counter & latency tracker.

---

## Quick Start

### 1. One-Click Launcher (Windows)
Double-click `start.bat` or run:
```powershell
.\start.bat
```

### 2. Manual Start

**Backend (FastAPI):**
```powershell
uv run python server.py
```
Backend runs at `http://localhost:8000`.

**Frontend (React + Vite):**
```powershell
cd frontend
npm run dev
```
React UI runs at `http://localhost:5173`.

---

## API Endpoints

- `GET /api/health` — System status, loaded model info, and class mappings.
- `POST /api/start` — Activates native OpenCV camera and inference worker.
- `POST /api/stop` — Safely stops native camera and releases hardware.
- `GET /api/status` — Live telemetry (detections, violations count, FPS).
- `GET /api/video_feed` — MJPEG live stream with bounding boxes plotted.
- `POST /api/detect_frame` — Runs YOLO detection on an uploaded frame from the browser camera.
- `POST /api/config` — Updates confidence threshold.

