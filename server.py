import io
import time
import base64
import threading
from typing import Optional, List, Dict, Any

import cv2
import requests
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from ultralytics import YOLO

app = FastAPI(title="MineGuard AI - PPE Detection Server", version="1.0.0")

# Enable CORS for React frontend (Vite runs on 5173 by default)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load and warmup YOLO model
print("[MineGuard] Loading YOLO model 'best.pt'...")
model = YOLO("best.pt")
# Model warmup
dummy = np.zeros((320, 320, 3), dtype=np.uint8)
_ = model.predict(source=dummy, conf=0.5, verbose=False)
print(f"[MineGuard] Model loaded successfully! Classes: {model.names}")

VIOLATION_KEYWORDS = {"no helmet", "bare head", "no boots", "no gloves", "no suit on"}

def is_violation_class(name: str) -> bool:
    lower_name = name.strip().lower()
    return any(v in lower_name for v in VIOLATION_KEYWORDS) or lower_name.startswith("no ")

def parse_yolo_result(result, original_w: int, original_h: int) -> Dict[str, Any]:
    boxes = result.boxes
    detections = []
    violations_count = 0
    compliant_count = 0

    if boxes is not None and len(boxes) > 0:
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i].item())
            class_name = result.names.get(class_id, f"Class {class_id}")
            confidence = round(float(boxes.conf[i].item()), 3)
            xyxy = [round(float(v), 1) for v in boxes.xyxy[i].tolist()]
            xyxyn = [round(float(v), 4) for v in boxes.xyxyn[i].tolist()]
            
            violation = is_violation_class(class_name)
            if violation:
                violations_count += 1
            else:
                compliant_count += 1

            detections.append({
                "id": i,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "confidence_pct": int(confidence * 100),
                "is_violation": violation,
                "bbox": {
                    "x1": xyxy[0],
                    "y1": xyxy[1],
                    "x2": xyxy[2],
                    "y2": xyxy[3],
                },
                "normalized_bbox": {
                    "x1": xyxyn[0],
                    "y1": xyxyn[1],
                    "x2": xyxyn[2],
                    "y2": xyxyn[3],
                }
            })

    status = "NO_OBJECTS"
    if violations_count > 0:
        status = "VIOLATION"
    elif compliant_count > 0:
        status = "COMPLIANT"

    return {
        "timestamp": time.time(),
        "status": status,
        "counts": {
            "total": len(detections),
            "violations": violations_count,
            "compliant": compliant_count
        },
        "detections": detections
    }


# ==========================================
# Native Camera Manager (OpenCV backend)
# ==========================================
# Roboflow Cloud Deployment Configuration
ROBOFLOW_API_KEY = "tRVHHVVhltdOXqileCgJ"
ROBOFLOW_PROJECT = "mineguard-detection-crele"
ROBOFLOW_VERSION = "7"
ROBOFLOW_FALLBACK_VERSION = "4"

def parse_roboflow_predictions(predictions: List[Dict[str, Any]], img_w: int, img_h: int) -> Dict[str, Any]:
    detections = []
    violations_count = 0
    compliant_count = 0

    for i, pred in enumerate(predictions):
        class_name = pred.get("class", "Unknown")
        confidence = round(float(pred.get("confidence", 0.0)), 3)
        cx, cy = pred.get("x", 0), pred.get("y", 0)
        w, h = pred.get("width", 0), pred.get("height", 0)

        x1 = max(0, int(cx - (w / 2)))
        y1 = max(0, int(cy - (h / 2)))
        x2 = min(img_w, int(cx + (w / 2)))
        y2 = min(img_h, int(cy + (h / 2)))

        violation = is_violation_class(class_name)
        if violation:
            violations_count += 1
        else:
            compliant_count += 1

        detections.append({
            "id": i,
            "class_id": i,
            "class_name": class_name,
            "confidence": confidence,
            "confidence_pct": int(confidence * 100),
            "is_violation": violation,
            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            },
            "normalized_bbox": {
                "x1": round(x1 / max(img_w, 1), 4),
                "y1": round(y1 / max(img_h, 1), 4),
                "x2": round(x2 / max(img_w, 1), 4),
                "y2": round(y2 / max(img_h, 1), 4),
            }
        })

    status = "NO_OBJECTS"
    if violations_count > 0:
        status = "VIOLATION"
    elif compliant_count > 0:
        status = "COMPLIANT"

    return {
        "timestamp": time.time(),
        "status": status,
        "counts": {
            "total": len(detections),
            "violations": violations_count,
            "compliant": compliant_count
        },
        "detections": detections
    }

def annotate_roboflow_frame(frame, detections):
    annotated = frame.copy()
    for det in detections:
        box = det["bbox"]
        is_violation = det["is_violation"]
        color = (0, 0, 255) if is_violation else (0, 255, 0)
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class_name']} {det['confidence_pct']}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - 22)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 3, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated

class NativeCameraManager:
    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running: bool = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.current_frame_bytes: Optional[bytes] = None
        self.latest_telemetry: Dict[str, Any] = {
            "active": False,
            "status": "STANDBY",
            "fps": 0,
            "counts": {"total": 0, "violations": 0, "compliant": 0},
            "detections": []
        }
        self.fps: float = 0.0
        self.conf_threshold: float = 0.45
        self.camera_index: int = 0

    def start(self, camera_index: int = 0, conf: float = 0.45):
        with self.lock:
            if self.is_running:
                return True, "Camera already running"
            
            self.camera_index = camera_index
            self.conf_threshold = conf
            
            # Open camera
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                return False, f"Could not open camera {self.camera_index}"
            
            # Set resolution for crisp 720p performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            self.is_running = True
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()
            return True, "Camera started successfully"

    def stop(self):
        with self.lock:
            if not self.is_running:
                return True, "Camera was not running"
            self.is_running = False
            
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
            
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.current_frame_bytes = None
            self.latest_telemetry["active"] = False
            self.latest_telemetry["status"] = "STANDBY"
            self.latest_telemetry["fps"] = 0
            
        return True, "Camera stopped successfully"

    def _worker(self):
        prev_time = time.time()
        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                break
                
            success, frame = self.cap.read()
            if not success or frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            
            # Predict with YOLO
            results = model.predict(source=frame, conf=self.conf_threshold, verbose=False)
            result = results[0]
            
            # Plot boxes directly on frame
            annotated_frame = result.plot()
            
            # Calculate FPS
            curr_time = time.time()
            fps_calc = 1.0 / max(curr_time - prev_time, 0.001)
            prev_time = curr_time
            self.fps = round(fps_calc, 1)

            # Telemetry
            telemetry = parse_yolo_result(result, w, h)
            telemetry["active"] = True
            telemetry["fps"] = self.fps
            
            # Encode frame to JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            _, buffer = cv2.imencode('.jpg', annotated_frame, encode_param)
            frame_bytes = buffer.tobytes()

            with self.lock:
                self.current_frame_bytes = frame_bytes
                self.latest_telemetry = telemetry

            time.sleep(0.005)

camera_manager = NativeCameraManager()


# ==========================================
# REST API Endpoints
# ==========================================
@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "service": "MineGuard Detection AI",
        "model": "YOLO (best.pt)",
        "classes": model.names,
        "is_camera_active": camera_manager.is_running
    }

@app.post("/api/start")
def start_camera(camera_index: int = 0, conf: float = 0.45):
    success, msg = camera_manager.start(camera_index=camera_index, conf=conf)
    return {"success": success, "message": msg, "active": camera_manager.is_running}

@app.post("/api/stop")
def stop_camera():
    success, msg = camera_manager.stop()
    return {"success": success, "message": msg, "active": camera_manager.is_running}

@app.get("/api/status")
def get_status():
    return {
        "active": camera_manager.is_running,
        "fps": camera_manager.fps,
        "conf_threshold": camera_manager.conf_threshold,
        "telemetry": camera_manager.latest_telemetry
    }

@app.post("/api/config")
def update_config(conf: float = 0.45):
    camera_manager.conf_threshold = conf
    return {"conf_threshold": conf}

# MJPEG Stream for native camera
def generate_mjpeg_stream():
    standby_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(standby_img, "MINEGUARD AI - CAMERA STANDBY", (100, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 2)
    cv2.putText(standby_img, "Click 'Start Detection' to activate live feed", (110, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1)
    _, standby_encoded = cv2.imencode('.jpg', standby_img)
    standby_bytes = standby_encoded.tobytes()

    while True:
        if camera_manager.is_running and camera_manager.current_frame_bytes is not None:
            frame_data = camera_manager.current_frame_bytes
        else:
            frame_data = standby_bytes

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        time.sleep(0.033)  # ~30 fps cap

@app.get("/api/video_feed")
def video_feed():
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# ==========================================
# Frame Detection API (Browser WebCam Mode)
# ==========================================
@app.post("/api/detect_frame")
async def detect_frame(
    file: UploadFile = File(...),
    conf: float = Form(0.45),
    return_annotated: bool = Form(True)
):
    """
    Accepts an uploaded frame from the browser webcam (HTML5 getUserMedia),
    runs YOLO detection, and returns telemetry + base64 annotated frame with bounding boxes.
    """
    t0 = time.time()
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Failed to decode image"})

    h, w = frame.shape[:2]

    # YOLO prediction
    results = model.predict(source=frame, conf=conf, verbose=False)
    result = results[0]

    # Telemetry
    telemetry = parse_yolo_result(result, w, h)
    telemetry["inference_time_ms"] = round((time.time() - t0) * 1000, 1)

    annotated_base64 = None
    if return_annotated:
        annotated_frame = result.plot()
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        _, buffer = cv2.imencode('.jpg', annotated_frame, encode_param)
        annotated_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

    return {
        "telemetry": telemetry,
        "annotated_image": annotated_base64,
        "width": w,
        "height": h
    }


@app.post("/api/detect_roboflow")
async def detect_roboflow(
    file: UploadFile = File(...),
    conf: float = Form(0.45),
    return_annotated: bool = Form(True),
    version: Optional[str] = Form("7")
):
    """
    Accepts an uploaded frame, runs detection via the Roboflow Cloud Server API,
    and returns telemetry + base64 annotated frame with bounding boxes.
    """
    t0 = time.time()
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Failed to decode image"})

    h, w = frame.shape[:2]
    image_base64 = base64.b64encode(contents).decode("utf-8")

    selected_version = version or ROBOFLOW_VERSION
    conf_pct = max(10, min(95, int(conf * 100)))
    url = f"https://detect.roboflow.com/{ROBOFLOW_PROJECT}/{selected_version}?api_key={ROBOFLOW_API_KEY}&confidence={conf_pct}&overlap=50"

    try:
        response = requests.post(
            url,
            data=image_base64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=12.0
        )
        if response.status_code != 200 and selected_version == "7":
            # Fallback to version 4 if version 7 is still initializing on cloud
            fallback_url = f"https://detect.roboflow.com/{ROBOFLOW_PROJECT}/{ROBOFLOW_FALLBACK_VERSION}?api_key={ROBOFLOW_API_KEY}&confidence={conf_pct}&overlap=50"
            fb_resp = requests.post(
                fallback_url,
                data=image_base64,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=12.0
            )
            if fb_resp.status_code == 200:
                response = fb_resp
                selected_version = ROBOFLOW_FALLBACK_VERSION

        if response.status_code != 200:
            return JSONResponse(
                status_code=response.status_code,
                content={"error": f"Roboflow API error ({response.status_code}): {response.text}"}
            )

        res_json = response.json()
        predictions = res_json.get("predictions", [])
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to connect to Roboflow server: {str(e)}"}
        )

    telemetry = parse_roboflow_predictions(predictions, w, h)
    telemetry["inference_time_ms"] = round((time.time() - t0) * 1000, 1)
    telemetry["model_engine"] = f"Roboflow Cloud ({ROBOFLOW_PROJECT}/{selected_version})"

    annotated_base64 = None
    if return_annotated:
        annotated_frame = annotate_roboflow_frame(frame, telemetry["detections"])
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        _, buffer = cv2.imencode('.jpg', annotated_frame, encode_param)
        annotated_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

    return {
        "telemetry": telemetry,
        "annotated_image": annotated_base64,
        "width": w,
        "height": h,
        "model_version": selected_version
    }

# ==========================================
# WebSocket for high-speed Browser Webcam Streaming
# ==========================================
@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """
    Bidirectional streaming over WebSocket:
    Browser sends binary JPEG frame or JSON frame,
    Server responds immediately with telemetry and annotated frame.
    """
    await websocket.accept()
    try:
        while True:
            # Receive frame data (bytes)
            data = await websocket.receive_bytes()
            t0 = time.time()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_json({"error": "Decode error"})
                continue

            h, w = frame.shape[:2]
            results = model.predict(source=frame, conf=0.45, verbose=False)
            result = results[0]

            telemetry = parse_yolo_result(result, w, h)
            telemetry["inference_time_ms"] = round((time.time() - t0) * 1000, 1)

            annotated_frame = result.plot()
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            _, buffer = cv2.imencode('.jpg', annotated_frame, encode_param)
            b64_img = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

            await websocket.send_json({
                "telemetry": telemetry,
                "annotated_image": b64_img
            })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS Error]: {e}")

import os
from fastapi.staticfiles import StaticFiles

# Mount static build files from frontend/dist if built
dist_dir = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.exists(dist_dir):
    app.mount("/", StaticFiles(directory=dist_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)

