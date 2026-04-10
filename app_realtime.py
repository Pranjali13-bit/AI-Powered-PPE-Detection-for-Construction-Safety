"""
app_realtime.py — Upgraded Flask backend with MJPEG stream endpoint
====================================================================
Replaces app.py for live camera deployments.

New endpoints:
  GET  /stream          — MJPEG stream (embed in <img src="/stream">)
  POST /api/start_camera — Start camera with config
  POST /api/stop_camera  — Stop camera
  GET  /api/counts       — Current worker/PPE counts (JSON, poll this)

Existing endpoints from app.py are preserved.
"""

import os
import cv2
import json
import base64
import numpy as np
import threading
import time
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from PIL import Image
import io

# Import the real-time pipeline components
from realtime_pipeline import (
    InferenceEngine, SimpleTracker, PPEAssociator,
    HUDRenderer, FPSMeter, Detection, Worker, PPE_CLASSES
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# ─── Camera Manager (singleton) ──────────────────────────────────────────────

class CameraManager:
    def __init__(self):
        self.cap = None
        self.running = False
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_annotated = None
        self.latest_counts = {}
        self.fps_meter = FPSMeter()
        self._thread = None

        # Pipeline components
        self.engine   = None
        self.tracker  = SimpleTracker()
        self.assoc    = PPEAssociator()
        self.renderer = HUDRenderer()
        self.violation_log = []

    def start(self, source=0, model_path="models/best.pt", input_size=640):
        if self.running:
            return {"status": "already_running"}

        self.engine = InferenceEngine(model_path, "cpu", input_size, skip_frames=1)
        self.cap = cv2.VideoCapture(source if isinstance(source, int)
                                    else source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return {"status": "started", "source": source}

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        return {"status": "stopped"}

    def _loop(self):
        while self.running:
            if not self.cap or not self.cap.isOpened():
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            with self.lock:
                self.latest_frame = frame

            # Run pipeline
            try:
                detections = self.engine.infer(frame)
                detections = self.tracker.update(detections)
                workers    = self.assoc.associate(detections)

                self.fps_meter.tick()
                annotated = self.renderer.render(
                    frame, workers, detections, self.fps_meter.fps
                )

                # Build counts
                total = len(workers)
                compliant = sum(1 for w in workers.values() if w.compliant)
                helmets  = sum(1 for w in workers.values() if w.has_helmet)
                vests    = sum(1 for w in workers.values() if w.has_vest)
                goggles  = sum(1 for w in workers.values() if w.has_goggles)

                counts = {
                    "fps": round(self.fps_meter.fps, 1),
                    "total_workers": total,
                    "compliant": compliant,
                    "violations": total - compliant,
                    "helmets": helmets,
                    "vests": vests,
                    "goggles": goggles,
                    "compliance_rate": round(compliant / total * 100, 1) if total else 0,
                }

                with self.lock:
                    self.latest_annotated = annotated
                    self.latest_counts = counts

                # Log violations
                if total - compliant > 0:
                    self.violation_log.append({
                        "time": time.strftime("%H:%M:%S"),
                        **counts
                    })

            except Exception as e:
                print(f"[CAM] Pipeline error: {e}")
                with self.lock:
                    self.latest_annotated = frame

    def get_jpeg_frame(self):
        """Encode latest annotated frame as JPEG bytes."""
        with self.lock:
            frame = self.latest_annotated
        if frame is None:
            # Return a placeholder frame
            ph = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(ph, "No camera feed", (180, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
            frame = ph
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return buf.tobytes()

    def get_counts(self):
        with self.lock:
            return dict(self.latest_counts)


camera = CameraManager()


# ─── MJPEG Stream Generator ───────────────────────────────────────────────────

def generate_stream():
    """Generator for MJPEG streaming."""
    while True:
        frame_bytes = camera.get_jpeg_frame()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes +
            b"\r\n"
        )
        time.sleep(0.033)  # ~30 FPS cap on stream


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/stream")
def stream():
    """MJPEG live stream. Use as: <img src='/stream'>"""
    return Response(
        stream_with_context(generate_stream()),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/api/start_camera", methods=["POST"])
def start_camera():
    data = request.json or {}
    source     = data.get("source", 0)
    model_path = data.get("model", "models/best.pt")
    input_size = data.get("input_size", 640)

    # Convert "0" → 0
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    result = camera.start(source, model_path, input_size)
    return jsonify(result)

@app.route("/api/stop_camera", methods=["POST"])
def stop_camera():
    return jsonify(camera.stop())

@app.route("/api/counts")
def get_counts():
    """Poll this endpoint every 500ms to update your dashboard."""
    return jsonify({
        "success": True,
        "data": camera.get_counts(),
        "timestamp": time.time(),
    })

@app.route("/api/violations")
def get_violations():
    """Last 50 violation events."""
    return jsonify({
        "success": True,
        "violations": camera.violation_log[-50:],
        "total": len(camera.violation_log),
    })

@app.route("/api/status")
def status():
    return jsonify({
        "camera_running": camera.running,
        "model_loaded": camera.engine is not None,
        "counts": camera.get_counts(),
    })

# ─── Single image detection (kept from original app.py) ──────────────────────

@app.route("/api/detect", methods=["POST"])
def detect():
    try:
        if request.is_json:
            data = request.json
            img_data = data.get("image", "")
            if "," in img_data:
                img_data = img_data.split(",")[1]
            img_bytes = base64.b64decode(img_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            file = request.files.get("file")
            if not file:
                return jsonify({"error": "No file provided"}), 400
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image_np is None:
            return jsonify({"error": "Could not read image"}), 400

        # Use camera engine if available, else create temporary one
        engine = camera.engine
        if engine is None:
            engine = InferenceEngine("models/best.pt")

        detections = engine.infer(image_np)
        workers    = PPEAssociator().associate(detections)

        annotated = HUDRenderer().render(image_np, workers, detections, fps=0)
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        result_b64 = base64.b64encode(buf).decode("utf-8")

        return jsonify({
            "success": True,
            "annotated_image": f"data:image/jpeg;base64,{result_b64}",
            "worker_count": len(workers),
            "detections": [
                {"class_name": d.class_name, "confidence": round(d.confidence * 100, 1),
                 "bbox": list(d.bbox)}
                for d in detections
            ],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    print("\n🦺 PPE Guard Real-Time Server")
    print("   Stream:  http://localhost:5000/stream")
    print("   Status:  http://localhost:5000/api/status")
    print("   Start:   POST http://localhost:5000/api/start_camera")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
