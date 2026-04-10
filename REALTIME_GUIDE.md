# 🦺 PPE Guard — Real-Time Detection System
### Complete Guide for Construction Site Live Camera Feeds

---

## 📦 New Files (add these to your project)

```
ppe_detector/
├── realtime_pipeline.py   ← Core: threads, tracker, person-PPE association, HUD
├── train_optimized.py     ← Upgraded training with augmentation tuned for sites
├── app_realtime.py        ← Flask backend with MJPEG stream + counts API
├── optimize.py            ← Benchmarking, ONNX export, TensorRT, INT8
└── (existing files kept)
```

---

## ⚡ Quick Start — Live Camera

```bash
# 1. Install dependencies
pip install ultralytics opencv-python flask numpy

# 2. Run live detection (webcam)
python realtime_pipeline.py --source 0

# 3. Or with an IP camera
python realtime_pipeline.py --source rtsp://192.168.1.64/stream

# 4. Keyboard shortcuts:
#    Q = quit
#    S = save snapshot
```

---

## 🌐 Web Server with Live Stream

```bash
# 1. Start server
python app_realtime.py

# 2. Start camera via API
curl -X POST http://localhost:5000/api/start_camera \
     -H "Content-Type: application/json" \
     -d '{"source": 0, "model": "models/best.pt"}'

# 3. Embed live stream in your HTML:
#    <img src="http://localhost:5000/stream">

# 4. Poll counts every 500ms:
#    GET http://localhost:5000/api/counts
```

Response from `/api/counts`:
```json
{
  "fps": 22.4,
  "total_workers": 3,
  "compliant": 2,
  "violations": 1,
  "helmets": 3,
  "vests": 2,
  "goggles": 2,
  "compliance_rate": 66.7
}
```

---

## 🤖 Training (Optimized)

```bash
# Auto-download dataset + train
export ROBOFLOW_API_KEY=your_key_here
python train_optimized.py --download

# Or manually place dataset in ./dataset/ then:
python train_optimized.py

# After training, export for faster inference:
python optimize.py --export-onnx
```

---

## 🚀 Performance Optimization

### Step 1: Benchmark your hardware
```bash
python optimize.py --benchmark --model models/best.pt
```

### Step 2: Find optimal resolution
```bash
python optimize.py --benchmark-all
```

### Step 3: Export to ONNX (~2x faster on CPU)
```bash
python optimize.py --export-onnx
# Then use: python realtime_pipeline.py --model models/best.onnx
```

### Step 4: TensorRT (GPU only — 3-5x faster)
```bash
python optimize.py --export-trt
```

---

## 📊 Hardware Performance Guide

| Hardware | Model | Size | Expected FPS |
|---|---|---|---|
| Laptop CPU (i5/i7) | yolov8n | 416 | ~12-18 FPS |
| Desktop CPU (i9) | yolov8n | 640 | ~15-20 FPS |
| NVIDIA RTX 3060 | yolov8s | 640 | ~80-120 FPS |
| Jetson Xavier NX | yolov8s | 640 | ~25-35 FPS |

---

## 🎯 Model Architecture Decision

**Use YOLOv8n for real-time on CPU, YOLOv8s if you have a GPU.**

Single-stage detection (all PPE in one model) vs cascaded (person → PPE):
- **Single-stage** (recommended): simpler, faster, good enough
- **Cascaded**: slightly better accuracy but 2x slower — use only if missing detections are high

---

## 🏷️ Annotation Strategy

**Use separate objects** (not person-attributes):

```
Class 0: helmet
Class 1: goggles
Class 2: ear_protection
Class 3: gloves
Class 4: safety_shoes
Class 5: vest
Class 6: face_shield
Class 7: person
```

The `PPEAssociator` in `realtime_pipeline.py` handles matching PPE → persons automatically.

**Dataset minimum:** 500 images for PoC, 2000+ for production.

**Annotation tips:**
- Include negative examples (no PPE)
- Annotate even partially visible PPE
- Include different: lighting, angles, distances, weather
- If using Roboflow, enable `Auto-augmentation` — saves time

---

## 🔍 Person-PPE Association Logic

The `PPEAssociator` in `realtime_pipeline.py`:

1. Expands each person bbox by 30% margin
2. Checks if PPE center-point falls inside expanded person box
3. If multiple persons qualify → assigns to nearest by center distance
4. Falls back to proximity check (150px) if no overlap found
5. Labels each worker as compliant only if **helmet + vest** both detected

To change compliance rules (e.g. require goggles too), edit line in `realtime_pipeline.py`:
```python
w.compliant = w.has_helmet and w.has_vest  # add: and w.has_goggles
```

---

## 🔄 Tracking

The `SimpleTracker` in `realtime_pipeline.py` uses IoU-based matching — no extra dependencies.

For production with many crossing paths, upgrade to ByteTrack (built into ultralytics):
```python
# In InferenceEngine.infer():
results = model.track(frame, persist=True, tracker="bytetrack.yaml")
# Use results[0].boxes.id for track IDs
```

---

## 📡 Multiple Cameras

To handle multiple camera streams simultaneously:

```python
from app_realtime import CameraManager

cam1 = CameraManager()
cam2 = CameraManager()

cam1.start(source=0, model_path="models/best.pt")
cam2.start(source="rtsp://192.168.1.64/stream")

# Each manager runs its own processing thread
# Expose /stream1 and /stream2 on different endpoints
```

---

## 🚨 Alert System Integration

Violations are logged in `camera.violation_log`. 
To send alerts (e.g., email/Telegram/webhook):

```python
# In CameraManager._loop(), after violation detection:
import requests

if counts["violations"] > 0:
    requests.post("https://hooks.slack.com/...", json={
        "text": f"⚠️ PPE Violation: {counts['violations']} workers not compliant"
    })
```

---

## 🐳 Docker Deployment

```dockerfile
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "app_realtime:app", "--workers=1", "--threads=4", "--bind=0.0.0.0:5000"]
```

> ⚠️ Use `--workers=1` — the CameraManager is not fork-safe. Use `--threads` instead.

---

## 🐞 Troubleshooting

| Problem | Fix |
|---|---|
| Low FPS on CPU | Use `--size 416` and export to ONNX |
| RTSP camera lag | Set `cv2.CAP_PROP_BUFFERSIZE = 1` (already done in CaptureThread) |
| PPE not matching workers | Increase `expand_margin` in PPEAssociator (default 0.3) |
| False positives on ground PPE | Raise confidence threshold: `conf=0.45` |
| Workers not tracked | Switch to ByteTrack: `model.track(persist=True)` |
| Multiple workers merged | Lower `iou_threshold` in SimpleTracker (default 0.3) |

---

*Built on YOLOv8 by Ultralytics — Project by Sarthak Shirdhankar*
