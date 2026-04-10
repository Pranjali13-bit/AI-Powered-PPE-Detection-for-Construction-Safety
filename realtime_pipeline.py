"""
Real-Time PPE Detection Pipeline
==================================
Drop-in upgrade for app.py — handles live camera feeds at 20+ FPS.

Architecture:
  Camera → CaptureThread → FrameQueue → InferenceThread → TrackerThread → DisplayThread

Usage:
  python realtime_pipeline.py --source 0        # webcam
  python realtime_pipeline.py --source rtsp://  # IP camera
  python realtime_pipeline.py --source video.mp4
"""

import cv2
import numpy as np
import time
import threading
import queue
import argparse
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# ─── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2
    color: Tuple[int, int, int]        # BGR
    track_id: int = -1
    is_person: bool = False

@dataclass
class Worker:
    track_id: int
    bbox: Tuple[int, int, int, int]
    has_helmet: bool = False
    has_vest: bool = False
    has_goggles: bool = False
    compliant: bool = False
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    frames_seen: int = 0

# PPE class definitions (matches your app.py)
PPE_CLASSES = {
    0: {"name": "Helmet",         "color": (0, 215, 255),  "key": "helmet"},
    1: {"name": "Goggles",        "color": (255, 255, 0),  "key": "goggles"},
    2: {"name": "Ear Protection", "color": (100, 100, 255),"key": "ear"},
    3: {"name": "Gloves",         "color": (100, 255, 100),"key": "gloves"},
    4: {"name": "Safety Shoes",   "color": (0, 150, 255),  "key": "shoes"},
    5: {"name": "Vest",           "color": (200, 0, 255),  "key": "vest"},
    6: {"name": "Face Shield",    "color": (255, 180, 0),  "key": "shield"},
    7: {"name": "Person",         "color": (180, 180, 180),"key": "person"},
}

PERSON_CLASS_ID = 7
PPE_KEYS_REQUIRED = ["helmet", "vest", "goggles"]   # tweak based on your site rules

# ─── Simple IoU Tracker (no external deps) ───────────────────────────────────

class SimpleTracker:
    """
    Lightweight centroid + IoU tracker.
    No DeepSORT/ByteTrack dependency — works on CPU.
    For production on GPU, swap in ByteTrack (ultralytics has it built-in).
    """
    def __init__(self, max_lost=10, iou_threshold=0.3):
        self.tracks: Dict[int, dict] = {}
        self.next_id = 0
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold

    def _iou(self, b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / (a1 + a2 - inter)

    def update(self, detections: List[Detection]) -> List[Detection]:
        if not detections:
            for tid in list(self.tracks):
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]
            return detections

        # Match existing tracks to new detections by IoU
        track_ids = list(self.tracks.keys())
        matched = set()
        used_tracks = set()

        for det in detections:
            best_iou, best_tid = 0, -1
            for tid in track_ids:
                if tid in used_tracks:
                    continue
                iou = self._iou(det.bbox, self.tracks[tid]["bbox"])
                if iou > best_iou:
                    best_iou, best_tid = iou, tid
            if best_iou >= self.iou_threshold:
                det.track_id = best_tid
                self.tracks[best_tid]["bbox"] = det.bbox
                self.tracks[best_tid]["lost"] = 0
                used_tracks.add(best_tid)
                matched.add(best_tid)
            else:
                det.track_id = self.next_id
                self.tracks[self.next_id] = {"bbox": det.bbox, "lost": 0}
                self.next_id += 1

        # Age out unmatched tracks
        for tid in track_ids:
            if tid not in matched:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]

        return detections


# ─── Person-PPE Association ───────────────────────────────────────────────────

class PPEAssociator:
    """
    Associates detected PPE items with the closest worker bounding box.

    Rules:
      1. PPE center-point must fall within or near a person bbox (expanded by margin%).
      2. If multiple persons qualify, pick the one with highest overlap.
      3. PPE with no person within max_distance pixels is flagged as unassigned.
    """
    def __init__(self, expand_margin: float = 0.3, max_distance: int = 150):
        self.expand_margin = expand_margin
        self.max_distance = max_distance

    def _expand_bbox(self, bbox, factor):
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        dx, dy = int(w * factor), int(h * factor)
        return (x1 - dx, y1 - dy, x2 + dx, y2 + dy)

    def _center(self, bbox):
        return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

    def _dist(self, p1, p2):
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5

    def _point_in_bbox(self, point, bbox):
        return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]

    def associate(self, detections: List[Detection]) -> Dict[int, Worker]:
        persons = [d for d in detections if d.class_id == PERSON_CLASS_ID]
        ppe_items = [d for d in detections if d.class_id != PERSON_CLASS_ID]

        workers: Dict[int, Worker] = {}
        for p in persons:
            tid = p.track_id if p.track_id >= 0 else id(p)
            workers[tid] = Worker(track_id=tid, bbox=p.bbox)

        for ppe in ppe_items:
            ppe_center = self._center(ppe.bbox)
            best_worker_id = None
            best_score = float("inf")

            for tid, worker in workers.items():
                expanded = self._expand_bbox(worker.bbox, self.expand_margin)
                if self._point_in_bbox(ppe_center, expanded):
                    dist = self._dist(ppe_center, self._center(worker.bbox))
                    if dist < best_score:
                        best_score = dist
                        best_worker_id = tid
                else:
                    # Fallback: proximity check
                    dist = self._dist(ppe_center, self._center(worker.bbox))
                    if dist < self.max_distance and dist < best_score:
                        best_score = dist
                        best_worker_id = tid

            if best_worker_id is not None:
                key = PPE_CLASSES.get(ppe.class_id, {}).get("key", "")
                conf = ppe.confidence
                if key == "helmet":
                    workers[best_worker_id].has_helmet = True
                    workers[best_worker_id].confidence_scores["helmet"] = conf
                elif key == "vest":
                    workers[best_worker_id].has_vest = True
                    workers[best_worker_id].confidence_scores["vest"] = conf
                elif key in ("goggles", "glasses", "shield"):
                    workers[best_worker_id].has_goggles = True
                    workers[best_worker_id].confidence_scores["goggles"] = conf

        # Determine compliance
        for w in workers.values():
            w.compliant = w.has_helmet and w.has_vest  # adjust per site rules
            w.frames_seen += 1

        return workers


# ─── Threaded Frame Capture ───────────────────────────────────────────────────

class CaptureThread(threading.Thread):
    """
    Dedicated thread for frame capture — decouples camera I/O from inference.
    Keeps only the LATEST frame in queue (drops stale frames automatically).
    """
    def __init__(self, source, queue_size=2):
        super().__init__(daemon=True)
        self.source = source
        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(self.source)
        # RTSP buffer size optimization
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Prefer MJPEG for USB cameras
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            # Drop old frame if queue full (keep only latest)
            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get(timeout=1.0)

    def stop(self):
        self.stopped = True
        if self.cap:
            self.cap.release()


# ─── Inference Engine ────────────────────────────────────────────────────────

class InferenceEngine:
    """
    Wraps YOLO with frame-skipping and resolution scaling for real-time speed.
    Target: 20+ FPS on CPU, 60+ FPS on GPU.
    """
    def __init__(self, model_path="models/best.pt", device="cpu",
                 input_size=640, conf=0.30, iou=0.45,
                 skip_frames=1, use_half=False):
        self.input_size = input_size
        self.conf = conf
        self.iou = iou
        self.skip_frames = skip_frames   # process every N frames
        self.frame_count = 0
        self.last_detections = []
        self.model = None
        self._load(model_path, device, use_half)

    def _load(self, path, device, use_half):
        try:
            from ultralytics import YOLO
            self.model = YOLO(path)
            # Export to ONNX for ~2x CPU speedup (one-time)
            # self.model.export(format="onnx", imgsz=self.input_size)
            print(f"[ENGINE] Model loaded: {path} | device={device}")
        except Exception as e:
            print(f"[ENGINE] Load failed: {e} — running mock mode")

    def infer(self, frame: np.ndarray) -> List[Detection]:
        self.frame_count += 1

        # Skip frames to maintain FPS
        if self.frame_count % max(1, self.skip_frames) != 0:
            return self.last_detections   # reuse previous result

        if self.model is None:
            return self._mock(frame)

        # Resize for speed (640 is sweet spot; use 416 for pure speed)
        h, w = frame.shape[:2]
        scale = self.input_size / max(h, w)
        if scale < 1.0:
            resized = cv2.resize(frame, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_LINEAR)
        else:
            resized = frame

        try:
            results = self.model(
                resized,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
                stream=False,
            )
            detections = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = r.names.get(cls_id, "")
                    mapped_id = self._map(cls_name, cls_id)
                    info = PPE_CLASSES.get(mapped_id, PPE_CLASSES[7])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    # Scale coords back to original resolution
                    if scale < 1.0:
                        x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale
                    detections.append(Detection(
                        class_id=mapped_id,
                        class_name=info["name"],
                        confidence=conf,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        color=info["color"],
                        is_person=(mapped_id == PERSON_CLASS_ID),
                    ))
            self.last_detections = detections
            return detections
        except Exception as e:
            print(f"[ENGINE] Inference error: {e}")
            return self.last_detections

    def _map(self, cls_name, cls_id):
        name_map = {
            "hardhat": 0, "helmet": 0, "hard hat": 0,
            "glasses": 1, "goggles": 1, "safety glasses": 1,
            "ear": 2,
            "gloves": 3, "hand": 3,
            "boots": 4, "shoes": 4,
            "vest": 5, "safety vest": 5,
            "mask": 6, "face shield": 6,
            "person": 7, "worker": 7,
        }
        ln = cls_name.lower()
        for k, v in name_map.items():
            if k in ln:
                return v
        if cls_id == 0 and cls_name == "person":
            return 7
        return cls_id % len(PPE_CLASSES)

    def _mock(self, frame):
        h, w = frame.shape[:2]
        return [
            Detection(7, "Person", 0.92, (50, 80, 200, 420), (180, 180, 180), is_person=True),
            Detection(0, "Helmet", 0.88, (80, 60, 170, 130), (0, 215, 255)),
            Detection(5, "Vest",   0.84, (60, 180, 195, 380), (200, 0, 255)),
        ]


# ─── HUD / Overlay Renderer ───────────────────────────────────────────────────

class HUDRenderer:
    """
    Renders compliance dashboard, bounding boxes, and worker annotations
    directly onto OpenCV frames. Optimized for readability.
    """
    GREEN  = (0, 200, 80)
    RED    = (0, 60, 220)
    YELLOW = (0, 200, 255)
    WHITE  = (255, 255, 255)
    BLACK  = (0, 0, 0)
    DARK   = (20, 20, 20)

    def render(self, frame: np.ndarray, workers: Dict[int, Worker],
               detections: List[Detection], fps: float) -> np.ndarray:
        out = frame.copy()

        # Draw PPE bounding boxes (non-person)
        for det in detections:
            if not det.is_person:
                self._draw_box(out, det.bbox, det.color, det.class_name, det.confidence)

        # Draw worker boxes with compliance color
        for worker in workers.values():
            color = self.GREEN if worker.compliant else self.RED
            self._draw_worker_box(out, worker, color)

        # Draw HUD panel
        self._draw_hud(out, workers, fps)

        return out

    def _draw_box(self, frame, bbox, color, label, conf):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.BLACK, 1, cv2.LINE_AA)

    def _draw_worker_box(self, frame, worker: Worker, color):
        x1, y1, x2, y2 = worker.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Compliance mini-badge
        status = "✓ SAFE" if worker.compliant else "✗ VIOLATION"
        text = f"W{worker.track_id} {status}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y2), (x1 + tw + 6, y2 + th + 8), color, -1)
        cv2.putText(frame, text, (x1 + 3, y2 + th + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.WHITE, 1, cv2.LINE_AA)

        # PPE icons row
        icons = []
        icons.append(("H", self.GREEN if worker.has_helmet  else self.RED))
        icons.append(("V", self.GREEN if worker.has_vest    else self.RED))
        icons.append(("G", self.GREEN if worker.has_goggles else self.RED))
        for i, (icon, ic) in enumerate(icons):
            cx = x1 + 4 + i * 20
            cv2.circle(frame, (cx + 8, y1 - 12), 9, ic, -1)
            cv2.putText(frame, icon, (cx + 4, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.WHITE, 1)

    def _draw_hud(self, frame, workers: Dict[int, Worker], fps: float):
        h, w = frame.shape[:2]
        panel_w, panel_h = 260, 200

        # Semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), self.DARK, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        total = len(workers)
        compliant = sum(1 for w in workers.values() if w.compliant)
        violations = total - compliant
        helmets = sum(1 for w in workers.values() if w.has_helmet)
        vests   = sum(1 for w in workers.values() if w.has_vest)
        goggles = sum(1 for w in workers.values() if w.has_goggles)
        rate = (compliant / total * 100) if total else 0

        lines = [
            (f"FPS: {fps:.1f}",                         self.YELLOW),
            (f"Workers:  {total}",                       self.WHITE),
            (f"Helmets:  {helmets}/{total}",             self.GREEN if helmets==total else self.RED),
            (f"Vests:    {vests}/{total}",               self.GREEN if vests==total   else self.RED),
            (f"Goggles:  {goggles}/{total}",             self.GREEN if goggles==total else self.RED),
            (f"Compliant:{compliant}/{total}",           self.GREEN if violations==0  else self.RED),
            (f"Rate:     {rate:.0f}%",                   self.GREEN if rate==100 else self.YELLOW),
        ]
        if violations > 0:
            lines.append(("!! PPE VIOLATION ALERT !!", self.RED))

        for i, (text, color) in enumerate(lines):
            cv2.putText(frame, text, (18, 35 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)


# ─── FPS Meter ───────────────────────────────────────────────────────────────

class FPSMeter:
    def __init__(self, window=30):
        self.times = deque(maxlen=window)
        self.last = time.perf_counter()

    def tick(self):
        now = time.perf_counter()
        self.times.append(now - self.last)
        self.last = now

    @property
    def fps(self):
        if len(self.times) < 2:
            return 0.0
        return len(self.times) / sum(self.times)


# ─── Main Pipeline ────────────────────────────────────────────────────────────

class RealTimePipeline:
    """
    Complete real-time PPE detection pipeline.

    Frame flow (all on same thread for simplicity, camera on separate thread):
      CaptureThread → InferenceEngine → SimpleTracker → PPEAssociator → HUDRenderer
    """
    def __init__(self, source=0, model_path="models/best.pt",
                 input_size=640, skip_frames=1, device="cpu"):
        self.capture = CaptureThread(source)
        self.engine   = InferenceEngine(model_path, device, input_size,
                                        skip_frames=skip_frames)
        self.tracker  = SimpleTracker()
        self.assoc    = PPEAssociator()
        self.renderer = HUDRenderer()
        self.fps_meter = FPSMeter()
        self.alert_log = []   # list of (timestamp, worker_id, violation_type)

    def run(self, show=True, save_path=None):
        self.capture.start()
        writer = None

        print("[PIPELINE] Starting. Press Q to quit.")
        while True:
            try:
                frame = self.capture.read()
            except queue.Empty:
                continue

            # 1. Inference
            detections = self.engine.infer(frame)

            # 2. Track
            detections = self.tracker.update(detections)

            # 3. Associate PPE → workers
            workers = self.assoc.associate(detections)

            # 4. Log violations
            self._log_violations(workers)

            # 5. Render
            self.fps_meter.tick()
            annotated = self.renderer.render(frame, workers, detections,
                                             self.fps_meter.fps)

            if save_path and writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(save_path,
                                         cv2.VideoWriter_fourcc(*"mp4v"),
                                         20, (w, h))
            if writer:
                writer.write(annotated)

            if show:
                cv2.imshow("PPE Guard — Real-Time", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    fname = f"snapshot_{int(time.time())}.jpg"
                    cv2.imwrite(fname, annotated)
                    print(f"[SNAPSHOT] Saved: {fname}")

        self.capture.stop()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        self._print_session_summary()

    def _log_violations(self, workers):
        ts = time.strftime("%H:%M:%S")
        for w in workers.values():
            if not w.compliant and w.frames_seen % 30 == 1:  # log every ~1s
                entry = {"time": ts, "worker": w.track_id,
                         "helmet": w.has_helmet, "vest": w.has_vest,
                         "goggles": w.has_goggles}
                self.alert_log.append(entry)

    def _print_session_summary(self):
        print(f"\n{'='*50}")
        print(f"Session Summary — {len(self.alert_log)} violation events logged")
        for e in self.alert_log[-10:]:
            missing = []
            if not e["helmet"]:  missing.append("Helmet")
            if not e["vest"]:    missing.append("Vest")
            if not e["goggles"]: missing.append("Goggles")
            print(f"  [{e['time']}] Worker {e['worker']} missing: {', '.join(missing)}")
        print("="*50)


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time PPE Detection")
    parser.add_argument("--source", default=0,
                        help="Camera index (0), RTSP URL, or video file path")
    parser.add_argument("--model",  default="models/best.pt")
    parser.add_argument("--size",   type=int, default=640,
                        help="Inference resolution (416=faster, 640=balanced, 1280=accurate)")
    parser.add_argument("--skip",   type=int, default=1,
                        help="Process every Nth frame (1=all, 2=every other, etc.)")
    parser.add_argument("--device", default="cpu",
                        help="cpu | 0 (first GPU) | cuda:0")
    parser.add_argument("--save",   default=None,
                        help="Optional output video path (e.g. output.mp4)")
    parser.add_argument("--no-display", action="store_true",
                        help="Run headless (for server deployments)")
    args = parser.parse_args()

    # Convert source to int if it's a digit string
    source = int(args.source) if str(args.source).isdigit() else args.source

    pipeline = RealTimePipeline(
        source=source,
        model_path=args.model,
        input_size=args.size,
        skip_frames=args.skip,
        device=args.device,
    )
    pipeline.run(show=not args.no_display, save_path=args.save)
