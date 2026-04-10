import os
import cv2
import json
import base64
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response
from PIL import Image
import io
import time
import threading
from collections import deque

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'

model = None
USE_MOCK = False

# ── Construction-PPE 11-Class System ──────────────────────────────────────────
# Matches Ultralytics construction-ppe.yaml exactly
# Positive classes (PPE worn) → 0-4 + 6
# Negative classes (PPE missing) → 7-10 + 5 (none = no PPE at all)
PPE_CLASSES = {
    0:  {"name": "Helmet",      "icon": "🪖", "color": "#FFD700", "key": "helmet",   "type": "positive", "pair": 7},
    1:  {"name": "Gloves",      "icon": "🧤", "color": "#A8FF78", "key": "gloves",   "type": "positive", "pair": 9},
    2:  {"name": "Vest",        "icon": "🦺", "color": "#FF4FCE", "key": "vest",     "type": "positive", "pair": None},
    3:  {"name": "Boots",       "icon": "👟", "color": "#FF9500", "key": "boots",    "type": "positive", "pair": 10},
    4:  {"name": "Goggles",     "icon": "🥽", "color": "#00FFFF", "key": "goggles",  "type": "positive", "pair": 8},
    5:  {"name": "No PPE",      "icon": "🚫", "color": "#FF3333", "key": "none",     "type": "negative", "pair": None},
    6:  {"name": "Person",      "icon": "🧍", "color": "#B0BEC5", "key": "person",   "type": "neutral",  "pair": None},
    7:  {"name": "No Helmet",   "icon": "⛔", "color": "#FF3333", "key": "no_helmet","type": "negative", "pair": 0},
    8:  {"name": "No Goggles",  "icon": "⛔", "color": "#FF6B00", "key": "no_goggle","type": "negative", "pair": 4},
    9:  {"name": "No Gloves",   "icon": "⛔", "color": "#FF9900", "key": "no_gloves","type": "negative", "pair": 1},
    10: {"name": "No Boots",    "icon": "⛔", "color": "#FFCC00", "key": "no_boots", "type": "negative", "pair": 3},
}

# PPE items that have wearing/not-wearing counters
PPE_TRACKED = [
    {"key": "helmet",  "label": "Helmet",  "icon": "🪖", "pos_id": 0, "neg_id": 7},
    {"key": "vest",    "label": "Vest",    "icon": "🦺", "pos_id": 2, "neg_id": None},
    {"key": "goggles", "label": "Goggles", "icon": "🥽", "pos_id": 4, "neg_id": 8},
    {"key": "gloves",  "label": "Gloves",  "icon": "🧤", "pos_id": 1, "neg_id": 9},
    {"key": "boots",   "label": "Boots",   "icon": "👟", "pos_id": 3, "neg_id": 10},
]

# Roboflow / Kaggle class name → our class ID mapping
ROBOFLOW_CLASSES = {
    # Helmets
    "hardhat": 0, "helmet": 0, "hard hat": 0, "safety helmet": 0,
    # Gloves
    "gloves": 1, "safety gloves": 1, "hand": 1,
    # Vests
    "vest": 2, "safety vest": 2,
    # Boots / shoes
    "boots": 3, "safety shoes": 3, "shoes": 3, "footwear": 3,
    # Goggles
    "glasses": 4, "goggles": 4, "safety glasses": 4, "safety goggles": 4,
    # No PPE at all
    "none": 5,
    # Person
    "person": 6, "worker": 6,
    # Violations
    "no-hardhat": 7, "no hardhat": 7, "no helmet": 7, "no-helmet": 7,
    "no-goggle": 8, "no goggle": 8, "no goggles": 8, "no-goggles": 8,
    "no-gloves": 9, "no gloves": 9,
    "no-boots": 10, "no boots": 10,
    # Kaggle dataset extras
    "no-safety vest": 2,   # treat as vest being mapped
    "safety cone": 5, "machinery": 5, "vehicle": 5,
    "mask": 5, "face shield": 5,
    # Positive re-maps for Kaggle dataset
    "safety vest": 2,
}

# ── Model Loading ──────────────────────────────────────────────────────────────
from ultralytics import YOLO

def load_model():
    global model
    model = YOLO("yolov8n.pt")
    print("[MODEL] Loaded yolov8n.pt successfully")

# ── Class Name Mapper ──────────────────────────────────────────────────────────
def map_class(name, cid):
    """Map any model's class name → our 11-class PPE ID."""
    ln = name.lower().strip()
    # Exact match first
    if ln in ROBOFLOW_CLASSES:
        return ROBOFLOW_CLASSES[ln]
    # Partial match
    for k, v in ROBOFLOW_CLASSES.items():
        if k in ln:
            return v
    # Fallback: clamp to valid range
    return min(cid, 10)

# ── Per-PPE Statistics Builder ─────────────────────────────────────────────────
def build_ppe_stats(detections, workers):
    """
    Returns structured per-PPE stats:
    {
      "helmet":  {"wearing": 3, "not_wearing": 1, "total": 4, "pct": 75},
      "vest":    {...},
      ...
    }
    Also total workers + overall compliance.
    """
    stats = {}
    for ppe in PPE_TRACKED:
        key = ppe["key"]
        # Count from worker associations
        wearing = sum(1 for w in workers if w.get(f"has_{key}", False))
        # Count explicit negative detections (no-helmet, no-goggles, etc.)
        neg_id = ppe["neg_id"]
        explicit_neg = sum(1 for d in detections if d["class_id"] == neg_id) if neg_id is not None else 0
        total_workers = len(workers)
        not_wearing = max(total_workers - wearing, explicit_neg)
        pct = round(wearing / total_workers * 100) if total_workers > 0 else 0
        stats[key] = {
            "label":       ppe["label"],
            "icon":        ppe["icon"],
            "wearing":     wearing,
            "not_wearing": not_wearing,
            "total":       total_workers,
            "pct":         pct,
        }
    return stats

# ── Person-PPE Association ─────────────────────────────────────────────────────
def associate_ppe_to_persons(raw_detections, expand=0.25, max_dist=200):
    persons   = [d for d in raw_detections if d["class_id"] == 6]
    ppe_items = [d for d in raw_detections if d["class_id"] != 6]

    workers = []
    for i, p in enumerate(persons):
        x1, y1, x2, y2 = p["bbox"]
        workers.append({
            "id":           i + 1,
            "bbox":         p["bbox"],
            "person_conf":  p["confidence"],
            "has_helmet":   False,
            "has_vest":     False,
            "has_goggles":  False,
            "has_gloves":   False,
            "has_boots":    False,
            "violations":   [],   # list of negative classes found on this worker
            "ppe_found":    [],
            "confidence":   {},
            "compliant":    False,
        })

    for ppe in ppe_items:
        px = (ppe["bbox"][0] + ppe["bbox"][2]) // 2
        py = (ppe["bbox"][1] + ppe["bbox"][3]) // 2

        best_wid, best_score = None, float("inf")
        for w in workers:
            bx1, by1, bx2, by2 = w["bbox"]
            bw = bx2 - bx1; bh = by2 - by1
            ex1 = bx1 - int(bw * expand)
            ey1 = by1 - int(bh * expand)
            ex2 = bx2 + int(bw * expand)
            ey2 = by2 + int(bh * expand)
            inside = ex1 <= px <= ex2 and ey1 <= py <= ey2
            cx = (bx1 + bx2) // 2; cy = (by1 + by2) // 2
            dist = ((px - cx)**2 + (py - cy)**2) ** 0.5
            if inside and dist < best_score:
                best_score = dist; best_wid = w["id"]
            elif not inside and dist < max_dist and dist < best_score:
                best_score = dist; best_wid = w["id"]

        if best_wid is not None:
            target = next(w for w in workers if w["id"] == best_wid)
            key = PPE_CLASSES.get(ppe["class_id"], {}).get("key", "")
            cls_type = PPE_CLASSES.get(ppe["class_id"], {}).get("type", "")
            target["ppe_found"].append(ppe["class_name"])
            target["confidence"][key] = ppe["confidence"]

            if cls_type == "positive":
                if key == "helmet":   target["has_helmet"]  = True
                elif key == "vest":   target["has_vest"]    = True
                elif key == "goggles":target["has_goggles"] = True
                elif key == "gloves": target["has_gloves"]  = True
                elif key == "boots":  target["has_boots"]   = True
            elif cls_type == "negative":
                target["violations"].append(ppe["class_name"])

    for w in workers:
        # Compliant = must have helmet AND vest at minimum
        w["compliant"] = w["has_helmet"] and w["has_vest"]

    return workers, ppe_items

# ── Detection ──────────────────────────────────────────────────────────────────
def run_detection(image_np):
    if USE_MOCK or model is None:
        return mock_detection(image_np)
    try:
        results = model(image_np, conf=0.25, iou=0.45, verbose=False)
        dets = []
        for r in results:
            for box in r.boxes:
                cid   = int(box.cls[0])
                cname = r.names.get(cid, f"class_{cid}")
                mid   = map_class(cname, cid)
                info  = PPE_CLASSES.get(mid, PPE_CLASSES[6])
                conf  = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                dets.append({
                    "class_id":       mid,
                    "class_name":     info["name"],
                    "icon":           info["icon"],
                    "color":          info["color"],
                    "type":           info["type"],
                    "confidence":     round(conf * 100, 1),
                    "bbox":           [x1, y1, x2, y2],
                    "original_class": cname,
                })
        return dets
    except Exception as e:
        print(f"[DETECT] {e}")
        return mock_detection(image_np)

def mock_detection(image_np):
    """Demo mode: simulates 3 workers with realistic PPE split."""
    h, w = image_np.shape[:2]
    return [
        # Workers
        {"class_id":6,"class_name":"Person","icon":"🧍","color":"#B0BEC5","type":"neutral","confidence":91.7,"bbox":[int(w*.05),int(h*.1),int(w*.3),int(h*.95)],"original_class":"person"},
        {"class_id":6,"class_name":"Person","icon":"🧍","color":"#B0BEC5","type":"neutral","confidence":88.2,"bbox":[int(w*.35),int(h*.08),int(w*.65),int(h*.95)],"original_class":"person"},
        {"class_id":6,"class_name":"Person","icon":"🧍","color":"#B0BEC5","type":"neutral","confidence":82.1,"bbox":[int(w*.68),int(h*.1),int(w*.95),int(h*.9)],"original_class":"person"},
        # Worker 1: Helmet + Vest ✓
        {"class_id":0,"class_name":"Helmet","icon":"🪖","color":"#FFD700","type":"positive","confidence":87.3,"bbox":[int(w*.08),int(h*.08),int(w*.27),int(h*.22)],"original_class":"helmet"},
        {"class_id":2,"class_name":"Vest","icon":"🦺","color":"#FF4FCE","type":"positive","confidence":83.1,"bbox":[int(w*.08),int(h*.25),int(w*.28),int(h*.6)],"original_class":"vest"},
        # Worker 2: Helmet + Vest + Goggles ✓
        {"class_id":0,"class_name":"Helmet","icon":"🪖","color":"#FFD700","type":"positive","confidence":79.4,"bbox":[int(w*.38),int(h*.06),int(w*.62),int(h*.2)],"original_class":"helmet"},
        {"class_id":2,"class_name":"Vest","icon":"🦺","color":"#FF4FCE","type":"positive","confidence":85.2,"bbox":[int(w*.38),int(h*.25),int(w*.62),int(h*.6)],"original_class":"vest"},
        {"class_id":4,"class_name":"Goggles","icon":"🥽","color":"#00FFFF","type":"positive","confidence":71.0,"bbox":[int(w*.40),int(h*.17),int(w*.60),int(h*.25)],"original_class":"goggles"},
        # Worker 3: NO Helmet violation ✗
        {"class_id":7,"class_name":"No Helmet","icon":"⛔","color":"#FF3333","type":"negative","confidence":76.5,"bbox":[int(w*.70),int(h*.08),int(w*.93),int(h*.22)],"original_class":"no_helmet"},
        {"class_id":2,"class_name":"Vest","icon":"🦺","color":"#FF4FCE","type":"positive","confidence":80.0,"bbox":[int(w*.70),int(h*.25),int(w*.93),int(h*.6)],"original_class":"vest"},
    ]

# ── Image Annotation ───────────────────────────────────────────────────────────
def annotate_image(image_np, workers, ppe_items, detections=None):
    out = image_np.copy()

    def hex2bgr(h):
        h = h.lstrip("#")
        r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return (b, g, r)

    # Draw PPE boxes (thin)
    for p in ppe_items:
        x1, y1, x2, y2 = p["bbox"]
        col = hex2bgr(p["color"])
        cv2.rectangle(out, (x1,y1), (x2,y2), col, 1)
        label = f"{p['class_name']} {p['confidence']}%"
        cv2.putText(out, label, (x1, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

    # Draw worker boxes
    for w in workers:
        x1, y1, x2, y2 = w["bbox"]
        col = (0,200,80) if w["compliant"] else (0,60,220)
        cv2.rectangle(out, (x1,y1), (x2,y2), col, 3)

        label = f"W{w['id']} {'✓ SAFE' if w['compliant'] else '✗ VIOLATION'}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (x1, y1-th-10), (x1+tw+8, y1), col, -1)
        cv2.putText(out, label, (x1+4, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)

        # PPE status dots
        dot_y = y1 + 14
        items = [("H", w["has_helmet"]), ("V", w["has_vest"]), ("G", w["has_goggles"]),
                 ("GL", w["has_gloves"]), ("B", w["has_boots"])]
        for j, (lbl, found) in enumerate(items):
            dc = (0,200,80) if found else (0,60,220)
            cx = x1 + 12 + j*28
            cv2.circle(out, (cx, dot_y), 10, dc, -1)
            cv2.putText(out, lbl, (cx-6, dot_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1, cv2.LINE_AA)

    # ── HUD Panel ──────────────────────────────────────────────────────────────
    total     = len(workers)
    compliant = sum(1 for w in workers if w["compliant"])
    helmets   = sum(1 for w in workers if w["has_helmet"])
    vests     = sum(1 for w in workers if w["has_vest"])
    goggles   = sum(1 for w in workers if w["has_goggles"])
    gloves    = sum(1 for w in workers if w["has_gloves"])
    boots     = sum(1 for w in workers if w["has_boots"])
    rate      = int(compliant / total * 100) if total else 0

    panel_w, panel_h = 255, 230
    overlay = out.copy()
    cv2.rectangle(overlay, (8,8), (8+panel_w, 8+panel_h), (15,15,15), -1)
    cv2.addWeighted(overlay, 0.78, out, 0.22, 0, out)

    def hud_line(label, val_str, good, y):
        col = (0,200,80) if good else (0,60,220)
        cv2.putText(out, label, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200,200,200), 1, cv2.LINE_AA)
        cv2.putText(out, val_str, (210, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)

    cv2.putText(out, "PPE GUARD", (16,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,220,120), 2, cv2.LINE_AA)
    hud_line("Workers:",  str(total),              True,                50)
    hud_line("Helmets:",  f"{helmets}/{total}",     helmets==total,      73)
    hud_line("Vests:",    f"{vests}/{total}",       vests==total,        96)
    hud_line("Goggles:",  f"{goggles}/{total}",     goggles==total,     119)
    hud_line("Gloves:",   f"{gloves}/{total}",      gloves==total,      142)
    hud_line("Boots:",    f"{boots}/{total}",       boots==total,       165)
    hud_line("Compliant:",f"{compliant}/{total}",   compliant==total,   188)
    hud_line("Rate:",     f"{rate}%",               rate==100,          211)

    if total > 0 and compliant < total:
        viol_count = total - compliant
        cv2.putText(out, f"!! {viol_count} VIOLATION(S) !!", (16, 225),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,60,220), 2, cv2.LINE_AA)

    return out

def img2b64(img):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode()

def load_image_bytes(fb, filename=""):
    try:
        nparr = np.frombuffer(fb, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            return img
        pil = Image.open(io.BytesIO(fb)).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except:
        return None

# ── Camera State ───────────────────────────────────────────────────────────────
camera_state = {
    "running": False, "cap": None,
    "latest_annotated": None, "latest_counts": {},
    "thread": None, "lock": threading.Lock(),
}

def camera_loop():
    cs = camera_state
    fps_times = deque(maxlen=20)
    while cs["running"]:
        cap = cs.get("cap")
        if not cap or not cap.isOpened():
            time.sleep(0.05); continue
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02); continue
        t0 = time.perf_counter()
        dets = run_detection(frame)
        workers, ppe_items = associate_ppe_to_persons(dets)
        ann = annotate_image(frame, workers, ppe_items, dets)
        fps_times.append(time.perf_counter() - t0)
        fps = len(fps_times) / sum(fps_times) if fps_times else 0

        total     = len(workers)
        compliant = sum(1 for w in workers if w["compliant"])
        ppe_stats = build_ppe_stats(dets, workers)

        with cs["lock"]:
            cs["latest_annotated"] = ann
            cs["latest_counts"] = {
                "fps": round(fps, 1),
                "total": total,
                "compliant": compliant,
                "violations": total - compliant,
                "rate": int(compliant/total*100) if total else 0,
                "ppe_stats": ppe_stats,
                "workers": [
                    {"id": w["id"], "compliant": w["compliant"],
                     "has_helmet": w["has_helmet"], "has_vest": w["has_vest"],
                     "has_goggles": w["has_goggles"], "has_gloves": w["has_gloves"],
                     "has_boots": w["has_boots"], "violations": w["violations"],
                     "ppe_found": w["ppe_found"]}
                    for w in workers
                ],
            }

def gen_stream():
    while True:
        with camera_state["lock"]:
            frame = camera_state.get("latest_annotated")
        if frame is None:
            ph = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(ph, "Waiting for camera...", (140,240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100,100,100), 2)
            frame = ph
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        time.sleep(0.04)

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", demo_mode=USE_MOCK)

@app.route("/stream")
def stream():
    from flask import Response, stream_with_context
    return Response(stream_with_context(gen_stream()),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/camera/start", methods=["POST"])
def camera_start():
    cs = camera_state
    if cs["running"]:
        return jsonify({"status": "already_running"})
    data   = request.json or {}
    source = data.get("source", 0)
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        return jsonify({"error": "Cannot open camera"}), 400
    cs["cap"] = cap; cs["running"] = True
    t = threading.Thread(target=camera_loop, daemon=True)
    cs["thread"] = t; t.start()
    return jsonify({"status": "started"})

@app.route("/api/camera/stop", methods=["POST"])
def camera_stop():
    cs = camera_state
    cs["running"] = False
    if cs.get("cap"):
        cs["cap"].release(); cs["cap"] = None
    with cs["lock"]:
        cs["latest_annotated"] = None; cs["latest_counts"] = {}
    return jsonify({"status": "stopped"})

@app.route("/api/camera/counts")
def camera_counts():
    with camera_state["lock"]:
        return jsonify({"success": True, "data": camera_state["latest_counts"]})

@app.route("/api/detect", methods=["POST"])
def detect():
    try:
        if request.is_json:
            data = request.json
            img_data = data.get("image", "")
            if "," in img_data: img_data = img_data.split(",")[1]
            fb = base64.b64decode(img_data)
            image_np = load_image_bytes(fb, ".jpg")
        else:
            f = request.files.get("file")
            if not f: return jsonify({"error": "No file"}), 400
            image_np = load_image_bytes(f.read(), f.filename)

        if image_np is None:
            return jsonify({"error": "Cannot read image"}), 400

        dets = run_detection(image_np)
        workers, ppe_items = associate_ppe_to_persons(dets)
        ann = annotate_image(image_np, workers, ppe_items, dets)

        total     = len(workers)
        compliant = sum(1 for w in workers if w["compliant"])
        ppe_stats = build_ppe_stats(dets, workers)

        return jsonify({
            "success":         True,
            "annotated_image": f"data:image/jpeg;base64,{img2b64(ann)}",
            "detections":      dets,
            "workers":         workers,
            "count":           len(dets),
            "summary": {
                "total_workers":   total,
                "compliant":       compliant,
                "violations":      total - compliant,
                "compliance_rate": int(compliant/total*100) if total else 0,
                # Per-PPE breakdown
                "ppe_stats":       ppe_stats,
                # Legacy flat counts
                "helmets":  ppe_stats["helmet"]["wearing"],
                "vests":    ppe_stats["vest"]["wearing"],
                "goggles":  ppe_stats["goggles"]["wearing"],
                "gloves":   ppe_stats["gloves"]["wearing"],
                "boots":    ppe_stats["boots"]["wearing"],
            },
            "demo_mode": USE_MOCK,
        })
    except Exception as e:
        print(f"[API] {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/detect_video_frame", methods=["POST"])
def detect_video_frame():
    try:
        data = request.json
        img_data = data.get("frame", "")
        if "," in img_data: img_data = img_data.split(",")[1]
        nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_np is None: return jsonify({"error": "Invalid frame"}), 400

        dets = run_detection(image_np)
        workers, ppe_items = associate_ppe_to_persons(dets)
        ann = annotate_image(image_np, workers, ppe_items, dets)

        total     = len(workers)
        compliant = sum(1 for w in workers if w["compliant"])
        ppe_stats = build_ppe_stats(dets, workers)

        return jsonify({
            "success":         True,
            "detections":      dets,
            "workers":         workers,
            "annotated_frame": f"data:image/jpeg;base64,{img2b64(ann)}",
            "count":           len(dets),
            "summary": {
                "total_workers":   total,
                "compliant":       compliant,
                "violations":      total - compliant,
                "compliance_rate": int(compliant/total*100) if total else 0,
                "ppe_stats":       ppe_stats,
                "helmets":  ppe_stats["helmet"]["wearing"],
                "vests":    ppe_stats["vest"]["wearing"],
                "goggles":  ppe_stats["goggles"]["wearing"],
                "gloves":   ppe_stats["gloves"]["wearing"],
                "boots":    ppe_stats["boots"]["wearing"],
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/status")
def status():
    return jsonify({
        "model_loaded":    model is not None,
        "demo_mode":       USE_MOCK,
        "camera_running":  camera_state["running"],
        "ppe_classes":     [{**v, "id": k} for k, v in PPE_CLASSES.items()],
        "ppe_tracked":     PPE_TRACKED,
        "dataset":         "construction-ppe (11 classes)",
    })

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("models",  exist_ok=True)
    load_model()
    print("\n🦺 PPE Guard running → http://localhost:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
