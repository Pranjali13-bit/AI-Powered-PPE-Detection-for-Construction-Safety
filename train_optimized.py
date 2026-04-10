"""
Optimized PPE Detection Training Script
=========================================
Trains YOLOv8 for real-time construction site PPE detection.

Class Structure Decision (answered for you):
  Use SEPARATE objects, NOT person-attributes.
  Reason: Easier to annotate, faster inference, works better with
  association logic in realtime_pipeline.py

Recommended Dataset (free):
  https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety-g9u2k

Annotation Classes:
  0: helmet
  1: goggles / safety_glasses
  2: ear_protection
  3: gloves
  4: safety_shoes
  5: vest
  6: face_shield
  7: person

Minimum dataset: 500 images (200 okay for PoC, 1000+ for production)
"""

import os
import sys
import shutil
import yaml


# ─── Training Configuration ──────────────────────────────────────────────────

# Adjust these based on your hardware
HARDWARE = "cpu"          # "cpu" | "0" (first GPU) | "0,1" (multi-GPU)
USE_GPU  = HARDWARE != "cpu"

# Auto-scale batch size
BATCH_SIZE = 8 if not USE_GPU else 16    # 16 fits in ~6GB VRAM; use 32 for 12GB+
IMG_SIZE   = 640                          # 416 for max speed; 640 balanced; 1280 accurate
EPOCHS     = 100
WORKERS    = 4 if USE_GPU else 2

# Model choice:
#   yolov8n.pt  → fastest  (2.1M params) — good for edge/CPU
#   yolov8s.pt  → balanced (11M params) — recommended starting point
#   yolov8m.pt  → accurate (25M params) — if GPU available
BASE_MODEL = "yolov8n.pt"                 # swap to yolov8s.pt for better accuracy


def create_data_yaml():
    """Create dataset config — edit class names to match your annotations."""
    data = {
        "path": "./dataset",
        "train": "images/train",
        "val":   "images/val",
        "nc":    8,
        "names": {
            0: "helmet",
            1: "goggles",
            2: "ear_protection",
            3: "gloves",
            4: "safety_shoes",
            5: "vest",
            6: "face_shield",
            7: "person",
        }
    }
    # If using Roboflow Construction Safety dataset, use their class names:
    # roboflow_data = {
    #     "names": {0: "Hardhat", 1: "Mask", 2: "NO-Hardhat",
    #               3: "NO-Mask", 4: "NO-Safety Vest", 5: "Person",
    #               6: "Safety Cone", 7: "Safety Vest", 8: "machinery", 9: "vehicle"}
    # }
    return data


def download_dataset_instructions():
    print("""
╔══════════════════════════════════════════════════════════════╗
║         GET A FREE PPE DATASET (2 options)                   ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  OPTION 1 — Roboflow (Recommended, 5 min):                   ║
║    1. pip install roboflow                                    ║
║    2. Run: python train.py --download                        ║
║                                                              ║
║  OPTION 2 — Manual:                                          ║
║    1. Go to universe.roboflow.com                            ║
║    2. Search "Construction Site Safety"                      ║
║    3. Download → YOLOv8 format → extract to ./dataset/       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


def download_roboflow():
    """Auto-download the Roboflow Construction Site Safety dataset."""
    try:
        from roboflow import Roboflow
        api_key = os.environ.get("ROBOFLOW_API_KEY", "")
        if not api_key:
            print("Set ROBOFLOW_API_KEY env variable first:")
            print("  export ROBOFLOW_API_KEY=your_key_here")
            print("  (Get free key at roboflow.com)")
            return False

        rf = Roboflow(api_key=api_key)
        project = rf.workspace("roboflow-universe-projects").project(
            "construction-site-safety-g9u2k"
        )
        dataset = project.version(1).download("yolov8", location="./dataset")
        print(f"[DATASET] Downloaded to {dataset.location}")
        return True
    except Exception as e:
        print(f"[DATASET] Download failed: {e}")
        return False


def train(download=False):
    if download:
        if not download_roboflow():
            return

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Run: pip install ultralytics")
        sys.exit(1)

    if not os.path.exists("dataset/data.yaml"):
        download_dataset_instructions()
        return

    print("🦺 Starting PPE Training")
    print(f"   Model:   {BASE_MODEL}")
    print(f"   Device:  {HARDWARE}")
    print(f"   Batch:   {BATCH_SIZE}")
    print(f"   ImgSize: {IMG_SIZE}")
    print(f"   Epochs:  {EPOCHS}")

    model = YOLO(BASE_MODEL)

    results = model.train(
        data="dataset/data.yaml",
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=HARDWARE,
        workers=WORKERS,
        patience=20,          # early stopping: stop if no improvement for 20 epochs
        optimizer="AdamW",    # better than SGD for smaller datasets
        lr0=0.001,            # initial learning rate
        lrf=0.01,             # final LR = lr0 * lrf
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        # ── Augmentation (tuned for construction sites) ──
        augment=True,
        degrees=15.0,         # rotation: workers at angles
        scale=0.6,            # zoom variation: near/far workers
        shear=5.0,
        perspective=0.0005,
        fliplr=0.5,
        flipud=0.0,           # don't flip vertically (persons not upside-down)
        mosaic=1.0,           # mosaic augmentation: key for crowded scenes
        mixup=0.15,
        copy_paste=0.2,       # copy-paste: great for PPE on different backgrounds
        hsv_h=0.015,          # hue shift: different lighting conditions
        hsv_s=0.7,            # saturation: muddy/dirty PPE
        hsv_v=0.4,            # brightness: dawn/dusk/shadows
        # ── Output ──
        name="ppe_realtime",
        project="runs",
        exist_ok=True,
        verbose=True,
        plots=True,           # save training plots
    )

    # Save best model
    best = "runs/ppe_realtime/weights/best.pt"
    if os.path.exists(best):
        os.makedirs("models", exist_ok=True)
        shutil.copy(best, "models/best.pt")
        print("\n✅ Saved: models/best.pt")
    else:
        print("\n⚠️  Check runs/ for weights")

    # Validate
    print("\n📊 Validation Metrics:")
    val_model = YOLO("models/best.pt")
    metrics = val_model.val(data="dataset/data.yaml", imgsz=IMG_SIZE, verbose=False)
    print(f"   mAP50:    {metrics.box.map50:.3f}")
    print(f"   mAP50-95: {metrics.box.map:.3f}")
    print(f"   Precision:{metrics.box.mp:.3f}")
    print(f"   Recall:   {metrics.box.mr:.3f}")


def export_optimized(model_path="models/best.pt"):
    """
    Export to ONNX/TensorRT for faster inference.
    Call after training is complete.
    """
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)

        print("\n🚀 Exporting optimized models...")

        # ONNX — ~2x faster on CPU, works everywhere
        model.export(format="onnx", imgsz=IMG_SIZE, simplify=True, opset=12)
        print("   ✅ ONNX exported → models/best.onnx")

        # TorchScript — for PyTorch deployment
        model.export(format="torchscript", imgsz=IMG_SIZE)
        print("   ✅ TorchScript exported")

        # TensorRT (GPU only) — 3-5x faster, requires CUDA + TensorRT
        if USE_GPU:
            model.export(format="engine", imgsz=IMG_SIZE, half=True)
            print("   ✅ TensorRT engine exported (FP16)")

        print("\nTo use ONNX in inference:")
        print("   engine = InferenceEngine('models/best.onnx', ...)")

    except Exception as e:
        print(f"[EXPORT] Error: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true",
                        help="Auto-download Roboflow dataset")
    parser.add_argument("--export",   action="store_true",
                        help="Export trained model to ONNX/TRT")
    args = parser.parse_args()

    if args.export:
        export_optimized()
    else:
        train(download=args.download)
