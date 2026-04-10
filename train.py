"""
PPE Detection Model Training Script
=====================================
This script trains a YOLOv8 model on a PPE dataset.

STEP 1: Prepare your dataset
  - Collect 200+ images of construction workers with PPE
  - Annotate using Roboflow (roboflow.com) or LabelImg
  - Export in YOLOv8 format → creates dataset/
  - Folder structure:
      dataset/
        images/train/  ← training images
        images/val/    ← validation images
        labels/train/  ← YOLO .txt annotations
        labels/val/
        data.yaml      ← class definitions

STEP 2: Create data.yaml (example below)
  path: ./dataset
  train: images/train
  val: images/val
  names:
    0: helmet
    1: glasses
    2: ear_protection
    3: gloves
    4: safety_shoes
    5: safety_vest
    6: face_shield
    7: person

STEP 3: Run this script
  python train.py

RECOMMENDED FREE DATASETS (Roboflow Universe):
  - "Construction Site Safety" by Roboflow
  - "PPE Detection" datasets
  Download with: pip install roboflow
"""

import os
import sys

def train():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Install ultralytics first: pip install ultralytics")
        sys.exit(1)

    # Check dataset exists
    if not os.path.exists("dataset/data.yaml"):
        print("""
ERROR: dataset/data.yaml not found!

Quick start with a public dataset:
  1. Go to https://universe.roboflow.com/
  2. Search "PPE Construction Site Safety"
  3. Download → YOLOv8 format → extract to ./dataset/

Or use the Roboflow Python package:
  pip install roboflow
  from roboflow import Roboflow
  rf = Roboflow(api_key="YOUR_KEY")
  project = rf.workspace("roboflow-universe-projects").project("construction-site-safety-g9u2k")
  dataset = project.version(1).download("yolov8")
""")
        return

    print("🦺 Starting PPE Detection Model Training...")
    print("=" * 50)

    # Load pretrained YOLOv8 nano (fastest, good for 200 images)
    model = YOLO("yolov8n.pt")

    results = model.train(
        data="dataset/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,           # Early stopping
        augment=True,          # Data augmentation
        degrees=10,            # Rotation augmentation
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        name="ppe_detector",
        project="runs",
        exist_ok=True,
        device="cpu",          # Change to "0" for GPU
        workers=4,
        verbose=True,
    )

    # Copy best model to models/
    import shutil
    best = "runs/ppe_detector/weights/best.pt"
    if os.path.exists(best):
        os.makedirs("models", exist_ok=True)
        shutil.copy(best, "models/best.pt")
        print("\n✅ Model saved to models/best.pt")
        print("🚀 Now run: python app.py")
    else:
        print("\n⚠️  Training may have failed. Check runs/ folder.")

    # Validate
    print("\n📊 Running validation...")
    model = YOLO("models/best.pt")
    metrics = model.val(data="dataset/data.yaml")
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")

if __name__ == "__main__":
    train()
