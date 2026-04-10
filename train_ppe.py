"""
train_ppe.py — Train YOLOv8 on the Ultralytics Construction-PPE Dataset
=========================================================================
Dataset: https://docs.ultralytics.com/datasets/detect/construction-ppe/
         178.4 MB | 1,132 train | 143 val | 141 test
         License: AGPL-3.0 | ~200+ images per class

11 Classes:
  0: helmet       ← PPE worn
  1: gloves       ← PPE worn
  2: vest         ← PPE worn
  3: boots        ← PPE worn
  4: goggles      ← PPE worn
  5: none         ← worker with NO PPE
  6: Person       ← person detection
  7: no_helmet    ← violation: helmet missing
  8: no_goggle    ← violation: goggles missing
  9: no_gloves    ← violation: gloves missing
  10: no_boots    ← violation: boots missing

Usage:
  python train_ppe.py                  # train with auto-download
  python train_ppe.py --epochs 50      # faster training
  python train_ppe.py --model yolov8s  # larger model (GPU recommended)
  python train_ppe.py --validate-only  # run validation on existing model
"""

import os
import sys
import shutil
import argparse


# ── Training Config — tune these for your hardware ────────────────────────────
CONFIG = {
    "model":       "yolov8n",    # yolov8n=fastest, yolov8s=balanced, yolov8m=accurate
    "epochs":      100,
    "imgsz":       640,
    "batch":       16,           # lower to 8 if memory issues on CPU
    "device":      "cpu",        # "0" for GPU, "cpu" for CPU
    "workers":     2,
    "patience":    20,
    "output_name": "ppe_construction",
    "output_dir":  "runs",
    "final_model": "models/construction_ppe_best.pt",
}


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║         🦺 Construction-PPE Training — 11 Class YOLOv8              ║
╠══════════════════════════════════════════════════════════════════════╣
║  Classes:  helmet | gloves | vest | boots | goggles | Person        ║
║            no_helmet | no_goggle | no_gloves | no_boots | none      ║
║  Dataset:  Ultralytics Construction-PPE (auto-download)             ║
║  Size:     ~1,132 train + 143 val + 141 test images                 ║
║  Result:   models/construction_ppe_best.pt                          ║
╚══════════════════════════════════════════════════════════════════════╝
""")


def check_ultralytics():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError:
        print("❌ ultralytics not installed. Run:")
        print("   pip install ultralytics")
        sys.exit(1)


def train(cfg):
    YOLO = check_ultralytics()
    print_banner()

    model_file = f"{cfg['model']}.pt"
    print(f"⚙️  Config:")
    print(f"   Model:   {model_file}")
    print(f"   Epochs:  {cfg['epochs']}")
    print(f"   ImgSize: {cfg['imgsz']}")
    print(f"   Device:  {cfg['device']}")
    print(f"   Batch:   {cfg['batch']}")
    print()

    # Load pretrained model
    model = YOLO(model_file)

    print("🚀 Starting training on Construction-PPE dataset...")
    print("   (Dataset will auto-download ~178 MB on first run)")
    print()

    results = model.train(
        # ── Dataset ─────────────────────────────────────────────────────────
        data="construction-ppe.yaml",   # Ultralytics built-in — auto-downloads!
        # ── Core settings ────────────────────────────────────────────────────
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        device=cfg["device"],
        workers=cfg["workers"],
        # ── Optimizer ────────────────────────────────────────────────────────
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        patience=cfg["patience"],
        # ── Augmentation (tuned for construction sites) ───────────────────────
        augment=True,
        degrees=15.0,       # construction workers at angles
        scale=0.6,          # near and far workers
        shear=5.0,
        perspective=0.0005,
        fliplr=0.5,
        flipud=0.0,         # persons never upside-down
        mosaic=1.0,         # critical for crowded scene handling
        mixup=0.15,
        copy_paste=0.2,     # synthetic PPE on new backgrounds
        hsv_h=0.015,        # hue shift: different lighting
        hsv_s=0.7,          # saturation: dirty/muddy PPE
        hsv_v=0.4,          # brightness: dawn/dusk/shadows
        # ── Output ───────────────────────────────────────────────────────────
        name=cfg["output_name"],
        project=cfg["output_dir"],
        exist_ok=True,
        verbose=True,
        plots=True,
        save=True,
        save_period=-1,     # save only best + last
    )

    # Copy best model to models/
    best_path = f"{cfg['output_dir']}/{cfg['output_name']}/weights/best.pt"
    if os.path.exists(best_path):
        os.makedirs("models", exist_ok=True)
        shutil.copy(best_path, cfg["final_model"])
        print(f"\n✅ Best model saved: {cfg['final_model']}")
    else:
        print(f"\n⚠️  Best model not found at {best_path}, check {cfg['output_dir']}/")
        return

    # Run validation
    _validate(cfg["final_model"], cfg)


def _validate(model_path, cfg):
    YOLO = check_ultralytics()
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    print(f"\n📊 Validation on Construction-PPE test set...")
    model = YOLO(model_path)
    metrics = model.val(
        data="construction-ppe.yaml",
        imgsz=cfg["imgsz"],
        verbose=False,
        split="test",
    )

    m50 = metrics.box.map50
    m5095 = metrics.box.map
    prec = metrics.box.mp
    rec = metrics.box.mr

    print(f"\n{'─'*50}")
    print(f"  mAP50:         {m50:.3f}")
    print(f"  mAP50-95:      {m5095:.3f}")
    print(f"  Precision:     {prec:.3f}")
    print(f"  Recall:        {rec:.3f}")
    print(f"{'─'*50}")

    # Per-class breakdown
    print("\n📋 Per-class metrics:")
    class_names = {
        0: "helmet", 1: "gloves", 2: "vest", 3: "boots", 4: "goggles",
        5: "none", 6: "Person", 7: "no_helmet", 8: "no_goggle",
        9: "no_gloves", 10: "no_boots",
    }
    try:
        for i, (p, r, ap50) in enumerate(zip(
            metrics.box.p, metrics.box.r, metrics.box.ap50
        )):
            name = class_names.get(i, f"class_{i}")
            status = "✅" if ap50 >= 0.5 else "⚠️ "
            print(f"  {status} {name:<14} P={p:.2f}  R={r:.2f}  AP50={ap50:.3f}")
    except Exception:
        pass

    # Performance guidance
    print()
    if m50 >= 0.70:
        print("  🏆 Excellent — model ready for production deployment!")
    elif m50 >= 0.50:
        print("  ✅ Good — suitable for demo/capstone. Train more epochs for production.")
    elif m50 >= 0.35:
        print("  ⚠️  Moderate — increase epochs to 150+, consider yolov8s model.")
    else:
        print("  ❌ Low accuracy — try: more epochs, yolov8s model, lower conf threshold.")

    print(f"\n🔗 Load in app.py: model path = '{model_path}'")
    print("   app.py will auto-detect this as 'models/construction_ppe_best.pt'")


def print_dataset_info():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║         Construction-PPE Dataset Class Reference                     ║
╠══════╦═══════════════╦════════╦══════════════════════════════════════╣
║  ID  ║ Class         ║ Type   ║ Description                          ║
╠══════╬═══════════════╬════════╬══════════════════════════════════════╣
║   0  ║ helmet        ║ ✅ PPE  ║ Safety helmet being worn             ║
║   1  ║ gloves        ║ ✅ PPE  ║ Safety gloves being worn             ║
║   2  ║ vest          ║ ✅ PPE  ║ High-vis safety vest being worn      ║
║   3  ║ boots         ║ ✅ PPE  ║ Safety boots/shoes being worn        ║
║   4  ║ goggles       ║ ✅ PPE  ║ Safety goggles/glasses being worn    ║
║   5  ║ none          ║ 🚫 VIO  ║ Person with NO PPE at all            ║
║   6  ║ Person        ║ 👤 OBJ  ║ Worker / person detected             ║
║   7  ║ no_helmet     ║ ⛔ VIO  ║ Person without helmet (violation)    ║
║   8  ║ no_goggle     ║ ⛔ VIO  ║ Person without goggles (violation)   ║
║   9  ║ no_gloves     ║ ⛔ VIO  ║ Person without gloves (violation)    ║
║  10  ║ no_boots      ║ ⛔ VIO  ║ Person without boots (violation)     ║
╚══════╩═══════════════╩════════╩══════════════════════════════════════╝

This dual-class approach means the model detects BOTH:
  ✅ "worker has helmet"  →  class 0 (helmet)
  ⛔ "worker missing helmet" → class 7 (no_helmet)

So your per-PPE wearing/not-wearing counts come directly from the model.
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 on Construction-PPE dataset"
    )
    parser.add_argument("--model",  default="yolov8n",
                        choices=["yolov8n","yolov8s","yolov8m","yolov8l"],
                        help="Model size (default: yolov8n)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs (default: 100)")
    parser.add_argument("--batch",  type=int, default=16,
                        help="Batch size (default: 16, use 8 if OOM)")
    parser.add_argument("--imgsz",  type=int, default=640,
                        help="Image size (default: 640)")
    parser.add_argument("--device", default="cpu",
                        help="Device: cpu or 0 (GPU)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate existing model, skip training")
    parser.add_argument("--info",   action="store_true",
                        help="Show dataset class info and exit")
    args = parser.parse_args()

    if args.info:
        print_dataset_info()
        sys.exit(0)

    cfg = dict(CONFIG)
    cfg["model"]   = args.model
    cfg["epochs"]  = args.epochs
    cfg["batch"]   = args.batch
    cfg["imgsz"]   = args.imgsz
    cfg["device"]  = args.device

    if args.validate_only:
        _validate(cfg["final_model"], cfg)
    else:
        train(cfg)
