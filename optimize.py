"""
optimize.py — Model optimization and benchmarking for real-time PPE detection
===============================================================================
Run this AFTER training to get maximum performance.

Usage:
  python optimize.py --benchmark          # measure FPS on your hardware
  python optimize.py --export-onnx        # export to ONNX (~2x faster on CPU)
  python optimize.py --export-trt         # export TensorRT (GPU only, ~5x faster)
  python optimize.py --quantize-int8      # INT8 quantization (CPU edge devices)
"""

import cv2
import time
import numpy as np
import argparse
import os

# ─── Benchmark ───────────────────────────────────────────────────────────────

def benchmark(model_path, input_size=640, n_frames=100, device="cpu"):
    """
    Measure real inference FPS on a synthetic frame.
    Run this on your actual hardware to know what skip_frames to use.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("pip install ultralytics")
        return

    print(f"\n🔬 Benchmarking {model_path}")
    print(f"   Input size: {input_size}x{input_size}")
    print(f"   Device: {device}")
    print(f"   Frames: {n_frames}")

    model = YOLO(model_path)
    dummy = np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)

    # Warmup
    for _ in range(5):
        model(dummy, conf=0.3, verbose=False)

    # Benchmark
    times = []
    for i in range(n_frames):
        t0 = time.perf_counter()
        model(dummy, conf=0.3, verbose=False)
        times.append(time.perf_counter() - t0)

    avg_ms  = sum(times) / len(times) * 1000
    fps     = 1000 / avg_ms
    p95_ms  = sorted(times)[int(n_frames * 0.95)] * 1000

    print(f"\n{'─'*40}")
    print(f"  Avg latency:  {avg_ms:.1f} ms")
    print(f"  Avg FPS:      {fps:.1f}")
    print(f"  P95 latency:  {p95_ms:.1f} ms")
    print(f"{'─'*40}")

    # Recommendation
    if fps >= 25:
        print("  ✅ Excellent — can process all frames in real-time")
    elif fps >= 15:
        skip = max(1, int(30 / fps))
        print(f"  ⚠️  Use skip_frames={skip} to maintain smooth display")
    else:
        print("  ❌ Too slow — try: smaller model, lower input_size, or ONNX export")

    return fps


def benchmark_resolutions(model_path, device="cpu"):
    """Test multiple resolutions to find the speed/accuracy sweet spot."""
    print("\n📐 Resolution benchmark:")
    resolutions = [320, 416, 512, 640]
    for res in resolutions:
        fps = benchmark(model_path, input_size=res, n_frames=50, device=device)
        bar = "█" * int(fps / 3)
        print(f"  {res}px: {fps:.0f} FPS {bar}")


# ─── ONNX Export & Inference ─────────────────────────────────────────────────

def export_onnx(model_path="models/best.pt", input_size=640):
    """
    Export to ONNX format.
    Provides ~1.5-2x speedup on CPU vs PyTorch.
    Compatible with: OpenCV DNN, ONNX Runtime, TensorRT, CoreML.
    """
    from ultralytics import YOLO
    model = YOLO(model_path)
    export_path = model.export(
        format="onnx",
        imgsz=input_size,
        simplify=True,     # simplify graph for faster inference
        opset=12,          # compatible with ONNX Runtime 1.12+
        dynamic=False,     # fixed batch size = faster
    )
    print(f"✅ ONNX exported: {export_path}")
    print(f"   Load with: InferenceEngine('{export_path}')")
    return export_path


def export_tensorrt(model_path="models/best.pt", input_size=640):
    """
    Export to TensorRT (NVIDIA GPU only).
    Provides 3-5x speedup over PyTorch on NVIDIA GPUs.
    Requires: CUDA + TensorRT + nvidia-tensorrt Python package.
    """
    from ultralytics import YOLO
    model = YOLO(model_path)
    export_path = model.export(
        format="engine",
        imgsz=input_size,
        half=True,         # FP16 — ~2x memory savings, minimal accuracy loss
        device=0,
    )
    print(f"✅ TensorRT exported: {export_path}")
    return export_path


# ─── INT8 Quantization (for edge devices) ────────────────────────────────────

def quantize_int8(model_path="models/best.pt", calib_images="dataset/images/val",
                  input_size=640):
    """
    INT8 quantization via ONNX Runtime.
    Best for: Raspberry Pi, Jetson Nano, CPU-only servers.
    ~4x faster than FP32 with ~1% mAP drop.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        import onnx
    except ImportError:
        print("pip install onnxruntime onnx")
        return

    onnx_path = model_path.replace(".pt", ".onnx")
    if not os.path.exists(onnx_path):
        onnx_path = export_onnx(model_path, input_size)

    quantized_path = onnx_path.replace(".onnx", "_int8.onnx")
    quantize_dynamic(
        onnx_path,
        quantized_path,
        weight_type=QuantType.QInt8,
    )
    print(f"✅ INT8 model: {quantized_path}")

    # Compare sizes
    orig_mb = os.path.getsize(onnx_path) / 1e6
    quant_mb = os.path.getsize(quantized_path) / 1e6
    print(f"   Size: {orig_mb:.1f} MB → {quant_mb:.1f} MB ({quant_mb/orig_mb*100:.0f}%)")


# ─── NMS Tuning ──────────────────────────────────────────────────────────────

def tune_nms_thresholds(model_path, val_images="dataset/images/val",
                         val_labels="dataset/labels/val"):
    """
    Find optimal conf/iou thresholds for your specific scene density.
    Construction sites are dense — different from generic COCO settings.
    """
    from ultralytics import YOLO
    model = YOLO(model_path)

    print("\n🎯 NMS Threshold Tuning")
    configs = [
        {"conf": 0.20, "iou": 0.40},
        {"conf": 0.25, "iou": 0.45},
        {"conf": 0.30, "iou": 0.50},
        {"conf": 0.35, "iou": 0.55},
    ]

    for cfg in configs:
        metrics = model.val(
            data="dataset/data.yaml",
            conf=cfg["conf"],
            iou=cfg["iou"],
            verbose=False,
        )
        print(f"  conf={cfg['conf']} iou={cfg['iou']}  "
              f"mAP50={metrics.box.map50:.3f}  "
              f"mAP50-95={metrics.box.map:.3f}")

    print("\n  Recommendation for construction sites:")
    print("    Dense scenes (many workers):   conf=0.25, iou=0.40")
    print("    Sparse scenes (few workers):   conf=0.30, iou=0.50")
    print("    Speed priority (low accuracy): conf=0.40, iou=0.50")


# ─── Hardware-Specific Recommendations ───────────────────────────────────────

def print_hardware_guide():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║              HARDWARE + SETTINGS CHEAT SHEET                     ║
╠═══════════════════════╦══════════╦═══════════╦═══════════════════╣
║ Hardware              ║ Model    ║ Size      ║ Expected FPS      ║
╠═══════════════════════╬══════════╬═══════════╬═══════════════════╣
║ Laptop CPU (i5/i7)    ║ yolov8n  ║ 416       ║ ~12-18 FPS        ║
║ Laptop CPU (i5/i7)    ║ yolov8n  ║ 640       ║ ~6-10 FPS         ║
║ Desktop CPU (i9/Ryzen)║ yolov8n  ║ 640       ║ ~15-20 FPS        ║
║ NVIDIA RTX 3060       ║ yolov8s  ║ 640       ║ ~80-120 FPS       ║
║ NVIDIA RTX 3090       ║ yolov8m  ║ 1280      ║ ~60-80 FPS        ║
║ Jetson Nano           ║ yolov8n  ║ 320       ║ ~8-12 FPS         ║
║ Jetson Xavier NX      ║ yolov8s  ║ 640       ║ ~25-35 FPS        ║
║ Raspberry Pi 4        ║ yolov8n  ║ 320       ║ ~2-4 FPS (!)      ║
╚═══════════════════════╩══════════╩═══════════╩═══════════════════╝

For 20+ FPS real-time on CPU:
  1. Use yolov8n.pt (or yolov8n-ppe.pt if you retrain)
  2. Set input_size=416
  3. Export to ONNX: python optimize.py --export-onnx
  4. Use skip_frames=2 (process every 2nd frame, display at native FPS)
  5. Resize input frames to 640px wide before passing to model

For GPU deployment:
  1. Use yolov8s.pt or yolov8m.pt
  2. Export to TensorRT: python optimize.py --export-trt
  3. Set half=True (FP16) for 2x speed
  4. Batch size 4-8 for multiple camera streams
""")


# ─── Skip Frame Strategy ──────────────────────────────────────────────────────

def optimal_skip_frames(model_fps, target_display_fps=30):
    """
    Calculate skip_frames to achieve smooth display with slower inference.

    Example: model does 10 FPS, display target 30 FPS → skip=3
    (run inference every 3rd frame, reuse last result for other 2)
    """
    skip = max(1, round(target_display_fps / model_fps))
    actual_inference_fps = target_display_fps / skip
    print(f"\nModel FPS: {model_fps}")
    print(f"Target display FPS: {target_display_fps}")
    print(f"→ skip_frames={skip} (inference at {actual_inference_fps:.1f} FPS, display at {target_display_fps} FPS)")
    return skip


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark",    action="store_true")
    parser.add_argument("--benchmark-all",action="store_true")
    parser.add_argument("--export-onnx",  action="store_true")
    parser.add_argument("--export-trt",   action="store_true")
    parser.add_argument("--quantize-int8",action="store_true")
    parser.add_argument("--tune-nms",     action="store_true")
    parser.add_argument("--guide",        action="store_true")
    parser.add_argument("--model",   default="models/best.pt")
    parser.add_argument("--size",    type=int, default=640)
    parser.add_argument("--device",  default="cpu")
    args = parser.parse_args()

    if args.guide:
        print_hardware_guide()
    if args.benchmark:
        fps = benchmark(args.model, args.size, device=args.device)
        optimal_skip_frames(fps)
    if args.benchmark_all:
        benchmark_resolutions(args.model, args.device)
    if args.export_onnx:
        export_onnx(args.model, args.size)
    if args.export_trt:
        export_tensorrt(args.model, args.size)
    if args.quantize_int8:
        quantize_int8(args.model)
    if args.tune_nms:
        tune_nms_thresholds(args.model)

    if not any(vars(args).values()):
        print_hardware_guide()
