#!/usr/bin/env python3
"""
prepare_face_model.py  --  Task 6-1

Export a YOLOv8n face model to a FULL-INTEGER (INT8) TFLite, ready to be fed to
Vela for the Ethos-U55. Day-1 goal is "does dual-model run", NOT accuracy, so a
plain yolov8n is an acceptable stand-in if face weights are unavailable.

Run this on YOUR machine (needs internet + `pip install ultralytics`). It cannot
run in the agent sandbox.

    python prepare_face_model.py                 # tries yolov8n-face.pt
    python prepare_face_model.py --model yolov8n.pt --imgsz 192
    python prepare_face_model.py --fallback      # silently fall back to yolov8n.pt

Output: models/face_model_int8.tflite   (NOT yet Vela-compiled -> see Task 6-3)

Why FULL-INTEGER: Ethos-U55 only accelerates int8/uint8 tensors. The
`*_full_integer_quant.tflite` variant (int8 weights AND activations) is the one
Vela can map to the NPU; the plain `*_float*` / `*_dynamic*` variants cannot.
"""
import argparse
import binascii
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE.parent / "models"
OUT_TFLITE = MODELS_DIR / "face_model_int8.tflite"


def crc32_of(path: Path) -> str:
    data = path.read_bytes()
    return f"size={len(data)} B  CRC32=0x{binascii.crc32(data) & 0xFFFFFFFF:08X}"


def find_full_int8(export_dir: Path) -> Path | None:
    """Locate the all-int8 tflite among ultralytics' exported variants."""
    for pattern in ("*full_integer_quant.tflite", "*int8.tflite", "*integer_quant.tflite"):
        hits = sorted(export_dir.glob(pattern))
        # prefer 'full_integer_quant' if several match
        for h in hits:
            if "full_integer_quant" in h.name:
                return h
        if hits:
            return hits[0]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n-face.pt",
                    help="weights to export (default: yolov8n-face.pt)")
    ap.add_argument("--imgsz", type=int, default=192,
                    help="input size; 192 matches the old project (default: 192)")
    ap.add_argument("--fallback", action="store_true",
                    help="fall back to yolov8n.pt without prompting if --model is missing")
    ap.add_argument("--data", default=None,
                    help="optional calibration dataset yaml for INT8 (else ultralytics default)")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed.  pip install ultralytics", file=sys.stderr)
        return 2

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_name = args.model
    try:
        model = YOLO(model_name)
    except Exception as e:  # face weights may not exist on ultralytics assets
        print(f"WARN: could not load '{model_name}': {e}")
        if args.fallback or model_name != "yolov8n.pt":
            print("WARN: falling back to plain 'yolov8n.pt' (Day-1 only cares that it RUNS).")
            model_name = "yolov8n.pt"
            model = YOLO(model_name)
        else:
            return 3

    print(f"Exporting '{model_name}' -> INT8 TFLite, imgsz={args.imgsz} ...")
    export_kwargs = dict(format="tflite", imgsz=args.imgsz, int8=True)
    if args.data:
        export_kwargs["data"] = args.data
    exported = Path(model.export(**export_kwargs))
    # model.export returns a path to one tflite; the full-int8 variant lives in
    # the same *_saved_model directory.
    export_dir = exported.parent
    full_int8 = find_full_int8(export_dir) or exported

    shutil.copyfile(full_int8, OUT_TFLITE)
    print(f"\nPicked full-INT8 variant : {full_int8.name}")
    print(f"Copied to                : {OUT_TFLITE}")
    print(f"  {crc32_of(OUT_TFLITE)}")
    print("\nNEXT: run Vela on this file (Task 6-3) before copying to the SD card.")
    print("      Gesture model is ALREADY Vela-compiled -> do NOT re-Vela it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
