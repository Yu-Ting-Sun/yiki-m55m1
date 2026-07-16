#!/usr/bin/env python3
"""
selfie_to_frame.py — convert selfie photos into the raw camera-frame files the
firmware's photo enrollment consumes (main.cpp RUN_PHOTO_ENROLL).

The board treats each file exactly like a captured camera frame: 240x240
RGB565 little-endian (115200 bytes), and enrolls it under the label carried in
the file name (enroll_<label>.raw; a trailing -N marks extra reference photos
of the same person). This script only crops/resizes — all inference stays on
the board (the SD model is Vela-compiled and only runs on the Ethos-U).

Usage:
    pip install pillow            (pillow-heif optional, for iPhone HEIC)
    python selfie_to_frame.py --label user1 selfie.jpg
    python selfie_to_frame.py --label user1 a.jpg b.jpg c.jpg --zoom 1.5

Output: enroll_<label>.raw, enroll_<label>-2.raw, ... (+ *_preview.png to
eyeball each crop). Copy the .raw files into the SD card's faces\\ folder and
reboot the board — no reflash needed. The face should fill a good part of the
frame, like standing in front of the camera.
"""
import argparse
import sys
from pathlib import Path

try:
    from PIL import Image, ImageOps
except ImportError:
    print("ERROR: Pillow not installed.  pip install pillow", file=sys.stderr)
    sys.exit(2)

try:  # optional HEIC support for phone photos
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

CAM_W, CAM_H = 240, 240


def convert(photo: str, zoom: float, out: Path) -> None:
    img = Image.open(photo)
    img = ImageOps.exif_transpose(img)          # honor phone orientation
    img = img.convert("RGB")

    side = int(min(img.size) / max(zoom, 1.0))
    cx, cy = img.width // 2, img.height // 2
    img = img.crop((cx - side // 2, cy - side // 2,
                    cx - side // 2 + side, cy - side // 2 + side))
    img = img.resize((CAM_W, CAM_H), Image.LANCZOS)

    buf = bytearray(CAM_W * CAM_H * 2)
    px = img.load()
    i = 0
    for y in range(CAM_H):
        for x in range(CAM_W):
            r, g, b = px[x, y]
            v = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)   # RGB565
            buf[i]     = v & 0xFF                               # little-endian
            buf[i + 1] = v >> 8
            i += 2

    out.write_bytes(buf)
    img.save(out.with_name(out.stem + "_preview.png"))
    print(f"wrote {out} ({len(buf)} bytes)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("photos", nargs="+", help="selfie image(s) (jpg/png/heic/...)")
    ap.add_argument("--label", required=True,
                    help="user name to enroll as (goes into the file name)")
    ap.add_argument("--zoom", type=float, default=1.0,
                    help="center-crop zoom factor, >1 = tighter on the face "
                         "(default 1.0 = largest center square)")
    ap.add_argument("--outdir", default=".", help="output directory")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    for i, photo in enumerate(args.photos, start=1):
        suffix = "" if i == 1 else f"-{i}"
        convert(photo, args.zoom, outdir / f"enroll_{args.label}{suffix}.raw")

    print("copy the .raw file(s) to the SD card's faces\\ folder and reboot "
          "the board (RUN_PHOTO_ENROLL firmware).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
