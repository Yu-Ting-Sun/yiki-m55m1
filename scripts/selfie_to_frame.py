#!/usr/bin/env python3
"""
selfie_to_frame.py — convert a selfie photo into the raw camera-frame file the
firmware's photo-enroll one-shot consumes (main.cpp RUN_PHOTO_ENROLL).

The board treats the file exactly like a captured camera frame: 240x240
RGB565 little-endian (115200 bytes), then runs the identical detect -> crop ->
embed pipeline on it. So this script only crops/resizes — all inference stays
on the board (the SD model is Vela-compiled and only runs on the Ethos-U).

Usage:
    pip install pillow            (pillow-heif optional, for iPhone HEIC)
    python selfie_to_frame.py selfie.jpg
    python selfie_to_frame.py selfie.jpg --zoom 1.5   # tighter face crop

Output: enroll.raw (+ enroll_preview.png to eyeball the crop).
Copy enroll.raw to the SD card as 0:\\faces\\enroll.raw, set
RUN_PHOTO_ENROLL=1 / PHOTO_ENROLL_LABEL, build and flash. The face should
fill a good part of the frame, like standing in front of the camera.
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("photo", help="selfie image (jpg/png/heic/...)")
    ap.add_argument("--zoom", type=float, default=1.0,
                    help="center-crop zoom factor, >1 = tighter on the face "
                         "(default 1.0 = largest center square)")
    ap.add_argument("--out", default="enroll.raw", help="output raw file")
    args = ap.parse_args()

    img = Image.open(args.photo)
    img = ImageOps.exif_transpose(img)          # honor phone orientation
    img = img.convert("RGB")

    side = int(min(img.size) / max(args.zoom, 1.0))
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

    out = Path(args.out)
    out.write_bytes(buf)

    preview = out.with_name(out.stem + "_preview.png")
    img.save(preview)

    print(f"wrote {out} ({len(buf)} bytes) and {preview}")
    print("copy to SD as 0:\\faces\\enroll.raw, then flash with RUN_PHOTO_ENROLL=1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
