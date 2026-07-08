#!/usr/bin/env python3
"""
prepare_pictures.py — convert photos (JPG/PNG/HEIC*/BMP/WebP) into LCD-ready
24-bit BMPs for the board slideshow.

The firmware (Slideshow.c) renders uncompressed 24-bpp BMPs from the SD card's
`pictures` folder, centered on the panel. Default target = 800x480 (LT7381
panel per board_config.h). Use --size 480x272 for FSA506 or 320x240 for
ILI9341 if the board's panel define changes.

Usage:
    pip install pillow
    python prepare_pictures.py C:\\photos                    # whole folder
    python prepare_pictures.py a.jpg b.png --mode fill       # crop-to-fill
    python prepare_pictures.py C:\\photos --size 480x272 --out E:\\pictures

Then copy the output folder onto the SD card as `0:\\pictures\\`.
(*HEIC needs `pip install pillow-heif`; script degrades gracefully without it.)
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

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".tif", ".tiff"}


def collect(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files += sorted(q for q in p.iterdir() if q.suffix.lower() in EXTS)
        elif p.is_file():
            files.append(p)
        else:
            print(f"WARN: skip missing input {item}", file=sys.stderr)
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="photo files and/or folders")
    ap.add_argument("--size", default="800x480",
                    help="panel WxH (default 800x480 = LT7381)")
    ap.add_argument("--mode", choices=["fit", "fill"], default="fit",
                    help="fit = letterbox with black bars (default); fill = center-crop")
    ap.add_argument("--out", default=None,
                    help="output folder (default: ./pictures next to this script)")
    args = ap.parse_args()

    w, h = (int(v) for v in args.size.lower().split("x"))
    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parent.parent / "pictures"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = collect(args.inputs)
    if not files:
        print("ERROR: no input photos found", file=sys.stderr)
        return 1

    n = 0
    for src in files:
        try:
            img = Image.open(src)
            img = ImageOps.exif_transpose(img)      # honour phone rotation
            img = img.convert("RGB")
            if args.mode == "fill":
                img = ImageOps.fit(img, (w, h))
            else:
                img = ImageOps.pad(img, (w, h), color=(0, 0, 0))
            n += 1
            dst = out_dir / f"PIC{n:03d}.BMP"
            img.save(dst, format="BMP")             # 24-bpp uncompressed
            print(f"  {src.name:40s} -> {dst.name}")
        except Exception as e:
            print(f"WARN: skip {src.name}: {e}", file=sys.stderr)

    print(f"\n{n} photo(s) written to {out_dir}")
    print("Copy that folder to the SD card root as 'pictures' (0:\\pictures\\).")
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
