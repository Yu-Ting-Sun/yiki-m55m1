#!/usr/bin/env python3
"""
compute_crc.py  --  host-side IEEE-802.3 CRC32 (matches the firmware's CRC).

The on-device ModelFileReader uses reflected CRC32, poly 0xEDB88420,
init/xorout 0xFFFFFFFF -- identical to Python's binascii.crc32. Use this to
record the EXPECTED CRC of each .tflite for the report, then cross-check it
against what the board prints in [LOAD][OK].

    python compute_crc.py ../models/gesture_model.tflite ../models/face_model.tflite
    python compute_crc.py ../models/*.tflite
"""
import binascii
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: compute_crc.py <file.tflite> [more.tflite ...]", file=sys.stderr)
        return 1
    print(f"{'file':40s} {'size (B)':>12s}  {'CRC32':>10s}")
    print("-" * 66)
    rc = 0
    for a in argv:
        p = Path(a)
        if not p.is_file():
            print(f"{a:40s} {'<missing>':>12s}")
            rc = 1
            continue
        d = p.read_bytes()
        print(f"{p.name:40s} {len(d):12d}  0x{binascii.crc32(d) & 0xFFFFFFFF:08X}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
