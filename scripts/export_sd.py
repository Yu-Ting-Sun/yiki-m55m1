#!/usr/bin/env python3
"""export_sd.py — pull the frame-sync manifest and lay out an SD card image.

替代（尚未實作的）韌體 HTTP 同步迴圈：把 App 的旅程（照片、LABEL.JSON、
遊記 STORY.TXT/TIM/WAV）與家人自拍註冊檔從後端抓下來，直接排成板子
Slideshow / photo-enroll 期待的 SD 結構。用讀卡機整包拷進 SD 卡即可 demo，
現場不需要 Wi-Fi。

    python scripts/export_sd.py                          # 後端在本機:8000
    python scripts/export_sd.py --base http://127.0.0.1:8000 --out sd_export

輸出（拷到 SD 卡根目錄）：
    pictures/T0006/P0002.JPG, LABEL.JSON, STORY.TXT/TIM/WAV, VERSION.TXT
    faces/enroll_dad.raw ...

增量：資料夾裡 VERSION.TXT 與 manifest 的 version 相同就整趟跳過——
跟未來韌體同步迴圈的行為一致。

別忘了 SD 卡還要放（見 README「SD 卡」節）：face_mobilenet.tflite。
"""
import argparse
import sys
from pathlib import Path

import httpx


def fetch(client: httpx.Client, url: str, dest: Path) -> int:
    r = client.get(url, timeout=120)  # story.wav 第一次要跑 TTS，給寬一點
    r.raise_for_status()
    dest.write_bytes(r.content)
    return len(r.content)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--base", default="http://127.0.0.1:8000",
                    help="backend base URL (default: %(default)s)")
    ap.add_argument("--frame", type=int, default=1, help="frame id (default: 1)")
    ap.add_argument("--out", default="sd_export",
                    help="output directory (default: %(default)s)")
    args = ap.parse_args()

    out = Path(args.out)
    client = httpx.Client(base_url=args.base)

    manifest = client.get(f"/frames/{args.frame}/sync", timeout=30)
    manifest.raise_for_status()
    data = manifest.json()

    pictures = out / data.get("sd_root", "pictures")
    pictures.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    for trip in data["trips"]:
        folder = pictures / trip["folder"]
        ver_file = folder / "VERSION.TXT"
        if ver_file.exists() and ver_file.read_text().strip() == trip["version"]:
            print(f"  {trip['folder']}  ({trip['title']}) — 沒變，跳過")
            continue
        folder.mkdir(parents=True, exist_ok=True)

        total_bytes += fetch(client, trip["label_json"], folder / "LABEL.JSON")
        if trip["has_story"]:
            total_bytes += fetch(client, trip["story_txt"], folder / "STORY.TXT")
            total_bytes += fetch(client, trip["story_tim"], folder / "STORY.TIM")
            total_bytes += fetch(client, trip["story_wav"], folder / "STORY.WAV")
        for photo in trip["photos"]:
            total_bytes += fetch(client, photo["url"], folder / photo["name"])

        ver_file.write_text(trip["version"])
        print(f"  {trip['folder']}  ({trip['title']}) — "
              f"{len(trip['photos'])} 張照片"
              f"{'、含遊記' if trip['has_story'] else ''}"
              f"、參加者 {trip['users'] or '（未標記）'}")

    faces_dir = out / data.get("faces_root", "faces")
    for face in data.get("faces", []):
        faces_dir.mkdir(parents=True, exist_ok=True)
        total_bytes += fetch(client, face["url"], faces_dir / face["name"])
        print(f"  faces/{face['name']}")

    if data.get("truncated"):
        print("  ⚠ 旅程超過板子相簿上限，只匯出最新 15 趟")

    print(f"\n完成：{out}/（新下載 {total_bytes / 1024:.0f} KB）")
    print("把裡面的 pictures/ 與 faces/ 整包拷到 SD 卡根目錄。")
    print("SD 卡另需 face_mobilenet.tflite（人臉辨識模型，見 README）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
