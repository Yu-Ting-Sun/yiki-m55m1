# m55m1_dual_model_poc

Day-1 feasibility PoC: run **Face** + **Gesture** TFLite models concurrently
(time-sliced) on NuMaker-M55M1 (Cortex-M55 @ 220 MHz + Ethos-U55 256-MAC),
both loaded from SD card into non-overlapping HyperRAM slots.

Derived from `Visually-Impaired-Assistance-System` (strategy A: load .tflite
from SD into HyperRAM, infer from there). **Not** a from-scratch project.

## Status (per the agreed Task order)

| Task | Item | Status |
|------|------|--------|
| 1 | Folder structure + copy infra from old project | **DONE** |
| 2 | `MemoryLayout.h` (centralized HyperRAM map + static_assert) | **DONE** |
| 3 | Rewrite `ModelFileReader` (4 KB reads + CRC32) + `LoadModelToHyperRAM()` | **DONE** |
| 4 | `FaceModel` / `GestureModel` subclasses | **DONE** |
| 5 | Dual-model test `main.cpp` (load / infer / alternate + sanity check) | **DONE** |
| 6-1 | Face model export script (`scripts/prepare_face_model.py`) | **DONE** (you run it) |
| 6-2 | Gesture model = old od model, copied to `models/gesture_model.tflite` | **DONE** |
| 6-3 | Vela compile (FACE only — gesture already compiled) | pending (needs your ML env) |
| 7 | `final_report.md` from UART log | pending (needs hardware run) |

## Files copied verbatim from the old project

- `Device/SDCard/sdglue.c`     — FATFS SD0/SD1 mount
- `Device/HyperRAM/hyperram_code.c` + `Device/include/hyperram_code.h` — HyperRAM init / SPIM0 direct-map
- `BoardInit.cpp/.hpp`, `board_config.h`, `mpu_config_M55M1.h`, `ffconf_M55M1.h`
- `NPU/` — Ethos-U55 init / cache / profiler
- `KEIL/M55M1.scatter` — linker layout (unchanged)
- `ModelFileReader.{c,h}.orig` — baseline, kept for diff; rewritten in Task 3

## New files

- `MemoryLayout.h`  — HyperRAM addresses + 1 MB arena + `ModelInfo` registry
- `main.cpp` — Day-1 test harness (load+CRC selfcheck / infer x100 / alternate x500)
- `Model/{FaceModel,GestureModel}.{cpp,hpp}` — op-resolver subclasses (Transpose+EthosU)
- `ModelFileReader.{c,h}` — rewritten loader: 4 KB reads + CRC32 + `LoadToAddress()`
- `ModelLoader.{c,h}` — `LoadModelToHyperRAM()` (drive select + timing + log)
- `PerfTimer.h` — DWT-cycle-counter ms/us timer (`GetSystemTick_ms()`)
- `ModelFileReader.{c,h}.orig` — original baseline, kept only for diff reference

## HyperRAM layout (8 MB device)

```
0x82000000  +-------------------------+
            | 1 MB front guard        |  (linker-overflow margin)
0x82100000  +-------------------------+  FACE_MODEL_ADDR
            | FACE slot     (3 MB)    |
0x82400000  +-------------------------+  GESTURE_MODEL_ADDR
            | GESTURE slot  (3 MB)    |
0x82700000  +-------------------------+
            | 1 MB rear spare         |
0x82800000  +-------------------------+
```

> The original spec's `FACE=BASE+4MB / GESTURE=BASE+8MB / 4 MB slots` does not
> fit in 8 MB and fails the bounds assert. See the header comment in
> `MemoryLayout.h` for the full reasoning and a 4 MB-slot alternative.

## Build (Keil)

This project builds **only from inside the BSP** (the `.uvprojx` uses
`..\..\..\..\` relative paths to the BSP `Library/` + `ThirdParty/`). Open:

```
<BSP>\SampleCode\NuEdgeWise\m55m1_dual_model_poc\KEIL\DualModelPoC.uvprojx
```
where `<BSP>` = `...\ML_M55M1_SampleCode\M55M1BSP-3.01.003`.
(The copy under `contest\` will NOT build — paths only resolve from the BSP.)

`DualModelPoC.uvprojx` was derived from the old `ObjectTracker.uvprojx`:
- pruned vision groups (ByteTrack, Pattern, UVC, Display, ImageSensor) + old
  app files (VoicePlayer, YOLOv8nODPostProcessing, YOLOv8nODModel, Labels);
- added `ModelLoader.c`, `Model/FaceModel.cpp`, `Model/GestureModel.cpp`;
- removed the `ACTIVATION_BUF_SZ=0x00080000` Define (now owned by MemoryLayout.h).

### Prerequisites verified
- BSP `Library/` was empty (only folder skeleton) → **restored** from the
  NuML_Studio M55M1BSP copy (1140 .c / 1999 .h / 10 .lib). All includes resolve.
- Needs the Keil **M55M1 device pack (DFP)** installed (device `M55M1H2LJAE`).

### First-build gate
Build the **old** `ObjectTracker.uvprojx` first — if it links clean, the
restored Library + DFP + toolchain are all good, and this PoC should build too.

### SD card (drive 0: root)
`face_model.tflite` (Vela output) + `gesture_model.tflite` (already provided).
