# Model pipeline notes

Day-1 only proves the two models can **load + infer + alternate** without
crashing. Accuracy is out of scope; fake input tensors are fine.

## The two models

| Role | Source | Vela needed? | SD-card filename |
|------|--------|--------------|------------------|
| **Gesture** | reuse old `YOLOv8n-od.tflite` (already Vela-compiled) | **NO** | `gesture_model.tflite` |
| **Face** | export YOLOv8n(-face) INT8 via `scripts/prepare_face_model.py` | **YES** | `face_model.tflite` |

### ⚠️ Correction to the original Task 6-3
The original plan said "both models must be re-Vela-compiled". That is wrong for
the gesture model: the old `YOLOv8n-od.tflite` **already contains the `ethos-u`
custom operator** (verified — it has the `ethos-u` marker and zero `CONV_2D`
ops, meaning every conv was already folded into the NPU command stream by Vela).
Running Vela again on an already-compiled model fails. So:

- **Gesture**: copy as-is → `gesture_model.tflite`. Done (see below).
- **Face**: the ultralytics export is plain INT8 (real `CONV_2D` ops) → it DOES
  need Vela.

## Files in this folder

| File | Stage | Status |
|------|-------|--------|
| `gesture_model.tflite` | Vela-compiled, **SD-ready** | ✅ created (copied from old project) |
| `face_model_int8.tflite` | pre-Vela INT8 export | ⏳ run `prepare_face_model.py` (your ML env) |
| `compiled/face_model.tflite` | post-Vela, **SD-ready** | ⏳ Task 6-3 |

## Known-good values (expected CRC for cross-checking the board log)

| File | Size (B) | CRC32 (IEEE) |
|------|---------:|--------------|
| `gesture_model.tflite` | 2,431,552 | `0xEBDA1FD1` |
| `face_model.tflite` | TBD after Vela | TBD (`compute_crc.py`) |

Both fit the 3 MB HyperRAM slot (2.32 MB < 3 MB). ✔

## Vela command (Task 6-3 — FACE ONLY)

```bash
vela --accelerator-config=ethos-u55-256 \
     --system-config=Ethos_U55_High_End_Embedded \
     --memory-mode=Shared_Sram \
     --output-dir=models/compiled \
     models/face_model_int8.tflite
# then: rename models/compiled/face_model_int8_vela.tflite -> face_model.tflite
```

> Match the `--system-config` / `--memory-mode` to whatever the old
> `YOLOv8n-od.tflite` was built with, so both models assume the same Ethos-U
> memory model. If the old project's Vela ini is available, reuse it verbatim.

## Final SD-card contents (root of drive 0:)

```
face_model.tflite      <- Vela output (Task 6-3)
gesture_model.tflite   <- this folder, as-is
```
