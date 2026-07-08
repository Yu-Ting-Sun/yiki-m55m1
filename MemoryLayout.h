/**************************************************************************//**
 * @file     MemoryLayout.h
 * @brief    Centralized HyperRAM layout for the dual-model PoC.
 *
 * Single source of truth for every HyperRAM address. Compile-time asserts
 * guarantee the model slots never overlap and never exceed the 8 MB device.
 *
 * ---------------------------------------------------------------------------
 * NOTE on the change from the original spec
 * ---------------------------------------------------------------------------
 * The board has only 8 MB of HyperRAM (0x82000000 .. 0x82800000).
 * The original draft used:
 *     FACE    = BASE + 4MB  (0x82400000)
 *     GESTURE = BASE + 8MB  (0x82800000)   <-- already AT the end of HyperRAM
 *     SLOT    = 4MB
 * which fails the bounds check:  GESTURE(0x82800000) + 4MB = 0x82C00000
 * is 4 MB past the end of the device. Two 4 MB slots simply cannot fit in
 * 8 MB once you also leave any safety margin.
 *
 * The old project (single model) placed its model at 0x82400000, i.e. it left
 * a 4 MB gap at the front. That gap was *arbitrary*: no linker section is
 * mapped into the start of HyperRAM (the 512 KB tensor arena fits inside the
 * internal SRAM01 aliased region and never spills into SPIM0). So we only need
 * a small front guard against future linker overflow, not a full 4 MB.
 *
 * Chosen layout (models are ~2.3 MB each today):
 *     1 MB front guard  | FACE slot 3 MB | GESTURE slot 3 MB | 1 MB rear spare
 *   0x82000000        0x82100000      0x82400000          0x82700000  0x82800000
 *
 * 3 MB slots give ~30% headroom over the current 2.3 MB models. If you would
 * rather honour the original "4 MB per slot" intent, use the alternative block
 * at the bottom of this file (fills HyperRAM exactly, zero margin).
 ******************************************************************************/
#ifndef MEMORY_LAYOUT_H
#define MEMORY_LAYOUT_H

#include <stdint.h>
#include "ethosu_mem_config.h"   /* ACTIVATION_BUF_SECTION (.bss.NoInit.activation_buf_sram) */

/* HyperRAM window (SPIM0 direct map) -------------------------------------- */
#define HYPERRAM_BASE          (0x82000000UL)
#define HYPERRAM_SIZE          (0x00800000UL)   /* 8 MB */
#define HYPERRAM_END           (HYPERRAM_BASE + HYPERRAM_SIZE)

/* Front guard: margin against linker data overflowing from the SRAM01-aliased
 * region into the start of the SPIM0 window (see KEIL/M55M1.scatter,
 * region SRAM01_HYPERRAM + its ScatterAssert). 1 MB is generous for this PoC. */
#define HYPERRAM_FRONT_GUARD   (0x00100000UL)   /* 1 MB */

/* Per-model slot. 3 MB covers the current ~2.3 MB models with headroom. */
#define MODEL_SLOT_SIZE        (0x00300000UL)   /* 3 MB per slot */

#define MODEL_REGION_BASE      (HYPERRAM_BASE + HYPERRAM_FRONT_GUARD)        /* 0x82100000 */

#define FACE_MODEL_ADDR        (MODEL_REGION_BASE)                           /* 0x82100000 */
#define GESTURE_MODEL_ADDR     (MODEL_REGION_BASE + MODEL_SLOT_SIZE)         /* 0x82400000 */

/* SD-card model file names (FATFS drive 0:) ------------------------------- */
#define FACE_MODEL_FILE        "0:\\face_model.tflite"
#define GESTURE_MODEL_FILE     "0:\\gesture_model.tflite"

/* ----------------------------------------------------------------------------
 * Tensor arena (activation buffer), shared time-sliced by both models.
 *
 * Raised 512 KB -> 1 MB because the face model's Vela report needs 830.81 KB of
 * SRAM (the old 512 KB arena is too small).
 *
 * ⚠️ HEADS-UP: SRAM01 (the Ethos-U-reachable internal SRAM) is exactly 1 MB and
 * it must ALSO hold the heap + non-cacheable data. A full 1 MB arena therefore
 * cannot fit entirely in SRAM01 — its tail spills, via the scatter's
 * SRAM01_HYPERRAM region, into the start of the SPIM0 window (0x82000000+),
 * i.e. into the 1 MB HYPERRAM_FRONT_GUARD below. That is SAFE (no collision with
 * FACE @ 0x82100000), but the spilled part of the arena lives in HyperRAM and
 * is slower for the NPU. If Day-1 latency misses <100 ms, shrink HEAP_SIZE in
 * the scatter and/or trim the arena toward ~832 KB so it fits fully in SRAM01.
 * --------------------------------------------------------------------------*/
/* Two RESIDENT interpreters (design "B"): each model gets its own arena slice
 * out of ONE contiguous backing buffer for deterministic placement.
 * Slice order = GESTURE first, FACE second (see main.cpp) — measured on
 * board: gesture with its arena in HyperRAM ran 177 ms vs 30.6 ms in SRAM01
 * (5.8x), while face tolerates its arena TAIL in HyperRAM at no cost
 * (41.8 ms) because TFLM puts high-traffic activations at the arena HEAD.
 * HEAP_SIZE in the scatter was shrunk 256->64 KB to maximise the SRAM01 share.
 * Measured needs: face 851,204 B, gesture 299,848 B. */
#define FACE_ARENA_SZ        (0x000E0000UL)   /* 896 KB (measured 851 KB + headroom) */
#define GESTURE_ARENA_SZ     (0x00050000UL)   /* 320 KB (measured ~300 KB + headroom) */

/* Combined size — used for the backing buffer + MPU region + accounting.
 * Single source of truth = this header. The old Keil project defined
 * ACTIVATION_BUF_SZ via a project -D flag; keep the stale-flag tripwire. */
#ifdef ACTIVATION_BUF_SZ
#  if (ACTIVATION_BUF_SZ != (FACE_ARENA_SZ + GESTURE_ARENA_SZ))
#    error "ACTIVATION_BUF_SZ is already defined (stale Keil -D flag?) and disagrees with FACE_ARENA_SZ+GESTURE_ARENA_SZ. Remove it from the project Defines; MemoryLayout.h owns it."
#  endif
#else
#  define ACTIVATION_BUF_SZ  (FACE_ARENA_SZ + GESTURE_ARENA_SZ)   /* 1216 KB */
#endif

#ifndef ACTIVATION_BUF_ATTRIBUTE
#define ACTIVATION_BUF_ATTRIBUTE __attribute__((aligned(16), ACTIVATION_BUF_SECTION))
#endif

/* ----------------------------------------------------------------------------
 * Model registry entry — drives both the loader and the boot self-check.
 * The expected_size / expected_crc32 are the host-computed values (see
 * scripts/compute_crc.py); the firmware compares them against what it actually
 * loaded from SD to catch a corrupt / wrong / truncated model before inferring.
 * --------------------------------------------------------------------------*/
typedef struct
{
    const char *file;            /* FATFS path, e.g. FACE_MODEL_FILE     */
    uint32_t    expected_size;   /* bytes, from compute_crc.py           */
    uint32_t    expected_crc32;  /* IEEE CRC32, from compute_crc.py      */
    uint32_t    load_addr;       /* HyperRAM slot base                   */
    uint32_t    slot_size;       /* slot capacity (= MODEL_SLOT_SIZE)    */
} ModelInfo;

/* ----------------------------------------------------------------------------
 * Slideshow decode frame buffer — lives in the 1 MB HyperRAM REAR SPARE
 * (after the gesture model slot), so it coexists with both resident models.
 * 800x480 RGB565 = 768 KB <= 1 MB. Only the CPU touches it (decode target +
 * banded blit source); the NPU never needs to reach it.
 * --------------------------------------------------------------------------*/
#define SLIDESHOW_FB_ADDR      (GESTURE_MODEL_ADDR + MODEL_SLOT_SIZE)   /* 0x82700000 */
#define SLIDESHOW_FB_SIZE      (0x00100000UL)                           /* 1 MB spare */

_Static_assert(SLIDESHOW_FB_ADDR + SLIDESHOW_FB_SIZE <= HYPERRAM_END,
               "Slideshow frame buffer exceeds HyperRAM!");

/* ----------------------------------------------------------------------------
 * Compile-time safety checks (C11 _Static_assert; valid in C and C++11+)
 * --------------------------------------------------------------------------*/
_Static_assert(FACE_MODEL_ADDR >= HYPERRAM_BASE + HYPERRAM_FRONT_GUARD,
               "Face model intrudes into the HyperRAM front guard region!");

_Static_assert(FACE_MODEL_ADDR + MODEL_SLOT_SIZE <= GESTURE_MODEL_ADDR,
               "Face and Gesture model slots overlap!");

_Static_assert(GESTURE_MODEL_ADDR + MODEL_SLOT_SIZE <= HYPERRAM_END,
               "Gesture model slot exceeds the 8 MB HyperRAM device!");

/* ----------------------------------------------------------------------------
 * ALTERNATIVE (commented out): honour the original "4 MB per slot" wish.
 * Fills HyperRAM EXACTLY, no front guard, no rear spare. FACE sits at the very
 * start of HyperRAM, so it is the slot most exposed if linker data ever
 * overflows SRAM01 into SPIM0. Only enable if you accept that risk.
 *
 *   #define MODEL_SLOT_SIZE      (0x00400000UL)            // 4 MB
 *   #define FACE_MODEL_ADDR      (HYPERRAM_BASE)           // 0x82000000
 *   #define GESTURE_MODEL_ADDR   (HYPERRAM_BASE + 0x400000)// 0x82400000
 * --------------------------------------------------------------------------*/

#endif /* MEMORY_LAYOUT_H */
