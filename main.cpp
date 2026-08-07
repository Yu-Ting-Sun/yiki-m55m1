/**************************************************************************//**
 * @file     main.cpp
 * @brief    Day-1 dual-model feasibility PoC for NuMaker-M55M1.
 *
 * Proves Face + Gesture YOLOv8n models can co-reside in non-overlapping
 * HyperRAM slots and be inferred alternately on the single Ethos-U55, with a
 * boot-time integrity self-check (size + CRC32) and an NPU-actually-ran sanity
 * check (output tensor must not be all-zero).
 *
 * Test sequence:
 *   Test 1  Load both models from SD -> HyperRAM, verify size+CRC.
 *   Test 2  Face inference x100   (latency stats + output sanity).
 *   Test 3  Gesture inference x100 (latency stats + output sanity).
 *   Test 4  Alternating switch x500 (shared arena, re-Init per switch).
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include <cstdio>
#include <cstring>
#include <cstdint>

#include "NuMicro.h"
#include "BoardInit.hpp"
#include "log_macros.h"
#include "mpu_config_M55M1.h"     /* eMPU_ATTR_CACHEABLE_WTRA */

#include "MemoryLayout.h"
#include "ModelLoader.h"
#include "PerfTimer.h"
#include "FaceModel.hpp"
#include "GestureModel.hpp"
#include "Display.h"
#include "Slideshow.h"
#include "StoryUI.h"
#include "Camera.h"
#include "FaceDetect.hpp"
#include "FaceRecog.hpp"
#include "GestureLike.hpp"
#include "esp_probe.h"
#include "day2_test.h"
#include "day3_demo.h"
#include "SdSync.h"
#include "ff.h"                   /* photo-enroll one-shot reads the SD */

/* 1 = Day-2 Task 0: ESP-12F firmware probe. Runs INSTEAD of everything else
 *     (slideshow and dual-model tests) and never returns — set back to 0
 *     once the probe verdict is recorded. Takes priority over RUN_SLIDESHOW.
 *     Verdict 2026-07-07: AT_FIRMWARE_PRESENT @115200 (AT 1.7.0.0/SDK 3.0.0). */
#define RUN_ESP_PROBE      (0)   /* 0 for the slideshow milestone; flip to 1 to run the Day-2 ESP probe */

/* 1 = Day-2 Tasks 1-4: Wi-Fi + HTTP link validation against the FastAPI
 *     backend. Fill day2_config.h (SSID/password/backend IP) first!
 *     Runs after PerfTimer_Init (esp_at layer needs GetSystemTick_ms) and
 *     never returns. Priority: RUN_ESP_PROBE > RUN_DAY2_TESTS > slideshow. */
#define RUN_DAY2_TESTS     (0)

/* 1 = Day-3 demo: photo-frame layout (photos left 640x480, story rail right)
 *     with a LIVE LLM story — Wi-Fi up, POST /generate, GET /textimg, story
 *     on the rail, then the slideshow cycles photos in the left region.
 *     Needs: day2_config.h filled, backend running, SD photos in 0:\pictures.
 *     Story failure is non-fatal (photos still run). Never returns.
 *     Priority: RUN_ESP_PROBE > RUN_DAY2_TESTS > RUN_DAY3_DEMO > slideshow. */
#define RUN_DAY3_DEMO      (1)

/* 1 = SD sync over Wi-Fi (the App-to-frame pipe, no card swapping).
 *     Boot: register this board (ESP MAC) -> per-board frame_id + pair
 *     code shown on the story rail; a blank card bootstraps a full sync.
 *     Loop: every SDSYNC_CHECK_MS a ~100-byte doorbell poll — albums only
 *     download when the App pressed 「立即同步」 (on-demand, not periodic).
 *     Fails soft: no Wi-Fi/backend -> whatever is on the card keeps playing.
 *     New enroll files still need one reboot (enrollment runs at boot). */
#define RUN_SD_SYNC        (1)
#define SDSYNC_CHECK_MS    (10000u)

/* 1 = photo-frame mode: show 0:\pictures\*.bmp on the LCD forever and skip
 *     the dual-model tests (falls through to the tests only if the slideshow
 *     cannot start — missing folder / no BMPs / display init failure).
 * 0 = Day-1 dual-model validation tests. */
#define RUN_SLIDESHOW      (1)

#define SLIDESHOW_DIR      "0:\\pictures"
#define SLIDESHOW_HOLD_MS  (3000)

/* Simulated face-recognition verdict, until the camera path exists: set to a
 * user named in the album label.json files (e.g. "user1") to play only that
 * user's albums; NULL = no filter (play everything, same as before). */
#define SLIDESHOW_SIM_USER ((const char *)NULL)

/* 1 = Phase-2: HM1055 camera preview in the bottom-right corner of the photo
 *     region, refreshed continuously between photos. Camera init failure is
 *     non-fatal (plain slideshow keeps running). Only affects RUN_DAY3_DEMO. */
#define RUN_CAMERA_PREVIEW (1)

/* 1 = draw the live camera frame in the bottom-right corner of the photo
 *     region. 0 = the capture->detect->recognise pipeline still runs every
 *     hold window, but nothing is drawn — recognition works silently in the
 *     background and the frame stays a clean photo frame.
 *     Only meaningful when RUN_CAMERA_PREVIEW=1. */
#define SHOW_CAMERA_PREVIEW (0)

/* 1 = Phase-3: run face detection (compiled-in yolo-fastest_192_face +
 *     DetectorPostProcessing) on each captured frame and draw the face boxes
 *     onto the preview. Needs RUN_CAMERA_PREVIEW. Init failure is non-fatal
 *     (preview keeps running without boxes). */
#define RUN_FACE_DETECT    (1)

/* 1 = Phase-4: face recognition. On each detected face, crop -> FaceMobileNet
 *     embedding -> cosine-match against SD references (0:\faces\embeddings.txt).
 *     Recognised faces are re-drawn with a green box; result logged on UART.
 *     Needs RUN_FACE_DETECT and 0:\face_mobilenet.tflite on the SD card. Init
 *     failure is non-fatal (detection keeps running). */
#define RUN_FACE_RECOG     (1)

/* 1 = ENROLL mode: instead of recognising, capture the largest detected face,
 *     compute its embedding and APPEND "ENROLL_LABEL:embedding" to the SD
 *     reference file, then stop. Flash once per person (change ENROLL_LABEL),
 *     then flash again with RUN_FACE_ENROLL=0 to recognise. Needs RUN_FACE_RECOG. */
#define RUN_FACE_ENROLL    (0)
#define ENROLL_LABEL       "user1"
#define ENROLL_SAMPLES     (8)   /* how many embeddings to append per enroll run */

/* 1 = photo enrollment at boot (the App-registration path): every
 *     0:\faces\enroll_<label>.raw (240x240 RGB565 LE — made by
 *     scripts/selfie_to_frame.py from a phone selfie) is run through the
 *     IDENTICAL detect->crop->embed pipeline and enrolled as <label>
 *     (a trailing -N is stripped: enroll_user1-2.raw also enrolls user1,
 *     so one user can carry several reference photos). Files are renamed
 *     *.done afterwards, so new users register by just dropping files on
 *     the SD card — no reflash. Needs RUN_FACE_RECOG=1, RUN_FACE_ENROLL=0. */
#define RUN_PHOTO_ENROLL   (1)
#define PHOTO_ENROLL_MAX   (8)   /* max photos enrolled per boot */

/* 1 = 手勢按讚: run the MediaPipe hand-landmark model on each captured frame
 *     (model + arena in HyperRAM, see MemoryLayout.h demo plan) and detect a
 *     thumbs-up geometrically. A confirmed like is POSTed to the backend
 *     (SdSync_PostLike) with the album/photo on screen + recognised user.
 *     Needs RUN_CAMERA_PREVIEW + RUN_FACE_DETECT (init lives with the face
 *     pipeline) and 0:\hand_landmark.tflite on the SD card. Init failure is
 *     non-fatal (frame runs without the like feature). */
#define RUN_GESTURE_LIKE   (1)
#define LIKE_CONFIRM_HITS  (3)      /* consecutive thumbs-up frames to confirm */
#define LIKE_COOLDOWN_MS   (10000)  /* one gesture = one like */

/* Phase-5: live album filter. Single-frame cosine dips below the threshold
 * (green/red flicker), so the verdict is debounced before it drives the
 * slideshow: FILTER_SWITCH_HITS consecutive same-label recognitions switch
 * the filter to that user; nobody recognised for FILTER_CLEAR_MS clears it
 * (play everything). Slideshow_SetFilter takes effect at the next photo. */
#define FILTER_SWITCH_HITS (3)
#define FILTER_CLEAR_MS    (5000)

/* 1 = compile + run the Day-1 validation tests (and their whole UART log)
 *     whenever the slideshow doesn't take over.
 * 0 = Day-1 is signed off (see final_report.md): test code, boot log, tensor
 *     arenas and model objects are all compiled out. Flip back to 1 to re-run. */
#define RUN_DAY1_TESTS     (0)

/* Number of iterations per test (tune here). */
#define INFER_ITERS        (100)
#define ALT_SWITCHES       (500)   /* total model switches in Test 4 */

#if RUN_DAY1_TESTS

/*----------------------------------------------------------------------------
 * Tensor arenas — design "B": both interpreters stay RESIDENT (each Init'd
 * exactly once; MicroMutableOpResolver ops can only be registered once, so
 * re-Init is not allowed). One contiguous backing buffer, sliced, so placement
 * is deterministic: FACE first (max SRAM01 residency for the heavier model),
 * GESTURE after (its slice lands in the HyperRAM front guard).
 *--------------------------------------------------------------------------*/
namespace arm
{
namespace app
{
static uint8_t tensorArenas[FACE_ARENA_SZ + GESTURE_ARENA_SZ] ACTIVATION_BUF_ATTRIBUTE;
} /* namespace app */
} /* namespace arm */

/* Slice order chosen from measurement: gesture with its whole arena in
 * HyperRAM ran 177 ms (5.8x its SRAM figure of 30.6 ms), while face with only
 * its arena TAIL in HyperRAM stayed at 41.8 ms. TFLM allocates the traffic-
 * heavy non-persistent data (activations) from the arena HEAD and persistent
 * metadata from the TAIL — so: GESTURE first (fully inside SRAM01), FACE
 * second (its head stays in SRAM01, only the low-traffic tail spills into the
 * HyperRAM front guard). */
#define GESTURE_ARENA_PTR   (arm::app::tensorArenas)
#define FACE_ARENA_PTR      (arm::app::tensorArenas + GESTURE_ARENA_SZ)

/*----------------------------------------------------------------------------
 * Model registry (+ expected integrity values from scripts/compute_crc.py).
 *--------------------------------------------------------------------------*/
enum { MODEL_FACE = 0, MODEL_GESTURE, MODEL_COUNT };

static const ModelInfo kModels[MODEL_COUNT] =
{
    /* file,                size,    CRC32,       load_addr,          slot       */
    { FACE_MODEL_FILE,    2854848, 0x21B4B873, FACE_MODEL_ADDR,    MODEL_SLOT_SIZE },
    { GESTURE_MODEL_FILE, 2431552, 0xEBDA1FD1, GESTURE_MODEL_ADDR, MODEL_SLOT_SIZE },
};

static uint32_t s_loadedSize[MODEL_COUNT] = { 0, 0 };
static bool     s_loadOk[MODEL_COUNT]     = { false, false };
static bool     s_initOk[MODEL_COUNT]     = { false, false };

/* Persistent model objects (re-Init'd when switching over the shared arena). */
static arm::app::FaceModel    faceModel;
static arm::app::GestureModel gestureModel;

/*----------------------------------------------------------------------------
 * Small helpers
 *--------------------------------------------------------------------------*/
static uint32_t isqrt_u64(uint64_t v)
{
    uint64_t x = v, y = (x + 1) / 2;
    if (v == 0) return 0;
    while (y < x) { x = y; y = (x + v / x) / 2; }
    return (uint32_t)x;
}

/* Fill the input tensor with a varying non-zero pattern so a working NPU
 * produces non-trivial output (guards against "NPU never got data"). */
static void fill_input(arm::app::Model &model, uint32_t seed)
{
    TfLiteTensor *in = model.GetInputTensor(0);
    if (!in || !in->data.data) return;
    uint8_t *p = static_cast<uint8_t *>(in->data.data);
    for (size_t i = 0; i < in->bytes; i++)
        p[i] = (uint8_t)((i * 31u + seed * 7u + 1u) & 0xFFu);
}

/* Returns number of non-zero bytes in output tensor 0 (0 => NPU likely idle). */
static uint32_t output_nonzero_count(arm::app::Model &model)
{
    TfLiteTensor *out = model.GetOutputTensor(0);
    if (!out || !out->data.data) return 0;
    const uint8_t *p = static_cast<const uint8_t *>(out->data.data);
    uint32_t nz = 0;
    for (size_t i = 0; i < out->bytes; i++)
        if (p[i] != 0) nz++;
    return nz;
}

/* Cacheable-WTRA MPU region for the arena (mirrors the proven old project so
 * CPU<->NPU cache behaviour is identical). Requires the BSP InitPreDefMPURegion
 * helper + MAIR setup from mpu_config_M55M1.h. */
static void setup_arena_mpu(void)
{
    const ARM_MPU_Region_t mpuConfig[] =
    {
        {
            ARM_MPU_RBAR((unsigned int)arm::app::tensorArenas,
                         ARM_MPU_SH_NON, 0, 1, 1),
            ARM_MPU_RLAR((unsigned int)arm::app::tensorArenas + sizeof(arm::app::tensorArenas) - 1,
                         eMPU_ATTR_CACHEABLE_WTRA)
        },
    };
    InitPreDefMPURegion(&mpuConfig[0], sizeof(mpuConfig) / sizeof(mpuConfig[0]));
}

/*----------------------------------------------------------------------------
 * Test 1 — load both models + boot self-check (size + CRC32)
 *--------------------------------------------------------------------------*/
static bool test_load_both_models(void)
{
    printf("\n=== Test 1: Load Both Models (SD -> HyperRAM) ===\n");
    bool all_ok = true;

    for (int i = 0; i < MODEL_COUNT; i++)
    {
        const ModelInfo *m = &kModels[i];
        uint32_t size = 0, crc = 0;
        int rc = LoadModelToHyperRAM(m->file, m->load_addr, m->slot_size, &size, &crc);

        if (rc != 0)
        {
            all_ok = false;
            continue;
        }

        bool ok = true;
        if (size != m->expected_size)
        {
            printf_err("  SELFCHECK size mismatch: got %u, expected %u\n",
                       (unsigned)size, (unsigned)m->expected_size);
            ok = false;
        }
        if (crc != m->expected_crc32)
        {
            printf_err("  SELFCHECK CRC mismatch: got 0x%08X, expected 0x%08X\n",
                       (unsigned)crc, (unsigned)m->expected_crc32);
            ok = false;
        }
        if (ok)
            printf("  [SELFCHECK OK] size + CRC32 match expected\n");

        s_loadedSize[i] = size;
        s_loadOk[i]     = ok;
        all_ok          = all_ok && ok;
    }

    printf("Test 1 result: %s\n", all_ok ? "PASS" : "FAIL");
    return all_ok;
}

/*----------------------------------------------------------------------------
 * One-time interpreter construction. MicroMutableOpResolver rejects duplicate
 * op registration, so each Model object must be Init'd exactly once and then
 * stay resident — switching models later = just invoking the other object.
 *--------------------------------------------------------------------------*/
static bool init_model_once(arm::app::Model &model, int modelIdx,
                            uint8_t *arena, uint32_t arenaSz, const char *name)
{
    if (!s_loadOk[modelIdx])
    {
        printf_err("[INIT] %s skipped: model did not load/verify\n", name);
        return false;
    }

    uint32_t t0 = GetSystemTick_ms();
    bool ok = model.Init(arena, arenaSz,
                         (const uint8_t *)kModels[modelIdx].load_addr,
                         s_loadedSize[modelIdx]);
    uint32_t dt = GetSystemTick_ms() - t0;

    if (!ok)
    {
        printf_err("[INIT] %s model.Init() FAILED (arena %u KB @0x%08X)\n",
                   name, (unsigned)(arenaSz / 1024U), (unsigned)(uintptr_t)arena);
        return false;
    }

    printf("[INIT] %-8s interpreter resident: %u ms (arena %u KB @0x%08X)\n",
           name, (unsigned)dt, (unsigned)(arenaSz / 1024U), (unsigned)(uintptr_t)arena);
    s_initOk[modelIdx] = true;
    return true;
}

/*----------------------------------------------------------------------------
 * Inference benchmark (Tests 2 & 3) — model must already be Init'd.
 *--------------------------------------------------------------------------*/
static bool run_inference_bench(arm::app::Model &model, int modelIdx, const char *name)
{
    printf("\n=== Inference: %s (%d iterations) ===\n", name, INFER_ITERS);

    if (!s_initOk[modelIdx])
    {
        printf_err("  skipped: %s interpreter not initialised\n", name);
        return false;
    }

    uint64_t sum_us = 0, sumsq_us = 0;
    uint32_t max_us = 0, min_us = 0xFFFFFFFFu;
    uint32_t min_nz = 0xFFFFFFFFu;

    for (int i = 0; i < INFER_ITERS; i++)
    {
        fill_input(model, (uint32_t)i + 1u);

        uint32_t t0 = GetSystemTick_us();
        bool ran   = model.RunInference();
        uint32_t dt = GetSystemTick_us() - t0;

        if (!ran)
        {
            printf_err("  RunInference() failed at iter %d\n", i);
            return false;
        }

        uint32_t nz = output_nonzero_count(model);
        if (nz < min_nz) min_nz = nz;

        sum_us   += dt;
        sumsq_us += (uint64_t)dt * dt;
        if (dt > max_us) max_us = dt;
        if (dt < min_us) min_us = dt;
    }

    uint32_t avg_us = (uint32_t)(sum_us / INFER_ITERS);
    uint64_t var    = (sumsq_us / INFER_ITERS) - (uint64_t)avg_us * avg_us;
    uint32_t std_us = isqrt_u64(var);

    printf("  latency: avg=%u.%03u ms  max=%u.%03u ms  min=%u.%03u ms  std=%u us\n",
           avg_us / 1000, avg_us % 1000,
           max_us / 1000, max_us % 1000,
           min_us / 1000, min_us % 1000,
           std_us);
    printf("  output sanity: min non-zero bytes across runs = %u  -> %s\n",
           (unsigned)min_nz, (min_nz > 0) ? "OK (NPU produced data)"
                                          : "FAIL (all-zero output!)");

    /* BOTH criteria gate the verdict. (An earlier version returned only the
     * sanity check, so the final summary claimed PASS while the latency line
     * said FAIL — gesture @177 ms slipped through as a false PASS.) */
    bool latency_ok = (avg_us < 100000u);
    printf("  <100 ms target: %s\n", latency_ok ? "PASS" : "FAIL");

    return (min_nz > 0) && latency_ok;
}

/*----------------------------------------------------------------------------
 * Test 4 — alternating inference across two RESIDENT interpreters.
 * No re-Init per switch: MicroMutableOpResolver ops register once, so each
 * model was Init'd once (own arena slice) and switching = invoking the other
 * object. Switch overhead is therefore ~0; what we verify here is that 500
 * alternations run without crash, without cross-model state corruption, and
 * with sane (non-zero) NPU output every time.
 *--------------------------------------------------------------------------*/
static bool test_alternating(void)
{
    printf("\n=== Test 4: Alternating Inference (%d switches, resident interpreters) ===\n",
           ALT_SWITCHES);

    if (!s_initOk[MODEL_FACE] || !s_initOk[MODEL_GESTURE])
    {
        printf_err("  skipped: both interpreters must be initialised first\n");
        return false;
    }

    uint64_t inf_sum_us[MODEL_COUNT] = { 0, 0 };
    uint32_t inf_max_us[MODEL_COUNT] = { 0, 0 };
    uint32_t cnt[MODEL_COUNT]        = { 0, 0 };
    uint32_t min_nz_all              = 0xFFFFFFFFu;

    for (int i = 0; i < ALT_SWITCHES; i++)
    {
        int idx = i & 1;   /* alternate face / gesture */
        arm::app::Model &model = (idx == MODEL_FACE)
                                 ? static_cast<arm::app::Model &>(faceModel)
                                 : static_cast<arm::app::Model &>(gestureModel);

        fill_input(model, (uint32_t)i + 1u);

        uint32_t t0 = GetSystemTick_us();
        bool ran   = model.RunInference();
        uint32_t inf_us = GetSystemTick_us() - t0;

        if (!ran)
        {
            printf_err("  switch %d: RunInference() failed\n", i);
            printf("Test 4 result: FAIL at switch %d/%d\n", i, ALT_SWITCHES);
            return false;
        }

        uint32_t nz = output_nonzero_count(model);
        if (nz < min_nz_all) min_nz_all = nz;

        inf_sum_us[idx] += inf_us;
        if (inf_us > inf_max_us[idx]) inf_max_us[idx] = inf_us;
        cnt[idx]++;

        if (((i + 1) % 100) == 0)
            printf("  ...%d/%d switches ok\n", i + 1, ALT_SWITCHES);
    }

    for (int k = 0; k < MODEL_COUNT; k++)
    {
        if (cnt[k] == 0) continue;
        uint32_t inf_avg = (uint32_t)(inf_sum_us[k] / cnt[k]);
        printf("  %-8s: inference avg=%u.%03u ms  max=%u.%03u ms  (%u runs)\n",
               (k == MODEL_FACE) ? "FACE" : "GESTURE",
               inf_avg / 1000, inf_avg % 1000,
               inf_max_us[k] / 1000, inf_max_us[k] % 1000,
               (unsigned)cnt[k]);
    }
    printf("  switch overhead: ~0 (interpreters resident; no re-Init/re-alloc)\n");
    printf("  output sanity over all switches: min non-zero = %u -> %s\n",
           (unsigned)min_nz_all, (min_nz_all > 0) ? "OK" : "FAIL (all-zero!)");
    printf("Test 4 result: %s (%d/%d switches, no crash)\n",
           (min_nz_all > 0) ? "PASS" : "PARTIAL", ALT_SWITCHES, ALT_SWITCHES);

    return (min_nz_all > 0);
}

#endif /* RUN_DAY1_TESTS */

#if RUN_FACE_RECOG && !RUN_FACE_ENROLL
/*----------------------------------------------------------------------------
 * Phase-5: debounced recognition -> slideshow album filter. Called once per
 * captured camera frame with the recognised label, or NULL when nobody was
 * recognised this frame (no face, unknown face, or recog unavailable).
 *--------------------------------------------------------------------------*/
static char s_filterUser[32] = "";   /* active filter, "" = play all
                                        (also names the liker, Phase-6) */

static void SlideFilter_Update(const char *label)
{
    static char     candUser[32] = "";
    static int      candHits     = 0;
    static uint32_t lastSeenMs   = 0;

    if (label != NULL)
    {
        lastSeenMs = GetSystemTick_ms();

        if (strncmp(label, candUser, sizeof(candUser)) != 0)
        {
            strncpy(candUser, label, sizeof(candUser) - 1);
            candUser[sizeof(candUser) - 1] = '\0';
            candHits = 1;
        }
        else if (candHits < FILTER_SWITCH_HITS)
            candHits++;

        if (candHits >= FILTER_SWITCH_HITS &&
            strncmp(candUser, s_filterUser, sizeof(s_filterUser)) != 0)
        {
            strcpy(s_filterUser, candUser);
            int n = Slideshow_SetFilter(s_filterUser);
            printf("[FILTER] -> '%s' (%d album(s))\n", s_filterUser, n);
        }
    }
    else if (s_filterUser[0] != '\0' &&
             (GetSystemTick_ms() - lastSeenMs) > FILTER_CLEAR_MS)
    {
        s_filterUser[0] = '\0';
        candUser[0] = '\0';
        candHits    = 0;
        Slideshow_SetFilter(NULL);
        printf("[FILTER] -> all (nobody recognised for %u ms)\n",
               (unsigned)FILTER_CLEAR_MS);
    }
}
#endif /* RUN_FACE_RECOG && !RUN_FACE_ENROLL */

#if RUN_GESTURE_LIKE
/*----------------------------------------------------------------------------
 * Phase-6: debounced thumbs-up -> one like report to the backend.
 * LIKE_CONFIRM_HITS consecutive gesture frames confirm (kills single-frame
 * false positives); then LIKE_COOLDOWN_MS of silence so one held gesture
 * produces exactly one like.
 *--------------------------------------------------------------------------*/
static void LikeGesture_Update(int seen)
{
    static int      hits       = 0;
    static uint32_t lastFireMs = 0;

    if (seen <= 0)
    {
        hits = 0;
        return;
    }

    uint32_t now = GetSystemTick_ms();
    if (lastFireMs != 0 && (now - lastFireMs) < LIKE_COOLDOWN_MS)
        return;                          /* cooling down */
    if (++hits < LIKE_CONFIRM_HITS)
        return;

    hits       = 0;
    lastFireMs = now;

    char folder[64], photo[64];
    Slideshow_GetCurrent(folder, sizeof(folder), photo, sizeof(photo));

#if RUN_FACE_RECOG && !RUN_FACE_ENROLL
    const char *user = s_filterUser;
#else
    const char *user = "";
#endif

    printf("[LIKE] thumbs-up confirmed: %s/%s by '%s'\n",
           folder, photo, user[0] ? user : "(unknown)");

    /* Rail feedback reflects the actual delivery (POST is ~0.3 s). */
    if (SdSync_PostLike(folder, photo, user) == 0)
        StoryUI_ShowStatus("Like sent <3");
    else
        StoryUI_ShowStatus("Like failed :(");
}
#endif /* RUN_GESTURE_LIKE */

/*----------------------------------------------------------------------------
 * main
 *--------------------------------------------------------------------------*/
int main(void)
{
    BoardInit();          /* clocks, UART, HyperRAM map, SD mount, NPU init */

#if RUN_ESP_PROBE
    /* Must run before PerfTimer_Init(): the probe owns SysTick for delays. */
    esp_probe_run();      /* never returns */
#endif

    /* NOTE: no DWT register is read here anymore - touching DWT/CoreSight
     * without a live debug session hangs this board (standalone-reset bug). */
    if (PerfTimer_Init())             /* timing source: DWT, else SysTick */
        printf("[OK] timing source: DWT CYCCNT (cycle-accurate)\n");
    else
        printf("[OK] timing source: SysTick 64-bit (DWT disabled - see PerfTimer.h)\n");

#if RUN_DAY2_TESTS
    day2_test_run();      /* Wi-Fi + HTTP link validation; never returns */
#endif

#if RUN_DAY3_DEMO
    /* Day-3 demo: story rail + photo slideshow side by side. */
    if (Display_Init() == 0)
    {
        int d3 = day3_demo_run();
        if (d3 != 0)
            printf_err("Day-3 story failed (rc=%d) - running photos only\n", d3);

#if RUN_SD_SYNC
        /* Boot sync BEFORE the library scan and photo enrollment, so albums
         * and enroll files pushed from the App are picked up this boot. */
        if (SdSync_Boot() > 0)
            printf("[MAIN] SD content updated from backend\n");
#endif

        Slideshow_ReserveRight(STORYUI_RESERVED_PX);
        Slideshow_SetFilter(SLIDESHOW_SIM_USER);

        int slrc;

        if (Slideshow_LibScan(SLIDESHOW_DIR) < 0)
            slrc = -1;
        else
        {
#if RUN_CAMERA_PREVIEW
            /* Preview sits in the bottom-right corner of the photo region.
             * No camera (init fail) degrades to the plain slideshow. */
            bool     camOk = (Camera_Init() == 0);
#if SHOW_CAMERA_PREVIEW
            uint32_t camX  = Disaplay_GetLCDWidth() - STORYUI_RESERVED_PX - CAM_W;
            uint32_t camY  = Disaplay_GetLCDHeight() - CAM_H;
#endif
#if RUN_FACE_DETECT
            /* Face detection draws boxes onto the frame before it is blitted.
             * Init failure degrades to plain preview (no boxes). */
            bool fdOk = camOk && (FaceDetect_Init() == 0);
#if RUN_FACE_RECOG
            bool frOk = fdOk && (FaceRecog_Init() == 0);
#endif
#if RUN_GESTURE_LIKE
            /* Hand-landmark model + arena live at fixed HyperRAM addresses
             * (MemoryLayout.h demo plan). The linker knows nothing about
             * them, so first verify its SRAM01_HYPERRAM spill still ends
             * below the shrunken front guard. */
            bool glOk = false;
            {
                extern char Image$$SRAM01_HYPERRAM$$ZI$$Limit[];
                uint32_t spillEnd =
                    (uint32_t)(uintptr_t)Image$$SRAM01_HYPERRAM$$ZI$$Limit;

                if (spillEnd > DEMO_GUARD_END)
                {
                    /* printf_err is a two-statement macro — keep the braces */
                    printf_err("[GESTURE] arena spill 0x%08X exceeds guard "
                               "0x%08X - like feature disabled\n",
                               (unsigned)spillEnd, (unsigned)DEMO_GUARD_END);
                }
                else
                    glOk = camOk && (GestureLike_Init() == 0);
            }
#endif
#if RUN_FACE_ENROLL
            int  enrollSeen = 0;
#endif
            /* Single combined arena MPU setup (cacheable WTRA): the BSP
             * configures all app MPU regions in ONE InitPreDefMPURegion call,
             * so both arenas are set together here rather than per module. */
            if (fdOk
#if RUN_GESTURE_LIKE
                || glOk
#endif
               )
            {
                ARM_MPU_Region_t rg[3];
                uint32_t nrg = 0;
                void    *aBase; uint32_t aSize;

                if (fdOk)
                {
                    FaceDetect_GetArena(&aBase, &aSize);
                    rg[nrg].RBAR = ARM_MPU_RBAR((unsigned int)aBase, ARM_MPU_SH_NON, 0, 1, 1);
                    rg[nrg].RLAR = ARM_MPU_RLAR((unsigned int)aBase + aSize - 1, eMPU_ATTR_CACHEABLE_WTRA);
                    nrg++;
#if RUN_FACE_RECOG
                    if (frOk)
                    {
                        FaceRecog_GetArena(&aBase, &aSize);
                        rg[nrg].RBAR = ARM_MPU_RBAR((unsigned int)aBase, ARM_MPU_SH_NON, 0, 1, 1);
                        rg[nrg].RLAR = ARM_MPU_RLAR((unsigned int)aBase + aSize - 1, eMPU_ATTR_CACHEABLE_WTRA);
                        nrg++;
                    }
#endif
                }
#if RUN_GESTURE_LIKE
                /* HyperRAM arena gets the same WTRA policy the Day-1 spilled
                 * arena ran with (write-through so the NPU sees CPU-written
                 * inputs; the ethosu cache hooks handle the way back). */
                if (glOk)
                {
                    GestureLike_GetArena(&aBase, &aSize);
                    rg[nrg].RBAR = ARM_MPU_RBAR((unsigned int)aBase, ARM_MPU_SH_NON, 0, 1, 1);
                    rg[nrg].RLAR = ARM_MPU_RLAR((unsigned int)aBase + aSize - 1, eMPU_ATTR_CACHEABLE_WTRA);
                    nrg++;
                }
#endif
                InitPreDefMPURegion(&rg[0], nrg);
            }

#if RUN_PHOTO_ENROLL && RUN_FACE_RECOG && !RUN_FACE_ENROLL
            /* Photo enrollment: scan 0:\faces for enroll_<label>.raw files,
             * run each through the identical detect/crop/embed pipeline as
             * if it were a camera frame, enroll, then rename it *.done.
             * (Names are collected first — renaming while f_readdir walks
             * the directory could disturb the iteration.) */
            if (fdOk && frOk)
            {
                static char names[PHOTO_ENROLL_MAX][40];
                int     nFound = 0;
                DIR     dj;
                FILINFO fno;

                if (f_opendir(&dj, "0:\\faces") == FR_OK)
                {
                    while (nFound < PHOTO_ENROLL_MAX &&
                           f_readdir(&dj, &fno) == FR_OK && fno.fname[0])
                    {
                        size_t n = strlen(fno.fname);
                        if (n >= sizeof(names[0]) ||
                            n < 12 ||                       /* enroll_x.raw */
                            strncmp(fno.fname, "enroll_", 7) != 0 ||
                            strcmp(&fno.fname[n - 4], ".raw") != 0)
                            continue;
                        strcpy(names[nFound++], fno.fname);
                    }
                    f_closedir(&dj);
                }

                for (int pi = 0; pi < nFound; pi++)
                {
                    char   label[24], rawPath[56], donePath[56];
                    size_t ll = strlen(names[pi]) - 7 - 4;
                    if (ll >= sizeof(label)) continue;
                    memcpy(label, &names[pi][7], ll);
                    label[ll] = '\0';

                    /* enroll_user1-2.raw -> user1 (extra reference photos) */
                    char *dash = strrchr(label, '-');
                    if (dash && dash[1])
                    {
                        bool digits = true;
                        for (char *p = dash + 1; *p; p++)
                            if (*p < '0' || *p > '9') digits = false;
                        if (digits) *dash = '\0';
                    }

                    snprintf(rawPath,  sizeof(rawPath),  "0:\\faces\\%s", names[pi]);
                    snprintf(donePath, sizeof(donePath), "0:\\faces\\%s", names[pi]);
                    memcpy(&donePath[strlen(donePath) - 4], ".done", 6);

                    FIL pf;
                    if (f_open(&pf, rawPath, FA_READ) != FR_OK)
                        continue;

                    uint16_t *frame = (uint16_t *)Camera_GetFrame();
                    UINT      br    = 0;
                    FRESULT   fr2   = f_read(&pf, frame, CAM_W * CAM_H * 2, &br);
                    f_close(&pf);

                    if (fr2 == FR_OK && br == CAM_W * CAM_H * 2)
                    {
                        FaceBox tb;
                        if (FaceDetect_Run(frame, CAM_W, CAM_H) > 0 &&
                            FaceDetect_GetTopBox(&tb))
                        {
                            if (FaceRecog_Enroll(frame, CAM_W, CAM_H, &tb,
                                                 label) == 0)
                            {
                                f_unlink(donePath);
                                f_rename(rawPath, donePath);
                                printf("[PHOTO-ENROLL] '%s' enrolled from %s\n",
                                       label, rawPath);
                            }
                        }
                        else
                            printf("[PHOTO-ENROLL] no face detected in %s\n",
                                   rawPath);

#if SHOW_CAMERA_PREVIEW
                        Camera_Blit(camX, camY);   /* show photo + box briefly */
#endif
                    }
                    else
                        printf("[PHOTO-ENROLL] bad size in %s: read %u, want %u\n",
                               rawPath, (unsigned)br, (unsigned)(CAM_W * CAM_H * 2));
                }
            }
#endif
#endif
#endif

            for (;;)
            {
                slrc = Slideshow_ShowNext();
                if (slrc != 0) break;

#if RUN_SD_SYNC
                /* Doorbell between photos: a tiny /pending GET; the actual
                 * download runs only when the App pressed 「立即同步」. On
                 * fresh content, rescan so it enters the rotation now. */
                {
                    static uint32_t s_lastCheckMs = 0;
                    uint32_t now = GetSystemTick_ms();

                    if (now - s_lastCheckMs >= SDSYNC_CHECK_MS)
                    {
                        s_lastCheckMs = now;
                        if (SdSync_Poll() > 0)
                        {
                            Slideshow_LibScan(SLIDESHOW_DIR);
                            continue;           /* show the new content now */
                        }
                    }
                }
#endif

                /* Hold window: this idle time is where the camera (and later
                 * the face-recognition inference) runs. */
                uint32_t t0 = GetSystemTick_ms();

                while ((GetSystemTick_ms() - t0) < SLIDESHOW_HOLD_MS)
                {
#if RUN_CAMERA_PREVIEW
                    if (camOk)
                    {
                        if (Camera_Capture() != 0)
                        {
                            camOk = false;      /* capture died: stop trying */
                            continue;
                        }

                        uint16_t *frame = (uint16_t *)Camera_GetFrame();

#if RUN_GESTURE_LIKE
                        /* Gesture first: the frame is still clean here (face
                         * boxes are drawn into it by FaceDetect_Run below). */
                        if (glOk)
                            LikeGesture_Update(GestureLike_Run(frame, CAM_W, CAM_H));
#endif

#if RUN_FACE_DETECT
#if RUN_FACE_RECOG && !RUN_FACE_ENROLL
                        const char *seenUser = NULL;
#endif
                        if (fdOk && FaceDetect_Run(frame, CAM_W, CAM_H) > 0)
                        {
                            FaceBox tb;
                            if (FaceDetect_GetTopBox(&tb))
                            {
#if RUN_FACE_RECOG
#if RUN_FACE_ENROLL
                                if (frOk && enrollSeen < ENROLL_SAMPLES)
                                {
                                    FaceRecog_Enroll(frame, CAM_W, CAM_H, &tb, ENROLL_LABEL);
                                    if (++enrollSeen >= ENROLL_SAMPLES)
                                        printf("[ENROLL] captured %d samples — done\n", ENROLL_SAMPLES);
                                }
#else
                                if (frOk && FaceRecog_Run(frame, CAM_W, CAM_H, &tb) == 1)
                                    seenUser = FaceRecog_GetLabel();
#endif
#endif
                            }
                        }
#if RUN_FACE_RECOG && !RUN_FACE_ENROLL
                        SlideFilter_Update(seenUser);
#endif
#endif
#if SHOW_CAMERA_PREVIEW
                        Camera_Blit(camX, camY);
#endif
                        continue;
                    }
#endif
                    Display_Delay(50);
                }
            }
        }

        printf_err("Slideshow could not run (rc=%d) - keeping story screen\n", slrc);
        for (;;) __WFI();     /* keep whatever is on screen; demo is over */
    }
    printf_err("Display_Init failed (check LCD panel define in board_config.h)\n");
#endif

#if RUN_SLIDESHOW && !RUN_DAY3_DEMO
    /* Photo-frame milestone: cycle SD-card photos on the LCD. Never returns
     * while it has photos to show; on failure fall through to the tests. */
    if (Display_Init() == 0)
    {
        Slideshow_SetFilter(SLIDESHOW_SIM_USER);
        int slrc = Slideshow_Run(SLIDESHOW_DIR, SLIDESHOW_HOLD_MS);
        printf_err("Slideshow could not run (rc=%d) - continuing with tests\n", slrc);
    }
    else
    {
        printf_err("Display_Init failed (check LCD panel define in board_config.h) - continuing with tests\n");
    }
#endif

#if RUN_DAY1_TESTS
    printf("\n");
    printf("==============================================\n");
    printf(" M55M1 Dual-Model PoC - Day 1 Validation\n");
    printf("==============================================\n");
    printf(" HyperRAM   : FACE @0x%08X  GESTURE @0x%08X  slot=%u KB\n",
           (unsigned)FACE_MODEL_ADDR, (unsigned)GESTURE_MODEL_ADDR,
           (unsigned)(MODEL_SLOT_SIZE / 1024));
    printf(" arenas     : FACE %u KB @0x%08X | GESTURE %u KB @0x%08X (resident)\n",
           (unsigned)(FACE_ARENA_SZ / 1024),    (unsigned)(uintptr_t)FACE_ARENA_PTR,
           (unsigned)(GESTURE_ARENA_SZ / 1024), (unsigned)(uintptr_t)GESTURE_ARENA_PTR);

    bool t1 = test_load_both_models();

    setup_arena_mpu();    /* after load, before inference */

    /* Build both interpreters exactly once; they stay resident from here on. */
    printf("\n=== Interpreter Init (once per model) ===\n");
    init_model_once(faceModel,    MODEL_FACE,    FACE_ARENA_PTR,    FACE_ARENA_SZ,    "FACE");
    init_model_once(gestureModel, MODEL_GESTURE, GESTURE_ARENA_PTR, GESTURE_ARENA_SZ, "GESTURE");

    bool t2 = run_inference_bench(faceModel,    MODEL_FACE,    "FACE");
    bool t3 = run_inference_bench(gestureModel, MODEL_GESTURE, "GESTURE");
    bool t4 = test_alternating();

    /* ---- Summary mapped to the Day-1 success criteria ---- */
    printf("\n==============================================\n");
    printf(" Day-1 Summary\n");
    printf("==============================================\n");
    printf(" [%s] Both models load to non-overlapping addrs + CRC OK\n", t1 ? "PASS" : "FAIL");
    printf(" [%s] Face inference < 100 ms + output valid\n",            t2 ? "PASS" : "FAIL");
    printf(" [%s] Gesture inference < 100 ms + output valid\n",         t3 ? "PASS" : "FAIL");
    printf(" [%s] %d alternating switches, no crash, output valid\n",   t4 ? "PASS" : "FAIL", ALT_SWITCHES);
    printf(" Overall: %s\n", (t1 && t2 && t3 && t4) ? "DAY-1 PASS" : "NEEDS REVIEW");
    printf("==============================================\n");
    printf("=== All Tests Done ===\n");
#else
    printf("\n(Day-1 validation compiled out - RUN_DAY1_TESTS=1 to re-run; results in final_report.md)\n");
#endif /* RUN_DAY1_TESTS */

    while (1) { __WFI(); }
}
