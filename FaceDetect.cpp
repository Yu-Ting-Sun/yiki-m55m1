/**************************************************************************//**
 * @file     FaceDetect.cpp
 * @brief    Phase-3 live face detection glue — see FaceDetect.hpp.
 *
 * Reuses the BSP FaceDetection pipeline verbatim: the compiled-in
 * yolo-fastest_192_face Vela model (arm::app::nn::GetModelPointer), the
 * NNModel interpreter (AddEthosU) and the anchor-based DetectorPostProcess.
 * Only the plumbing around them (arena, MPU, RGB565->grayscale resize and
 * box drawing) lives here.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "FaceDetect.hpp"

#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>

#include "NuMicro.h"
#include "mpu_config_M55M1.h"          /* eMPU_ATTR_CACHEABLE_WTRA */

#include "NNModel.hpp"                 /* interpreter + extern anchor1/anchor2 */
#include "DetectorPostProcessing.hpp"
#include "DetectionResult.hpp"

/* The compiled-in model provides these in namespace arm::app::nn (no header,
 * same extern trick the FaceDetection sample uses in its main.cpp). */
namespace arm { namespace app { namespace nn {
extern const uint8_t *GetModelPointer();
extern size_t         GetModelLen();
} } }

/*----------------------------------------------------------------------------
 * Dedicated tensor arena. yolo-fastest_192_face is tiny; the BSP sample gives
 * it 1 MB but that whole budget cannot coexist with the 112 KB camera frame
 * inside the 1 MB SRAM01. 512 KB keeps the arena entirely in NPU-reachable
 * SRAM01 (Ethos-U55 cannot touch SPIM0/EBI). If NNModel.Init() reports it is
 * too small on the VM build, bump this — the failure is loud and non-fatal.
 *--------------------------------------------------------------------------*/
#define FACEDET_ARENA_SZ   (0x00080000UL)   /* 512 KB */

__attribute__((section(".bss.NoInit.activation_buf_sram"), aligned(16)))
static uint8_t s_arena[FACEDET_ARENA_SZ];

/* Draw threshold + colour for face boxes (red in RGB565). */
#define FACEDET_BOX_COLOR  (0xF800)
#define FACEDET_BOX_THICK  (2)

static arm::app::NNModel s_model;
static bool          s_ready = false;
static int           s_inCols = 0, s_inRows = 0;   /* model input W, H */
static TfLiteTensor *s_in   = nullptr;
static TfLiteTensor *s_out0 = nullptr;
static TfLiteTensor *s_out1 = nullptr;

/* Post-processor is built lazily on the first Run (it needs the frame size);
 * it holds a const ref to s_ppParams, so both must be static and outlive it. */
static std::vector<arm::app::object_detection::DetectionResult> s_results;
static arm::app::object_detection::PostProcessParams           s_ppParams;
static arm::app::DetectorPostProcess                          *s_post = nullptr;
static int s_postW = 0, s_postH = 0;

/* Cacheable WTRA over the arena, mirroring the FaceDetection sample so the
 * CPU<->NPU cache behaviour is identical. BoardInit sets the base regions;
 * this appends one for our arena. */
static void setup_arena_mpu(void)
{
    const ARM_MPU_Region_t cfg[] =
    {
        {
            ARM_MPU_RBAR((unsigned int)s_arena, ARM_MPU_SH_NON, 0, 1, 1),
            ARM_MPU_RLAR((unsigned int)s_arena + sizeof(s_arena) - 1,
                         eMPU_ATTR_CACHEABLE_WTRA)
        },
    };
    InitPreDefMPURegion(&cfg[0], sizeof(cfg) / sizeof(cfg[0]));
}

extern "C" int FaceDetect_Init(void)
{
    if (!s_model.Init(s_arena, sizeof(s_arena),
                      arm::app::nn::GetModelPointer(),
                      arm::app::nn::GetModelLen()))
    {
        printf("[FACEDET] model Init failed (arena %u KB too small?)\n",
               (unsigned)(FACEDET_ARENA_SZ / 1024U));
        return -1;
    }

    setup_arena_mpu();

    s_in   = s_model.GetInputTensor(0);
    s_out0 = s_model.GetOutputTensor(0);
    s_out1 = s_model.GetOutputTensor(1);

    TfLiteIntArray *shape = s_model.GetInputShape(0);
    if (!s_in || !s_out0 || !s_out1 || !shape || shape->size < 4)
    {
        printf("[FACEDET] unexpected tensor layout\n");
        return -2;
    }

    s_inRows = shape->data[arm::app::NNModel::ms_inputRowsIdx];
    s_inCols = shape->data[arm::app::NNModel::ms_inputColsIdx];
    const int nCh = shape->data[arm::app::NNModel::ms_inputChannelsIdx];

    if (nCh != 1 || s_in->bytes != (size_t)(s_inCols * s_inRows))
    {
        printf("[FACEDET] expected %dx%dx1 grayscale input, got %dx%dx%d (%u B)\n",
               s_inCols, s_inRows, s_inCols, s_inRows, nCh, (unsigned)s_in->bytes);
        return -3;
    }

    printf("[FACEDET] ready: input %dx%dx1, arena %u KB\n",
           s_inCols, s_inRows, (unsigned)(FACEDET_ARENA_SZ / 1024U));
    s_ready = true;
    return 0;
}

/* Nearest-neighbour downscale of an RGB565 frame into the model's grayscale
 * int8 input tensor. Luma = (77R + 150G + 29B) >> 8; int8 = luma - 128, which
 * is exactly the quantisation the FaceDetection sample applies. */
static void preprocess(const uint16_t *frame, int w, int h)
{
    int8_t *dst = static_cast<int8_t *>(s_in->data.data);

    for (int y = 0; y < s_inRows; y++)
    {
        const uint16_t *srow = frame + (y * h / s_inRows) * w;
        int8_t         *drow = dst + y * s_inCols;

        for (int x = 0; x < s_inCols; x++)
        {
            uint16_t px = srow[x * w / s_inCols];
            int r = ((px >> 11) & 0x1F) * 255 / 31;
            int g = ((px >> 5)  & 0x3F) * 255 / 63;
            int b = ( px        & 0x1F) * 255 / 31;
            int gray = (r * 77 + g * 150 + b * 29) >> 8;   /* 0..255 */
            drow[x] = static_cast<int8_t>(gray - 128);
        }
    }
}

static void draw_rect(uint16_t *fb, int W, int H,
                      int x0, int y0, int bw, int bh, uint16_t c, int th)
{
    int x1 = x0 + bw, y1 = y0 + bh;
    int xa = std::max(0, x0), xb = std::min(W, x1);
    int ya = std::max(0, y0), yb = std::min(H, y1);

    for (int t = 0; t < th; t++)
    {
        int yt = y0 + t, ybt = y1 - 1 - t;
        if (yt  >= 0 && yt  < H) for (int x = xa; x < xb; x++) fb[yt  * W + x] = c;
        if (ybt >= 0 && ybt < H) for (int x = xa; x < xb; x++) fb[ybt * W + x] = c;
        int xl = x0 + t, xr = x1 - 1 - t;
        if (xl >= 0 && xl < W) for (int y = ya; y < yb; y++) fb[y * W + xl] = c;
        if (xr >= 0 && xr < W) for (int y = ya; y < yb; y++) fb[y * W + xr] = c;
    }
}

extern "C" int FaceDetect_Run(uint16_t *frameRGB565, int w, int h)
{
    if (!s_ready) return -1;

    /* (Re)build the post-processor if the frame size changed. */
    if (!s_post || w != s_postW || h != s_postH)
    {
        delete s_post;
        s_ppParams = arm::app::object_detection::PostProcessParams{
            s_inRows, s_inCols, h, w, anchor1, anchor2};
        s_post = new arm::app::DetectorPostProcess(s_out0, s_out1, s_results, s_ppParams);
        s_postW = w;
        s_postH = h;
    }

    preprocess(frameRGB565, w, h);

    if (!s_model.RunInference())
    {
        printf("[FACEDET] RunInference failed\n");
        return -2;
    }

    s_results.clear();
    s_post->RunPostProcess(s_results);

    for (const auto &box : s_results)
        draw_rect(frameRGB565, w, h,
                  box.m_x0, box.m_y0, box.m_w, box.m_h,
                  FACEDET_BOX_COLOR, FACEDET_BOX_THICK);

    return (int)s_results.size();
}
