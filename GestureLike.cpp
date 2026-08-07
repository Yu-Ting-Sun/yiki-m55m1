/**************************************************************************//**
 * @file     GestureLike.cpp
 * @brief    Thumbs-up detection glue — see GestureLike.hpp.
 *
 * Pipeline per frame: RGB565 240x240 -> nearest-neighbour resize to the
 * model's 224x224x3 int8 input (int8 = channel - 128, exactly what the BSP
 * HandPoseRecognition sample does) -> inference -> dequantise the presence
 * score + 21 screen-space landmarks -> geometric thumbs-up rule.
 *
 * Landmark indices (MediaPipe): 0 wrist; 1-4 thumb (CMC,MCP,IP,TIP);
 * 5-8 index (MCP,PIP,DIP,TIP); 9-12 middle; 13-16 ring; 17-20 pinky.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "GestureLike.hpp"

#include <cstdio>
#include <cstdint>
#include <cmath>

#include "NuMicro.h"

#include "MemoryLayout.h"          /* HAND_MODEL_ADDR / HAND_ARENA_ADDR ... */
#include "ModelLoader.h"
#include "HandLandmarkModel.hpp"

/* Output tensor indices (from the BSP HandPoseRecognition sample). */
#define OUT_IDX_HANDEDNESS   (0)
#define OUT_IDX_WORLD        (1)
#define OUT_IDX_PRESENCE     (2)
#define OUT_IDX_SCREEN       (3)

/* A hand is only considered when the model is at least this sure. */
#define PRESENCE_THRESHOLD   (0.6f)

/* Hand must be reasonably close to the camera: wrist-to-middle-MCP distance
 * in model-input pixels (224x224). Rejects tiny far-away false hands. */
#define MIN_HAND_SPAN_PX     (18.0f)

static arm::app::HandLandmarkModel s_model;
static bool          s_ready   = false;
static int           s_inCols  = 0, s_inRows = 0;
static TfLiteTensor *s_in      = nullptr;
static TfLiteTensor *s_outScr  = nullptr;
static TfLiteTensor *s_outPres = nullptr;

extern "C" void GestureLike_GetArena(void **base, uint32_t *size)
{
    if (base) *base = (void *)HAND_ARENA_ADDR;
    if (size) *size = (uint32_t)HAND_ARENA_SZ;
}

extern "C" int GestureLike_Init(void)
{
    uint32_t loadedSz = 0, crc = 0;

    if (LoadModelToHyperRAM(HAND_MODEL_FILE, HAND_MODEL_ADDR,
                            HAND_MODEL_MAXSZ, &loadedSz, &crc) != 0)
    {
        printf("[GESTURE] SD load failed (%s missing on card?)\n", HAND_MODEL_FILE);
        return -1;
    }

    if (!s_model.Init((uint8_t *)HAND_ARENA_ADDR, HAND_ARENA_SZ,
                      (const uint8_t *)HAND_MODEL_ADDR, loadedSz))
    {
        printf("[GESTURE] model Init failed (arena %u KB too small?)\n",
               (unsigned)(HAND_ARENA_SZ / 1024U));
        return -2;
    }

    s_in      = s_model.GetInputTensor(0);
    s_outScr  = s_model.GetOutputTensor(OUT_IDX_SCREEN);
    s_outPres = s_model.GetOutputTensor(OUT_IDX_PRESENCE);

    TfLiteIntArray *shape = s_model.GetInputShape(0);
    if (!s_in || !s_outScr || !s_outPres || !shape || shape->size < 4)
    {
        printf("[GESTURE] unexpected tensor layout\n");
        return -3;
    }

    s_inRows = shape->data[arm::app::HandLandmarkModel::ms_inputRowsIdx];
    s_inCols = shape->data[arm::app::HandLandmarkModel::ms_inputColsIdx];
    const int nCh = shape->data[arm::app::HandLandmarkModel::ms_inputChannelsIdx];

    if (nCh != 3 || s_in->bytes != (size_t)(s_inCols * s_inRows * 3) ||
        s_outScr->bytes < 21 * 3)
    {
        printf("[GESTURE] expected %dx%dx3 input / 63B screen output, got "
               "%dch (%u B in, %u B out)\n", s_inCols, s_inRows, nCh,
               (unsigned)s_in->bytes, (unsigned)s_outScr->bytes);
        return -4;
    }

    printf("[GESTURE] ready: input %dx%dx3, model %u KB @0x%08X, arena %u KB "
           "@0x%08X (HyperRAM)\n", s_inCols, s_inRows,
           (unsigned)(loadedSz / 1024U), (unsigned)HAND_MODEL_ADDR,
           (unsigned)(HAND_ARENA_SZ / 1024U), (unsigned)HAND_ARENA_ADDR);
    s_ready = true;
    return 0;
}

/* Nearest-neighbour RGB565 -> RGB888 resize straight into the input tensor,
 * quantised the way the sample does it: int8 = channel(0..255) - 128. */
static void preprocess(const uint16_t *frame, int w, int h)
{
    int8_t *dst = static_cast<int8_t *>(s_in->data.data);

    for (int y = 0; y < s_inRows; y++)
    {
        const uint16_t *srow = frame + (y * h / s_inRows) * w;

        for (int x = 0; x < s_inCols; x++)
        {
            uint16_t px = srow[x * w / s_inCols];
            int r = ((px >> 11) & 0x1F) * 255 / 31;
            int g = ((px >> 5)  & 0x3F) * 255 / 63;
            int b = ( px        & 0x1F) * 255 / 31;
            *dst++ = static_cast<int8_t>(r - 128);
            *dst++ = static_cast<int8_t>(g - 128);
            *dst++ = static_cast<int8_t>(b - 128);
        }
    }
}

static inline float dequant1(const TfLiteTensor *t, int i)
{
    const TfLiteAffineQuantization *q =
        (const TfLiteAffineQuantization *)t->quantization.params;
    return q->scale->data[0] *
           ((float)t->data.int8[i] - (float)q->zero_point->data[0]);
}

static inline float d2(const float *p, int a, int b)
{
    float dx = p[a * 3] - p[b * 3];
    float dy = p[a * 3 + 1] - p[b * 3 + 1];
    return dx * dx + dy * dy;
}

extern "C" int GestureLike_Run(const uint16_t *frameRGB565, int w, int h)
{
    if (!s_ready) return -1;

    preprocess(frameRGB565, w, h);

    if (!s_model.RunInference())
    {
        printf("[GESTURE] RunInference failed\n");
        return -2;
    }

    if (dequant1(s_outPres, 0) < PRESENCE_THRESHOLD)
        return 0;                       /* no hand in frame */

    /* 21 x (x, y, z) in model-input pixel coordinates (224x224 space). */
    float lm[21 * 3];
    for (int i = 0; i < 21 * 3; i++)
        lm[i] = dequant1(s_outScr, i);

    /* Reject tiny/far hands: span = wrist(0) .. middle MCP(9). */
    float span2 = d2(lm, 0, 9);
    if (span2 < MIN_HAND_SPAN_PX * MIN_HAND_SPAN_PX)
        return 0;
    float span = sqrtf(span2);

    /* Thumb extended upward (screen y grows downward):
     * TIP(4) above IP(3) above MCP(2), and TIP well above the wrist. */
    float thumbTipY = lm[4 * 3 + 1], thumbIpY = lm[3 * 3 + 1];
    float thumbMcpY = lm[2 * 3 + 1], wristY   = lm[0 * 3 + 1];

    bool thumbUp = (thumbTipY < thumbIpY) && (thumbIpY < thumbMcpY) &&
                   ((wristY - thumbTipY) > 0.55f * span);

    /* The other four fingers curled into a fist: each fingertip closer to the
     * wrist than its PIP joint (works regardless of hand rotation). */
    static const uint8_t tips[4] = { 8, 12, 16, 20 };
    static const uint8_t pips[4] = { 6, 10, 14, 18 };
    bool curled = true;
    for (int f = 0; f < 4; f++)
        if (d2(lm, tips[f], 0) >= d2(lm, pips[f], 0))
            curled = false;

    return (thumbUp && curled) ? 1 : 0;
}
