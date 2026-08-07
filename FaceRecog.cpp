/**************************************************************************//**
 * @file     FaceRecog.cpp
 * @brief    Phase-4 face recognition glue — see FaceRecog.hpp.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "FaceRecog.hpp"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

#include "NuMicro.h"
#include "ff.h"

#include "MemoryLayout.h"          /* FACE_MODEL_ADDR, MODEL_SLOT_SIZE */
#include "ModelLoader.h"           /* LoadModelToHyperRAM */
#include "FaceMobileNetModel.hpp"  /* embedding interpreter + Model.hpp */
#include "Recognizer.hpp"
#include "RecognitionResult.hpp"
#include "Labels.hpp"

/* Embedding model: SD file -> HyperRAM. Address/size come from the DAY-3
 * demo HyperRAM plan in MemoryLayout.h (EMBED_MODEL_ADDR @ 0x82080000,
 * 3.25 MB slot — face_mobilenet.tflite is ~3.17 MB). */
#define EMBED_MODEL_FILE   "0:\\face_mobilenet.tflite"

/* Sample uses 460 KB; 512 KB gives headroom. Tail may spill into the HyperRAM
 * front guard (safe, slower) — the embedding runs once per detection, not per
 * frame, so that is acceptable. */
#define EMBED_ARENA_SZ     (0x00080000UL)             /* 512 KB */

#define EMBED_REF_DIR      "0:\\faces"
#define EMBED_REF_FILE     "0:\\faces\\embeddings.txt"

/* 0.6 rejects photo-enrolled references: live-vs-photo cosine peaks at ~0.53
 * (live-vs-live is 0.8+), so the App/selfie registration path needs 0.50. */
#define RECOG_THRESHOLD    (0.50)
#define RECOG_BOX_COLOR    (0x07E0)   /* green in RGB565 */
#define RECOG_BOX_THICK    (2)

__attribute__((section(".bss.NoInit.activation_buf_sram"), aligned(16)))
static uint8_t s_arena[EMBED_ARENA_SZ];

static arm::app::FaceMobileNetModel s_model;
static arm::app::Recognizer         s_recognizer;
static std::vector<S_LABEL_INFO>    s_labels;

static char          s_lastLabel[32] = "";
static bool          s_ready   = false;
static int           s_inCols  = 0, s_inRows = 0;
static TfLiteTensor *s_in      = nullptr;
static TfLiteTensor *s_out     = nullptr;

extern "C" void FaceRecog_GetArena(void **base, uint32_t *size)
{
    if (base) *base = s_arena;
    if (size) *size = (uint32_t)sizeof(s_arena);
}

extern "C" int FaceRecog_Init(void)
{
    uint32_t loadedSz = 0, crc = 0;
    if (LoadModelToHyperRAM(EMBED_MODEL_FILE, EMBED_MODEL_ADDR,
                            EMBED_MODEL_MAXSZ, &loadedSz, &crc) != 0)
    {
        printf("[FACEREC] SD load failed (%s missing on card?)\n", EMBED_MODEL_FILE);
        return -1;
    }

    if (!s_model.Init(s_arena, sizeof(s_arena),
                      (const uint8_t *)EMBED_MODEL_ADDR, loadedSz))
    {
        printf("[FACEREC] model Init failed (arena %u KB too small?)\n",
               (unsigned)(EMBED_ARENA_SZ / 1024U));
        return -2;
    }

    s_in  = s_model.GetInputTensor(0);
    s_out = s_model.GetOutputTensor(0);
    TfLiteIntArray *shape = s_model.GetInputShape(0);
    if (!s_in || !s_out || !shape || shape->size < 4)
    {
        printf("[FACEREC] unexpected tensor layout\n");
        return -3;
    }

    s_inRows = shape->data[arm::app::FaceMobileNetModel::ms_inputRowsIdx];
    s_inCols = shape->data[arm::app::FaceMobileNetModel::ms_inputColsIdx];
    const int nCh = shape->data[arm::app::FaceMobileNetModel::ms_inputChannelsIdx];
    if (nCh != 3 || s_in->bytes != (size_t)(s_inCols * s_inRows * 3))
    {
        printf("[FACEREC] expected %dx%dx3 RGB input, got %dx%dx%d (%u B)\n",
               s_inCols, s_inRows, s_inCols, s_inRows, nCh, (unsigned)s_in->bytes);
        return -4;
    }

    s_recognizer.SetThreshold(RECOG_THRESHOLD);

    /* Reference embeddings are optional at enroll time. */
    size_t n = 0;
    if (ParserLabelVectorFromFile(EMBED_REF_FILE, s_labels, &n))
        printf("[FACEREC] loaded %u reference embedding(s) from %s\n",
               (unsigned)s_labels.size(), EMBED_REF_FILE);
    else
        printf("[FACEREC] no references yet (%s) — enroll first\n", EMBED_REF_FILE);

    printf("[FACEREC] ready: input %dx%dx3, arena %u KB\n",
           s_inCols, s_inRows, (unsigned)(EMBED_ARENA_SZ / 1024U));
    s_ready = true;
    return 0;
}

/* Crop the face box from the RGB565 frame, nearest-neighbour resize into the
 * model's RGB888 input, then quantise int8 = uint8 - 128 (same as the BSP
 * FaceRecognition sample). */
static bool run_embedding(const uint16_t *frame, int fw, int fh, const FaceBox *box)
{
    if (!box || box->w <= 0 || box->h <= 0) return false;

    uint8_t *u = static_cast<uint8_t *>(s_in->data.data);

    for (int y = 0; y < s_inRows; y++)
    {
        int sy = box->y + y * box->h / s_inRows;
        if (sy < 0) sy = 0; else if (sy >= fh) sy = fh - 1;
        const uint16_t *srow = frame + sy * fw;
        uint8_t *drow = u + (size_t)y * s_inCols * 3;

        for (int x = 0; x < s_inCols; x++)
        {
            int sx = box->x + x * box->w / s_inCols;
            if (sx < 0) sx = 0; else if (sx >= fw) sx = fw - 1;
            uint16_t px = srow[sx];
            drow[x * 3 + 0] = (uint8_t)(((px >> 11) & 0x1F) * 255 / 31);   /* R */
            drow[x * 3 + 1] = (uint8_t)(((px >> 5)  & 0x3F) * 255 / 63);   /* G */
            drow[x * 3 + 2] = (uint8_t)(( px        & 0x1F) * 255 / 31);   /* B */
        }
    }

    int8_t *s = static_cast<int8_t *>(s_in->data.data);
    for (size_t i = 0; i < s_in->bytes; i++)
        s[i] = static_cast<int8_t>((int)u[i] - 128);

    return s_model.RunInference();
}

/* Dequantise the output tensor into a float embedding (mirrors Recognizer). */
static void read_embedding(std::vector<float> &emb)
{
    uint32_t dim = 1;
    for (int i = 0; i < s_out->dims->size; i++) dim *= s_out->dims->data[i];

    arm::app::QuantParams q = arm::app::GetTensorQuantParams(s_out);
    emb.resize(dim);

    if (s_out->type == kTfLiteInt8)
        for (uint32_t i = 0; i < dim; i++)
            emb[i] = q.scale * ((float)s_out->data.int8[i] - q.offset);
    else if (s_out->type == kTfLiteUInt8)
        for (uint32_t i = 0; i < dim; i++)
            emb[i] = q.scale * ((float)s_out->data.uint8[i] - q.offset);
    else /* kTfLiteFloat32 */
        for (uint32_t i = 0; i < dim; i++)
            emb[i] = s_out->data.f[i];
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

extern "C" int FaceRecog_Run(uint16_t *frame, int fw, int fh, const FaceBox *box)
{
    if (!s_ready) return -1;
    if (!run_embedding(frame, fw, fh, box))
    {
        printf("[FACEREC] inference failed\n");
        return -2;
    }

    arm::app::RecognitionResult result;
    bool ok = false;
    if (!s_labels.empty())
        ok = s_recognizer.GetRecognitionResults(s_out, result, s_labels);

    if (ok && result.m_recognize)
    {
        printf("[FACEREC] recognized: %s (%.3f)\n", result.m_label.c_str(), result.m_predict);
        strncpy(s_lastLabel, result.m_label.c_str(), sizeof(s_lastLabel) - 1);
        s_lastLabel[sizeof(s_lastLabel) - 1] = '\0';
        draw_rect(frame, fw, fh, box->x, box->y, box->w, box->h,
                  RECOG_BOX_COLOR, RECOG_BOX_THICK);
        return 1;
    }

    printf("[FACEREC] unknown (best %.3f)\n", result.m_predict);
    return 0;
}

extern "C" const char *FaceRecog_GetLabel(void)
{
    return s_lastLabel;
}

extern "C" int FaceRecog_Enroll(uint16_t *frame, int fw, int fh,
                                const FaceBox *box, const char *label)
{
    if (!s_ready) return -1;
    if (!run_embedding(frame, fw, fh, box))
    {
        printf("[FACEREC] enroll inference failed\n");
        return -2;
    }

    std::vector<float> emb;
    read_embedding(emb);

    f_mkdir(EMBED_REF_DIR);   /* ignore FR_EXIST */

    FIL fil;
    if (f_open(&fil, EMBED_REF_FILE, FA_WRITE | FA_OPEN_APPEND) != FR_OK)
    {
        printf("[FACEREC] cannot open %s for append\n", EMBED_REF_FILE);
        return -3;
    }

    /* Format MUST match ParserLabelVectorFromFile (Labels.cpp): a ':'-
     * delimited line "label:v0:v1:...:vN:" — every value, including the last,
     * is followed by ':' so the parser (which only emits a token when it finds
     * the next ':') captures all of them. Comma separators leave fParam empty. */
    UINT bw;
    char buf[32];
    int len = snprintf(buf, sizeof(buf), "%s:", label);
    f_write(&fil, buf, len, &bw);
    for (size_t i = 0; i < emb.size(); i++)
    {
        len = snprintf(buf, sizeof(buf), "%.6f:", emb[i]);
        f_write(&fil, buf, len, &bw);
    }
    f_write(&fil, "\n", 1, &bw);
    f_close(&fil);

    /* Also register in-memory so the new reference matches immediately,
     * without a reboot/re-init (used by the photo-enroll one-shot). */
    s_labels.push_back({label, emb});

    printf("[FACEREC] enrolled '%s' (%u-dim) -> %s\n",
           label, (unsigned)emb.size(), EMBED_REF_FILE);
    return 0;
}

extern "C" int FaceRecog_ForgetLabel(const char *label)
{
    if (!label || !label[0]) return -1;

    /* 1) In-RAM references: recognition stops matching immediately. */
    int removed = 0;
    for (size_t i = s_labels.size(); i-- > 0; )
        if (s_labels[i].szLable == label)
        {
            s_labels.erase(s_labels.begin() + (long)i);
            removed++;
        }

    /* 2) SD file: stream-filter embeddings.txt line by line, dropping the
     * label's lines ("label:v0:v1:...:"). Works even when recognition never
     * initialised (file may still hold the label from an earlier boot). */
    FIL src, dst;
    if (f_open(&src, EMBED_REF_FILE, FA_READ) != FR_OK)
        return removed;                 /* no reference file: RAM-only purge */

    const char *tmpPath = EMBED_REF_DIR "\\embeddings.tmp";
    if (f_open(&dst, tmpPath, FA_CREATE_ALWAYS | FA_WRITE) != FR_OK)
    {
        f_close(&src);
        printf("[FACEREC] forget '%s': cannot create temp file\n", label);
        return -2;
    }

    char rbuf[512], lbl[32];
    UINT br = 0, bw = 0;
    int  li = 0;
    bool inLabel = true, keep = true, dropped = false, ioErr = false;

    for (;;)
    {
        if (f_read(&src, rbuf, sizeof(rbuf), &br) != FR_OK) { ioErr = true; break; }
        if (br == 0) break;

        for (UINT i = 0; i < br && !ioErr; i++)
        {
            char c = rbuf[i];

            if (inLabel)
            {
                if (c == ':' || c == '\n' || li >= (int)sizeof(lbl) - 1)
                {
                    lbl[li] = '\0';
                    keep = (strcmp(lbl, label) != 0);
                    if (!keep) dropped = true;
                    if (keep &&
                        (f_write(&dst, lbl, (UINT)li, &bw) != FR_OK ||
                         f_write(&dst, &c, 1, &bw) != FR_OK))
                        ioErr = true;
                    inLabel = (c == '\n');
                    li = 0;
                }
                else
                    lbl[li++] = c;
            }
            else
            {
                if (keep && f_write(&dst, &c, 1, &bw) != FR_OK)
                    ioErr = true;
                if (c == '\n') inLabel = true;
            }
        }
        if (ioErr) break;
    }

    /* Trailing partial label (file without final newline): keep it. */
    if (!ioErr && inLabel && li > 0 &&
        f_write(&dst, lbl, (UINT)li, &bw) != FR_OK)
        ioErr = true;

    f_close(&src);
    f_close(&dst);

    if (ioErr)
    {
        f_unlink(tmpPath);
        printf("[FACEREC] forget '%s': SD I/O error, file kept\n", label);
        return -2;
    }

    if (dropped)
    {
        f_unlink(EMBED_REF_FILE);
        f_rename(tmpPath, EMBED_REF_FILE);
        printf("[FACEREC] forgot '%s' (%d RAM ref(s), file rewritten)\n",
               label, removed);
    }
    else
        f_unlink(tmpPath);              /* label not in file: nothing changed */

    return removed + (dropped ? 1 : 0);
}
