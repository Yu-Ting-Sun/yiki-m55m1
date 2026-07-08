/**************************************************************************//**
 * @file     Slideshow.c
 * @brief    SD-card photo slideshow on the LCD — JPG (on-device decode) + BMP.
 *
 * JPEG: decoded on-device with TJpgDec (baseline JPEG, no malloc, static
 * pool). Phone photos (e.g. 4032x3024) use TJpgDec's built-in 1/2 1/4 1/8
 * descaling: the largest scale whose output fits the panel is chosen, decoded
 * into a HyperRAM frame buffer (rear spare @SLIDESHOW_FB_ADDR), then blitted
 * to the LCD in bands through an SRAM2 bounce buffer. Progressive JPEGs are
 * not decodable on this RAM class and are skipped with a message.
 *
 * BMP: uncompressed 24-bpp, streamed straight from SD in bands (no FB needed).
 *
 * PNG/HEIC: skipped with a hint to use scripts/prepare_pictures.py.
 *
 * Buffers: band buffers + TJpgDec pool live in `.bss.vram.data` (SRAM2,
 * 320 KB, CPU/PDMA-only — never used by the models or the NPU).
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "Slideshow.h"

#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "NuMicro.h"
#include "ff.h"
#include "Display.h"
#include "MemoryLayout.h"   /* SLIDESHOW_FB_ADDR / _SIZE */
#include "tjpgd.h"

/* Show the file name in the bottom-left corner of each photo. */
#define SLIDE_CAPTION       (1)

#define SLIDE_MAX_W         (800)   /* LT7381 horizontal resolution */
#define SLIDE_BAND_ROWS     (20)
#define SLIDE_MAX_ROWBYTES  (((SLIDE_MAX_W * 3u) + 3u) & ~3u)

/* TJpgDec working pool. Requirement (R0.03, JD_FASTDECODE=2): ~3.5 KB base
 * + 6 KB huffman LUT + JD_SZBUF (4 KB) stream buffer + MCU work area. 32 KB
 * leaves ample headroom. */
#define SLIDE_JD_POOL_SZ    (32 * 1024)

/* Band/bounce buffers -> SRAM2 (.bss.vram.data). */
__attribute__((section(".bss.vram.data"), aligned(32)))
static uint8_t  s_bandRaw[SLIDE_BAND_ROWS * SLIDE_MAX_ROWBYTES];   /* BMP rows */
__attribute__((section(".bss.vram.data"), aligned(32)))
static uint16_t s_bandPix[SLIDE_BAND_ROWS * SLIDE_MAX_W];          /* RGB565 band */
__attribute__((section(".bss.vram.data"), aligned(8)))
static uint8_t  s_jdPool[SLIDE_JD_POOL_SZ];                        /* TJpgDec pool */

static FIL s_photoFile;   /* FIL is large; keep off the stack */

/* Columns at the right of the panel the slideshow must not paint (story panel). */
static uint32_t s_rightReserve = 0;

void Slideshow_ReserveRight(uint32_t px) { s_rightReserve = px; }

static uint16_t le16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t le32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/*----------------------------------------------------------------------------
 * File-name classification
 *--------------------------------------------------------------------------*/
static bool ext_is(const char *name, const char *ext)   /* ext lowercase, with dot */
{
    size_t n = strlen(name), e = strlen(ext);
    if (n < e + 1) return false;
    const char *p = name + n - e;
    for (size_t i = 0; i < e; i++)
    {
        char c = p[i];
        if (c >= 'A' && c <= 'Z') c = (char)(c + 32);
        if (c != ext[i]) return false;
    }
    return true;
}

static bool name_is_bmp(const char *n)  { return ext_is(n, ".bmp"); }
static bool name_is_jpg(const char *n)  { return ext_is(n, ".jpg") || ext_is(n, ".jpeg"); }
static bool name_is_png(const char *n)  { return ext_is(n, ".png"); }
static bool name_is_heic(const char *n) { return ext_is(n, ".heic") || ext_is(n, ".heif"); }

/*----------------------------------------------------------------------------
 * Common: blit one RGB565 band buffer to the LCD
 *--------------------------------------------------------------------------*/
static void blit_band(const uint16_t *pix, uint32_t x, uint32_t y,
                      uint32_t w, uint32_t h)
{
    S_DISP_RECT rect;
    rect.u32TopLeftX     = x;
    rect.u32TopLeftY     = y;
    rect.u32BottonRightX = x + w - 1u;
    rect.u32BottonRightY = y + h - 1u;
    Display_FillRect((uint16_t *)pix, &rect, 1);
}

/*----------------------------------------------------------------------------
 * JPEG path (TJpgDec)
 *--------------------------------------------------------------------------*/
typedef struct
{
    FIL      *fp;      /* open file being decoded                  */
    uint16_t *fb;      /* decode target (HyperRAM frame buffer)    */
    uint32_t  fbW;     /* scaled image width = fb row stride       */
} JpegSession;

/* TJpgDec input callback: read (or skip, when buf==NULL) len bytes. */
static size_t jpeg_in_cb(JDEC *jd, uint8_t *buf, size_t len)
{
    JpegSession *s = (JpegSession *)jd->device;

    if (buf)
    {
        UINT br = 0;
        if (f_read(s->fp, buf, (UINT)len, &br) != FR_OK)
            return 0;
        return (size_t)br;
    }
    return (f_lseek(s->fp, f_tell(s->fp) + len) == FR_OK) ? len : 0;
}

/* TJpgDec output callback: copy an RGB565 block into the frame buffer. */
static int jpeg_out_cb(JDEC *jd, void *bitmap, JRECT *rect)
{
    JpegSession    *s   = (JpegSession *)jd->device;
    const uint16_t *src = (const uint16_t *)bitmap;
    uint32_t        w   = (uint32_t)(rect->right - rect->left + 1);
    uint32_t        h   = (uint32_t)(rect->bottom - rect->top + 1);

    for (uint32_t y = 0; y < h; y++)
        memcpy(s->fb + (uint32_t)(rect->top + y) * s->fbW + rect->left,
               src + y * w, w * 2u);
    return 1;   /* continue decoding */
}

/**
 * Decode + display one JPEG. 0 on success; negative on skip/error:
 *  -1 open failed, -10 jd_prepare failed (bad/unsupported JPEG, e.g.
 *  progressive), -11 too large even at 1/8 scale, -12 decode error.
 */
static int show_jpg(const char *path, uint32_t panelW, uint32_t panelH)
{
    JDEC        jd;
    JpegSession sess;
    JRESULT     jr;

    if (f_open(&s_photoFile, path, FA_OPEN_EXISTING | FA_READ) != FR_OK)
        return -1;

    sess.fp  = &s_photoFile;
    sess.fb  = (uint16_t *)SLIDESHOW_FB_ADDR;
    sess.fbW = 0;

    jr = jd_prepare(&jd, jpeg_in_cb, s_jdPool, sizeof(s_jdPool), &sess);
    if (jr != JDR_OK)
    {
        f_close(&s_photoFile);
        printf("[SLIDESHOW] jd_prepare rc=%d (progressive JPEG or corrupt file?)\n", (int)jr);
        return -10;
    }

    /* Pick the largest output (smallest scale n) that fits the region; if
     * even 1/8 is too big (12-MP photo into the 360-px photo region), keep
     * the 1/8 decode and nearest-resample down during the blit. */
    uint8_t  scale;
    uint32_t sw = 0, sh = 0;

    for (scale = 0; scale <= 3; scale++)
    {
        sw = (uint32_t)jd.width  >> scale;
        sh = (uint32_t)jd.height >> scale;
        if (sw <= panelW && sh <= panelH) break;
    }
    if (sw == 0 || sh == 0 || sw * sh * 2u > SLIDESHOW_FB_SIZE)
    {
        f_close(&s_photoFile);
        printf("[SLIDESHOW] %ux%u undecodable (exceeds FB even at 1/8 scale)\n",
               (unsigned)jd.width, (unsigned)jd.height);
        return -11;
    }

    sess.fbW = sw;   /* decode target stride = scaled width */

    jr = jd_decomp(&jd, jpeg_out_cb, scale);
    f_close(&s_photoFile);

    if (jr != JDR_OK)
    {
        printf("[SLIDESHOW] jd_decomp rc=%d\n", (int)jr);
        return -12;
    }

    /* Contain-fit in both directions: TJpgDec's power-of-2 scales rarely
     * land on the region size, so nearest-resample while blitting — down
     * when even 1/8 is too big, up (capped at 2x so tiny files don't turn
     * to mush) to fill the region, e.g. 12-MP landscape 504x378 -> 640x480. */
    uint32_t ow = panelW;
    uint32_t oh = sh * panelW / sw;
    if (oh > panelH)
    {
        oh = panelH;
        ow = sw * panelH / sh;
    }
    if (ow == 0 || oh == 0 || ow > 2u * sw || oh > 2u * sh)
    {
        ow = sw;   /* tiny image: just center it unscaled */
        oh = sh;
    }

    /* Banded blit: HyperRAM FB -> SRAM2 bounce -> LCD, centered. */
    uint32_t offX = (panelW - ow) / 2u;
    uint32_t offY = (panelH - oh) / 2u;

    if (ow < panelW || oh < panelH)
    {
        /* Not full-bleed: clear the photo region so a previous larger
         * photo can't linger around this one. */
        S_DISP_RECT r = { 0, 0, panelW - 1u, panelH - 1u };
        Display_ClearRect(C_BLACK, &r);
    }

    for (uint32_t y = 0; y < oh; y += SLIDE_BAND_ROWS)
    {
        uint32_t n = oh - y;
        if (n > SLIDE_BAND_ROWS) n = SLIDE_BAND_ROWS;

        if (ow == sw && oh == sh)
            memcpy(s_bandPix, sess.fb + y * sw, n * sw * 2u);
        else
        {
            for (uint32_t r = 0; r < n; r++)
            {
                const uint16_t *srow = sess.fb + ((y + r) * sh / oh) * sw;
                uint16_t       *dst  = s_bandPix + r * ow;

                for (uint32_t x = 0; x < ow; x++)
                    dst[x] = srow[x * sw / ow];
            }
        }
        blit_band(s_bandPix, offX, offY + y, ow, n);
    }
    return 0;
}

/*----------------------------------------------------------------------------
 * BMP path (24-bpp uncompressed, streamed in bands; no frame buffer)
 *--------------------------------------------------------------------------*/
static int show_bmp(const char *path, uint32_t panelW, uint32_t panelH)
{
    UINT    br;
    uint8_t hdr[54];

    if (f_open(&s_photoFile, path, FA_OPEN_EXISTING | FA_READ) != FR_OK)
        return -1;

    if (f_read(&s_photoFile, hdr, sizeof(hdr), &br) != FR_OK || br != sizeof(hdr) ||
        hdr[0] != 'B' || hdr[1] != 'M')
    {
        f_close(&s_photoFile);
        return -2;
    }

    uint32_t offBits = le32(hdr + 10);
    int32_t  w       = (int32_t)le32(hdr + 18);
    int32_t  h       = (int32_t)le32(hdr + 22);
    uint16_t bpp     = le16(hdr + 28);
    uint32_t comp    = le32(hdr + 30);

    int topdown = 0;
    if (h < 0) { topdown = 1; h = -h; }

    if (bpp != 24 || comp != 0 || w <= 0 || h <= 0)
    {
        f_close(&s_photoFile);
        return -3;
    }
    if ((uint32_t)w > panelW || (uint32_t)h > panelH || w > SLIDE_MAX_W)
    {
        f_close(&s_photoFile);
        return -4;
    }

    uint32_t rowBytes = (((uint32_t)w * 3u) + 3u) & ~3u;
    uint32_t offX = (panelW - (uint32_t)w) / 2u;
    uint32_t offY = (panelH - (uint32_t)h) / 2u;

    if ((uint32_t)w < panelW || (uint32_t)h < panelH)
    {
        /* Not full-bleed: clear the photo region so a previous larger
         * photo can't linger around this one. */
        S_DISP_RECT r = { 0, 0, panelW - 1u, panelH - 1u };
        Display_ClearRect(C_BLACK, &r);
    }

    f_lseek(&s_photoFile, offBits);

    uint32_t fileRow = 0;
    while (fileRow < (uint32_t)h)
    {
        uint32_t n = (uint32_t)h - fileRow;
        if (n > SLIDE_BAND_ROWS) n = SLIDE_BAND_ROWS;

        if (f_read(&s_photoFile, s_bandRaw, rowBytes * n, &br) != FR_OK ||
            br != rowBytes * n)
        {
            f_close(&s_photoFile);
            return -5;
        }

        /* top-down BMP: rows [fileRow..]; bottom-up: reversed block */
        uint32_t bandTop = topdown ? fileRow : ((uint32_t)h - fileRow - n);

        for (uint32_t r = 0; r < n; r++)
        {
            const uint8_t *src  = s_bandRaw + r * rowBytes;
            uint32_t imgY       = topdown ? (fileRow + r)
                                          : ((uint32_t)h - 1u - (fileRow + r));
            uint16_t *dst       = s_bandPix + (imgY - bandTop) * (uint32_t)w;

            for (int32_t x = 0; x < w; x++)
            {
                uint8_t b  = src[3 * x + 0];
                uint8_t g  = src[3 * x + 1];
                uint8_t rr = src[3 * x + 2];
                dst[x] = (uint16_t)(((rr & 0xF8u) << 8) | ((g & 0xFCu) << 3) | (b >> 3));
            }
        }

        blit_band(s_bandPix, offX, offY + bandTop, (uint32_t)w, n);
        fileRow += n;
    }

    f_close(&s_photoFile);
    return 0;
}

/*----------------------------------------------------------------------------
 * Main loop
 *--------------------------------------------------------------------------*/
int Slideshow_Run(const char *dirPath, uint32_t holdMs)
{
    DIR     dir;
    FILINFO fno;
    TCHAR   drv[] = { '0', ':', 0 };

    uint32_t panelW = Disaplay_GetLCDWidth();
    uint32_t panelH = Disaplay_GetLCDHeight();

    /* Photos letterbox into the region left of the reserved panel (StoryUI). */
    if (s_rightReserve < panelW)
        panelW -= s_rightReserve;

    f_chdrive(drv);

    if (f_opendir(&dir, dirPath) != FR_OK)
    {
        printf("[SLIDESHOW] cannot open %s (folder missing on SD?)\n", dirPath);
        return -1;
    }

    printf("[SLIDESHOW] %s -> %ux%u panel, %u ms per photo (JPG decoded on-device)\n",
           dirPath, (unsigned)panelW, (unsigned)panelH, (unsigned)holdMs);

    if (s_rightReserve == 0)
        Display_ClearLCD(C_BLACK);
    else
    {
        /* Clear only the photo region — keep the story panel intact. */
        S_DISP_RECT r = { 0, 0, panelW - 1u, panelH - 1u };
        Display_ClearRect(C_BLACK, &r);
    }

    uint32_t shownThisPass = 0;

    for (;;)
    {
        if (f_readdir(&dir, &fno) != FR_OK)
        {
            f_closedir(&dir);
            return -3;
        }

        if (fno.fname[0] == 0)                    /* end of directory */
        {
            if (shownThisPass == 0)
            {
                printf("[SLIDESHOW] no displayable photo (.jpg/.bmp) in %s\n", dirPath);
                f_closedir(&dir);
                return -2;
            }
            shownThisPass = 0;
            f_readdir(&dir, NULL);                /* rewind, loop forever */
            continue;
        }

        if (fno.fattrib & AM_DIR) continue;

        char path[300];
        int  rc;

        if (name_is_jpg(fno.fname))
        {
            snprintf(path, sizeof(path), "%s\\%s", dirPath, fno.fname);
            rc = show_jpg(path, panelW, panelH);
        }
        else if (name_is_bmp(fno.fname))
        {
            snprintf(path, sizeof(path), "%s\\%s", dirPath, fno.fname);
            rc = show_bmp(path, panelW, panelH);
        }
        else if (name_is_png(fno.fname) || name_is_heic(fno.fname))
        {
            printf("[SLIDESHOW] skip %s (PNG/HEIC not decodable on-device; "
                   "convert with scripts/prepare_pictures.py)\n", fno.fname);
            continue;
        }
        else
            continue;   /* not a picture */

        if (rc != 0)
        {
            printf("[SLIDESHOW] skip %s (rc=%d)\n", fno.fname, rc);
            continue;
        }

#if SLIDE_CAPTION
        Display_PutText(fno.fname, (uint32_t)strlen(fno.fname),
                        4, panelH - 20, C_WHITE, C_BLACK, false, 1);
#endif
        shownThisPass++;
        Display_Delay(holdMs);
    }
}
