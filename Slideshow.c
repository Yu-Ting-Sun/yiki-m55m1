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
#include "StoryUI.h"        /* story rail follows the album on screen */
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
 * Photo library — albums are the subfolders of the root, each optionally
 * carrying a label.json ({"users": ["user1", ...]}, hand-written or later
 * emitted by the backend). Loose files in the root form an unlabeled
 * pseudo-album (index 0) that only plays when no filter is active, so a flat
 * pictures/ SD card from the pre-album era keeps working unchanged.
 *--------------------------------------------------------------------------*/
#define LIB_MAX_ALBUMS      (16)
#define LIB_MAX_USERS       (8)     /* per album */
#define LIB_NAME_LEN        (64)    /* album folder name */
#define LIB_USER_LEN        (24)
#define LIB_LABEL_FILE      "label.json"
#define LIB_LABEL_MAX       (512)   /* bytes of label.json parsed */

typedef struct
{
    char    name[LIB_NAME_LEN];     /* "" = loose files in the root */
    char    users[LIB_MAX_USERS][LIB_USER_LEN];
    uint8_t nUsers;
} Album;

static char    s_root[64];
static Album   s_albums[LIB_MAX_ALBUMS];
static uint8_t s_nAlbums;                    /* incl. pseudo-album 0 */

static char    s_filter[LIB_USER_LEN];       /* "" = play everything */
static uint8_t s_matched[LIB_MAX_ALBUMS];    /* album indices in play set */
static uint8_t s_nMatched;
static bool    s_restart;                    /* filter changed: reset cursor */

/* Iteration state across Slideshow_ShowNext() calls. */
static DIR     s_iterDir;
static bool    s_iterOpen;
static uint8_t s_cursor;                     /* index into s_matched */
static uint8_t s_albumsDone;                 /* albums exhausted this cycle */

/* What is on screen right now (for the gesture-like report). */
static char    s_curFolder[LIB_NAME_LEN];    /* "" = root pseudo-album */
static char    s_curPhoto[64];               /* "" = story-only slide  */
static uint32_t s_shownInCycle;
static uint32_t s_photosThisAlbum;           /* photos shown since album open */

/* Story rail state: which album's STORY.TIM is on the rail right now.
 * STORY.TIM sits next to the photos (backend sync / scripts/export_sd.py). */
#define ALBUM_STORY_FILE    "STORY.TIM"
static uint8_t s_storyAlbum = 0xFF;          /* album index shown in the rail */
static bool    s_storyHasStory = false;      /* rail shows a story (vs empty) */

static bool str_ieq(const char *a, const char *b)
{
    for (; *a && *b; a++, b++)
    {
        char ca = (*a >= 'A' && *a <= 'Z') ? (char)(*a + 32) : *a;
        char cb = (*b >= 'A' && *b <= 'Z') ? (char)(*b + 32) : *b;
        if (ca != cb) return false;
    }
    return *a == *b;
}

/* Naive fixed-structure parser: pull the quoted strings out of the "users"
 * array. No JSON library on this board; anything malformed just yields
 * fewer users, never a crash. */
static void lib_read_label(const char *dirPath, Album *al)
{
    char path[300], buf[LIB_LABEL_MAX + 1];
    UINT br = 0;

    al->nUsers = 0;

    snprintf(path, sizeof(path), "%s\\%s", dirPath, LIB_LABEL_FILE);
    if (f_open(&s_photoFile, path, FA_OPEN_EXISTING | FA_READ) != FR_OK)
        return;                     /* unlabeled album: plays only unfiltered */
    f_read(&s_photoFile, buf, LIB_LABEL_MAX, &br);
    f_close(&s_photoFile);
    buf[br] = 0;

    const char *p = strstr(buf, "\"users\"");
    if (!p || (p = strchr(p, '[')) == NULL) return;
    const char *end = strchr(p, ']');

    while (al->nUsers < LIB_MAX_USERS)
    {
        const char *q0 = strchr(p, '"');
        if (!q0 || (end && q0 > end)) break;
        const char *q1 = strchr(q0 + 1, '"');
        if (!q1) break;

        size_t n = (size_t)(q1 - q0) - 1u;
        if (n >= LIB_USER_LEN) n = LIB_USER_LEN - 1;
        memcpy(al->users[al->nUsers], q0 + 1, n);
        al->users[al->nUsers][n] = 0;
        al->nUsers++;
        p = q1 + 1;
    }
}

/* Rebuild s_matched from s_filter and flag the iterator for a restart. */
static void lib_apply_filter(void)
{
    s_nMatched = 0;

    for (uint8_t i = 0; i < s_nAlbums; i++)
    {
        if (s_filter[0] == 0)
        {
            s_matched[s_nMatched++] = i;    /* no filter: everything */
            continue;
        }
        for (uint8_t u = 0; u < s_albums[i].nUsers; u++)
        {
            if (str_ieq(s_albums[i].users[u], s_filter))
            {
                s_matched[s_nMatched++] = i;
                break;
            }
        }
    }

    if (s_filter[0] != 0 && s_nMatched == 0)
    {
        /* Unknown user: degrade to playing everything (same philosophy as
         * the Wi-Fi fallback — never a black screen). */
        printf("[SLIDESHOW] filter '%s' matches no album -> playing all\n", s_filter);
        for (uint8_t i = 0; i < s_nAlbums; i++)
            s_matched[s_nMatched++] = i;
    }

    s_restart = true;
}

/* Story rail follows the album on screen. Returns 0 iff the rail now shows
 * this album's story; albums without STORY.TIM (incl. the loose-file
 * pseudo-album) get the empty panel back. */
static int story_follow_album(uint8_t albumIdx)
{
    const Album *al;
    char path[300];
    int  rc = -4;

    if (albumIdx == s_storyAlbum)               /* rail already up to date */
        return s_storyHasStory ? 0 : -1;

    al = &s_albums[albumIdx];
    if (al->name[0])
    {
        snprintf(path, sizeof(path), "%s\\%s\\" ALBUM_STORY_FILE,
                 s_root, al->name);
        rc = StoryUI_ShowTextImageFile(path);
    }

    if (rc == 0)
        printf("[SLIDESHOW] story rail -> album '%s'\n", al->name);
    else
        StoryUI_DrawChrome();                   /* no story: empty panel */

    s_storyAlbum    = albumIdx;
    s_storyHasStory = (rc == 0);
    return s_storyHasStory ? 0 : -1;
}

int Slideshow_LibScan(const char *rootPath)
{
    DIR     dir;
    FILINFO fno;
    TCHAR   drv[] = { '0', ':', 0 };

    f_chdrive(drv);

    /* Album indices are about to change; forget what the rail shows. */
    s_storyAlbum    = 0xFF;
    s_storyHasStory = false;

    strncpy(s_root, rootPath, sizeof(s_root) - 1);
    s_root[sizeof(s_root) - 1] = 0;

    /* Pseudo-album 0: loose files directly in the root, no label. */
    s_albums[0].name[0] = 0;
    s_albums[0].nUsers  = 0;
    s_nAlbums = 1;

    if (f_opendir(&dir, rootPath) != FR_OK)
    {
        printf("[SLIDESHOW] cannot open %s (folder missing on SD?)\n", rootPath);
        s_nAlbums = 0;
        return -1;
    }

    while (f_readdir(&dir, &fno) == FR_OK && fno.fname[0] != 0)
    {
        if (!(fno.fattrib & AM_DIR)) continue;

        if (s_nAlbums >= LIB_MAX_ALBUMS)
        {
            printf("[SLIDESHOW] more than %d albums; extras ignored\n",
                   LIB_MAX_ALBUMS - 1);
            break;
        }

        Album *al = &s_albums[s_nAlbums++];
        strncpy(al->name, fno.fname, LIB_NAME_LEN - 1);
        al->name[LIB_NAME_LEN - 1] = 0;

        char sub[300];
        snprintf(sub, sizeof(sub), "%s\\%s", rootPath, al->name);
        lib_read_label(sub, al);

        printf("[SLIDESHOW] album '%s': %u labeled user(s)", al->name,
               (unsigned)al->nUsers);
        for (uint8_t u = 0; u < al->nUsers; u++)
            printf("%s%s", u ? ", " : " [", al->users[u]);
        printf(al->nUsers ? "]\n" : "\n");
    }
    f_closedir(&dir);

    lib_apply_filter();             /* re-apply whatever filter is stored */
    return (int)(s_nAlbums - 1);    /* number of real albums found */
}

int Slideshow_SetFilter(const char *user)
{
    if (user == NULL) user = "";
    strncpy(s_filter, user, sizeof(s_filter) - 1);
    s_filter[sizeof(s_filter) - 1] = 0;

    if (s_nAlbums == 0) return 0;   /* not scanned yet; LibScan will apply */

    lib_apply_filter();
    printf("[SLIDESHOW] filter '%s' -> %u album(s) in play set\n",
           s_filter[0] ? s_filter : "(none)", (unsigned)s_nMatched);
    return (int)s_nMatched;
}

int Slideshow_ShowNext(void)
{
    FILINFO fno;

    uint32_t panelW = Disaplay_GetLCDWidth();
    uint32_t panelH = Disaplay_GetLCDHeight();

    /* Photos letterbox into the region left of the reserved panel (StoryUI). */
    if (s_rightReserve < panelW)
        panelW -= s_rightReserve;

    if (s_nAlbums == 0 || s_nMatched == 0)
        return -1;                  /* Slideshow_LibScan not run / failed */

    if (s_restart)
    {
        if (s_iterOpen) { f_closedir(&s_iterDir); s_iterOpen = false; }
        s_cursor = 0;
        s_albumsDone = 0;
        s_shownInCycle = 0;
        s_restart = false;

        /* Clear the photo region so leftovers from the previous play set
         * (larger photo, stale caption) can't linger. */
        S_DISP_RECT r = { 0, 0, panelW - 1u, panelH - 1u };
        Display_ClearRect(C_BLACK, &r);
    }

    for (;;)
    {
        if (!s_iterOpen)
        {
            if (s_albumsDone >= s_nMatched)     /* full cycle completed */
            {
                if (s_shownInCycle == 0)
                {
                    printf("[SLIDESHOW] no displayable photo (.jpg/.bmp) in play set\n");
                    return -2;
                }
                s_albumsDone = 0;
                s_shownInCycle = 0;             /* wrap: loop forever */
            }

            const Album *al = &s_albums[s_matched[s_cursor]];
            char dirPath[300];

            if (al->name[0])
                snprintf(dirPath, sizeof(dirPath), "%s\\%s", s_root, al->name);
            else
                snprintf(dirPath, sizeof(dirPath), "%s", s_root);

            if (f_opendir(&s_iterDir, dirPath) != FR_OK)
            {
                s_albumsDone++;
                s_cursor = (uint8_t)((s_cursor + 1) % s_nMatched);
                continue;
            }
            s_iterOpen = true;
            s_photosThisAlbum = 0;
        }

        if (f_readdir(&s_iterDir, &fno) != FR_OK)
        {
            f_closedir(&s_iterDir);
            s_iterOpen = false;
            return -3;
        }

        if (fno.fname[0] == 0)                  /* album exhausted */
        {
            uint8_t doneIdx = s_matched[s_cursor];

            f_closedir(&s_iterDir);
            s_iterOpen = false;
            s_albumsDone++;
            s_cursor = (uint8_t)((s_cursor + 1) % s_nMatched);

            /* Story-only album (a trip with a story but no photos yet):
             * the story itself is the slide — black photo region + rail. */
            if (s_photosThisAlbum == 0 && story_follow_album(doneIdx) == 0)
            {
                S_DISP_RECT r = { 0, 0, panelW - 1u, panelH - 1u };
                Display_ClearRect(C_BLACK, &r);
                strncpy(s_curFolder, s_albums[doneIdx].name, sizeof(s_curFolder) - 1);
                s_curPhoto[0] = '\0';           /* story-only, no photo */
                s_shownInCycle++;
                return 0;
            }
            continue;
        }

        if (fno.fattrib & AM_DIR) continue;

        const Album *al = &s_albums[s_matched[s_cursor]];
        char path[300];
        int  rc;

        if (al->name[0])
            snprintf(path, sizeof(path), "%s\\%s\\%s", s_root, al->name, fno.fname);
        else
            snprintf(path, sizeof(path), "%s\\%s", s_root, fno.fname);

        if (name_is_jpg(fno.fname))
            rc = show_jpg(path, panelW, panelH);
        else if (name_is_bmp(fno.fname))
            rc = show_bmp(path, panelW, panelH);
        else if (name_is_png(fno.fname) || name_is_heic(fno.fname))
        {
            printf("[SLIDESHOW] skip %s (PNG/HEIC not decodable on-device; "
                   "convert with scripts/prepare_pictures.py)\n", fno.fname);
            continue;
        }
        else
            continue;   /* not a picture (label.json lands here too) */

        if (rc != 0)
        {
            printf("[SLIDESHOW] skip %s (rc=%d)\n", fno.fname, rc);
            continue;
        }

#if SLIDE_CAPTION
        {
            char caption[96];
            if (al->name[0])
                snprintf(caption, sizeof(caption), "%s/%s", al->name, fno.fname);
            else
                snprintf(caption, sizeof(caption), "%s", fno.fname);
            Display_PutText(caption, (uint32_t)strlen(caption),
                            4, panelH - 20, C_WHITE, C_BLACK, false, 1);
        }
#endif
        /* Photo is up: make the story rail match the album it came from. */
        story_follow_album(s_matched[s_cursor]);

        strncpy(s_curFolder, al->name, sizeof(s_curFolder) - 1);
        strncpy(s_curPhoto, fno.fname, sizeof(s_curPhoto) - 1);

        s_photosThisAlbum++;
        s_shownInCycle++;
        return 0;
    }
}

void Slideshow_GetCurrent(char *folder, int folderCap, char *photo, int photoCap)
{
    if (folder && folderCap > 0)
    {
        strncpy(folder, s_curFolder, (size_t)folderCap - 1);
        folder[folderCap - 1] = '\0';
    }
    if (photo && photoCap > 0)
    {
        strncpy(photo, s_curPhoto, (size_t)photoCap - 1);
        photo[photoCap - 1] = '\0';
    }
}

/*----------------------------------------------------------------------------
 * Main loop (kept as a convenience wrapper over the step API)
 *--------------------------------------------------------------------------*/
int Slideshow_Run(const char *dirPath, uint32_t holdMs)
{
    if (Slideshow_LibScan(dirPath) < 0)
        return -1;

    printf("[SLIDESHOW] %s: %u album(s), %u ms per photo (JPG decoded on-device)\n",
           dirPath, (unsigned)(s_nAlbums - 1), (unsigned)holdMs);

    for (;;)
    {
        int rc = Slideshow_ShowNext();
        if (rc != 0)
            return rc;
        Display_Delay(holdMs);
    }
}
