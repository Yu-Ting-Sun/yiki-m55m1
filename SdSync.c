/**************************************************************************//**
 * @file     SdSync.c
 * @brief    Frame sync over Wi-Fi (see SdSync.h).
 *
 * Manifest (backend main.py frame_sync):
 *   {"sd_root":"pictures","faces_root":"faces",
 *    "trips":[{"folder":"T0006","version":"ab12cd34","has_story":true,
 *              "label_json":"/trips/6/label.json",
 *              "story_tim":"/trips/6/story.tim",
 *              "photos":[{"name":"P0002.JPG","url":"/photos/2/board"}]}, ...],
 *    "faces":[{"name":"enroll_dad.raw","url":"/faces/file/enroll_dad.raw"}]}
 *
 * Per album: skip whole album when SD VERSION.TXT == manifest version, else
 * download every file and write VERSION.TXT last (torn sync self-heals).
 * STORY.WAV / STORY.TXT are not fetched — no audio path on this board yet
 * and the rail renders the pre-drawn STORY.TIM.
 *
 * Buffers: downloads land in the slideshow HyperRAM frame buffer (idle
 * while a sync runs between photos); the manifest text sits in the tail of
 * the same 1 MB slot, so the module costs zero SRAM.
 *
 * JSON is walked with a tiny fixed-structure scanner (no JSON lib on the
 * board): escape-aware string skipping + depth-matched {} / [] iteration.
 * The only free text (trip titles) stays inside quoted strings, which the
 * scanner steps over, so it cannot be mistaken for a key.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "SdSync.h"

#include <stdio.h>
#include <string.h>

#include "ff.h"
#include "MemoryLayout.h"   /* SLIDESHOW_FB_ADDR / _SIZE (download buffer) */
#include "StoryUI.h"
#include "esp_at.h"
#include "esp_http.h"

/* Fresh clone: fall back to placeholder creds so the project still builds. */
#if defined(__has_include) && !__has_include("day2_config.h")
#include "day2_config.example.h"
#else
#include "day2_config.h"
#endif

#define SYNC_FRAME_ID       (1)         /* demo frame seeded by the backend */
#define SYNC_MANIFEST_TMO   (15000)
#define SYNC_FILE_TMO       (30000)     /* enroll raw 115 KB @ ~11.5 KB/s   */

/* Slideshow frame buffer, borrowed while the slideshow is between photos:
 * first 768 KB = file downloads, last 256 KB = manifest text. */
#define DL_BUF      ((uint8_t *)SLIDESHOW_FB_ADDR)
#define DL_CAP      (768 * 1024)
#define MAN_BUF     ((char *)(SLIDESHOW_FB_ADDR + DL_CAP))
#define MAN_CAP     ((int)SLIDESHOW_FB_SIZE - DL_CAP)

static bool s_wifiUp = false;
static FIL  s_file;                     /* FIL is large; keep off the stack */

/*----------------------------------------------------------------------------
 * Tiny fixed-structure JSON scanning (byte-based, escape-aware)
 *--------------------------------------------------------------------------*/

/* p points at an opening quote; return the char AFTER the closing quote. */
static const char *str_skip(const char *p, const char *end)
{
    for (p++; p < end; p++)
    {
        if (*p == '\\') { p++; continue; }
        if (*p == '"')  return p + 1;
    }
    return NULL;
}

/* Find `"key"` inside [obj, end); return the char after the closing quote. */
static const char *find_key(const char *obj, const char *end, const char *key)
{
    char pat[32];
    int  n = snprintf(pat, sizeof(pat), "\"%s\"", key);

    if (n <= 0 || n >= (int)sizeof(pat))
        return NULL;
    for (const char *p = obj; p + n <= end; p++)
        if (memcmp(p, pat, (size_t)n) == 0)
            return p + n;
    return NULL;
}

/* Copy the string value of `"key": "value"`; 0 on success. */
static int get_str(const char *obj, const char *end, const char *key,
                   char *out, int cap)
{
    const char *p = find_key(obj, end, key);

    if (!p) return -1;
    while (p < end && (*p == ':' || *p == ' ')) p++;
    if (p >= end || *p != '"') return -1;

    const char *e = str_skip(p, end);
    if (!e) return -1;

    int n = (int)(e - 1 - (p + 1));
    if (n >= cap) n = cap - 1;
    memcpy(out, p + 1, (size_t)n);      /* backend fields carry no escapes */
    out[n] = '\0';
    return 0;
}

static bool get_bool(const char *obj, const char *end, const char *key)
{
    const char *p = find_key(obj, end, key);

    if (!p) return false;
    while (p < end && (*p == ':' || *p == ' ')) p++;
    return (p + 4 <= end && memcmp(p, "true", 4) == 0);
}

/* Locate `"key": [ ... ]`; returns just after '[', sets *arrEnd to the
 * matching ']'. Depth-matched and string-aware. */
static const char *find_array(const char *obj, const char *end,
                              const char *key, const char **arrEnd)
{
    const char *p = find_key(obj, end, key);

    if (!p) return NULL;
    while (p < end && *p != '[') p++;
    if (p >= end) return NULL;

    int depth = 0;
    for (const char *q = p; q < end; q++)
    {
        if (*q == '"')
        {
            q = str_skip(q, end);
            if (!q) return NULL;
            q--;
            continue;
        }
        if (*q == '[') depth++;
        else if (*q == ']' && --depth == 0)
        {
            *arrEnd = q;
            return p + 1;
        }
    }
    return NULL;
}

/* Next {...} object at/after cur; sets *objEnd past the matching '}'. */
static const char *next_obj(const char *cur, const char *arrEnd,
                            const char **objEnd)
{
    const char *p = cur;

    while (p < arrEnd && *p != '{') p++;
    if (p >= arrEnd) return NULL;

    int depth = 0;
    for (const char *q = p; q < arrEnd; q++)
    {
        if (*q == '"')
        {
            q = str_skip(q, arrEnd);
            if (!q) return NULL;
            q--;
            continue;
        }
        if (*q == '{') depth++;
        else if (*q == '}' && --depth == 0)
        {
            *objEnd = q + 1;
            return p;
        }
    }
    return NULL;
}

/*----------------------------------------------------------------------------
 * SD + HTTP helpers
 *--------------------------------------------------------------------------*/

static bool sd_exists(const char *path)
{
    if (f_open(&s_file, path, FA_OPEN_EXISTING | FA_READ) != FR_OK)
        return false;
    f_close(&s_file);
    return true;
}

static bool version_matches(const char *dir, const char *ver)
{
    char path[96], cur[9];
    UINT br = 0;

    snprintf(path, sizeof(path), "%s\\VERSION.TXT", dir);
    if (f_open(&s_file, path, FA_OPEN_EXISTING | FA_READ) != FR_OK)
        return false;
    f_read(&s_file, cur, 8, &br);
    f_close(&s_file);
    return (br == 8 && strncmp(cur, ver, 8) == 0);
}

static void version_write(const char *dir, const char *ver)
{
    char path[96];
    UINT bw = 0;

    snprintf(path, sizeof(path), "%s\\VERSION.TXT", dir);
    if (f_open(&s_file, path, FA_CREATE_ALWAYS | FA_WRITE) == FR_OK)
    {
        f_write(&s_file, ver, (UINT)strlen(ver), &bw);
        f_close(&s_file);
    }
}

/* GET urlPath -> SD sdPath (whole file through DL_BUF). 0 on success. */
static int dl_to_file(const char *urlPath, const char *sdPath)
{
    int blen = 0;
    int st = http_get_binary(BACKEND_HOST, BACKEND_PORT, urlPath,
                             DL_BUF, DL_CAP, &blen, SYNC_FILE_TMO);

    if (st != 200)
    {
        printf("[SYNC] GET %s -> %d\n", urlPath, st);
        return (st < 0) ? st : -st;
    }

    if (f_open(&s_file, sdPath, FA_CREATE_ALWAYS | FA_WRITE) != FR_OK)
    {
        printf("[SYNC] cannot create %s\n", sdPath);
        return -200;
    }

    UINT    bw = 0;
    FRESULT fr = f_write(&s_file, DL_BUF, (UINT)blen, &bw);
    f_close(&s_file);

    if (fr != FR_OK || (int)bw != blen)
    {
        printf("[SYNC] short write %s (%u/%d)\n", sdPath, (unsigned)bw, blen);
        return -201;
    }

    printf("[SYNC] %s (%d B)\n", sdPath, blen);
    return 0;
}

static int wifi_up(void)
{
    char ip[20];

    if (esp_at_init() != ESP_OK)        return -1;
    if (esp_wifi_set_mode(1) != ESP_OK) return -2;
    if (esp_wifi_connect(WIFI_SSID, WIFI_PWD) != ESP_OK) return -3;
    if (esp_wifi_get_ip(ip) != ESP_OK)  return -4;

    printf("[SYNC] WiFi up, IP %s\n", ip);
    return 0;
}

/*----------------------------------------------------------------------------
 * Album / faces sync
 *--------------------------------------------------------------------------*/

/* Returns 1 if the album was (re)downloaded, 0 if up to date / unusable. */
static int sync_trip(const char *obj, const char *objEnd,
                     const char *sdRoot, bool verbose)
{
    char folder[24], ver[16], url[96], dir[64], sdPath[112], name[32];

    if (get_str(obj, objEnd, "folder", folder, sizeof(folder)) != 0 ||
        get_str(obj, objEnd, "version", ver, sizeof(ver)) != 0)
        return 0;

    snprintf(dir, sizeof(dir), "0:\\%s\\%s", sdRoot, folder);
    if (version_matches(dir, ver))
        return 0;

    printf("[SYNC] album %s: updating\n", folder);
    if (verbose)
    {
        char msg[24];
        snprintf(msg, sizeof(msg), "Sync %s", folder);
        StoryUI_ShowStatus(msg);
    }
    f_mkdir(dir);                       /* FR_EXIST is fine */

    int fails = 0;

    if (get_str(obj, objEnd, "label_json", url, sizeof(url)) == 0)
    {
        snprintf(sdPath, sizeof(sdPath), "%s\\LABEL.JSON", dir);
        if (dl_to_file(url, sdPath) != 0) fails++;
    }

    if (get_bool(obj, objEnd, "has_story") &&
        get_str(obj, objEnd, "story_tim", url, sizeof(url)) == 0)
    {
        snprintf(sdPath, sizeof(sdPath), "%s\\STORY.TIM", dir);
        if (dl_to_file(url, sdPath) != 0) fails++;
    }

    const char *aEnd, *oEnd;
    const char *a = find_array(obj, objEnd, "photos", &aEnd);
    if (a)
    {
        for (const char *o = next_obj(a, aEnd, &oEnd); o;
             o = next_obj(oEnd, aEnd, &oEnd))
        {
            if (get_str(o, oEnd, "name", name, sizeof(name)) != 0 ||
                get_str(o, oEnd, "url", url, sizeof(url)) != 0)
                continue;
            snprintf(sdPath, sizeof(sdPath), "%s\\%s", dir, name);
            if (dl_to_file(url, sdPath) != 0) fails++;
        }
    }

    if (fails == 0)
        version_write(dir, ver);        /* only a complete album gets a version */
    else
        printf("[SYNC] album %s: %d file(s) failed, will retry next pass\n",
               folder, fails);
    return 1;
}

/* Download enrollment raws we have never seen (neither .raw nor .raw.done
 * on the card). Returns the number of new files. */
static int sync_faces(const char *man, const char *end, const char *facesRoot)
{
    char name[48], url[96], sdPath[112], donePath[120];
    int  added = 0;

    const char *aEnd, *oEnd;
    const char *a = find_array(man, end, "faces", &aEnd);
    if (!a)
        return 0;

    char dir[32];
    snprintf(dir, sizeof(dir), "0:\\%s", facesRoot);
    f_mkdir(dir);

    for (const char *o = next_obj(a, aEnd, &oEnd); o;
         o = next_obj(oEnd, aEnd, &oEnd))
    {
        if (get_str(o, oEnd, "name", name, sizeof(name)) != 0 ||
            get_str(o, oEnd, "url", url, sizeof(url)) != 0)
            continue;

        snprintf(sdPath, sizeof(sdPath), "%s\\%s", dir, name);
        snprintf(donePath, sizeof(donePath), "%s.done", sdPath);
        if (sd_exists(sdPath) || sd_exists(donePath))
            continue;                   /* already fetched / already enrolled */

        if (dl_to_file(url, sdPath) == 0)
            added++;
    }
    return added;
}

int SdSync_Run(bool verbose)
{
    char path[32];
    int  blen = 0, st;

    snprintf(path, sizeof(path), "/frames/%d/sync", SYNC_FRAME_ID);

    st = http_get_binary(BACKEND_HOST, BACKEND_PORT, path,
                         (uint8_t *)MAN_BUF, MAN_CAP - 1, &blen,
                         SYNC_MANIFEST_TMO);
    if (st < 0 && !s_wifiUp)
    {
        /* First contact (or the Day-3 demo never joined): bring Wi-Fi up. */
        if (verbose) StoryUI_ShowStatus("WiFi connecting");
        if (wifi_up() != 0)
        {
            if (verbose) StoryUI_ShowStatus("WiFi failed");
            printf("[SYNC] WiFi bring-up failed\n");
            return -1;
        }
        s_wifiUp = true;
        st = http_get_binary(BACKEND_HOST, BACKEND_PORT, path,
                             (uint8_t *)MAN_BUF, MAN_CAP - 1, &blen,
                             SYNC_MANIFEST_TMO);
    }
    if (st < 0)
    {
        printf("[SYNC] manifest transport error %d\n", st);
        return -2;
    }
    s_wifiUp = true;                    /* a served request proves the link */
    if (st != 200)
    {
        printf("[SYNC] manifest HTTP %d\n", st);
        return -3;
    }

    char       *man = MAN_BUF;
    const char *end = man + blen;
    man[blen] = '\0';

    char sdRoot[16]   = "pictures";
    char facesRoot[16] = "faces";
    get_str(man, end, "sd_root", sdRoot, sizeof(sdRoot));
    get_str(man, end, "faces_root", facesRoot, sizeof(facesRoot));

    char rootDir[32];
    snprintf(rootDir, sizeof(rootDir), "0:\\%s", sdRoot);
    f_mkdir(rootDir);

    int changed = 0;
    const char *aEnd, *oEnd;
    const char *a = find_array(man, end, "trips", &aEnd);
    if (a)
    {
        for (const char *o = next_obj(a, aEnd, &oEnd); o;
             o = next_obj(oEnd, aEnd, &oEnd))
            changed += sync_trip(o, oEnd, sdRoot, verbose);
    }

    int newFaces = sync_faces(man, end, facesRoot);
    if (newFaces > 0)
        printf("[SYNC] %d new face file(s) downloaded - reboot to enroll\n",
               newFaces);

    if (verbose)
        StoryUI_ShowStatus(changed ? "Sync done" : "Up to date");
    if (changed)
        printf("[SYNC] %d album(s) updated\n", changed);
    return changed;
}
