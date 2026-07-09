/**************************************************************************//**
 * @file     day3_demo.c
 * @brief    Day-3 on-board demo (see day3_demo.h).
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "day3_demo.h"

#include <stdio.h>
#include <string.h>

#include "PerfTimer.h"
#include "StoryUI.h"
/* Fresh clone: fall back to placeholder creds so the project still builds.
 * Copy day2_config.example.h -> day2_config.h and fill in real values. */
#if defined(__has_include) && !__has_include("day2_config.h")
#include "day2_config.example.h"
#else
#include "day2_config.h"
#endif
#include "esp_at.h"
#include "esp_http.h"

#define DAY3_FACE_ID        1
#define GEN_TIMEOUT_MS      60000   /* /generate holds while the LLM thinks (10-30 s) */
#define IMG_TIMEOUT_MS      20000   /* 37.5 KB @ ~11.5 KB/s over the AT link ~ 3.3 s */

#define STR_(x) #x
#define STR(x)  STR_(x)

/* /textimg response: headers (~200 B) + TIM4 body, in SRAM2 (CPU-only). */
__attribute__((section(".bss.vram.data"), aligned(4)))
static uint8_t s_timBuf[STORYUI_TIM4_MAX + 512];

static char s_json[768];            /* /generate response (zh+en story + ids) */
static char s_audioId[24];

/* Pull "audio_id":"<hex>" out of the /generate JSON (whitespace-tolerant). */
static int extract_audio_id(const char *json, char *out, int out_sz)
{
    const char *p = strstr(json, "\"audio_id\"");

    if (!p) return -1;
    p = strchr(p + 10, ':');
    if (!p) return -1;
    p = strchr(p, '"');
    if (!p) return -1;
    p++;

    int n = 0;
    while (p[n] && p[n] != '"' && n < out_sz - 1)
    {
        out[n] = p[n];
        n++;
    }
    if (n == 0 || p[n] != '"')
        return -1;
    out[n] = '\0';
    return 0;
}

int day3_demo_run(void)
{
    char     ip[20];
    char     path[48];
    int      rc, bodyLen;
    uint32_t t0, tStart;

    printf("\n=== Day 3 demo: layout + live LLM story ===\n");
    tStart = GetSystemTick_ms();

    StoryUI_DrawChrome();
    StoryUI_ShowStatus("WiFi connecting");

    /* --- Wi-Fi up (same bring-up day2_test validated) -------------------- */
    if ((rc = esp_at_init()) != ESP_OK || (rc = esp_wifi_set_mode(1)) != ESP_OK)
    {
        printf("[DAY3] ESP init failed: %d\n", rc);
        StoryUI_ShowStatus("WiFi module err");
        return -1;
    }

    t0 = GetSystemTick_ms();
    if ((rc = esp_wifi_connect(WIFI_SSID, WIFI_PWD)) != ESP_OK)
    {
        printf("[DAY3] join \"%s\" failed: %d (2.4 GHz only?)\n", WIFI_SSID, rc);
        StoryUI_ShowStatus("WiFi join fail");
        return -2;
    }
    if ((rc = esp_wifi_get_ip(ip)) != ESP_OK)
    {
        printf("[DAY3] no DHCP lease: %d\n", rc);
        StoryUI_ShowStatus("No IP address");
        return -3;
    }
    printf("[DAY3] Wi-Fi up in %u ms, IP %s\n",
           (unsigned)(GetSystemTick_ms() - t0), ip);

    /* --- POST /generate --------------------------------------------------- */
    StoryUI_ShowStatus("Writing story");
    t0 = GetSystemTick_ms();
    rc = http_post_json_ex(BACKEND_HOST, BACKEND_PORT, "/generate",
                           "{\"face_id\":" STR(DAY3_FACE_ID) "}",
                           s_json, sizeof(s_json), GEN_TIMEOUT_MS);
    if (rc != 200)
    {
        printf("[DAY3] POST /generate failed: %d (backend up? firewall?)\n", rc);
        StoryUI_ShowStatus("Backend error");
        return -4;
    }
    printf("[DAY3] /generate 200 in %u ms\n", (unsigned)(GetSystemTick_ms() - t0));
    printf("[DAY3] body: %s\n", s_json);   /* full story lands in the UART log */

    if (extract_audio_id(s_json, s_audioId, sizeof(s_audioId)) != 0)
    {
        printf("[DAY3] no audio_id in response\n");
        StoryUI_ShowStatus("Bad response");
        return -5;
    }

    /* --- GET /textimg (TIM4 story image) ---------------------------------- */
    StoryUI_ShowStatus("Loading story");
    snprintf(path, sizeof(path), "/textimg/%s", s_audioId);
    t0 = GetSystemTick_ms();
    rc = http_get_binary(BACKEND_HOST, BACKEND_PORT, path,
                         s_timBuf, (int)sizeof(s_timBuf), &bodyLen, IMG_TIMEOUT_MS);
    if (rc != 200 || bodyLen < 8)
    {
        printf("[DAY3] GET %s failed: rc=%d len=%d\n", path, rc, bodyLen);
        StoryUI_ShowStatus("Image dl failed");
        return -6;
    }
    printf("[DAY3] /textimg 200: %d bytes in %u ms\n",
           bodyLen, (unsigned)(GetSystemTick_ms() - t0));

    rc = StoryUI_ShowTextImage(s_timBuf, (uint32_t)bodyLen);
    if (rc != 0)
    {
        printf("[DAY3] TIM4 blit failed: %d\n", rc);
        StoryUI_ShowStatus("Bad image data");
        return -7;
    }

    printf("[DAY3] story on screen — total %u ms from boot of demo\n",
           (unsigned)(GetSystemTick_ms() - tStart));
    return 0;
}
