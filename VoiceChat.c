/**************************************************************************//**
 * @file     VoiceChat.c
 * @brief    Ask 小憶 over Wi-Fi. See VoiceChat.h.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include <stdio.h>
#include <string.h>

#include "PerfTimer.h"      /* GetSystemTick_ms */
#include "StoryUI.h"
#include "SdSync.h"         /* SdSync_FrameId, SdSync_DownloadToFile */
#include "Audio.h"
#include "esp_http.h"
#include "VoiceChat.h"

/* Fresh clone: fall back to placeholder creds so the project still builds. */
#if defined(__has_include) && !__has_include("day2_config.h")
#include "day2_config.example.h"
#else
#include "day2_config.h"
#endif

/* From the last body byte sent: the LLM (3-7 s measured) plus the ~37 KB
 * answer image over the AT link (~3.3 s at 115200). */
#define VOICE_TIMEOUT_MS    (60000)

/* Answer image: TIM4 + HTTP headers, in SRAM2 (CPU-only, like Day-3's). */
__attribute__((section(".bss.vram.data"), aligned(4)))
static uint8_t s_answer[STORYUI_TIM4_MAX + 512];
static int     s_answerLen;     /* valid TIM4 bytes in s_answer, 0 = none */
static char    s_voiceId[20];   /* backend's id for the last answer, "" = none */
static int     s_voiceFrame;    /* the frame id that answer belongs to */

/* 小憶's spoken answer lands here before it plays (it is far too big for a
 * RAM buffer: ~8 KB per second of speech). */
#define VOICE_REPLY_FILE    "0:\\REPLY.ULW"

/* Query-string escape: album/photo/user names are plain ASCII today, but a
 * stray space or '&' must not split the query. Truncates to fit dst. */
static void url_encode(char *dst, int cap, const char *src)
{
    static const char hex[] = "0123456789ABCDEF";
    int n = 0;

    for (; src && *src; src++)
    {
        unsigned char c = (unsigned char)*src;
        int plain = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                    (c >= '0' && c <= '9') ||
                    c == '-' || c == '_' || c == '.' || c == '~';

        if (n + (plain ? 1 : 3) >= cap)
            break;
        if (plain)
        {
            dst[n++] = (char)c;
        }
        else
        {
            dst[n++] = '%';
            dst[n++] = hex[c >> 4];
            dst[n++] = hex[c & 0x0F];
        }
    }
    dst[n] = '\0';
}

int VoiceChat_Ask(const uint8_t *ulaw, uint32_t n, uint32_t rateHz,
                  const char *folder, const char *photo, const char *user,
                  char *action, int actCap)
{
    char path[224], f[48], p[64], u[64];
    char act[32] = "", vid[20] = "";    /* stay "" if the request fails early */
    http_hdr_out_t hdrs[2] =
    {
        { "x-frame-action:", act, (int)sizeof(act) },
        { "x-voice-id:",     vid, (int)sizeof(vid) },
    };
    int  frameId = SdSync_FrameId();
    int  st, answerLen = 0;

    if (action && actCap > 0)
        action[0] = '\0';
    s_answerLen  = 0;
    s_voiceId[0] = '\0';
    if (frameId < 0)
        return VOICECHAT_ERR_OFFLINE;

    url_encode(f, sizeof(f), folder);
    url_encode(p, sizeof(p), photo);
    url_encode(u, sizeof(u), user);
    snprintf(path, sizeof(path),
             "/frames/%d/voice?rate=%u&folder=%s&photo=%s&user=%s",
             frameId, (unsigned)rateHz, f, p, u);

    uint32_t t0 = GetSystemTick_ms();
    st = http_post_binary(BACKEND_HOST, BACKEND_PORT, path, ulaw, (int)n,
                          s_answer, (int)sizeof(s_answer), &answerLen,
                          VOICE_TIMEOUT_MS, hdrs, 2);
    printf("[VOICE] POST %s (%u B) -> %d, answer %d B, action '%s', %u ms\n",
           path, (unsigned)n, st, answerLen, act,
           (unsigned)(GetSystemTick_ms() - t0));

    /* The answer image comes back short now and then (the Wi-Fi module drops
     * bytes when its buffer fills). The backend keeps it under this id, so
     * fetch it again instead of paying for the whole answer a second time. */
    for (int attempt = 0; st != 200 && vid[0] && attempt < 2; attempt++)
    {
        snprintf(path, sizeof(path), "/frames/%d/voice/%s/image", frameId, vid);
        answerLen = 0;
        st = http_get_binary(BACKEND_HOST, BACKEND_PORT, path, s_answer,
                             (int)sizeof(s_answer), &answerLen, VOICE_TIMEOUT_MS);
        printf("[VOICE] re-fetch answer -> %d, %d B\n", st, answerLen);
    }

    if (st != 200)
        return VOICECHAT_ERR_HTTP;
    if (StoryUI_ShowTextImage(s_answer, (uint32_t)answerLen) != 0)
        return VOICECHAT_ERR_IMAGE;
    s_answerLen  = answerLen;
    s_voiceFrame = frameId;
    strncpy(s_voiceId, vid, sizeof(s_voiceId) - 1);
    s_voiceId[sizeof(s_voiceId) - 1] = '\0';

    if (action && actCap > 0)
    {
        strncpy(action, act, (size_t)actCap - 1);
        action[actCap - 1] = '\0';
    }
    return 0;
}

int VoiceChat_RedrawAnswer(void)
{
    if (s_answerLen <= 0)
        return -1;
    return StoryUI_ShowTextImage(s_answer, (uint32_t)s_answerLen);
}

int VoiceChat_SpeakAnswer(int (*cancel)(void))
{
    char url[96];
    int  rc;

    if (s_voiceId[0] == '\0')
        return -1;

    /* The backend starts this TTS as it sends the answer, so it is usually
     * ready by the time the rail has been drawn; its endpoint waits out the
     * rest. Straight to the card: speech is too big for a RAM buffer. */
    snprintf(url, sizeof(url), "/frames/%d/voice/%s/audio",
             s_voiceFrame, s_voiceId);
    rc = SdSync_DownloadToFile(url, VOICE_REPLY_FILE);
    if (rc != 0)
    {
        printf("[VOICE] no speech for this answer (rc=%d)\n", rc);
        return -2;
    }
    return Audio_PlayUlawFile(VOICE_REPLY_FILE, cancel);
}
