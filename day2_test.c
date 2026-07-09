/**************************************************************************//**
 * @file     day2_test.c
 * @brief    Day-2 Task 4: link validation main
 *           M55M1 -> UART8 -> ESP-12F -> Wi-Fi -> FastAPI backend -> back.
 *
 * Test 1  Join Wi-Fi (station) + report IP.
 * Test 2  HTTP GET  /ping   -> expect 200 + JSON body.
 * Test 3  HTTP POST /story  -> expect 200 + story text (face_id=1).
 * Test 4  Stability: 100x GET /ping, per-request latency stats.
 *
 * Success criteria (from the Day-2 brief):
 *   [1] ESP joins Wi-Fi via AT commands       (Test 1)
 *   [2] HTTP GET/POST reach the backend       (Tests 2+3)
 *   [3] Responses come back intact            (Tests 2+3 status/body)
 *   [4] 100 consecutive requests, no drop     (Test 4)
 *   [5] Round-trip latency < 1 s              (Test 4 avg; max reported too)
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "NuMicro.h"
#include "PerfTimer.h"
#include "esp_at.h"
#include "esp_http.h"
/* Fresh clone: fall back to placeholder creds so the project still builds. */
#if defined(__has_include) && !__has_include("day2_config.h")
#include "day2_config.example.h"
#else
#include "day2_config.h"
#endif
#include "day2_test.h"

#define STABILITY_ITERS   100
#define RESP_SZ           1024

static char s_resp[RESP_SZ];

/* results shared with the summary */
static int      s_wifiOk = 0, s_getOk = 0, s_postOk = 0;
static int      s_stabOkCnt = 0;
static uint32_t s_latAvg = 0, s_latMin = 0, s_latMax = 0;

static const char *rc_str(int rc)
{
    switch (rc)
    {
        case ESP_ERR_TIMEOUT: return "TIMEOUT";
        case ESP_ERR_ERROR:   return "ERROR reply";
        case ESP_ERR_PARAM:   return "bad param";
        case ESP_ERR_STATE:   return "bad state";
        case HTTP_ERR_PARSE:  return "HTTP parse failed";
        default:              return "unknown";
    }
}

/*------------------------------------------------------------------
 * Test 1: AT alive -> station mode -> join AP -> IP
 *----------------------------------------------------------------*/
static void test_wifi_connect(void)
{
    char     ip[16] = "";
    int      rc;
    uint32_t t0;

    printf("\n=== Test 1: Wi-Fi connect ===\n");

    if ((rc = esp_at_init()) != ESP_OK)
    {
        printf("  [FAIL] esp_at_init: %s\n", rc_str(rc));
        return;
    }
    printf("  [OK] AT firmware alive @115200, echo off, CIPMUX=0\n");

    if ((rc = esp_wifi_set_mode(1)) != ESP_OK)
    {
        printf("  [FAIL] CWMODE=1 (station): %s\n", rc_str(rc));
        return;
    }

    printf("  joining \"%s\" ...\n", WIFI_SSID);
    t0 = GetSystemTick_ms();
    rc = esp_wifi_connect(WIFI_SSID, WIFI_PWD);

    if (rc != ESP_OK)
    {
        printf("  [FAIL] CWJAP: %s (check SSID/password, 2.4 GHz only)\n", rc_str(rc));
        return;
    }
    printf("  joined in %u ms\n", (unsigned)(GetSystemTick_ms() - t0));

    if ((rc = esp_wifi_get_ip(ip)) != ESP_OK)
    {
        printf("  [FAIL] CIFSR/no DHCP lease: %s\n", rc_str(rc));
        return;
    }

    printf("  [OK] station IP: %s\n", ip);
    printf("Test 1 result: PASS\n");
    s_wifiOk = 1;
}

/*------------------------------------------------------------------
 * Test 2: GET /ping
 *----------------------------------------------------------------*/
static void test_http_ping(void)
{
    int      st;
    uint32_t t0, dt;

    printf("\n=== Test 2: HTTP GET /ping ===\n");

    if (!s_wifiOk)
    {
        printf("  skipped: Wi-Fi not connected\n");
        return;
    }

    t0 = GetSystemTick_ms();
    st = http_get(BACKEND_HOST, BACKEND_PORT, "/ping", s_resp, RESP_SZ);
    dt = GetSystemTick_ms() - t0;

    if (st < 0)
    {
        printf("  [FAIL] %s (backend up? firewall? host %s:%d)\n",
               rc_str(st), BACKEND_HOST, BACKEND_PORT);
        return;
    }

    printf("  status=%d  round-trip=%u ms\n", st, (unsigned)dt);
    printf("  body: %s\n", s_resp);
    printf("Test 2 result: %s\n", (st == 200) ? "PASS" : "FAIL");
    s_getOk = (st == 200);
}

/*------------------------------------------------------------------
 * Test 3: POST /story (face_id = 1)
 *----------------------------------------------------------------*/
static void test_http_post(void)
{
    char     body[96];
    int      st;
    uint32_t t0, dt, ms;

    printf("\n=== Test 3: HTTP POST /story ===\n");

    if (!s_wifiOk)
    {
        printf("  skipped: Wi-Fi not connected\n");
        return;
    }

    ms = GetSystemTick_ms();
    snprintf(body, sizeof(body), "{\"face_id\":1,\"timestamp\":%u.%03u}",
             (unsigned)(ms / 1000u), (unsigned)(ms % 1000u));
    printf("  request body: %s\n", body);

    t0 = GetSystemTick_ms();
    st = http_post_json(BACKEND_HOST, BACKEND_PORT, "/story", body, s_resp, RESP_SZ);
    dt = GetSystemTick_ms() - t0;

    if (st < 0)
    {
        printf("  [FAIL] %s\n", rc_str(st));
        return;
    }

    printf("  status=%d  round-trip=%u ms\n", st, (unsigned)dt);
    printf("  body: %s\n", s_resp);   /* story text is UTF-8 Chinese */
    printf("Test 3 result: %s\n", (st == 200) ? "PASS" : "FAIL");
    s_postOk = (st == 200);
}

/*------------------------------------------------------------------
 * Test 4: 100x GET /ping — drop count + latency stats
 *----------------------------------------------------------------*/
static void test_stability(void)
{
    uint64_t sum = 0;
    uint32_t lmin = 0xFFFFFFFFu, lmax = 0;
    int      okCnt = 0, failCnt = 0, worstStreak = 0, streak = 0;
    int      i;

    printf("\n=== Test 4: stability, %dx GET /ping ===\n", STABILITY_ITERS);

    if (!s_wifiOk)
    {
        printf("  skipped: Wi-Fi not connected\n");
        return;
    }

    for (i = 1; i <= STABILITY_ITERS; i++)
    {
        uint32_t t0 = GetSystemTick_ms();
        int      st = http_get(BACKEND_HOST, BACKEND_PORT, "/ping", s_resp, RESP_SZ);
        uint32_t dt = GetSystemTick_ms() - t0;

        if (st == 200)
        {
            okCnt++;
            streak = 0;
            sum += dt;
            if (dt < lmin) lmin = dt;
            if (dt > lmax) lmax = dt;
        }
        else
        {
            failCnt++;
            if (++streak > worstStreak) worstStreak = streak;
            printf("  #%03d FAIL (%s, %u ms)\n", i, rc_str(st), (unsigned)dt);
        }

        if ((i % 10) == 0)
            printf("  ...%d/%d  (ok=%d fail=%d)\n", i, STABILITY_ITERS, okCnt, failCnt);
    }

    s_stabOkCnt = okCnt;

    if (okCnt > 0)
    {
        s_latAvg = (uint32_t)(sum / (uint32_t)okCnt);
        s_latMin = lmin;
        s_latMax = lmax;
        printf("  latency over %d OK requests: avg=%u ms  min=%u ms  max=%u ms\n",
               okCnt, (unsigned)s_latAvg, (unsigned)s_latMin, (unsigned)s_latMax);
    }

    printf("  drops: %d (worst consecutive: %d)\n", failCnt, worstStreak);
    printf("  RX health: ring overflow=%u, FIFO overrun=%u\n",
           (unsigned)esp_ll_rx_overflow(), (unsigned)esp_ll_rx_hw_errors());
    printf("Test 4 result: %s\n",
           (okCnt == STABILITY_ITERS) ? "PASS" : "FAIL");
}

/*------------------------------------------------------------------
 * Entry
 *----------------------------------------------------------------*/
void day2_test_run(void)
{
    printf("\n");
    printf("==============================================\n");
    printf(" M55M1 Day 2 - Wi-Fi Communication Validation\n");
    printf("==============================================\n");
    printf(" backend  : http://%s:%d  (FastAPI /ping, /story)\n",
           BACKEND_HOST, BACKEND_PORT);
    printf(" Wi-Fi    : \"%s\" (2.4 GHz)\n", WIFI_SSID);

    test_wifi_connect();
    test_http_ping();
    test_http_post();
    test_stability();

    printf("\n==============================================\n");
    printf(" Day-2 Summary\n");
    printf("==============================================\n");
    printf(" [%s] AT commands join Wi-Fi + get IP\n",       s_wifiOk ? "PASS" : "FAIL");
    printf(" [%s] HTTP GET  /ping  -> 200 + body\n",        s_getOk  ? "PASS" : "FAIL");
    printf(" [%s] HTTP POST /story -> 200 + story\n",       s_postOk ? "PASS" : "FAIL");
    printf(" [%s] %d/%d requests OK, no drop\n",
           (s_stabOkCnt == STABILITY_ITERS) ? "PASS" : "FAIL",
           s_stabOkCnt, STABILITY_ITERS);
    printf(" [%s] round-trip avg %u ms (min %u / max %u) < 1000 ms\n",
           (s_stabOkCnt > 0 && s_latAvg < 1000u) ? "PASS" : "FAIL",
           (unsigned)s_latAvg, (unsigned)s_latMin, (unsigned)s_latMax);
    printf(" Overall: %s\n",
           (s_wifiOk && s_getOk && s_postOk &&
            s_stabOkCnt == STABILITY_ITERS && s_latAvg < 1000u)
               ? "DAY-2 PASS" : "NEEDS REVIEW");
    printf("==============================================\n");
    printf("=== Day 2 Tests Done === (paste this whole log back for day2_report.md)\n");

    while (1)
    {
        __WFI();
    }
}
