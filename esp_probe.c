/**************************************************************************//**
 * @file     esp_probe.c
 * @brief    Day-2 Task 0: probe the on-board ESP-12F to find out what firmware
 *           it runs (AT firmware / something else / dead), before any Wi-Fi
 *           work is attempted.
 *
 * RESULT (board run 2026-07-07): AT_FIRMWARE_PRESENT @115200 —
 * AT 1.7.0.0 (Aug 16 2018) / SDK 3.0.0, Wroom-02 bin v1.7.0, 16Mbit flash.
 * Boot ROM chatter clean at 74880, ring overflow 0. Kept for re-runs after
 * any reflash of the module.
 *
 * Wiring (confirmed from the official BSP SecureOTADemo sample):
 *   - ESP-12F UART  : UART8, TXD = PJ0, RXD = PJ1
 *   - ESP-12F RESET : PD2 (low pulse resets)
 * UART8 hardware access lives in esp_at.c's esp_ll_* layer (shared IRQ
 * handler + ring buffer); this file only sequences the probe.
 *
 * Probe sequence:
 *   Step 1  HW-reset while listening at 74880 baud — the ESP8266 boot ROM
 *           always prints its banner there (26 MHz crystal). Readable text
 *           proves the module has power and executes code.
 *   Step 2  Baud scan {115200, 9600, 74880, 57600, 38400}: "AT" -> "OK"?
 *   Step 3  If OK: AT+GMR (firmware version) + AT+UART_CUR?.
 *   Step 4  Machine-parsable verdict block.
 *
 * Runs BEFORE PerfTimer_Init() and never returns; delays use
 * CLK_SysTickDelay (fine here — nothing else owns SysTick yet).
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "NuMicro.h"
#include "esp_at.h"        /* esp_ll_* low-level UART8 layer */
#include "esp_probe.h"

/*----------------------------------------------------------------------------
 * Small helpers
 *--------------------------------------------------------------------------*/
static void delay_ms(uint32_t ms)
{
    while (ms--)
        CLK_SysTickDelay(1000);            /* 1000 us busy-wait */
}

/* Print a captured buffer with control/binary bytes made visible:
 * printable ASCII as-is, LF as newline, CR dropped, the rest as [xx]. */
static void print_escaped(const char *buf, int len)
{
    int i;

    for (i = 0; i < len; i++)
    {
        uint8_t c = (uint8_t)buf[i];

        if (c == '\n')
            printf("\n");
        else if (c == '\r')
            ; /* drop */
        else if (c >= 0x20 && c < 0x7F)
            printf("%c", c);
        else
            printf("[%02X]", c);
    }

    if (len > 0)
        printf("\n");
}

static void print_hex_head(const char *buf, int len, int max_bytes)
{
    int n = (len < max_bytes) ? len : max_bytes;
    int i;

    printf("        hex[0..%d]:", n - 1);

    for (i = 0; i < n; i++)
        printf(" %02X", (uint8_t)buf[i]);

    printf("\n");
}

/* Collect ESP output into buf.
 *  - stops early when "OK\r\n" / "ERROR" / "FAIL" is seen (if stop_on_status)
 *  - stops when idle_ms passes with data already received and nothing new
 *  - always stops at timeout_ms
 * Returns byte count; *saw_ok / *saw_err report the status match. */
static int collect_response(char *buf, int maxlen,
                            uint32_t timeout_ms, uint32_t idle_ms,
                            int stop_on_status, int *saw_ok, int *saw_err)
{
    int      len = 0;
    uint32_t waited = 0, idle = 0;

    *saw_ok  = 0;
    *saw_err = 0;
    buf[0]   = '\0';

    while (waited < timeout_ms)
    {
        uint8_t ch;
        int     got = 0;

        while (len < maxlen - 1 && esp_ll_getc(&ch))
        {
            buf[len++] = (char)ch;
            got = 1;
        }

        buf[len] = '\0';

        if (got)
        {
            idle = 0;

            if (stop_on_status)
            {
                if (strstr(buf, "OK\r\n") != NULL)
                {
                    *saw_ok = 1;
                    /* grace period: catch bytes trailing right behind "OK" */
                    delay_ms(30);

                    while (len < maxlen - 1 && esp_ll_getc(&ch))
                        buf[len++] = (char)ch;

                    buf[len] = '\0';
                    break;
                }

                if (strstr(buf, "ERROR") != NULL || strstr(buf, "FAIL") != NULL)
                {
                    *saw_err = 1;
                    break;
                }
            }
        }
        else
        {
            idle++;

            if (len > 0 && idle >= idle_ms)
                break;
        }

        delay_ms(1);
        waited++;
    }

    return len;
}

static void send_cmd(const char *cmd)
{
    esp_ll_flush_rx();
    esp_ll_write(cmd, (uint32_t)strlen(cmd));
}

/*----------------------------------------------------------------------------
 * Probe main
 *--------------------------------------------------------------------------*/
#define RESP_BUF_SZ  2048

static char s_resp[RESP_BUF_SZ];

void esp_probe_run(void)
{
    static const uint32_t bauds[] = { 115200, 9600, 74880, 57600, 38400 };
    const int   nBauds = (int)(sizeof(bauds) / sizeof(bauds[0]));
    int         bytesAtBaud[8] = { 0 };
    uint32_t    foundBaud = 0;
    int         i, try_, len, ok, err;

    printf("\n");
    printf("==============================================\n");
    printf(" M55M1 Day 2 - Task 0: ESP-12F firmware probe\n");
    printf("==============================================\n");
    printf(" ESP UART : UART8  TXD=PJ0  RXD=PJ1  (per BSP SecureOTADemo)\n");
    printf(" ESP RST  : PD2\n");

    /*------------------------------------------------------------------
     * Step 1: hardware reset, listen to boot chatter at 74880
     * (ESP8266 boot ROM baud with the ESP-12F's 26 MHz crystal).
     *----------------------------------------------------------------*/
    printf("\n[STEP 1] HW reset via PD2, capturing boot output @74880...\n");
    esp_ll_init(74880);
    delay_ms(50);
    esp_ll_flush_rx();
    esp_ll_hw_reset();

    len = collect_response(s_resp, RESP_BUF_SZ, 2000, 400, 0, &ok, &err);

    if (len > 0)
    {
        printf("  -> %d bytes of boot output:\n", len);
        printf("---------- boot chatter (74880) ----------\n");
        print_escaped(s_resp, len);
        printf("-------------------------------------------\n");
        printf("  NOTE: readable 'ets Jan  8 2013 / rst cause / boot mode' text\n");
        printf("        = module alive. Garbage AFTER that line is normal (the\n");
        printf("        flashed firmware talks at its own baud, e.g. 115200).\n");
    }
    else
    {
        printf("  -> NOTHING received. Module may be unpowered/held in reset,\n");
        printf("     or TX/RX wiring differs from the BSP sample.\n");
    }

    /* Give AT firmware (if any) time to finish booting before the scan. */
    delay_ms(800);

    /*------------------------------------------------------------------
     * Step 2: baud scan with "AT"
     *----------------------------------------------------------------*/
    printf("\n[STEP 2] Baud scan: sending AT, waiting for OK...\n");

    for (i = 0; i < nBauds && foundBaud == 0; i++)
    {
        printf("  baud %6u: ", (unsigned)bauds[i]);
        esp_ll_set_baud(bauds[i]);
        delay_ms(30);

        for (try_ = 1; try_ <= 3; try_++)
        {
            send_cmd("AT\r\n");
            len = collect_response(s_resp, RESP_BUF_SZ, 1000, 200, 1, &ok, &err);
            bytesAtBaud[i] += len;

            if (ok)
            {
                printf("OK (try %d)\n", try_);
                foundBaud = bauds[i];
                break;
            }
        }

        if (foundBaud == 0)
        {
            if (bytesAtBaud[i] > 0)
            {
                printf("no OK, %d bytes of noise/garbage\n", bytesAtBaud[i]);
                print_hex_head(s_resp, len, 32);
            }
            else
            {
                printf("silent\n");
            }
        }
    }

    /*------------------------------------------------------------------
     * Step 3: firmware identification
     *----------------------------------------------------------------*/
    if (foundBaud != 0)
    {
        printf("\n[STEP 3] AT firmware detected @ %u baud. Querying version...\n",
               (unsigned)foundBaud);

        send_cmd("AT+GMR\r\n");
        len = collect_response(s_resp, RESP_BUF_SZ, 3000, 300, 1, &ok, &err);
        printf("---------- AT+GMR ----------\n");
        print_escaped(s_resp, len);
        printf("----------------------------\n");

        send_cmd("AT+UART_CUR?\r\n");
        len = collect_response(s_resp, RESP_BUF_SZ, 1000, 200, 1, &ok, &err);
        printf("---------- AT+UART_CUR? (ERROR = old AT fw, that's fine) ----------\n");
        print_escaped(s_resp, len);
        printf("--------------------------------------------------------------------\n");
    }

    /*------------------------------------------------------------------
     * Step 4: verdict
     *----------------------------------------------------------------*/
    printf("\n==============================================\n");
    printf(" TASK 0 VERDICT\n");
    printf("==============================================\n");

    if (foundBaud != 0)
    {
        printf(" RESULT   : AT_FIRMWARE_PRESENT\n");
        printf(" BAUD     : %u\n", (unsigned)foundBaud);
        printf(" UART     : UART8 (TXD=PJ0 RXD=PJ1), reset=PD2\n");
        printf(" NEXT     : proceed to Task 1 (AT driver layer)\n");
    }
    else
    {
        int anyBytes = 0;

        for (i = 0; i < nBauds; i++)
            anyBytes += bytesAtBaud[i];

        if (anyBytes > 0)
        {
            printf(" RESULT   : ALIVE_BUT_NO_AT\n");
            printf(" DETAIL   : ESP transmits (%d bytes total) but never answers OK.\n", anyBytes);
            printf("            Likely a non-AT firmware (NodeMCU/Arduino/custom),\n");
            printf("            or its baud is outside the scan set.\n");
            printf(" NEXT     : reflash Espressif AT firmware, or tell me the\n");
            printf("            boot-chatter text above so I can identify it.\n");
        }
        else
        {
            printf(" RESULT   : NO_RESPONSE\n");
            printf(" DETAIL   : not a single byte at any baud, incl. boot chatter.\n");
            printf("            Check: module power / EN pin, PD2 reset polarity,\n");
            printf("            or UART wiring differs from BSP sample.\n");
            printf(" NEXT     : paste this log back; we debug wiring before firmware.\n");
        }
    }

    printf(" RX stats : ring overflow=%u bytes, UART FIFO overrun events=%u\n",
           (unsigned)esp_ll_rx_overflow(), (unsigned)esp_ll_rx_hw_errors());
    printf("==============================================\n");
    printf(" Probe done. Press RESET to run again.\n");

    while (1)
    {
        __WFI();
    }
}
