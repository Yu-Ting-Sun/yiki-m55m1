/**************************************************************************//**
 * @file     esp_at.c
 * @brief    ESP-12F AT-command driver (see esp_at.h for the layer contract).
 *
 * Verified module (Task-0 probe): AT 1.7.0.0 / SDK 3.0.0, 115200 8N1.
 * The module reports +UART_CUR flow-control=1 (its RTS output is enabled);
 * we deliberately ignore it — every payload we send is far below the ESP's
 * RX buffer, and the M55M1 side runs no HW flow control (matches the BSP
 * SecureOTADemo wiring). Revisit only if Day-3 ever streams big uploads.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "NuMicro.h"
#include "PerfTimer.h"      /* GetSystemTick_ms(); PerfTimer_Init() is the caller's job */
#include "esp_at.h"

/*---------------------------------------------------------------------------
 * Board wiring (per BSP SecureOTADemo, proven by the Task-0 probe)
 *-------------------------------------------------------------------------*/
#define ESP_UART            UART8
#define ESP_UART_IRQ        UART8_IRQn
#define ESP_RST_PIN         PD2

/* Set 1 to mirror every AT exchange on the debug console (very chatty). */
#define ESP_AT_TRACE        (0)

#if ESP_AT_TRACE
    #define AT_TRACE(...)   printf(__VA_ARGS__)
#else
    #define AT_TRACE(...)
#endif

/*---------------------------------------------------------------------------
 * Low-level layer: UART8 + RX ring buffer + PD2 reset
 *-------------------------------------------------------------------------*/
#define RX_RING_SIZE        4096u          /* power of two */

static volatile uint8_t  s_rxRing[RX_RING_SIZE];
static volatile uint32_t s_rxHead = 0;
static volatile uint32_t s_rxTail = 0;
static volatile uint32_t s_rxOverflow = 0;
static volatile uint32_t s_rxHwErr    = 0;

void UART8_IRQHandler(void)
{
    if (UART_GET_INT_FLAG(ESP_UART, UART_INTSTS_RDAINT_Msk))
    {
        while (UART_IS_RX_READY(ESP_UART))
        {
            uint8_t  ch   = (uint8_t)UART_READ(ESP_UART);
            uint32_t next = (s_rxHead + 1u) & (RX_RING_SIZE - 1u);

            if (next != s_rxTail)
            {
                s_rxRing[s_rxHead] = ch;
                s_rxHead = next;
            }
            else
            {
                s_rxOverflow++;
            }
        }
    }

    if (UART_GET_INT_FLAG(ESP_UART, UART_INTSTS_BUFERRIF_Msk))
    {
        s_rxHwErr++;
        ESP_UART->FIFOSTS |= (UART_FIFOSTS_RXOVIF_Msk | UART_FIFOSTS_TXOVIF_Msk);
    }
}

void esp_ll_init(uint32_t baud)
{
    SYS_UnlockReg();
    CLK_EnableModuleClock(UART8_MODULE);
    CLK_SetModuleClock(UART8_MODULE, CLK_UARTSEL1_UART8SEL_HXT,
                       CLK_UARTDIV1_UART8DIV(1));
    CLK_EnableModuleClock(GPIOD_MODULE);   /* PD2 reset line */
    CLK_EnableModuleClock(GPIOJ_MODULE);   /* PJ0/PJ1 UART pins */
    SET_UART8_TXD_PJ0();
    SET_UART8_RXD_PJ1();
    SYS_LockReg();

    GPIO_SetMode(PD, BIT2, GPIO_MODE_OUTPUT);
    ESP_RST_PIN = 1;                       /* deassert reset */

    esp_ll_set_baud(baud);
    esp_ll_flush_rx();
    NVIC_EnableIRQ(ESP_UART_IRQ);
}

void esp_ll_set_baud(uint32_t baud)
{
    UART_Open(ESP_UART, baud);             /* 8N1 + baud (clock src already set) */
    UART_ENABLE_INT(ESP_UART, (UART_INTEN_RDAIEN_Msk | UART_INTEN_BUFERRIEN_Msk));
}

void esp_ll_write(const void *data, uint32_t len)
{
    UART_Write(ESP_UART, (uint8_t *)data, len);
}

int esp_ll_getc(uint8_t *ch)
{
    if (s_rxTail == s_rxHead)
        return 0;

    *ch = s_rxRing[s_rxTail];
    s_rxTail = (s_rxTail + 1u) & (RX_RING_SIZE - 1u);
    return 1;
}

void esp_ll_flush_rx(void)
{
    s_rxTail = s_rxHead;
}

/* Timer-free busy wait so this works both before PerfTimer_Init() (probe)
 * and after (AT layer). ~4 cycles per iteration is close enough here. */
static void ll_burn_ms(uint32_t ms)
{
    volatile uint32_t n = (SystemCoreClock / 4000u) * ms;
    while (n--) { }
}

void esp_ll_hw_reset(void)
{
    ESP_RST_PIN = 0;
    ll_burn_ms(20);
    ESP_RST_PIN = 1;
}

uint32_t esp_ll_rx_overflow(void)  { return s_rxOverflow; }
uint32_t esp_ll_rx_hw_errors(void) { return s_rxHwErr;    }

/*---------------------------------------------------------------------------
 * AT layer plumbing
 *-------------------------------------------------------------------------*/
#define AT_RESP_SZ          1024

static char s_atResp[AT_RESP_SZ];

static uint32_t now_ms(void)
{
    return GetSystemTick_ms();
}

/* Collect module output into resp until a success or failure token shows up.
 * Success tokens: ok_tok (usually "OK\r\n").
 * Failure tokens: "ERROR" or "FAIL" (covers SEND FAIL, +CWJAP ... FAIL).
 * Returns ESP_OK / ESP_ERR_ERROR / ESP_ERR_TIMEOUT; resp always NUL-ended. */
static int at_wait(const char *ok_tok, uint32_t timeout_ms,
                   char *resp, int resp_sz)
{
    int      len = 0;
    uint32_t t0  = now_ms();

    resp[0] = '\0';

    while ((now_ms() - t0) < timeout_ms)
    {
        uint8_t ch;

        while (len < resp_sz - 1 && esp_ll_getc(&ch))
        {
            resp[len++] = (char)ch;
            resp[len]   = '\0';

            if (strstr(resp, ok_tok) != NULL)
            {
                AT_TRACE("[AT<] %s", resp);
                return ESP_OK;
            }

            if (strstr(resp, "ERROR") != NULL || strstr(resp, "FAIL") != NULL)
            {
                AT_TRACE("[AT<] %s", resp);
                return ESP_ERR_ERROR;
            }
        }

        if (len >= resp_sz - 1)            /* buffer full without a verdict */
            return ESP_ERR_ERROR;
    }

    AT_TRACE("[AT<timeout] %s", resp);
    return ESP_ERR_TIMEOUT;
}

/* Send one AT command line and wait for its verdict. */
static int at_cmd(const char *cmd, const char *ok_tok, uint32_t timeout_ms)
{
    AT_TRACE("[AT>] %s", cmd);
    esp_ll_flush_rx();
    esp_ll_write(cmd, (uint32_t)strlen(cmd));
    return at_wait(ok_tok, timeout_ms, s_atResp, AT_RESP_SZ);
}

/*---------------------------------------------------------------------------
 * AT layer — basic control
 *-------------------------------------------------------------------------*/
int esp_at_check(void)
{
    return at_cmd("AT\r\n", "OK\r\n", 1000);
}

static int at_apply_base_config(void)
{
    int rc;

    if ((rc = at_cmd("ATE0\r\n", "OK\r\n", 1000)) != ESP_OK)      /* echo off */
        return rc;

    return at_cmd("AT+CIPMUX=0\r\n", "OK\r\n", 1000);             /* single conn */
}

int esp_at_init(void)
{
    int rc = ESP_ERR_TIMEOUT;
    int i;

    esp_ll_init(115200);                   /* Task-0 verdict: AT fw @115200 */

    for (i = 0; i < 3; i++)
    {
        if ((rc = esp_at_check()) == ESP_OK)
            break;
    }

    if (rc != ESP_OK)                      /* unresponsive: one hard reset try */
    {
        if ((rc = esp_at_reset()) != ESP_OK)
            return rc;
        return ESP_OK;                     /* esp_at_reset re-applied config */
    }

    return at_apply_base_config();
}

int esp_at_reset(void)
{
    int rc = at_cmd("AT+RST\r\n", "ready", 5000);

    if (rc != ESP_OK)                      /* soft reset failed: yank PD2 */
    {
        esp_ll_flush_rx();
        esp_ll_hw_reset();
        rc = at_wait("ready", 5000, s_atResp, AT_RESP_SZ);
        if (rc != ESP_OK)
            return rc;
    }

    /* "ready" arrives before Wi-Fi bring-up chatter; settle, then re-config
     * (ATE0/CIPMUX are volatile across reset). */
    {
        uint32_t t0 = now_ms();
        while ((now_ms() - t0) < 300u) { }
    }

    return at_apply_base_config();
}

/*---------------------------------------------------------------------------
 * AT layer — Wi-Fi
 *-------------------------------------------------------------------------*/
int esp_wifi_set_mode(int mode)
{
    char cmd[32];

    if (mode < 1 || mode > 3)
        return ESP_ERR_PARAM;

    snprintf(cmd, sizeof(cmd), "AT+CWMODE_CUR=%d\r\n", mode);
    return at_cmd(cmd, "OK\r\n", 2000);
}

int esp_wifi_connect(const char *ssid, const char *pwd)
{
    char cmd[160];

    if (!ssid || !pwd)
        return ESP_ERR_PARAM;

    /* NOTE: SSID/password containing " , or \ would need AT escaping. */
    snprintf(cmd, sizeof(cmd), "AT+CWJAP_CUR=\"%s\",\"%s\"\r\n", ssid, pwd);
    return at_cmd(cmd, "OK\r\n", 20000);   /* join + DHCP can take a while */
}

int esp_wifi_get_ip(char *ip_out)
{
    int rc;

    if (!ip_out)
        return ESP_ERR_PARAM;

    ip_out[0] = '\0';

    if ((rc = at_cmd("AT+CIFSR\r\n", "OK\r\n", 2000)) != ESP_OK)
        return rc;

    /* Response holds: +CIFSR:STAIP,"192.168.x.x" */
    {
        const char *p = strstr(s_atResp, "STAIP,\"");
        int i = 0;

        if (!p)
            return ESP_ERR_ERROR;

        p += 7;
        while (p[i] && p[i] != '"' && i < 15)
        {
            ip_out[i] = p[i];
            i++;
        }
        ip_out[i] = '\0';
    }

    if (strcmp(ip_out, "0.0.0.0") == 0)    /* joined nothing / no DHCP lease */
        return ESP_ERR_STATE;

    return ESP_OK;
}

int esp_wifi_get_mac(char *mac_out)
{
    int rc;

    if (!mac_out)
        return ESP_ERR_PARAM;

    mac_out[0] = '\0';

    if ((rc = at_cmd("AT+CIFSR\r\n", "OK\r\n", 2000)) != ESP_OK)
        return rc;

    /* Response holds: +CIFSR:STAMAC,"aa:bb:cc:dd:ee:ff" — the board's
     * stable hardware identity (used by /frames/register). */
    {
        const char *p = strstr(s_atResp, "STAMAC,\"");
        int i = 0;

        if (!p)
            return ESP_ERR_ERROR;

        p += 8;
        while (p[i] && p[i] != '"' && i < 17)
        {
            mac_out[i] = p[i];
            i++;
        }
        mac_out[i] = '\0';
    }

    return mac_out[0] ? ESP_OK : ESP_ERR_ERROR;
}

/*---------------------------------------------------------------------------
 * AT layer — TCP (single connection, CIPMUX=0)
 *-------------------------------------------------------------------------*/
int esp_tcp_connect(const char *host, int port)
{
    char cmd[128];

    if (!host || port <= 0 || port > 65535)
        return ESP_ERR_PARAM;

    snprintf(cmd, sizeof(cmd), "AT+CIPSTART=\"TCP\",\"%s\",%d\r\n", host, port);
    return at_cmd(cmd, "OK\r\n", 10000);   /* covers DNS + TCP handshake */
}

int esp_tcp_send(const char *data, int len)
{
    char cmd[32];
    int  rc;

    if (!data || len <= 0 || len > 2048)   /* 2048 = AT+CIPSEND hard limit */
        return ESP_ERR_PARAM;

    snprintf(cmd, sizeof(cmd), "AT+CIPSEND=%d\r\n", len);

    if ((rc = at_cmd(cmd, ">", 2000)) != ESP_OK)
        return rc;

    esp_ll_write(data, (uint32_t)len);
    return at_wait("SEND OK", 5000, s_atResp, AT_RESP_SZ);
}

/* Demultiplex "+IPD,<len>:<payload>" frames into buf. Returns payload bytes
 * collected (>= 0), ESP_ERR_TIMEOUT if the deadline passed with zero bytes.
 * Ends early once the peer closes the connection ("CLOSED") — which, with
 * "Connection: close" HTTP, is exactly the end-of-response signal. */
int esp_tcp_recv(char *buf, int max_len, int timeout_ms)
{
    enum { SCAN, LEN, DATA } st = SCAN;
    const char *tag       = "+IPD,";
    const char *closedTag = "CLOSED";
    int  tagPos = 0, closedPos = 0;
    int  frameLeft = 0;
    int  total = 0;
    uint32_t t0 = now_ms();

    if (!buf || max_len <= 0)
        return ESP_ERR_PARAM;

    while ((now_ms() - t0) < (uint32_t)timeout_ms)
    {
        uint8_t ch;

        if (!esp_ll_getc(&ch))
            continue;

        switch (st)
        {
            case SCAN:
                /* match "+IPD," */
                if ((char)ch == tag[tagPos])
                {
                    if (tag[++tagPos] == '\0')
                    {
                        st = LEN;
                        frameLeft = 0;
                        tagPos = 0;
                    }
                }
                else
                {
                    tagPos = ((char)ch == tag[0]) ? 1 : 0;
                }

                /* watch for "CLOSED" between frames */
                if ((char)ch == closedTag[closedPos])
                {
                    if (closedTag[++closedPos] == '\0')
                        return total;      /* peer closed: response complete */
                }
                else
                {
                    closedPos = ((char)ch == closedTag[0]) ? 1 : 0;
                }
                break;

            case LEN:
                if (ch >= '0' && ch <= '9')
                    frameLeft = frameLeft * 10 + (ch - '0');
                else if ((char)ch == ':')
                    st = (frameLeft > 0) ? DATA : SCAN;
                else
                    st = SCAN;             /* malformed; resync */
                break;

            case DATA:
                if (total < max_len)
                    buf[total++] = (char)ch;
                /* else: frame bigger than caller buffer — drain and drop */

                if (--frameLeft == 0)
                    st = SCAN;

                if (total >= max_len)      /* caller buffer full: stop here */
                    return total;
                break;
        }
    }

    return (total > 0) ? total : ESP_ERR_TIMEOUT;
}

int esp_tcp_close(void)
{
    int rc = at_cmd("AT+CIPCLOSE\r\n", "OK\r\n", 2000);

    if (rc == ESP_ERR_ERROR)               /* link was already closed — fine */
        return ESP_OK;

    return rc;
}
