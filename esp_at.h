/**************************************************************************//**
 * @file     esp_at.h
 * @brief    Day-2 Task 1: ESP-12F AT-command driver for NuMaker-M55M1.
 *
 * Two layers:
 *   esp_ll_*  — UART8 hardware layer (clock/pins/IRQ/ring buffer/PD2 reset).
 *               Owns UART8_IRQHandler. Shared with esp_probe.c.
 *   esp_*     — AT command layer (Wi-Fi + single-connection TCP), verified
 *               against the on-board module: AT 1.7.0.0 / SDK 3.0.0 @115200
 *               (Task-0 probe, 2026-07-07).
 *
 * TIMING CONTRACT: the AT layer takes all timeouts from GetSystemTick_ms()
 * (PerfTimer.h), so PerfTimer_Init() MUST run before any esp_at/esp_wifi/
 * esp_tcp call. Never call CLK_SysTickDelay() after PerfTimer_Init() — it
 * reprograms SysTick and kills the pmu fallback counter this board relies on.
 * The esp_ll_* layer itself is timer-free (busy-wait only in esp_ll_hw_reset)
 * so esp_probe.c may keep using it before PerfTimer_Init().
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef ESP_AT_H
#define ESP_AT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*---------------------------------------------------------------------------
 * Return codes (0 = success, negative = failure)
 *-------------------------------------------------------------------------*/
#define ESP_OK             (0)
#define ESP_ERR_TIMEOUT    (-1)   /* expected reply never arrived            */
#define ESP_ERR_ERROR      (-2)   /* module replied ERROR / FAIL             */
#define ESP_ERR_PARAM      (-3)   /* bad argument                            */
#define ESP_ERR_STATE      (-4)   /* wrong link state (e.g. already open)    */

/*---------------------------------------------------------------------------
 * Low-level UART8 layer (TXD=PJ0, RXD=PJ1, reset=PD2 — per BSP SecureOTADemo)
 *-------------------------------------------------------------------------*/
void     esp_ll_init(uint32_t baud);      /* clocks + pins + IRQ + ring reset */
void     esp_ll_set_baud(uint32_t baud);  /* reconfigure UART8, keep IRQ armed */
void     esp_ll_write(const void *data, uint32_t len);   /* blocking TX */
int      esp_ll_getc(uint8_t *ch);        /* 1 = got byte, 0 = ring empty */
void     esp_ll_flush_rx(void);           /* drop everything buffered so far */
void     esp_ll_hw_reset(void);           /* PD2 low ~20 ms pulse (busy-wait) */
uint32_t esp_ll_rx_overflow(void);        /* bytes dropped: ring full */
uint32_t esp_ll_rx_hw_errors(void);       /* UART FIFO overrun events */

/*---------------------------------------------------------------------------
 * AT layer — basic control
 *-------------------------------------------------------------------------*/
int esp_at_init(void);         /* esp_ll_init(115200) + AT alive + ATE0 + CIPMUX=0 */
int esp_at_reset(void);        /* AT+RST (hw-reset fallback), wait "ready", re-init */
int esp_at_check(void);        /* AT -> OK */

/*---------------------------------------------------------------------------
 * AT layer — Wi-Fi (station)
 *-------------------------------------------------------------------------*/
int esp_wifi_set_mode(int mode);                          /* AT+CWMODE_CUR (1=STA) */
int esp_wifi_connect(const char *ssid, const char *pwd);  /* AT+CWJAP_CUR, <=20 s */
int esp_wifi_get_ip(char *ip_out);                        /* AT+CIFSR; >=16 bytes */
int esp_wifi_get_mac(char *mac_out);                      /* AT+CIFSR STAMAC; >=18 bytes
                                                             ("aa:bb:cc:dd:ee:ff") */

/*---------------------------------------------------------------------------
 * AT layer — single TCP connection (CIPMUX=0)
 *-------------------------------------------------------------------------*/
int esp_tcp_connect(const char *host, int port);          /* AT+CIPSTART */
int esp_tcp_send(const char *data, int len);              /* AT+CIPSEND + payload */
int esp_tcp_recv(char *buf, int max_len, int timeout_ms); /* +IPD demux; >=0 bytes
                                                             (returns early once the
                                                             peer closes the link) */
int esp_tcp_close(void);                                  /* AT+CIPCLOSE */

#ifdef __cplusplus
}
#endif

#endif /* ESP_AT_H */
