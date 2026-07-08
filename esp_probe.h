/**************************************************************************//**
 * @file     esp_probe.h
 * @brief    Day-2 Task 0: on-board ESP-12F firmware probe (AT / non-AT / dead).
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef ESP_PROBE_H
#define ESP_PROBE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Runs the full probe sequence and prints a verdict on the debug console.
 * Never returns (ends in while(1)). Call right after BoardInit(), BEFORE
 * PerfTimer_Init() — the probe reprograms SysTick via CLK_SysTickDelay(). */
void esp_probe_run(void);

#ifdef __cplusplus
}
#endif

#endif /* ESP_PROBE_H */
