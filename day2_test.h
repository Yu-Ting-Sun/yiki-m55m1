/**************************************************************************//**
 * @file     day2_test.h
 * @brief    Day-2 Task 4: full-link integration tests
 *           (M55M1 -> ESP-12F -> Wi-Fi -> FastAPI -> back).
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef DAY2_TEST_H
#define DAY2_TEST_H

#ifdef __cplusplus
extern "C" {
#endif

/* Runs Tests 1-4 + summary on the debug console. Never returns.
 * Call AFTER PerfTimer_Init() — the esp_at layer times out via
 * GetSystemTick_ms(). */
void day2_test_run(void);

#ifdef __cplusplus
}
#endif

#endif /* DAY2_TEST_H */
