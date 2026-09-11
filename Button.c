/**************************************************************************//**
 * @file     Button.c
 * @brief    Polled, debounced board push-button (active low). See Button.h.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include <stdint.h>

#include "NuMicro.h"
#include "PerfTimer.h"      /* GetSystemTick_ms */
#include "Button.h"

/* NuMaker-M55M1 user buttons. Both pull their pin LOW when pressed and float
 * otherwise, so the internal pull-ups are enabled below.
 *
 * CONFIRMED ON HARDWARE 2026-09-11 with a temporary two-pin probe (both
 * candidates pulled up, pressing each in turn): PI.11 and PH.1 are BOTH
 * wired to a user button. The original PI.11 choice was right; an early
 * test that looked like "the button does nothing" was just the other
 * button being pressed. Either one unlocks, so whoever is standing at the
 * frame cannot pick the wrong one. */
#define BUTTON_A_PORT       PI
#define BUTTON_A_PIN        (11)
#define BUTTON_B_PORT       PH
#define BUTTON_B_PIN        (1)

#define BUTTON_A_DOWN()     ((BUTTON_A_PORT->PIN & (1u << BUTTON_A_PIN)) == 0u)
#define BUTTON_B_DOWN()     ((BUTTON_B_PORT->PIN & (1u << BUTTON_B_PIN)) == 0u)
#define BUTTON_IS_DOWN()    (BUTTON_A_DOWN() || BUTTON_B_DOWN())

/* The only caller polls this from the slideshow loop, and one turn of that
 * loop is a camera capture plus a hand-landmark inference — 150-200 ms, not
 * the "few tens of ms" a level debounce assumes. At that rate a quick tap is
 * sampled at most ONCE, which the original "level stable across two polls,
 * then report on release" debounce would have dropped entirely.
 *
 * Sampling three orders of magnitude slower than contact bounce makes level
 * debouncing pointless anyway, so this reports the FIRST sample that sees a
 * button down and re-arms only once a sample sees both up again — one event
 * per physical press, acted on immediately. BUTTON_RETRIGGER_MS is belt and
 * braces against a bounce landing on two consecutive samples. */
#define BUTTON_RETRIGGER_MS (200)

int Button_Init(void)
{
    GPIO_SetMode(BUTTON_A_PORT, (1u << BUTTON_A_PIN), GPIO_MODE_INPUT);
    GPIO_SetPullCtl(BUTTON_A_PORT, (1u << BUTTON_A_PIN), GPIO_PUSEL_PULL_UP);
    GPIO_SetMode(BUTTON_B_PORT, (1u << BUTTON_B_PIN), GPIO_MODE_INPUT);
    GPIO_SetPullCtl(BUTTON_B_PORT, (1u << BUTTON_B_PIN), GPIO_PUSEL_PULL_UP);
    return 0;
}

int Button_Pressed(void)
{
    static int      armed  = 1;   /* 1 = a new press may be reported */
    static uint32_t lastMs = 0;

    if (!BUTTON_IS_DOWN())
    {
        armed = 1;
        return 0;
    }

    if (!armed)
        return 0;                 /* still held down from the last report */

    uint32_t now = GetSystemTick_ms();

    if (lastMs != 0 && (now - lastMs) < BUTTON_RETRIGGER_MS)
        return 0;

    armed  = 0;
    lastMs = now;
    return 1;
}
