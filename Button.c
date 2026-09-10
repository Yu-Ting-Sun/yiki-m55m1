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

/* NuMaker-M55M1 user button BTN0. Both buttons pull the pin LOW when pressed
 * and float otherwise, so the internal pull-up is enabled below.
 * VERIFY AGAINST THE BOARD SCHEMATIC before trusting a "button does nothing"
 * result: BTN1 on this board is PH.1 — swap the two defines to use it. */
#define BUTTON_PORT         PI
#define BUTTON_PIN          (11)

#define BUTTON_DEBOUNCE_MS  (30)

#define BUTTON_IS_DOWN()    ((BUTTON_PORT->PIN & (1u << BUTTON_PIN)) == 0u)

int Button_Init(void)
{
    GPIO_SetMode(BUTTON_PORT, (1u << BUTTON_PIN), GPIO_MODE_INPUT);
    GPIO_SetPullCtl(BUTTON_PORT, (1u << BUTTON_PIN), GPIO_PUSEL_PULL_UP);
    return 0;
}

int Button_Pressed(void)
{
    static int      stable    = 0;   /* last debounced level: 1 = down */
    static int      raw       = 0;   /* last sampled level             */
    static uint32_t changedMs = 0;

    int now = BUTTON_IS_DOWN() ? 1 : 0;

    if (now != raw)
    {
        raw       = now;
        changedMs = GetSystemTick_ms();
        return 0;
    }

    if (now == stable || (GetSystemTick_ms() - changedMs) < BUTTON_DEBOUNCE_MS)
        return 0;

    stable = now;

    /* Report on release, so a held button fires exactly once. */
    return (stable == 0) ? 1 : 0;
}
