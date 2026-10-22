/**************************************************************************//**
 * @file     Button.c
 * @brief    Polled board push-buttons (active low). See Button.h.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include <stdint.h>
#include <stdio.h>

#include "NuMicro.h"
#include "PerfTimer.h"      /* GetSystemTick_ms */
#include "Button.h"

/* NuMaker-M55M1 user buttons. Both pull their pin LOW when pressed and float
 * otherwise, so the internal pull-ups are enabled below.
 *
 * CONFIRMED ON HARDWARE 2026-09-11 with a temporary two-pin probe (both
 * candidates pulled up, pressing each in turn): PI.11 and PH.1 are BOTH
 * wired to a user button.
 *
 * Silkscreen BTN0 = PI.11 and BTN1 = PH.1, as the PR #1 author noted —
 * VERIFIED 2026-09-19: pressing the button printed BTN0 on the board logged
 * "BTN0 (PI.11) pressed" and started a recording. */
#define BTN0_PORT           PI
#define BTN0_PIN            (11)
#define BTN0_NAME           "BTN0 (PI.11)"

#define BTN1_PORT           PH
#define BTN1_PIN            (1)
#define BTN1_NAME           "BTN1 (PH.1)"

/* The only caller polls from the slideshow loop, and one turn of that loop
 * is a camera capture plus a hand-landmark inference — 150-200 ms, not the
 * "few tens of ms" a level debounce assumes. At that rate a quick tap is
 * sampled at most ONCE, which a "level stable across two polls, then report
 * on release" debounce would drop entirely.
 *
 * Sampling three orders of magnitude slower than contact bounce makes level
 * debouncing pointless anyway, so each button reports the FIRST sample that
 * sees it down and re-arms only once a sample sees it up again — one event
 * per physical press, acted on immediately. BUTTON_RETRIGGER_MS is belt and
 * braces against a bounce landing on two consecutive samples. */
#define BUTTON_RETRIGGER_MS (200)

typedef struct
{
    GPIO_T     *port;
    uint32_t    pin;
    uint32_t    mask;       /* BUTTON_BTN0 / BUTTON_BTN1 */
    const char *name;
    int         armed;      /* 1 = a new press may be reported */
    uint32_t    lastMs;
} ButtonState;

static ButtonState s_btn[2];

static void setup_pin(GPIO_T *port, uint32_t pin)
{
    GPIO_SetMode(port, (1u << pin), GPIO_MODE_INPUT);
    GPIO_SetPullCtl(port, (1u << pin), GPIO_PUSEL_PULL_UP);
}

int Button_Init(void)
{
    s_btn[0].port   = BTN0_PORT;
    s_btn[0].pin    = BTN0_PIN;
    s_btn[0].mask   = BUTTON_BTN0;
    s_btn[0].name   = BTN0_NAME;

    s_btn[1].port   = BTN1_PORT;
    s_btn[1].pin    = BTN1_PIN;
    s_btn[1].mask   = BUTTON_BTN1;
    s_btn[1].name   = BTN1_NAME;

    for (int i = 0; i < 2; i++)
    {
        s_btn[i].armed  = 1;
        s_btn[i].lastMs = 0;
        setup_pin(s_btn[i].port, s_btn[i].pin);
    }
    return 0;
}

static int poll_one(ButtonState *b)
{
    if ((b->port->PIN & (1u << b->pin)) != 0u)      /* up */
    {
        b->armed = 1;
        return 0;
    }

    if (!b->armed)
        return 0;                 /* still held down from the last report */

    uint32_t now = GetSystemTick_ms();

    if (b->lastMs != 0 && (now - b->lastMs) < BUTTON_RETRIGGER_MS)
        return 0;

    b->armed  = 0;
    b->lastMs = now;
    return 1;
}

int Button_IsDown(uint32_t mask)
{
    for (int i = 0; i < 2; i++)
        if (s_btn[i].mask == mask)
            return (s_btn[i].port->PIN & (1u << s_btn[i].pin)) == 0u;
    return 0;
}

uint32_t Button_Poll(void)
{
    uint32_t hit = 0;

    for (int i = 0; i < 2; i++)
    {
        if (poll_one(&s_btn[i]))
        {
            hit |= s_btn[i].mask;
            printf("[BTN] %s pressed\n", s_btn[i].name);
        }
    }
    return hit;
}
