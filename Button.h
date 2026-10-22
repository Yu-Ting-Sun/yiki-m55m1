/**************************************************************************//**
 * @file     Button.h
 * @brief    Polled board push-buttons (active low).
 *
 * The photo frame's only physical inputs:
 *   BTN0  push-to-talk with 小憶
 *   BTN1  drop the recognition lock early (main.cpp) instead of waiting out
 *         the re-check window
 *
 * Polling (not EXINT) on purpose: the only caller is the slideshow hold
 * window, which already ticks every frame, and an ISR would have to hand the
 * event over anyway.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef __BUTTON_H__
#define __BUTTON_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BUTTON_BTN0     (1u << 0)
#define BUTTON_BTN1     (1u << 1)

/**
 * @brief  Configure both button pins as pulled-up inputs. Safe to call once,
 *         after BoardInit().
 * @return 0 always (a GPIO input cannot fail to configure).
 */
int Button_Init(void);

/**
 * @brief  Poll both buttons and report one press per physical push each.
 *         A press is reported on the first poll that sees the button down and
 *         the button re-arms only once a poll sees it up, so holding it does
 *         not repeat. Safe to poll slowly: the caller's loop turns every
 *         150-200 ms and a press still registers from a single sample.
 *
 *         Call ONCE per loop turn and dispatch on the result — a second call
 *         in the same turn finds the presses already consumed.
 * @return bitmask of BUTTON_BTN0 / BUTTON_BTN1 newly pressed on this call
 */
uint32_t Button_Poll(void);

/**
 * @brief  Raw level right now, no edge logic and no side effects — for
 *         hold-to-talk, which needs to know when BTN0 is let go.
 * @param  mask  BUTTON_BTN0 or BUTTON_BTN1
 * @return 1 if that button is held down, else 0
 */
int Button_IsDown(uint32_t mask);

#ifdef __cplusplus
}
#endif

#endif /* __BUTTON_H__ */
