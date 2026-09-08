/**************************************************************************//**
 * @file     Button.h
 * @brief    Polled, debounced board push-button (active low).
 *
 * The photo frame has no other user input: the recognition lock (main.cpp)
 * uses this to let the person in front of the frame drop the lock manually
 * instead of waiting for the re-check window.
 *
 * Polling (not EXINT) on purpose: the only caller is the slideshow hold
 * window, which already ticks every frame, and an ISR would have to hand the
 * event over anyway.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef __BUTTON_H__
#define __BUTTON_H__

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief  Configure the button pin as a pulled-up input. Safe to call once,
 *         after BoardInit().
 * @return 0 always (a GPIO input cannot fail to configure).
 */
int Button_Init(void);

/**
 * @brief  Poll the button and report one press per physical push.
 *         Debounced by BUTTON_DEBOUNCE_MS of a stable level; the press is
 *         reported on the release edge, so holding the button down does not
 *         repeat. Call regularly (every few tens of ms) or presses are missed.
 * @return 1 if a press completed since the last call, 0 otherwise.
 */
int Button_Pressed(void);

#ifdef __cplusplus
}
#endif

#endif /* __BUTTON_H__ */
