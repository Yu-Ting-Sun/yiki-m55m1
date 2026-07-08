/**************************************************************************//**
 * @file     day3_demo.h
 * @brief    Day-3 on-board demo: show the photo-frame layout with a live
 *           LLM story on the LCD (voice dropped by design decision).
 *
 * Flow: draw layout chrome -> Wi-Fi up (day2_config.h creds) ->
 *       POST /generate {face_id} -> GET /textimg/{audio_id} (TIM4, ~37 KB)
 *       -> StoryUI_ShowTextImage. The photo slideshow is the CALLER's job
 *       (call Slideshow_ReserveRight(STORYUI_RESERVED_PX) + Slideshow_Run
 *       afterwards — the story rail stays on screen).
 *
 * Requires: Display_Init() done, PerfTimer_Init() done (esp_at timing
 * contract — see esp_at.h).
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef DAY3_DEMO_H
#define DAY3_DEMO_H

#ifdef __cplusplus
extern "C" {
#endif

/** @return 0 = story is on screen; negative = failed (an error message is
 *  left on the story rail; the caller should still run the slideshow). */
int day3_demo_run(void);

#ifdef __cplusplus
}
#endif

#endif /* DAY3_DEMO_H */
