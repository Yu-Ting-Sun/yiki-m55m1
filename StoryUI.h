/**************************************************************************//**
 * @file     StoryUI.h
 * @brief    Day-3 screen layout: photo slideshow region + story strip.
 *
 * 800x480 LT7381 panel split (landscape-photo first, per user):
 *   x   0..639  photo region 640x480 — landscape 4:3 phone photos fill it
 *               edge to edge; portrait ones center on the black background
 *   x 640..643  amber divider
 *   x 644..799  story rail 156x480: backend-rendered Chinese text image
 *               (TIM4, place/date header + story), or ASCII via 8x16 font
 *
 * TIM4 payload (produced by backend textimg.py):
 *   [0..3] "TIM4"  [4..5] width LE  [6..7] height LE
 *   [8..]  4-bpp grayscale, 2 px/byte (high nibble = left px), rows top-down;
 *          0 = background .. 15 = full text color (mapped through a 16-entry
 *          RGB565 LUT so the anti-aliasing survives).
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef __STORY_UI_H__
#define __STORY_UI_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define STORYUI_PHOTO_W     (640u)  /* slideshow area width               */
#define STORYUI_DIVIDER_X   (640u)
#define STORYUI_DIVIDER_W   (4u)
#define STORYUI_PANEL_X     (644u)
#define STORYUI_PANEL_W     (156u)
#define STORYUI_PANEL_H     (480u)

/* Columns the photo region must leave alone (pass to Slideshow_ReserveRight). */
#define STORYUI_RESERVED_PX (STORYUI_DIVIDER_W + STORYUI_PANEL_W)

/* Max size of a TIM4 download buffer: header + 156x480 @ 4 bpp. */
#define STORYUI_TIM4_MAX    (8u + (STORYUI_PANEL_W / 2u) * STORYUI_PANEL_H)

/** Paint the divider and the empty story panel. Call once after Display_Init
 *  (and again if a full-screen clear wiped the layout). */
void StoryUI_DrawChrome(void);

/** Blit a TIM4 story image (from GET /textimg/{id}) into the panel.
 *  @return 0 ok; -1 bad header; -2 size mismatch; -3 too large. */
int StoryUI_ShowTextImage(const uint8_t *tim, uint32_t len);

/** Load a TIM4 file from SD (e.g. "0:\\pictures\\T0007\\STORY.TIM" written
 *  by the backend sync / scripts/export_sd.py) and blit it into the panel.
 *  @return 0 ok; -4 open failed; -5 read failed; else ShowTextImage codes. */
int StoryUI_ShowTextImageFile(const char *path);

/** One line of ASCII status ("Generating story...") centered in the panel. */
void StoryUI_ShowStatus(const char *asciiText);

/** Fallback option C: word-wrap an ASCII story into the panel. */
void StoryUI_ShowAsciiStory(const char *asciiText);

#ifdef __cplusplus
}
#endif

#endif /* __STORY_UI_H__ */
