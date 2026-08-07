/**************************************************************************//**
 * @file     GestureLike.hpp
 * @brief    Thumbs-up (「按讚」) detection on camera frames.
 *
 * Runs the MediaPipe hand-landmark model (21 keypoints, from the BSP
 * HandPoseRecognition sample) on each captured RGB565 frame, then applies a
 * geometric rule: thumb extended upward + the four other fingers curled.
 * No extra classifier model is needed.
 *
 * Model + tensor arena both live in HyperRAM (see the DAY-3 demo plan in
 * MemoryLayout.h) — inference is ~5x slower than SRAM01 (~150 ms), which is
 * fine: it runs in the slideshow hold window and a like only needs a few
 * frames per second. Init failure is loud but non-fatal (photo frame keeps
 * running without the like feature).
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef __GESTURE_LIKE_HPP__
#define __GESTURE_LIKE_HPP__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief  Load 0:\hand_landmark.tflite into its HyperRAM slot and build the
 *         resident interpreter (arena also in HyperRAM).
 * @return 0 ready; negative on SD-load / interpreter failure (non-fatal).
 */
int GestureLike_Init(void);

/**
 * @brief  Arena base/size for the combined MPU setup in main() (same pattern
 *         as FaceDetect_GetArena / FaceRecog_GetArena).
 */
void GestureLike_GetArena(void **base, uint32_t *size);

/**
 * @brief  Run hand-landmark inference on one RGB565 frame and judge the pose.
 * @return 1 thumbs-up seen this frame, 0 not seen, negative on error.
 *         Callers debounce (consecutive hits) and rate-limit themselves.
 */
int GestureLike_Run(const uint16_t *frameRGB565, int w, int h);

#ifdef __cplusplus
}
#endif

#endif /* __GESTURE_LIKE_HPP__ */
