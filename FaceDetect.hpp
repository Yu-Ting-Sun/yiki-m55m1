/**************************************************************************//**
 * @file     FaceDetect.hpp
 * @brief    Phase-3 live face detection glue for the photo-frame demo.
 *
 * Wraps the BSP FaceDetection pipeline (compiled-in yolo-fastest_192_face
 * Vela model + anchor-based DetectorPostProcessing) so the main slideshow
 * loop can run one detection on a captured camera frame during the photo
 * hold window and draw the resulting face boxes onto that same RGB565 frame
 * before it is blitted to the LCD.
 *
 * Self-contained on purpose: the RGB565 -> 192x192 grayscale-int8 resize and
 * the box drawing are done in plain C++ here (no omv/fb_alloc), so the only
 * external dependencies are the model + TFLM + DetectorPostProcessing that
 * the FaceDetection sample already proves on this board.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef __FACE_DETECT_HPP__
#define __FACE_DETECT_HPP__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief  A detected face box in frame (source-image) pixel coordinates. */
typedef struct
{
    int x, y, w, h;
} FaceBox;

/**
 * @brief  Build the face-detection interpreter over its own tensor arena and
 *         set the arena MPU policy (cacheable WTRA, same as the BSP sample).
 *         Safe to call once, after BoardInit().
 * @return 0 on success; negative if the model fails to load / the arena is
 *         too small (the demo should then keep running plain preview).
 */
int FaceDetect_Init(void);

/**
 * @brief  Run one detection on an RGB565 frame and draw the face boxes onto
 *         it in place. The frame is both the model input source (resized to
 *         the model's grayscale input) and the draw target.
 * @param[in,out] frameRGB565  Frame buffer, w*h RGB565 pixels, modified in place.
 * @param[in]     w,h          Frame dimensions in pixels.
 * @return number of faces drawn (>=0), or negative on inference error.
 */
int FaceDetect_Run(uint16_t *frameRGB565, int w, int h);

/**
 * @brief  Get the largest face box from the most recent FaceDetect_Run.
 * @param[out] out  Filled with the largest box (by area) if any.
 * @return 1 if a box was returned, 0 if the last run found no face.
 */
int FaceDetect_GetTopBox(FaceBox *out);

/**
 * @brief  Expose the detection tensor arena so the caller can set the arena
 *         MPU policy for all app arenas in one InitPreDefMPURegion() call.
 */
void FaceDetect_GetArena(void **base, uint32_t *size);

#ifdef __cplusplus
}
#endif

#endif /* __FACE_DETECT_HPP__ */
