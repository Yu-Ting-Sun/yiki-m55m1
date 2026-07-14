/**************************************************************************//**
 * @file     Camera.h
 * @brief    HM1055 (CCAP) capture + LCD preview for the photo frame.
 *
 * Phase-2 scope: bring the camera up and prove frames reach the LCD. The
 * capture buffer is RGB565 at CAM_W x CAM_H (CCAP scales in hardware), the
 * same frame later feeds face detection (Phase 3).
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef __CAMERA_H__
#define __CAMERA_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CAM_W   (240)
#define CAM_H   (240)

/**
 * @brief  Enable the CCAP clock, configure the sensor pins and init the
 *         HM1055 (QVGA YUV422 in, RGB565 CAM_WxCAM_H out, aspect kept).
 * @return 0 on success; negative if the sensor does not answer (not
 *         mounted / wiring) — callers should treat that as "no camera"
 *         and keep the plain slideshow running.
 */
int Camera_Init(void);

/**
 * @brief  Capture one frame into the internal buffer (blocking, ~1 frame
 *         time) and blit it to the LCD at (x, y).
 * @return 0 shown; negative on capture timeout.
 */
int Camera_PreviewTick(uint32_t x, uint32_t y);

/**
 * @brief  Latest captured frame (RGB565, CAM_W x CAM_H) for inference.
 */
const uint16_t *Camera_GetFrame(void);

#ifdef __cplusplus
}
#endif

#endif /* __CAMERA_H__ */
