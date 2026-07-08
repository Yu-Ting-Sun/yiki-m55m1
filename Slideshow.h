/**************************************************************************//**
 * @file     Slideshow.h
 * @brief    SD-card photo slideshow on the LCD (24-bit BMP files).
 *
 * Photos are plain uncompressed 24-bpp BMPs (any size up to the panel size;
 * smaller images are centered). Use scripts/prepare_pictures.py to batch
 * convert JPG/PNG to the right BMPs. Call Display_Init() before Slideshow_Run.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef __SLIDESHOW_H__
#define __SLIDESHOW_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief  Reserve the rightmost @p px columns of the panel (the slideshow
 *         will letterbox photos into the remaining left region and never
 *         paint into the reserve). Used by the Day-3 story panel (StoryUI.h:
 *         STORYUI_RESERVED_PX). Call before Slideshow_Run; 0 = full screen.
 */
void Slideshow_ReserveRight(uint32_t px);

/**
 * @brief  Loop forever showing every *.bmp in @p dirPath on the LCD.
 * @param  dirPath  FATFS directory, e.g. "0:\\pictures".
 * @param  holdMs   How long each photo stays on screen (ms).
 * @return Only returns on failure:
 *         -1 directory missing/unopenable;
 *         -2 a full pass found no displayable BMP;
 *         -3 directory read error.
 */
int Slideshow_Run(const char *dirPath, uint32_t holdMs);

#ifdef __cplusplus
}
#endif

#endif /* __SLIDESHOW_H__ */
