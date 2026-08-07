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
 * @brief  Scan the photo library. Albums are the subfolders of @p rootPath;
 *         each may carry a label.json ({"users": ["user1", ...]}) naming the
 *         people that album relates to. Loose files directly in the root are
 *         an unlabeled pseudo-album that only plays when no filter is set.
 * @param  rootPath  FATFS directory, e.g. "0:\\pictures".
 * @return Number of albums found (>= 0), or -1 if the root is unopenable.
 */
int Slideshow_LibScan(const char *rootPath);

/**
 * @brief  Restrict playback to the albums whose label.json lists @p user
 *         (case-insensitive). NULL or "" clears the filter (play everything).
 *         A user matching no album degrades to playing everything. Takes
 *         effect at the next Slideshow_ShowNext() call; may be called at any
 *         time (e.g. from the face-recognition loop).
 * @return Number of albums in the resulting play set.
 */
int Slideshow_SetFilter(const char *user);

/**
 * @brief  Decode + display the next photo of the play set (albums cycle in
 *         scan order, looping forever). Blocks only for the decode itself —
 *         pacing (hold time) is the caller's job, so camera capture and
 *         inference can run between photos.
 * @return 0 photo shown; -1 library not scanned; -2 a full pass found no
 *         displayable photo; -3 directory read error.
 */
int Slideshow_ShowNext(void);

/**
 * @brief  Album folder + photo file currently on screen (for the gesture-like
 *         report). folder "" = root pseudo-album; photo "" = story-only slide.
 *         Either output may be NULL if not wanted.
 */
void Slideshow_GetCurrent(char *folder, int folderCap, char *photo, int photoCap);

/**
 * @brief  Convenience wrapper: Slideshow_LibScan + ShowNext/Delay forever.
 * @param  dirPath  FATFS directory, e.g. "0:\\pictures".
 * @param  holdMs   How long each photo stays on screen (ms).
 * @return Only returns on failure (codes as above).
 */
int Slideshow_Run(const char *dirPath, uint32_t holdMs);

#ifdef __cplusplus
}
#endif

#endif /* __SLIDESHOW_H__ */
