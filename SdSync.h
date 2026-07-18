/**************************************************************************//**
 * @file     SdSync.h
 * @brief    Frame sync: mirror the backend's albums + faces onto the SD card.
 *
 * Pull model — the board polls GET /frames/<id>/sync and downloads every
 * new/changed album (photos, LABEL.JSON, STORY.TIM, VERSION.TXT) plus the
 * faces/enroll_*.raw enrollment files. No card swapping.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef SD_SYNC_H
#define SD_SYNC_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief  One sync pass. Brings Wi-Fi up on demand (day2_config.h) the
 *         first time a request fails; a torn/partial album (power loss,
 *         Wi-Fi drop) has no VERSION.TXT yet and self-heals next pass.
 * @param  verbose  paint progress into the story rail (use at boot).
 * @return >0 number of albums created/updated (caller should re-run
 *         Slideshow_LibScan), 0 nothing changed, <0 network error.
 */
int SdSync_Run(bool verbose);

#ifdef __cplusplus
}
#endif

#endif /* SD_SYNC_H */
