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
 * @brief  Boot sequence: Wi-Fi up (day2_config.h), register this board's
 *         identity (ESP MAC) via POST /frames/register — the backend hands
 *         back a per-board frame_id + 6-digit pair code, which is shown on
 *         the story rail for the App to pair with. Then an initial sync
 *         runs only if the SD library is empty (first run) or the App has
 *         already requested one.
 * @return >0 albums updated (re-run Slideshow_LibScan), 0 nothing to do,
 *         <0 Wi-Fi/backend unreachable (card content keeps playing).
 */
int SdSync_Boot(void);

/**
 * @brief  Doorbell poll (call between photos, every ~10 s): one tiny
 *         GET /frames/<id>/pending; only when the App pressed 「立即同步」
 *         does the actual album download run.
 * @return >0 albums updated (re-run Slideshow_LibScan), 0 idle,
 *         <0 offline (boot never got Wi-Fi up).
 */
int SdSync_Poll(void);

/**
 * @brief  One full sync pass against /frames/<id>/sync. A torn/partial
 *         album (power loss, Wi-Fi drop) has no VERSION.TXT yet and
 *         self-heals on the next pass.
 * @param  verbose  paint progress into the story rail.
 * @return >0 albums updated, 0 nothing changed, <0 network error.
 */
int SdSync_Run(bool verbose);

/**
 * @brief  Report a thumbs-up (「按讚」) to the backend:
 *         POST /frames/<id>/like {"folder","photo","user"}.
 *         Fire-and-forget — the caller only logs a failure.
 * @param  folder  album on screen, e.g. "T0006" ("" allowed).
 * @param  photo   photo file on screen, e.g. "P0012.JPG" ("" = story slide).
 * @param  user    recognised face label ("" = unknown family member).
 * @return 0 delivered, <0 offline / HTTP error.
 */
int SdSync_PostLike(const char *folder, const char *photo, const char *user);

/**
 * @brief  This board's frame id, for other /frames/<id>/... requests (e.g.
 *         VoiceChat). Valid once SdSync_Boot has run.
 * @return frame id, or -1 while offline (no Wi-Fi link to talk over).
 */
int SdSync_FrameId(void);

/**
 * @brief  Download one backend URL straight to a file on the SD card, with
 *         the same retry as album files. Big payloads stream through the
 *         HyperRAM buffer, so this costs no SRAM (used for 小憶's spoken
 *         answer, which is far too big for a RAM buffer).
 * @param  urlPath  e.g. "/frames/4/voice/ab12cd34/audio"
 * @param  sdPath   e.g. "0:\\REPLY.ULW" (created/overwritten)
 * @return 0 saved, <0 HTTP/SD error.
 */
int SdSync_DownloadToFile(const char *urlPath, const char *sdPath);

#ifdef __cplusplus
}
#endif

#endif /* SD_SYNC_H */
