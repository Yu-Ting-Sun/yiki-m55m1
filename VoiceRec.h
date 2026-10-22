/**************************************************************************//**
 * @file     VoiceRec.h
 * @brief    Capture from the on-board PDM microphone (DMIC0).
 *
 * The recording half of push-to-talk with 小憶: the frame records the
 * question, VoiceChat uploads it, and the backend hands it to the LLM
 * (speech understanding and the reply in one call).
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef __VOICEREC_H__
#define __VOICEREC_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief  Bring up DMIC0 on PB4/PB5 at 16 kHz and print the achieved rate.
 *         Needs PerfTimer_Init() first (captures time on GetSystemTick_ms).
 * @return 0 always; a dead mic shows up in the capture report.
 */
int VoiceRec_Init(void);

/** Longest utterance VoiceRec_Capture keeps: 8 s of 8 kHz mu-law, 64 KB. */
#define VOICEREC_MAX_MS     (8000u)

/**
 * @brief  Asked about every 10 ms of audio whether to keep recording.
 * @param  elapsedMs  audio recorded so far
 * @return nonzero to keep going, 0 to stop now
 */
typedef int (*VoiceRec_KeepFn)(uint32_t elapsedMs);

/**
 * @brief  Record speech as 8 kHz G.711 mu-law - the upload format: 8 KB/s, so
 *         a 4 s question is 32 KB. Blocks for the whole recording.
 *
 *         The mic is sampled at 16 kHz; each pair of samples is averaged into
 *         one 8 kHz sample (a 2-tap low-pass and the decimation in one step)
 *         and encoded. DC is already gone: the DMIC high-pass filter is on.
 * @param  maxMs      upper bound, clamped to VOICEREC_MAX_MS
 * @param  keepGoing  hold-to-talk hook (e.g. "button still down?"); NULL
 *                    records the full maxMs
 * @param  nBytes     receives the bytes actually captured
 * @return pointer to the internal buffer, valid until the next capture
 */
const uint8_t *VoiceRec_Capture(uint32_t maxMs, VoiceRec_KeepFn keepGoing,
                                uint32_t *nBytes);

/**
 * @brief  The rate captures are really sampled at, for labelling the upload:
 *         the DMIC clock only divides down to about 16.5 kHz here (see
 *         VoiceRec.c), so this is ~8263 Hz, not a nominal 8000.
 */
uint32_t VoiceRec_RateHz(void);

#ifdef __cplusplus
}
#endif

#endif /* __VOICEREC_H__ */
