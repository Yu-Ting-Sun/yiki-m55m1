/**************************************************************************//**
 * @file     Audio.h
 * @brief    Play a story out of the board's phone jack (NAU8822 over I2S0).
 *
 * So 小憶 can read a trip's story to a viewer who no longer reads small text.
 * The board has no speaker: the handbook's board table lists a "Phone-jack
 * (耳機介面)" as the only audio output, so headphones or a powered speaker
 * have to be plugged in (and JK-EN, PD1, pulled low - Audio_Init does that).
 *
 * Audio arrives as 8 kHz G.711 mu-law (STORY.ULW next to the album's photos,
 * fetched by SdSync): 8 KB per second of speech, a quarter of the 16-bit WAV
 * the backend makes, which matters on an 11.5 KB/s Wi-Fi link.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef __AUDIO_H__
#define __AUDIO_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AUDIO_ERR_CODEC     (-1)    /* codec did not answer on I2C          */
#define AUDIO_ERR_DMA       (-2)    /* no PDMA channel for I2S0 TX          */
#define AUDIO_ERR_FILE      (-3)    /* no such recording on the card        */
#define AUDIO_ERR_STATE     (-4)    /* Audio_Init not run (or it failed)    */

/**
 * @brief  Bring up I2C3 + the NAU8822 codec and I2S0 at 16 kHz, enable the
 *         phone jack and claim a PDMA channel. Needs PerfTimer_Init (I2C
 *         waits are timed) and BoardInit's clocks.
 * @return 0 ready; AUDIO_ERR_CODEC / AUDIO_ERR_DMA (playback then no-ops).
 */
int Audio_Init(void);

/**
 * @brief  Play an 8 kHz mu-law file from the SD card, blocking until it ends.
 *         Each sample is sent twice so the codec can stay at 16 kHz, and to
 *         both channels.
 * @param  path    e.g. "0:\\pictures\\T0016\\STORY.ULW"
 * @param  cancel  polled every ~128 ms; return nonzero to stop early
 *                 (a button press). May be NULL.
 * @return 0 played to the end; 1 stopped by `cancel`; else AUDIO_ERR_*
 */
int Audio_PlayUlawFile(const char *path, int (*cancel)(void));

#ifdef __cplusplus
}
#endif

#endif /* __AUDIO_H__ */
