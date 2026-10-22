/**************************************************************************//**
 * @file     VoiceChat.h
 * @brief    Ask 小憶: one recorded question in, her answer on the story rail.
 *
 * The recording (VoiceRec) goes to the backend's POST /frames/<id>/voice
 * together with what the frame is showing; the backend transcribes and
 * answers in one LLM call and sends the answer back already drawn as a
 * story-rail image (TIM4), which lands on the rail as is.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef __VOICECHAT_H__
#define __VOICECHAT_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VOICECHAT_ERR_OFFLINE   (-1)    /* no Wi-Fi link (SdSync_FrameId)    */
#define VOICECHAT_ERR_HTTP      (-2)    /* upload/answer failed or not 200    */
#define VOICECHAT_ERR_IMAGE     (-3)    /* answer arrived but is no TIM4      */

/**
 * @brief  Upload one question and put 小憶's answer on the rail. Blocks for
 *         the upload, the LLM and the answer download (~10 s at 115200).
 * @param  ulaw    8-bit mu-law mono, as from VoiceRec_Capture
 * @param  n       bytes in ulaw
 * @param  rateHz  the rate it was really sampled at (VoiceRec_RateHz)
 * @param  folder  album on screen, e.g. "T0019" ("" allowed)
 * @param  photo   photo on screen, e.g. "P0025.JPG" ("" = story slide)
 * @param  user    recognised viewer ("" = nobody recognised)
 * @param  action  receives what 小憶 asked the frame to do on top of her
 *                 answer (X-Frame-Action), "" for nothing; one of the
 *                 VOICECHAT_ACT_* tokens. May be NULL.
 * @param  actCap  size of action
 * @return 0 answer on the rail, else VOICECHAT_ERR_*
 */
int VoiceChat_Ask(const uint8_t *ulaw, uint32_t n, uint32_t rateHz,
                  const char *folder, const char *photo, const char *user,
                  char *action, int actCap);

/* Frame actions 小憶's tools can ask for (backend VOICE_TOOLS). */
#define VOICECHAT_ACT_SYNC      "sync"    /* run a sync pass now (new albums) */
#define VOICECHAT_ACT_ALBUM     "album="  /* + folder ("T0016"): play it now  */
#define VOICECHAT_ACT_PHOTO     "photo="  /* + next | prev | pause | resume   */
#define VOICECHAT_ACT_STORY     "story="  /* + read: read this album's story  */

/**
 * @brief  Put the last answer back on the rail — for when a photo change
 *         the answer itself asked for drew the album story over it.
 * @return 0 ok; -1 no answer yet; else StoryUI_ShowTextImage codes.
 */
int VoiceChat_RedrawAnswer(void);

/**
 * @brief  Play the last answer out loud (the backend synthesizes it in 小憶's
 *         voice as it sends the answer; the board downloads it to the card
 *         and plays it). Needs Audio_Init and a plugged-in phone jack.
 * @param  cancel  polled during playback; nonzero stops it (a button press)
 * @return 0 spoken; 1 stopped early; -1 no answer to speak; -2 download
 *         failed (no speech this time); else Audio_PlayUlawFile codes.
 */
int VoiceChat_SpeakAnswer(int (*cancel)(void));

#ifdef __cplusplus
}
#endif

#endif /* __VOICECHAT_H__ */
