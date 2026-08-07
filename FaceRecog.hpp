/**************************************************************************//**
 * @file     FaceRecog.hpp
 * @brief    Phase-4 face recognition (embedding + cosine match) glue.
 *
 * Runs the BSP FaceMobileNet embedding model on a detected face crop and
 * matches the resulting embedding against reference embeddings enrolled on
 * the SD card (0:\faces\embeddings.txt, one "label:e0,e1,..." per line).
 *
 * On-device enrollment writes those references with the SAME model, so the
 * enrolled and queried embeddings live in exactly the same space (no
 * offline/on-device INT8 mismatch). The crop/resize/quantise is done in plain
 * C++ here (no omv), identical for enroll and query so any preprocessing
 * quirk cancels out in the cosine comparison.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef __FACE_RECOG_HPP__
#define __FACE_RECOG_HPP__

#include <stdint.h>
#include "FaceDetect.hpp"   /* FaceBox */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief  SD-load the embedding model into HyperRAM, build the interpreter
 *         over its own arena, and load the SD reference embeddings. Does NOT
 *         set the arena MPU — the caller sets all app MPU regions in one
 *         InitPreDefMPURegion() call (see FaceRecog_GetArena).
 * @return 0 on success; negative on model-load / init / SD failure.
 */
int FaceRecog_Init(void);

/** @brief  Expose the arena for the caller's combined MPU setup. */
void FaceRecog_GetArena(void **base, uint32_t *size);

/**
 * @brief  Recognise the face in `box` (frame pixel coords). Logs the result to
 *         UART and, if recognised, overpaints the box green on the frame.
 * @return 1 recognised, 0 unknown (below threshold / no references), <0 error.
 */
int FaceRecog_Run(uint16_t *frame, int fw, int fh, const FaceBox *box);

/**
 * @brief  Label recognised by the most recent FaceRecog_Run() that returned 1.
 *         Only valid right after such a call (Phase-5 filter debounce input).
 */
const char *FaceRecog_GetLabel(void);

/**
 * @brief  Compute the embedding for the face in `box` and append
 *         "label:e0,e1,...\n" to 0:\faces\embeddings.txt.
 * @return 0 on success; negative on inference / SD-write failure.
 */
int FaceRecog_Enroll(uint16_t *frame, int fw, int fh, const FaceBox *box,
                     const char *label);

/**
 * @brief  Remove every reference for `label`: in-RAM entries (recognition
 *         stops matching immediately) AND the label's lines in
 *         0:\faces\embeddings.txt (stream-filtered rewrite). Used by the
 *         sync prune when a face is deleted in the App. Safe to call when
 *         recognition never initialised.
 * @return number of references removed (0 = label unknown), <0 on SD error.
 */
int FaceRecog_ForgetLabel(const char *label);

#ifdef __cplusplus
}
#endif

#endif /* __FACE_RECOG_HPP__ */
