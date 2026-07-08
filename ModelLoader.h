/**************************************************************************//**
 * @file     ModelLoader.h
 * @brief    Thin presentation wrapper around ModelFileReader_LoadToAddress():
 *           selects the SD drive, times the load, and prints a one-line report.
 *
 * (Implements Task 3-2 "LoadModelToHyperRAM". Placed in its own module rather
 * than main.cpp so the loader is reusable and main.cpp stays focused on the
 * test sequence.)
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef __MODEL_LOADER_H__
#define __MODEL_LOADER_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief  Load a .tflite from SD into a HyperRAM slot, with timing + logging.
 *
 * @param  filename   FATFS path, e.g. FACE_MODEL_FILE.
 * @param  dst_addr   HyperRAM slot base, e.g. FACE_MODEL_ADDR.
 * @param  max_size   Slot capacity, e.g. MODEL_SLOT_SIZE.
 * @param  out_size   [out] bytes loaded (may be NULL).
 * @param  out_crc32  [out] CRC32 of loaded bytes (may be NULL).
 * @return 0 on success, negative E_MODEL_LOAD_RESULT on failure.
 */
int LoadModelToHyperRAM(const char *filename,
                        uint32_t    dst_addr,
                        uint32_t    max_size,
                        uint32_t   *out_size,
                        uint32_t   *out_crc32);

#ifdef __cplusplus
}
#endif

#endif /* __MODEL_LOADER_H__ */
