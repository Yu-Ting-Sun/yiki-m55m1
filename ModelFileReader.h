/**************************************************************************//**
 * @file     ModelFileReader.h
 * @brief    Read a .tflite model file from SD card (FATFS) into a target
 *           address, with 4 KB block reads and a CRC32 integrity check.
 *
 * Rewritten from the original Visually-Impaired-Assistance-System version:
 *   - block size raised 512 B -> 4096 B (fewer f_read calls, faster load)
 *   - CRC32 (IEEE 802.3) computed during load to catch SD bit errors
 *   - one-shot LoadToAddress() replaces the piecemeal Initialize/Read/Finish API
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef __MODEL_FILE_READER_H__
#define __MODEL_FILE_READER_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Read granularity. 4 KB matches a typical SD/FATFS cluster sector burst and
 * cuts the number of f_read() calls for a 2.3 MB model from ~4700 to ~590. */
#define MODEL_READ_BLOCK    (4096U)

/* Return codes for the loader. 0 = success, negative = failure. */
typedef enum
{
    MODEL_LOAD_OK             =  0,
    MODEL_LOAD_ERR_OPEN       = -1,   /* f_open failed (file missing / no SD)   */
    MODEL_LOAD_ERR_EMPTY      = -2,   /* file size is 0                         */
    MODEL_LOAD_ERR_TOO_LARGE  = -3,   /* file size > max_size (would overrun)   */
    MODEL_LOAD_ERR_READ       = -4,   /* f_read returned an error mid-file      */
    MODEL_LOAD_ERR_SHORT_READ = -5    /* EOF before expected size was read      */
} E_MODEL_LOAD_RESULT;

/**
 * @brief  Incrementally update an IEEE-802.3 CRC32 over a buffer.
 * @param  crc   Running CRC value. Seed the FIRST call with 0xFFFFFFFF.
 * @param  data  Pointer to the bytes to fold in.
 * @param  len   Number of bytes.
 * @return Updated running CRC. XOR the final result with 0xFFFFFFFF.
 * @note   Table is lazily built on first use.
 */
uint32_t ModelFileReader_CRC32Update(uint32_t crc, const void *data, uint32_t len);

/**
 * @brief  Open @p filename, validate size, stream it in MODEL_READ_BLOCK
 *         chunks into @p dst_addr, and compute its CRC32.
 *
 * @param  filename  FATFS path, e.g. "0:\\face_model.tflite".
 * @param  dst_addr  Destination (HyperRAM slot base).
 * @param  max_size  Slot capacity in bytes; load aborts if the file is bigger.
 * @param  out_size  [out] bytes actually loaded (may be NULL).
 * @param  out_crc32 [out] finalized CRC32 of the loaded bytes (may be NULL).
 * @return MODEL_LOAD_OK or a negative E_MODEL_LOAD_RESULT.
 *
 * @note  max_size is taken here (not in the caller) on purpose: the slot bound
 *        must be enforced *before* any byte is written, so a bad/oversized file
 *        can never overrun the neighbouring model's slot.
 */
int ModelFileReader_LoadToAddress(const char *filename,
                                  void       *dst_addr,
                                  uint32_t    max_size,
                                  uint32_t   *out_size,
                                  uint32_t   *out_crc32);

/** @brief  Human-readable string for an E_MODEL_LOAD_RESULT code. */
const char *ModelFileReader_StrError(int rc);

#ifdef __cplusplus
}
#endif

#endif /* __MODEL_FILE_READER_H__ */
