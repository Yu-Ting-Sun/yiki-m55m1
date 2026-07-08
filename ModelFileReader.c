/**************************************************************************//**
 * @file     ModelFileReader.c
 * @brief    SD-card (FATFS) .tflite loader with 4 KB reads + CRC32 check.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "ModelFileReader.h"

#include <stdio.h>
#include <string.h>
#include "NuMicro.h"    /* SCB cache-maintenance ops (core_cm55.h) */
#include "ff.h"     /* FatFs */

/* FIL is large (~hundreds of bytes); keep it static to spare the stack,
 * matching the original implementation's approach. */
static FIL s_modelFile;

/*----------------------------------------------------------------------------
 * CRC32 (IEEE 802.3 / zlib): reflected, poly 0xEDB88420, init/xorout 0xFFFFFFFF.
 * Table is built once on first use to avoid a 1 KB const blob in the image.
 *--------------------------------------------------------------------------*/
static uint32_t s_crcTable[256];
static int      s_crcTableReady = 0;

static void crc32_build_table(void)
{
    for (uint32_t i = 0; i < 256; i++)
    {
        uint32_t c = i;
        for (int k = 0; k < 8; k++)
            c = (c & 1U) ? (0xEDB88320U ^ (c >> 1)) : (c >> 1);
        /* 0xEDB88320 = reflected IEEE-802.3 poly. A previous typo (0xEDB88420)
         * made every on-board CRC differ from binascii.crc32 — verified: the
         * "wrong" board CRCs matched host CRCs computed with the typo'd poly,
         * proving the SD->HyperRAM data was byte-perfect all along. */
        s_crcTable[i] = c;
    }
    s_crcTableReady = 1;
}

uint32_t ModelFileReader_CRC32Update(uint32_t crc, const void *data, uint32_t len)
{
    const uint8_t *p = (const uint8_t *)data;

    if (!s_crcTableReady)
        crc32_build_table();

    while (len--)
        crc = s_crcTable[(crc ^ *p++) & 0xFFU] ^ (crc >> 8);

    return crc;
}

/*----------------------------------------------------------------------------
 * One-shot loader
 *--------------------------------------------------------------------------*/
int ModelFileReader_LoadToAddress(const char *filename,
                                  void       *dst_addr,
                                  uint32_t    max_size,
                                  uint32_t   *out_size,
                                  uint32_t   *out_crc32)
{
    FRESULT  res;
    uint32_t fsize;
    uint32_t off = 0;
    uint32_t crc = 0xFFFFFFFFU;
    uint8_t *dst = (uint8_t *)dst_addr;

    res = f_open(&s_modelFile, (const TCHAR *)filename, FA_OPEN_EXISTING | FA_READ);
    if (res != FR_OK)
        return MODEL_LOAD_ERR_OPEN;

    fsize = (uint32_t)f_size(&s_modelFile);
    if (fsize == 0)
    {
        f_close(&s_modelFile);
        return MODEL_LOAD_ERR_EMPTY;
    }
    if (fsize > max_size)
    {
        f_close(&s_modelFile);
        return MODEL_LOAD_ERR_TOO_LARGE;
    }

    while (off < fsize)
    {
        UINT     br   = 0;
        uint32_t want = fsize - off;
        if (want > MODEL_READ_BLOCK)
            want = MODEL_READ_BLOCK;

        res = f_read(&s_modelFile, dst + off, want, &br);
        if (res != FR_OK)
        {
            f_close(&s_modelFile);
            return MODEL_LOAD_ERR_READ;
        }
        if (br == 0)            /* unexpected EOF */
            break;

        off += br;
    }

    f_close(&s_modelFile);

    if (off < fsize)
        return MODEL_LOAD_ERR_SHORT_READ;

#if defined(__DCACHE_PRESENT) && (__DCACHE_PRESENT == 1U)
    /* SD -> memory transfers are a MIX of SDH DMA (writes HyperRAM directly,
     * bypassing the D-cache) and FatFs CPU memcpy for the sub-sector tail
     * (lands in the D-cache, maybe never flushed). Make the CPU view and the
     * physical HyperRAM coherent before hashing:
     *   clean      = flush CPU-written (dirty) lines out to HyperRAM
     *   invalidate = drop stale lines so the CRC pass re-reads HyperRAM
     * The NPU fetches physical HyperRAM, so hashing after this maintenance
     * validates exactly the bytes the NPU will see. */
    SCB_CleanDCache_by_Addr((uint32_t *)dst, (int32_t)fsize);
    SCB_InvalidateDCache_by_Addr((uint32_t *)dst, (int32_t)fsize);
#endif

    /* Second pass: CRC what is actually in HyperRAM. */
    crc = ModelFileReader_CRC32Update(crc, dst, fsize) ^ 0xFFFFFFFFU;
    if (out_size)  *out_size  = fsize;
    if (out_crc32) *out_crc32 = crc;
    return MODEL_LOAD_OK;
}

const char *ModelFileReader_StrError(int rc)
{
    switch (rc)
    {
        case MODEL_LOAD_OK:             return "OK";
        case MODEL_LOAD_ERR_OPEN:       return "open failed (file missing / no SD)";
        case MODEL_LOAD_ERR_EMPTY:      return "file is empty";
        case MODEL_LOAD_ERR_TOO_LARGE:  return "file larger than slot";
        case MODEL_LOAD_ERR_READ:       return "f_read error";
        case MODEL_LOAD_ERR_SHORT_READ: return "short read (truncated file)";
        default:                        return "unknown error";
    }
}
