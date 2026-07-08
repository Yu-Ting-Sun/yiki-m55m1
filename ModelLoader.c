/**************************************************************************//**
 * @file     ModelLoader.c
 * @brief    LoadModelToHyperRAM(): drive select + timing + logging wrapper.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "ModelLoader.h"
#include "ModelFileReader.h"
#include "PerfTimer.h"

#include <stdio.h>
#include "ff.h"     /* f_chdrive */

int LoadModelToHyperRAM(const char *filename,
                        uint32_t    dst_addr,
                        uint32_t    max_size,
                        uint32_t   *out_size,
                        uint32_t   *out_crc32)
{
    TCHAR    sd_path[] = { '0', ':', 0 };   /* SD = FATFS drive 0: */
    uint32_t size = 0, crc = 0, t0, dt;
    int      rc;

    f_chdrive(sd_path);

    t0 = GetSystemTick_ms();
    rc = ModelFileReader_LoadToAddress(filename, (void *)dst_addr, max_size, &size, &crc);
    dt = GetSystemTick_ms() - t0;

    if (rc != MODEL_LOAD_OK)
    {
        printf("[LOAD][FAIL] %-22s rc=%d (%s)  dst=0x%08X  slot=%uKB\n",
               filename, rc, ModelFileReader_StrError(rc),
               (unsigned)dst_addr, (unsigned)(max_size / 1024U));
        return rc;
    }

    /* size in KB with one decimal, avoiding float printf */
    printf("[LOAD][OK]   %-22s dst=0x%08X  size=%u B (%u.%01u KB)  CRC32=0x%08X  %u ms\n",
           filename, (unsigned)dst_addr,
           (unsigned)size,
           (unsigned)(size / 1024U), (unsigned)((size % 1024U) * 10U / 1024U),
           (unsigned)crc, (unsigned)dt);

    /* Diagnostic dump: a valid .tflite has 'TFL3' (54 46 4C 33) at bytes 4-7.
     * Bytes 0-3 are the flatbuffer root offset and legitimately DIFFER per
     * file (e.g. 0x20 face-vela, 0x28 gesture) — do not treat them as fixed.
     * The loader clean+invalidated the D-cache before returning, so these
     * volatile reads reflect the REAL HyperRAM content the NPU sees.
     * Garbage here = the HyperRAM write path itself is broken. */
    printf("  First 32B @ 0x%08X: ", (unsigned)dst_addr);
    for (int i = 0; i < 32; i++)
        printf("%02X ", *((volatile uint8_t *)dst_addr + i));
    printf(" (bytes 4-7 must be 54 46 4C 33 = 'TFL3')\n");

    if (out_size)  *out_size  = size;
    if (out_crc32) *out_crc32 = crc;
    return MODEL_LOAD_OK;
}
