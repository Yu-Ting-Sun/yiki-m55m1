/**************************************************************************//**
 * @file     VoiceRec.c
 * @brief    Capture from the on-board PDM microphone. See VoiceRec.h.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "NuMicro.h"
#include "PerfTimer.h"      /* GetSystemTick_ms */
#include "VoiceRec.h"

/* The on-board microphone is a PDM MEMS mic on DMIC0 (PB4 CLK, PB5 DAT): the
 * eBook's board table lists "PDM Dig. Mic", and the BSP's keyword-spotting
 * sample (KeywordSpotting_EdgeImpulse/BoardInit.cpp) wires it exactly so, on
 * channel 0. An earlier revision recorded through the NAU8822 codec's MIC
 * input instead - that input belongs to the phone jack, so with nothing
 * plugged in it only ever returned the noise of a floating input.
 *
 * Clocking departs from that sample on purpose. It reprograms APLL1 to
 * 196.608 MHz so 16 kHz divides exactly; here APLL1 (220 MHz) also clocks the
 * SD host, which every boot loads models and photos from. So DMIC runs off
 * APLL1/2 = 110 MHz as it is, DMIC_SetSampleRate picks the nearest divider,
 * and init prints the rate it actually got. */
#define VR_DMIC         DMIC0
#define VR_IN_RATE      (16000u)
#define VR_OUT_RATE     (8000u)
#define VR_SETTLE_MS    (50u)    /* MEMS wake-up + decimator settling, dropped */

static uint32_t s_inRate = 0;

int VoiceRec_Init(void)
{
    SYS_UnlockReg();

    SET_DMIC0_CLK_PB4();
    SET_DMIC0_DAT_PB5();

    CLK_SetModuleClock(DMIC0_MODULE, CLK_DMICSEL_DMIC0SEL_APLL1_DIV2, MODULE_NoMsk);
    CLK_EnableModuleClock(DMIC0_MODULE);
    SYS_ResetModule(SYS_DMIC0RST);

    SYS_LockReg();

    DMIC_Open(VR_DMIC);
    DMIC_SET_DOWNSAMPLE(VR_DMIC, DMIC_DOWNSAMPLE_256);     /* as the KWS sample */
    DMIC_SetFIFOWidth(VR_DMIC, DMIC_FIFOWIDTH_16);
    s_inRate = DMIC_SetSampleRate(VR_DMIC, VR_IN_RATE);

    printf("[VREC] DMIC0 (PB4/PB5) at %u Hz (asked %u)\n",
           (unsigned)s_inRate, (unsigned)VR_IN_RATE);

    /* Channel stays off until a capture wants samples: left on with nobody
     * draining it, the FIFO would just sit in overflow in between. */
    return 0;
}

static void rx_start(void)
{
    DMIC_ClearFIFO(VR_DMIC);
    DMIC_EnableChMsk(VR_DMIC, DMIC_CTL_CHEN0_Msk);   /* also turns on the HPF */
}

static void rx_stop(void)
{
    DMIC_DisableChMsk(VR_DMIC, DMIC_CTL_CHEN0_Msk);
}

/*----------------------------------------------------------------------------
 * Capture: 8 kHz G.711 mu-law into SRAM2 (the NPU never needs this buffer).
 *--------------------------------------------------------------------------*/
#define VR_MAX_BYTES    (VOICEREC_MAX_MS * VR_OUT_RATE / 1000u)

__attribute__((section(".bss.vram.data"), aligned(4)))
static uint8_t s_ulaw[VR_MAX_BYTES];

/* Standard G.711 mu-law: bias, find the segment, keep 4 mantissa bits. */
static uint8_t mulaw_encode(int32_t pcm)
{
    const int32_t BIAS = 0x84, CLIP = 32635;
    int32_t sign = 0;

    if (pcm < 0)
    {
        sign = 0x80;
        pcm  = -pcm;
    }
    if (pcm > CLIP)
        pcm = CLIP;
    pcm += BIAS;

    int32_t exponent = 7;
    for (int32_t mask = 0x4000; (pcm & mask) == 0 && exponent > 0; mask >>= 1)
        exponent--;

    int32_t mantissa = (pcm >> (exponent + 3)) & 0x0F;
    return (uint8_t)~(sign | (exponent << 4) | mantissa);
}

const uint8_t *VoiceRec_Capture(uint32_t maxMs, VoiceRec_KeepFn keepGoing,
                                uint32_t *nBytes)
{
    uint32_t ms = (maxMs > VOICEREC_MAX_MS) ? VOICEREC_MAX_MS : maxMs;

    uint32_t want     = ms * VR_OUT_RATE / 1000u;
    uint32_t got      = 0, reads = 0;
    int32_t  half     = 0, peak = 0;
    int      haveHalf = 0;
    uint64_t sumSq    = 0;

    rx_start();

    uint32_t t0 = GetSystemTick_ms();
    while ((GetSystemTick_ms() - t0) < VR_SETTLE_MS)
        if (!DMIC_IS_FIFOEMPTY(VR_DMIC))
            (void)DMIC_READ_DATA(VR_DMIC);

    uint32_t t1     = GetSystemTick_ms();
    uint32_t budget = ms + 500u;                  /* mic stopped clocking? bail */

    while (got < want)
    {
        if (DMIC_IS_FIFOEMPTY(VR_DMIC))
        {
            if ((GetSystemTick_ms() - t1) > budget)
                break;
            continue;
        }

        int32_t s = (int16_t)(DMIC_READ_DATA(VR_DMIC) & 0xFFFFu);
        reads++;

        /* 16 kHz -> 8 kHz: average each pair (2-tap low-pass + decimate). */
        if (!haveHalf)
        {
            half     = s;
            haveHalf = 1;
            continue;
        }
        haveHalf = 0;
        s = (half + s) / 2;

        int32_t a = s < 0 ? -s : s;
        if (a > peak)
            peak = a;
        sumSq += (uint64_t)((int64_t)s * s);

        s_ulaw[got++] = mulaw_encode(s);

        /* Every 10 ms of audio: still talking? (a GPIO read - quick
         * enough that the 16 kHz FIFO never backs up meanwhile) */
        if (keepGoing && (got % (VR_OUT_RATE / 100u)) == 0u &&
            !keepGoing(got * 1000u / VR_OUT_RATE))
            break;
    }

    rx_stop();

    /* reads over elapsed time checks that one FIFO read really is one 16 kHz
     * sample; peak/rms tell how much gain headroom real speech leaves. */
    uint32_t dt = GetSystemTick_ms() - t1;
    printf("[VREC] captured %u B (%u ms audio) in %u ms, %u reads, peak %d, rms %u\n",
           (unsigned)got, (unsigned)(got * 1000u / VR_OUT_RATE), (unsigned)dt,
           (unsigned)reads, (int)peak,
           got ? (unsigned)sqrtf((float)sumSq / (float)got) : 0u);

    if (nBytes)
        *nBytes = got;
    return s_ulaw;
}

uint32_t VoiceRec_RateHz(void)
{
    return s_inRate ? s_inRate / 2u : VR_OUT_RATE;
}
