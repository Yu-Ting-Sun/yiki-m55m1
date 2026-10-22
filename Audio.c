/**************************************************************************//**
 * @file     Audio.c
 * @brief    NAU8822 playback over I2S0. See Audio.h.
 *
 * Wiring and codec registers follow the BSP sample
 * SampleCode/StdDriver/I2S_WAVEPlayer (same board): I2S0 on PI6-PI10, the
 * codec's control port on I2C3 (PG0/PG1, address 0x1A), the codec as I2S
 * master off a 12 MHz MCLK, and JK-EN (PD1) low for the phone jack.
 *
 * The sample owns PDMA0 and its IRQ outright; here the display driver
 * already does (Device/Display/drv_pdma.c), so playback borrows a channel
 * from it instead of a second PDMA0_IRQHandler.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include <stdio.h>
#include <string.h>

#include "NuMicro.h"
#include "PerfTimer.h"          /* GetSystemTick_ms (I2C waits, reporting) */
#include "ff.h"
#include "Device/Display/drv_pdma.h"
#include "Audio.h"

#define AU_I2C              I2C3
#define AU_CODEC_ADDR       (0x1A)
#define AU_I2C_TIMEOUT_MS   (50u)   /* codec missing: fail, never hang       */

/* The codec runs at 16 kHz and the file is 8 kHz: every sample goes out
 * twice. One DMA buffer is 2048 stereo frames = 128 ms, which is a lot of
 * slack for the SD read that refills the other one. */
#define AU_RATE_HZ          (16000u)
#define AU_FRAMES           (2048u)
#define AU_ULAW_CHUNK       (AU_FRAMES / 2u)

/* A buffer takes 128 ms to play. If one is not handed back well inside this,
 * the codec is not clocking us (nothing plugged into the jack does not do
 * that - a dead PLL would) and playback gives up instead of hanging. */
#define AU_STALL_MS         (1000u)

/* One 32-bit word per stereo frame (both halves carry the same sample).
 * DCache is on (system_M55M1.h), so each filled buffer is cleaned before
 * the PDMA reads it. */
__attribute__((section(".bss.vram.data"), aligned(DCACHE_LINE_SIZE)))
static uint32_t s_pcm[2][AU_FRAMES];
static uint8_t  s_ulaw[AU_ULAW_CHUNK];
static int16_t  s_ulawToPcm[256];

static volatile uint8_t s_bufFull[2];   /* 1 = holds audio the DMA still owes */
static volatile uint8_t s_playIdx;      /* buffer the DMA is draining         */
static int      s_dmaCh = -1;
static int      s_ready;
static DSCT_T  *s_desc[2];              /* ping-pong scatter-gather nodes     */

/*----------------------------------------------------------------------------
 * NAU8822 control port (I2C3)
 *--------------------------------------------------------------------------*/
static int i2c_wait(void)
{
    uint32_t t0 = GetSystemTick_ms();

    while (!(AU_I2C->CTL0 & I2C_CTL0_SI_Msk))
        if ((GetSystemTick_ms() - t0) > AU_I2C_TIMEOUT_MS)
            return -1;
    return 0;
}

/* 9-bit value into a 7-bit register address, as the codec's 2-byte frame. */
static int codec_write(uint8_t addr, uint16_t data)
{
    int rc = -1;

    I2C_START(AU_I2C);
    if (i2c_wait() != 0)
        goto done;

    I2C_SET_DATA(AU_I2C, AU_CODEC_ADDR << 1);
    I2C_SET_CONTROL_REG(AU_I2C, I2C_CTL_SI);
    if (i2c_wait() != 0)
        goto done;

    I2C_SET_DATA(AU_I2C, (uint8_t)((addr << 1) | (data >> 8)));
    I2C_SET_CONTROL_REG(AU_I2C, I2C_CTL_SI);
    if (i2c_wait() != 0)
        goto done;

    I2C_SET_DATA(AU_I2C, (uint8_t)(data & 0x00FF));
    I2C_SET_CONTROL_REG(AU_I2C, I2C_CTL_SI);
    if (i2c_wait() != 0)
        goto done;

    rc = 0;
done:
    I2C_STOP(AU_I2C);
    return rc;
}

/* Register values are the BSP sample's, for 16 kHz playback. */
static int codec_setup(void)
{
    int rc = 0;

    rc |= codec_write(0, 0x000);     /* reset all registers */
    {
        uint32_t t0 = GetSystemTick_ms();
        while ((GetSystemTick_ms() - t0) < 10u) { }
    }

    rc |= codec_write(1,  0x02F);
    rc |= codec_write(2,  0x1BF);    /* L/R headphone, ADC mix/boost, ADC */
    rc |= codec_write(3,  0x07F);    /* L/R main mixer, DAC */
    rc |= codec_write(4,  0x010);    /* 16-bit words, I2S format, stereo */
    rc |= codec_write(5,  0x000);    /* no companding, no loopback */
    rc |= codec_write(10, 0x008);    /* DAC unmuted, 128x oversampling */
    rc |= codec_write(14, 0x108);
    rc |= codec_write(15, 0x1EF);
    rc |= codec_write(16, 0x1EF);
    rc |= codec_write(44, 0x000);    /* LLIN/RLIN not connected to PGA */
    rc |= codec_write(47, 0x050);
    rc |= codec_write(48, 0x050);
    rc |= codec_write(50, 0x001);    /* left DAC -> left mixer */
    rc |= codec_write(51, 0x001);    /* right DAC -> right mixer */
    /* Headphone volume 0 dB, both channels at once (bit 8 = update). The
     * sample leaves the reset defaults; say it out loud instead. */
    rc |= codec_write(52, 0x139);
    rc |= codec_write(53, 0x139);

    /* 12.288 MHz PLL path + 16 kHz filters (sample's ConfigSampleRate). */
    rc |= codec_write(36, 0x008);
    rc |= codec_write(37, 0x00C);
    rc |= codec_write(38, 0x093);
    rc |= codec_write(39, 0x0E9);
    rc |= codec_write(6,  0x1AD);    /* divide by 6 -> 16 kHz */
    rc |= codec_write(7,  0x006);    /* 16 kHz internal filter coefficients */
    return rc;
}

/*----------------------------------------------------------------------------
 * PDMA: two buffers chained into a loop; the callback marks the finished one
 * empty so the fill loop can refill it while the other one plays.
 *--------------------------------------------------------------------------*/
static void dma_done_cb(void *userData, uint32_t events)
{
    (void)userData;

    if (events & NU_PDMA_EVENT_TRANSFER_DONE)
    {
        s_bufFull[s_playIdx] = 0;
        s_playIdx ^= 1u;
    }
}

int Audio_Init(void)
{
    struct nu_pdma_chn_cb cb;

    for (int i = 0; i < 256; i++)
    {
        /* G.711 mu-law -> PCM16, the inverse of VoiceRec's encoder. */
        uint32_t u   = (uint32_t)(~i) & 0xFFu;
        int32_t  mag = (int32_t)(((((u & 0x0Fu) << 3) + 0x84u) << ((u >> 4) & 0x07u)) - 0x84u);
        s_ulawToPcm[i] = (int16_t)((u & 0x80u) ? -mag : mag);
    }

    SYS_UnlockReg();
    CLK_EnableModuleClock(I2S0_MODULE);
    CLK_EnableModuleClock(I2C3_MODULE);
    /* drv_pdma.c does not clock the PDMAs itself - Display.c does, and it
     * may not have run yet. Without this its init writes land nowhere and
     * nothing ever transfers. Enabling an already-enabled clock is free. */
    CLK_EnableModuleClock(PDMA0_MODULE);
    CLK_EnableModuleClock(PDMA1_MODULE);
    CLK_EnableModuleClock(GPIOD_MODULE);    /* PD1 = JK-EN  */
    CLK_EnableModuleClock(GPIOG_MODULE);    /* I2C3         */
    CLK_EnableModuleClock(GPIOI_MODULE);    /* I2S0         */

    SET_I2S0_BCLK_PI6();
    SET_I2S0_MCLK_PI7();
    SET_I2S0_DI_PI8();
    SET_I2S0_DO_PI9();
    SET_I2S0_LRCK_PI10();
    GPIO_ENABLE_SCHMITT_TRIGGER(PI, BIT6);

    SET_I2C3_SDA_PG1();
    SET_I2C3_SCL_PG0();
    GPIO_ENABLE_SCHMITT_TRIGGER(PG, BIT0);

    /* I2S0 off HXT, as the BSP sample: nothing here touches APLL1, which
     * also clocks the SD host (see VoiceRec.c). */
    CLK_SetModuleClock(I2S0_MODULE, CLK_I2SSEL_I2S0SEL_HXT, CLK_I2SDIV_I2S0DIV(2));
    SYS_LockReg();

    I2C_Open(AU_I2C, 100000);

    /* Slave: the codec is the I2S master (register 6), clocked off our MCLK. */
    I2S_Open(I2S0, I2S_MODE_SLAVE, AU_RATE_HZ, I2S_DATABIT_16,
             I2S_STEREO, I2S_FORMAT_I2S);

    /* JK-EN low enables the phone jack on the NuMaker board. */
    SET_GPIO_PD1();
    GPIO_SetMode(PD, BIT1, GPIO_MODE_OUTPUT);
    PD1 = 0;

    if (codec_setup() != 0)
    {
        printf("[AUDIO] NAU8822 not answering on I2C3 (PG0/PG1)\n");
        return AUDIO_ERR_CODEC;
    }

    I2S_EnableMCLK(I2S0, 12000000);
    I2S0->CTL0 |= I2S_CTL0_ORDER_Msk;

    s_dmaCh = nu_pdma_channel_allocate(PDMA_I2S0_TX);
    if (s_dmaCh < 0)
    {
        printf("[AUDIO] no PDMA channel for I2S0 TX\n");
        return AUDIO_ERR_DMA;
    }
    if (nu_pdma_sgtbls_allocate(s_desc, 2) != 0)
    {
        printf("[AUDIO] no scatter-gather descriptors\n");
        nu_pdma_channel_free(s_dmaCh);
        s_dmaCh = -1;
        return AUDIO_ERR_DMA;
    }

    cb.m_eCBType      = eCBType_Event;
    cb.m_pfnCBHandler = dma_done_cb;
    cb.m_pvUserData   = NULL;
    cb.m_u32Reserved  = 0;
    nu_pdma_callback_register(s_dmaCh, &cb);
    /* The driver gates event callbacks on this filter; without it the
     * "buffer done" callback never runs and playback stalls after the two
     * buffers filled before the start. */
    nu_pdma_filtering_set(s_dmaCh, NU_PDMA_EVENT_TRANSFER_DONE);

    s_ready = 1;
    printf("[AUDIO] NAU8822 ready on I2S0 @ %u Hz (phone jack)\n",
           (unsigned)AU_RATE_HZ);
    return 0;
}

/*----------------------------------------------------------------------------
 * Playback
 *--------------------------------------------------------------------------*/

/* mu-law from the card -> one DMA buffer; returns the bytes read (0 = end).
 * A short read is padded with silence so the buffer always plays whole. */
static uint32_t fill_buffer(FIL *fp, uint32_t *dst)
{
    UINT     br = 0;
    uint32_t n  = 0;

    if (f_read(fp, s_ulaw, AU_ULAW_CHUNK, &br) != FR_OK)
        br = 0;

    for (UINT i = 0; i < br; i++)
    {
        uint32_t lr = (uint32_t)(uint16_t)s_ulawToPcm[s_ulaw[i]];

        lr |= lr << 16;                  /* same sample to both channels */
        dst[n++] = lr;                   /* 8 kHz -> 16 kHz: twice each  */
        dst[n++] = lr;
    }
    while (n < AU_FRAMES)
        dst[n++] = 0;

    SCB_CleanDCache_by_Addr((uint32_t *)dst, AU_FRAMES * sizeof(uint32_t));
    return (uint32_t)br;
}

int Audio_PlayUlawFile(const char *path, int (*cancel)(void))
{
    FIL      file;
    uint32_t played = 0, t0;
    int      fillIdx = 0, stopped = 0, stalled = 0, eof = 0;

    if (!s_ready)
        return AUDIO_ERR_STATE;
    if (f_open(&file, path, FA_OPEN_EXISTING | FA_READ) != FR_OK)
        return AUDIO_ERR_FILE;

    s_bufFull[0] = s_bufFull[1] = 0;
    s_playIdx    = 0;

    /* Both buffers before the first sample leaves: the codec must never see
     * a gap it would play as a click. */
    for (int i = 0; i < 2; i++)
    {
        uint32_t got = fill_buffer(&file, s_pcm[i]);

        played += got;
        s_bufFull[i] = 1;
        if (got < AU_ULAW_CHUNK)
            eof = 1;
    }

    if (nu_pdma_desc_setup(s_dmaCh, s_desc[0], 32, (uint32_t)&s_pcm[0][0],
                           (uint32_t)&I2S0->TXFIFO, AU_FRAMES, s_desc[1], 0) != 0 ||
        nu_pdma_desc_setup(s_dmaCh, s_desc[1], 32, (uint32_t)&s_pcm[1][0],
                           (uint32_t)&I2S0->TXFIFO, AU_FRAMES, s_desc[0], 0) != 0 ||
        nu_pdma_sg_transfer(s_dmaCh, s_desc[0], 0) != 0)
    {
        f_close(&file);
        printf("[AUDIO] PDMA setup failed\n");
        return AUDIO_ERR_DMA;
    }

    I2S_ENABLE_TXDMA(I2S0);
    I2S_ENABLE_TX(I2S0);
    t0 = GetSystemTick_ms();

    fillIdx = 0;
    while (!eof && !stopped)
    {
        /* Wait for the DMA to hand this buffer back, then refill it. */
        uint32_t waited = GetSystemTick_ms();

        while (s_bufFull[fillIdx])
        {
            if (cancel && cancel())
            {
                stopped = 1;
                break;
            }
            if ((GetSystemTick_ms() - waited) > AU_STALL_MS)
            {
                printf("[AUDIO] codec is not draining the buffers\n");
                stalled = 1;
                break;
            }
        }
        if (stopped || stalled)
            break;

        uint32_t got = fill_buffer(&file, s_pcm[fillIdx]);

        played += got;
        s_bufFull[fillIdx] = 1;
        if (got < AU_ULAW_CHUNK)
            eof = 1;
        fillIdx ^= 1;
    }

    /* Let what is already queued drain (both buffers at most), so the last
     * words are not cut off. */
    if (!stopped && !stalled)
    {
        uint32_t quiet = GetSystemTick_ms();

        while ((s_bufFull[0] || s_bufFull[1]) &&
               (GetSystemTick_ms() - quiet) < (2u * AU_FRAMES * 1000u / AU_RATE_HZ + 50u))
        {
            if (cancel && cancel())
            {
                stopped = 1;
                break;
            }
        }
    }

    nu_pdma_channel_terminate(s_dmaCh);
    I2S_DISABLE_TX(I2S0);
    I2S_DISABLE_TXDMA(I2S0);
    f_close(&file);

    printf("[AUDIO] %s: %u B (%u ms audio) in %u ms%s\n", path,
           (unsigned)played, (unsigned)(played * 1000u / (AU_RATE_HZ / 2u)),
           (unsigned)(GetSystemTick_ms() - t0),
           stalled ? ", stalled" : stopped ? ", stopped" : "");
    if (stalled)
        return AUDIO_ERR_DMA;
    return stopped ? 1 : 0;
}
