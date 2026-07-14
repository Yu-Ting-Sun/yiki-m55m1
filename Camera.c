/**************************************************************************//**
 * @file     Camera.c
 * @brief    HM1055 (CCAP) capture + LCD preview for the photo frame.
 *
 * The CCAP DMA target must be cache-coherent: the buffer lives in
 * .bss.sram.data, which the scatter file places in the SRAM01 alias window
 * (SRAM_NONCACHEABLE region) — same convention the BSP FaceDetection sample
 * relies on for its frame buffers.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "Camera.h"

#include <stdio.h>
#include <stdbool.h>

#include "NuMicro.h"
#include "ImageSensor.h"
#include "Display.h"

__attribute__((section(".bss.sram.data"), aligned(32)))
static uint16_t s_camFrame[CAM_W * CAM_H];

static bool s_camReady = false;

int Camera_Init(void)
{
    SYS_UnlockReg();
    CLK_EnableModuleClock(CCAP0_MODULE);
    SYS_ResetModule(SYS_CCAP0RST);
    SYS_LockReg();

    if (ImageSensor_Init() != 0)
    {
        printf("[CAMERA] HM1055 init failed (module mounted? ribbon seated?)\n");
        return -1;
    }

    ImageSensor_Config(eIMAGE_FMT_RGB565, CAM_W, CAM_H, true);

    s_camReady = true;
    printf("[CAMERA] HM1055 up: %ux%u RGB565 preview\n",
           (unsigned)CAM_W, (unsigned)CAM_H);
    return 0;
}

int Camera_Capture(void)
{
    if (!s_camReady) return -1;

    ImageSensor_TriggerCapture((uint32_t)s_camFrame);

    if (ImageSensor_WaitCaptureDone() != 0)
    {
        printf("[CAMERA] capture timeout\n");
        return -2;
    }
    return 0;
}

void Camera_Blit(uint32_t x, uint32_t y)
{
    if (!s_camReady) return;

    S_DISP_RECT rect;
    rect.u32TopLeftX     = x;
    rect.u32TopLeftY     = y;
    rect.u32BottonRightX = x + CAM_W - 1u;
    rect.u32BottonRightY = y + CAM_H - 1u;
    Display_FillRect(s_camFrame, &rect, 1);
}

int Camera_PreviewTick(uint32_t x, uint32_t y)
{
    int rc = Camera_Capture();
    if (rc != 0) return rc;
    Camera_Blit(x, y);
    return 0;
}

const uint16_t *Camera_GetFrame(void)
{
    return s_camReady ? s_camFrame : (const uint16_t *)0;
}
