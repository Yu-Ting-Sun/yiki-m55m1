/**************************************************************************//**
 * @file     StoryUI.c
 * @brief    Story panel rendering right of the photo region (see StoryUI.h).
 *
 * The 4-bpp story image is expanded to RGB565 through a 16-entry LUT
 * (background -> text color gradient) band by band in SRAM2, then blitted
 * with Display_FillRect — same path the slideshow uses.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "StoryUI.h"

#include <string.h>

#include "Display.h"

/* Palette (RGB565): warm dark background, warm white text, amber divider. */
#define UI_BG_COLOR     (0x18A2u)   /* #181410 */
#define UI_FG_COLOR     (0xF79Du)   /* #F7F2E9 */
#define UI_DIVIDER      (0xED09u)   /* #E8A34C */

#define UI_BAND_ROWS    (10u)

/* Band bounce buffer -> SRAM2 (CPU/PDMA only; NPU never touches SRAM2). */
__attribute__((section(".bss.vram.data"), aligned(32)))
static uint16_t s_uiBand[STORYUI_PANEL_W * UI_BAND_ROWS];

static uint16_t s_lut[16];
static int      s_lutReady = 0;

static void build_lut(void)
{
    int32_t bgR = (UI_BG_COLOR >> 11) & 0x1F, fgR = (UI_FG_COLOR >> 11) & 0x1F;
    int32_t bgG = (UI_BG_COLOR >> 5) & 0x3F,  fgG = (UI_FG_COLOR >> 5) & 0x3F;
    int32_t bgB = UI_BG_COLOR & 0x1F,          fgB = UI_FG_COLOR & 0x1F;

    for (int32_t i = 0; i < 16; i++)
    {
        int32_t r = bgR + ((fgR - bgR) * i) / 15;
        int32_t g = bgG + ((fgG - bgG) * i) / 15;
        int32_t b = bgB + ((fgB - bgB) * i) / 15;
        s_lut[i] = (uint16_t)((r << 11) | (g << 5) | b);
    }
    s_lutReady = 1;
}

static void fill_rect(uint32_t color, uint32_t x, uint32_t y,
                      uint32_t w, uint32_t h)
{
    S_DISP_RECT rect;
    rect.u32TopLeftX     = x;
    rect.u32TopLeftY     = y;
    rect.u32BottonRightX = x + w - 1u;
    rect.u32BottonRightY = y + h - 1u;
    Display_ClearRect(color, &rect);
}

void StoryUI_DrawChrome(void)
{
    fill_rect(UI_DIVIDER, STORYUI_DIVIDER_X, 0, STORYUI_DIVIDER_W, STORYUI_PANEL_H);
    fill_rect(UI_BG_COLOR, STORYUI_PANEL_X, 0, STORYUI_PANEL_W, STORYUI_PANEL_H);
}

int StoryUI_ShowTextImage(const uint8_t *tim, uint32_t len)
{
    if (len < 8u || memcmp(tim, "TIM4", 4) != 0)
        return -1;

    uint32_t w = (uint32_t)tim[4] | ((uint32_t)tim[5] << 8);
    uint32_t h = (uint32_t)tim[6] | ((uint32_t)tim[7] << 8);

    if (w == 0u || h == 0u || (w & 1u) != 0u ||
        w > STORYUI_PANEL_W || h > STORYUI_PANEL_H)
        return -3;
    if (len < 8u + (w / 2u) * h)
        return -2;

    if (!s_lutReady)
        build_lut();

    /* Smaller-than-panel images sit centered on the panel background. */
    if (w < STORYUI_PANEL_W || h < STORYUI_PANEL_H)
        fill_rect(UI_BG_COLOR, STORYUI_PANEL_X, 0, STORYUI_PANEL_W, STORYUI_PANEL_H);

    uint32_t offX = STORYUI_PANEL_X + (STORYUI_PANEL_W - w) / 2u;
    uint32_t offY = (STORYUI_PANEL_H - h) / 2u;
    const uint8_t *pix = tim + 8u;
    uint32_t rowBytes = w / 2u;

    for (uint32_t y0 = 0; y0 < h; y0 += UI_BAND_ROWS)
    {
        uint32_t n = h - y0;
        if (n > UI_BAND_ROWS) n = UI_BAND_ROWS;

        for (uint32_t r = 0; r < n; r++)
        {
            const uint8_t *src = pix + (y0 + r) * rowBytes;
            uint16_t      *dst = s_uiBand + r * w;

            for (uint32_t b = 0; b < rowBytes; b++)
            {
                uint8_t v = src[b];
                dst[2u * b]      = s_lut[v >> 4];
                dst[2u * b + 1u] = s_lut[v & 0x0F];
            }
        }

        S_DISP_RECT rect;
        rect.u32TopLeftX     = offX;
        rect.u32TopLeftY     = offY + y0;
        rect.u32BottonRightX = offX + w - 1u;
        rect.u32BottonRightY = offY + y0 + n - 1u;
        Display_FillRect(s_uiBand, &rect, 1);
    }
    return 0;
}

/* ---- ASCII helpers (status line + option-C fallback story) -------------- */

#define UI_TEXT_SCALE   (1)                          /* 156-px rail: 8x16   */
#define UI_CHAR_W       (FONT_WIDTH * UI_TEXT_SCALE)
#define UI_LINE_H       (FONT_HTIGHT * UI_TEXT_SCALE + 4)
#define UI_MAX_COLS     ((STORYUI_PANEL_W - 16u) / UI_CHAR_W)   /* 17 chars */
#define UI_MAX_LINES    (12u)

static void put_line_centered(const char *s, uint32_t chars, uint32_t y)
{
    uint32_t px = chars * UI_CHAR_W;
    uint32_t x  = STORYUI_PANEL_X;
    if (px < STORYUI_PANEL_W)
        x += (STORYUI_PANEL_W - px) / 2u;
    Display_PutText(s, chars, x, y, UI_FG_COLOR, UI_BG_COLOR, false, UI_TEXT_SCALE);
}

void StoryUI_ShowStatus(const char *asciiText)
{
    uint32_t n = (uint32_t)strlen(asciiText);
    if (n > UI_MAX_COLS) n = UI_MAX_COLS;

    fill_rect(UI_BG_COLOR, STORYUI_PANEL_X, 0, STORYUI_PANEL_W, STORYUI_PANEL_H);
    put_line_centered(asciiText, n,
                      (STORYUI_PANEL_H - FONT_HTIGHT * UI_TEXT_SCALE) / 2u);
}

void StoryUI_ShowAsciiStory(const char *asciiText)
{
    const char *lineStart[UI_MAX_LINES];
    uint32_t    lineLen[UI_MAX_LINES];
    uint32_t    lines = 0;
    const char *p = asciiText;

    /* Greedy word wrap into at most UI_MAX_LINES lines. */
    while (*p && lines < UI_MAX_LINES)
    {
        while (*p == ' ') p++;
        if (!*p) break;

        const char *end = p, *lastSpace = 0;
        while (*end && (uint32_t)(end - p) < UI_MAX_COLS)
        {
            if (*end == ' ') lastSpace = end;
            end++;
        }
        if (*end && lastSpace && (uint32_t)(end - p) >= UI_MAX_COLS)
            end = lastSpace;

        lineStart[lines] = p;
        lineLen[lines]   = (uint32_t)(end - p);
        lines++;
        p = end;
    }

    fill_rect(UI_BG_COLOR, STORYUI_PANEL_X, 0, STORYUI_PANEL_W, STORYUI_PANEL_H);

    uint32_t y = (STORYUI_PANEL_H - lines * UI_LINE_H) / 2u;
    for (uint32_t i = 0; i < lines; i++)
    {
        put_line_centered(lineStart[i], lineLen[i], y);
        y += UI_LINE_H;
    }
}
