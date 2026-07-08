# Render the story text into the LCD story rail image (Day 3 Task 3, option B).
#
# Layout v3 (landscape-first, per user): photo 640x480 on the left (landscape
# 4:3 phone photos fill it edge to edge; portrait ones centered on the warm-
# black background), 4-px divider, then this 156x480 story rail on the right:
# place + date header up top, the story in small horizontal lines below.
#
# Output format "TIM4" (Text IMage, 4-bpp grayscale), consumed by StoryUI.c:
#   offset 0  : magic "TIM4"
#   offset 4  : width  (uint16 LE)
#   offset 6  : height (uint16 LE)
#   offset 8  : pixels, 2 per byte, HIGH nibble = left pixel, rows top-down,
#               0 = background, 15 = full text color (board maps via LUT)
# 156x480 rail -> 8 + 37440 bytes. Grayscale keeps the CJK anti-aliasing
# (and the dimmed header) that a 1-bpp font blit would lose.
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PANEL_W = 156
PANEL_H = 480
PAD_X = 16
PAD_TOP = 30
PAD_BOTTOM = 30
FONT_PATH = "C:/Windows/Fonts/msjh.ttc"  # Microsoft JhengHei (Traditional Chinese)

HEADER_SIZE = 20         # place, e.g. 日月潭
HEADER_FILL = 175
SUB_SIZE = 13            # date line, dimmer
SUB_FILL = 120
RULE_FILL = 60           # hairline under the header block
HEADER_GAP = 22          # rule -> story block

STORY_SIZE = 19
STORY_LEADING = 28
STORY_MIN_SIZE = 15
MAX_LINES = 12

# Characters that must not begin a line (CJK punctuation kinsoku rule).
NO_LINE_START = "，。！？、；：）」』…‧．·,.!?;:)~"


def _fit_font(text: str, size: int, min_size: int, max_w: int) -> ImageFont.FreeTypeFont:
    """Largest font <= size that fits text in max_w (floor at min_size)."""
    while size > min_size:
        font = ImageFont.truetype(FONT_PATH, size)
        if font.getlength(text) <= max_w:
            return font
        size -= 1
    return ImageFont.truetype(FONT_PATH, min_size)


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    lines: list[str] = []
    cur = ""
    for ch in text.strip():
        if ch == "\n":
            lines.append(cur)
            cur = ""
            continue
        if not cur or ch in NO_LINE_START or font.getlength(cur + ch) <= max_w:
            cur += ch
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines


def render_story_gray(text: str, header: str = "", subheader: str = "") -> Image.Image:
    """Story rail -> anti-aliased 156x480 'L' image (0=bg)."""
    img = Image.new("L", (PANEL_W, PANEL_H), 0)
    draw = ImageDraw.Draw(img)
    usable_w = PANEL_W - 2 * PAD_X

    y = PAD_TOP
    if header:
        hfont = _fit_font(header, HEADER_SIZE, 14, usable_w)
        draw.text((PAD_X, y), header, fill=HEADER_FILL, font=hfont)
        y += hfont.size + 8
    if subheader:
        sfont = _fit_font(subheader, SUB_SIZE, 10, usable_w)
        draw.text((PAD_X, y), subheader, fill=SUB_FILL, font=sfont)
        y += sfont.size + 12
    if header or subheader:
        draw.line([(PAD_X, y), (PANEL_W - PAD_X, y)], fill=RULE_FILL, width=1)
        y += HEADER_GAP

    size = STORY_SIZE
    leading = STORY_LEADING
    while True:
        font = ImageFont.truetype(FONT_PATH, size)
        lines = _wrap(text, font, usable_w)
        if len(lines) <= MAX_LINES or size <= STORY_MIN_SIZE:
            break
        size -= 1          # very long story: shrink until it fits
        leading = size + 9

    # Story block vertically centered in the space below the header.
    block_h = len(lines) * leading
    y += max(0, (PANEL_H - PAD_BOTTOM - y - block_h) // 2)
    for line in lines:
        draw.text((PAD_X, y), line, fill=255, font=font)   # left-aligned
        y += leading
    return img


def pack_tim4(img: Image.Image) -> bytes:
    w, h = img.size
    px = img.load()
    out = bytearray()
    out += b"TIM4"
    out += w.to_bytes(2, "little") + h.to_bytes(2, "little")
    for y in range(h):
        for x in range(0, w, 2):
            hi = px[x, y] >> 4
            lo = (px[x + 1, y] >> 4) if x + 1 < w else 0
            out.append((hi << 4) | lo)
    return bytes(out)


def render_story_tim4(text: str, header: str = "", subheader: str = "",
                      preview_png: Path | None = None) -> bytes:
    img = render_story_gray(text, header, subheader)
    if preview_png is not None:
        # Browser-checkable preview in the board's actual colors.
        bg, fg = (0x18, 0x14, 0x10), (0xF7, 0xF2, 0xE9)
        rgb = Image.new("RGB", img.size, bg)
        tint = Image.new("RGB", img.size, fg)
        rgb.paste(tint, (0, 0), img)
        rgb.save(preview_png)
    return pack_tim4(img)
