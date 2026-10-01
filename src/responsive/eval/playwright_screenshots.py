from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from PIL import Image
from PIL import ImageDraw


async def _scroll_metrics(page: Any) -> tuple[int, int]:
    """
    Return (scroll_height, viewport_height) for the document scroll container.
    """
    data = await page.evaluate(
        """() => {
        const se = document.scrollingElement || document.documentElement;
        const sh = Math.max(se.scrollHeight || 0, document.documentElement.scrollHeight || 0, document.body?.scrollHeight || 0);
        const vh = window.innerHeight || document.documentElement.clientHeight || 0;
        return { sh, vh };
      }"""
    )
    try:
        sh = int(data.get("sh", 0))
    except Exception:
        sh = 0
    try:
        vh = int(data.get("vh", 0))
    except Exception:
        vh = 0
    return max(1, sh), max(1, vh)


async def _scroll_width(page: Any) -> int:
    data = await page.evaluate(
        """() => {
        const se = document.scrollingElement || document.documentElement;
        const sw = Math.max(
          se.scrollWidth || 0,
          document.documentElement.scrollWidth || 0,
          document.body?.scrollWidth || 0
        );
        return { sw };
      }"""
    )
    try:
        sw = int(data.get("sw", 0))
    except Exception:
        sw = 0
    return max(1, sw)


def _overlay_viewport_border_png(
    png_bytes: bytes,
    *,
    viewport_width: int,
    viewport_height: int,
    border_rgba: tuple[int, int, int, int] = (255, 0, 0, 128),
    border_width_px: int = 4,
) -> bytes:
    w = max(1, int(viewport_width))
    vh = max(1, int(viewport_height))
    bw = max(1, int(border_width_px))

    im = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    draw = ImageDraw.Draw(im, "RGBA")

    x1 = min(w - 1, im.width - 1)
    y1 = min(vh - 1, im.height - 1)
    if x1 >= 0 and y1 >= 0:
        draw.rectangle((0, 0, x1, y1), outline=border_rgba, width=bw)

    out = io.BytesIO()
    im.save(out, format="PNG")
    return out.getvalue()


async def screenshot_full_page_bordered(
    page: Any,
    *,
    viewport_width: int,
    output_path: Path | None = None,
    timeout_ms: int | None = None,
    animations: str | None = None,
    border_rgba: tuple[int, int, int, int] = (255, 0, 0, 128),
    border_width_px: int = 4,
) -> bytes | None:
    """
    Capture a full-page screenshot (full document width + height) and draw a border
    indicating the visible viewport area at the top of the page.
    """
    w_viewport = max(1, int(viewport_width))
    _, viewport_h = await _scroll_metrics(page)

    png = await page.screenshot(
        full_page=True,
        timeout=timeout_ms,
        animations=animations,
    )

    png_bordered = _overlay_viewport_border_png(
        png,
        viewport_width=w_viewport,
        viewport_height=viewport_h,
        border_rgba=border_rgba,
        border_width_px=border_width_px,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(png_bordered)
        return None
    return png_bordered


async def screenshot_full_height_clipped(
    page: Any,
    *,
    viewport_width: int,
    output_path: Path | None = None,
    timeout_ms: int | None = None,
    animations: str | None = None,
) -> bytes | None:
    """
    Capture a screenshot that matches the device viewport width (clips overflow),
    while covering the full scroll height.

    Why: Playwright's full_page=True captures full document width when the page
    overflows horizontally, which does not match what a real device shows.

    Returns PNG bytes if output_path is None; otherwise writes to output_path.
    """
    w = max(1, int(viewport_width))
    scroll_h, viewport_h = await _scroll_metrics(page)

    tiles: list[Image.Image] = []
    positions: list[int] = []

    # Ensure scrolling doesn't animate (reduces duplicate tiles on some pages).
    await page.evaluate(
        """() => {
        try {
          const style = document.createElement('style');
          style.id = '__responsivebench_scroll_fix__';
          style.textContent = 'html{scroll-behavior:auto !important;} *{scroll-behavior:auto !important;}';
          document.head && document.head.appendChild(style);
        } catch (e) {}
      }"""
    )

    overlap = 80
    step = max(1, viewport_h - overlap)

    y = 0
    while True:
        await page.evaluate(
            """y => {
            const se = document.scrollingElement || document.documentElement;
            se.scrollTop = y;
          }""",
            y,
        )
        await page.wait_for_timeout(80)

        png = await page.screenshot(
            clip={"x": 0, "y": 0, "width": w, "height": viewport_h},
            timeout=timeout_ms,
            animations=animations,
        )
        tiles.append(Image.open(io.BytesIO(png)).convert("RGBA"))
        positions.append(y)

        if y + viewport_h >= scroll_h:
            break
        y_next = y + step
        if y_next == y:
            break
        y = y_next

    # Stitch using the scroll positions as offsets; crop the final tile to exact height.
    stitched = Image.new("RGBA", (w, scroll_h))
    for im, y0 in zip(tiles, positions, strict=False):
        stitched.paste(im, (0, y0))

    out = io.BytesIO()
    stitched.save(out, format="PNG")
    png_all = out.getvalue()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(png_all)
        return None
    return png_all


async def screenshot_full_height_full_width(
    page: Any,
    *,
    output_path: Path | None = None,
    timeout_ms: int | None = None,
    animations: str | None = None,
) -> bytes | None:
    """
    Capture a screenshot covering full scroll height and full document width.
    Returns PNG bytes if output_path is None; otherwise writes to output_path.
    """
    scroll_h, viewport_h = await _scroll_metrics(page)
    full_w = await _scroll_width(page)

    tiles: list[Image.Image] = []
    positions: list[int] = []

    y = 0
    while y < scroll_h:
        tile_h = min(viewport_h, scroll_h - y)
        png = await page.screenshot(
            clip={"x": 0, "y": y, "width": full_w, "height": tile_h},
            timeout=timeout_ms,
            animations=animations,
        )
        tiles.append(Image.open(io.BytesIO(png)).convert("RGBA"))
        positions.append(y)
        if y + tile_h >= scroll_h:
            break
        y += tile_h

    stitched = Image.new("RGBA", (full_w, scroll_h))
    for im, y0 in zip(tiles, positions, strict=False):
        stitched.paste(im, (0, y0))

    out = io.BytesIO()
    stitched.save(out, format="PNG")
    png_all = out.getvalue()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(png_all)
        return None
    return png_all


async def screenshot_full_height_full_width_bordered(
    page: Any,
    *,
    viewport_width: int,
    output_path: Path | None = None,
    timeout_ms: int | None = None,
    animations: str | None = None,
    border_rgba: tuple[int, int, int, int] = (255, 0, 0, 128),
    border_width_px: int = 4,
) -> bytes | None:
    w_viewport = max(1, int(viewport_width))
    _, viewport_h = await _scroll_metrics(page)

    png = await screenshot_full_height_full_width(
        page,
        output_path=None,
        timeout_ms=timeout_ms,
        animations=animations,
    )
    if png is None:
        return None

    png_bordered = _overlay_viewport_border_png(
        png,
        viewport_width=w_viewport,
        viewport_height=viewport_h,
        border_rgba=border_rgba,
        border_width_px=border_width_px,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(png_bordered)
        return None
    return png_bordered


async def screenshot_full_height_clipped_bordered(
    page: Any,
    *,
    viewport_width: int,
    output_path: Path | None = None,
    timeout_ms: int | None = None,
    animations: str | None = None,
    border_rgba: tuple[int, int, int, int] = (255, 0, 0, 128),
    border_width_px: int = 4,
) -> bytes | None:
    w = max(1, int(viewport_width))
    _, viewport_h = await _scroll_metrics(page)

    png = await screenshot_full_height_clipped(
        page,
        viewport_width=w,
        output_path=None,
        timeout_ms=timeout_ms,
        animations=animations,
    )
    if png is None:
        return None

    png_bordered = _overlay_viewport_border_png(
        png,
        viewport_width=w,
        viewport_height=viewport_h,
        border_rgba=border_rgba,
        border_width_px=border_width_px,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(png_bordered)
        return None
    return png_bordered

