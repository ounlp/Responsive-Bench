"""
Sync screenshot capture for generated HTML at multiple viewport sizes.

Used by self_reflection mode to render the initial HTML at mobile, tablet,
and desktop sizes for LLM evaluation.
"""

import logging
from pathlib import Path
from typing import Literal
logger = logging.getLogger("responsive.runner.screenshots")

# Keep screenshots viewport-clipped (real device view), but full scroll height.
from PIL import Image
from PIL import ImageDraw
import io
import math

# Default viewports (same as eval llm_judge_screenshots)
DEFAULT_VIEWPORTS = {
    "mobile": {"width": 375, "height": 800},
    "tablet": {"width": 768, "height": 1024},
    "desktop": {"width": 1280, "height": 720},
}

# Output filenames per viewport
RENDER_FILENAMES = {
    "mobile": "render_mobile.png",
    "tablet": "render_tablet.png",
    "desktop": "render_desktop.png",
}


def capture_screenshots(
    html_path: Path,
    output_dir: Path,
    viewports: dict[str, dict[str, int]] | None = None,
    *,
    timeout_ms: int = 30000,
    headless: bool = True,
    mode: Literal["default", "bordered"] = "bordered",
) -> dict[str, Path]:
    """
    Render an HTML file at three viewport sizes and save screenshots.

    Args:
        html_path: Path to the HTML file (local file).
        output_dir: Directory to write PNG files (render_mobile.png, etc.).
        viewports: Optional dict of name -> {"width": int, "height": int}.
                   Defaults to mobile 375x800, tablet 768x1024, desktop 1280x720.
        timeout_ms: Page load timeout in milliseconds.
        headless: Whether to run the browser headless.

    Returns:
        Dict mapping viewport name ("mobile", "tablet", "desktop") to Path of saved PNG.

    Raises:
        ImportError: If playwright is not installed.
        RuntimeError: If screenshot capture fails.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as err:
        raise ImportError(
            "playwright not installed. Install with: pip install playwright && playwright install"
        ) from err

    viewports = viewports or DEFAULT_VIEWPORTS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    html_path = Path(html_path).resolve()
    if not html_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")

    # Build file:// URL (cross-platform)
    path_str = str(html_path).replace("\\", "/")
    if len(path_str) > 1 and path_str[1] == ":":
        path_str = "/" + path_str
    file_url = f"file://{path_str}"

    result_paths: dict[str, Path] = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        try:
            context = browser.new_context()
            context.set_default_timeout(timeout_ms)
            try:
                for viewport_name, size in viewports.items():
                    filename = RENDER_FILENAMES.get(viewport_name, f"render_{viewport_name}.png")
                    out_path = output_dir / filename
                    logger.debug(
                        "Rendering %s screenshot (%sx%s) -> %s",
                        viewport_name,
                        size["width"],
                        size["height"],
                        out_path,
                    )
                    page = context.new_page()
                    try:
                        page.set_viewport_size({"width": size["width"], "height": size["height"]})
                        page.goto(file_url, wait_until="networkidle")
                        page.wait_for_timeout(500)
                        _screenshot_full_height_clipped_sync(
                            page,
                            viewport_width=size["width"],
                        viewport_height=size["height"],
                            output_path=out_path,
                            timeout_ms=timeout_ms,
                        bordered=(mode == "bordered"),
                        border_rgba=(255, 0, 0, 128),
                        border_width_px=4,
                        )
                        result_paths[viewport_name] = out_path
                    finally:
                        page.close()
            finally:
                context.close()
        finally:
            browser.close()

    return result_paths


def capture_single_screenshot(
    html_path: Path,
    output_path: Path,
    *,
    width: int = 375,
    height: int = 688,
    timeout_ms: int = 30000,
    headless: bool = True,
    mode: Literal["default", "bordered"] = "bordered",
) -> None:
    """
    Render a single HTML file at a fixed viewport and save a full-page screenshot.

    Args:
        html_path: Path to the HTML file (local file).
        output_path: PNG file path to write.
        width: Viewport width in pixels (default: 375).
        height: Viewport height in pixels (default: 688).
        timeout_ms: Page load timeout in milliseconds.
        headless: Whether to run the browser headless.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as err:
        raise ImportError(
            "playwright not installed. Install with: pip install playwright && playwright install"
        ) from err

    html_path = Path(html_path).resolve()
    if not html_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build file:// URL (cross-platform)
    path_str = str(html_path).replace("\\", "/")
    if len(path_str) > 1 and path_str[1] == ":":
        path_str = "/" + path_str
    file_url = f"file://{path_str}"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        try:
            context = browser.new_context()
            context.set_default_timeout(timeout_ms)
            try:
                page = context.new_page()
                try:
                    page.set_viewport_size({"width": width, "height": height})
                    page.goto(file_url, wait_until="networkidle")
                    page.wait_for_timeout(500)
                    _screenshot_full_height_clipped_sync(
                        page,
                        viewport_width=width,
                        viewport_height=height,
                        output_path=output_path,
                        timeout_ms=timeout_ms,
                        bordered=(mode == "bordered"),
                        border_rgba=(255, 0, 0, 128),
                        border_width_px=4,
                    )
                    logger.debug(
                        "Saved single screenshot %s at %sx%s",
                        output_path,
                        width,
                        height,
                    )
                finally:
                    page.close()
            finally:
                context.close()
        finally:
            browser.close()


def _screenshot_full_height_clipped_sync(
    page,
    *,
    viewport_width: int,
    viewport_height: int,
    output_path: Path,
    timeout_ms: int = 30000,
    max_tile_height: int = 8000,
    bordered: bool = False,
    border_rgba: tuple[int, int, int, int] = (255, 0, 0, 128),
    border_width_px: int = 4,
) -> None:
    w = max(1, int(viewport_width))
    vh = max(1, int(viewport_height))
    full_w = page.evaluate(
        "Math.max(document.body.scrollWidth, document.documentElement.scrollWidth)"
    )
    try:
        full_w = int(full_w)
    except Exception:
        full_w = w
    full_w = max(1, full_w)
    h = page.evaluate(
        "Math.max(document.body.scrollHeight, document.documentElement.scrollHeight)"
    )
    try:
        h = int(h)
    except Exception:
        h = 1
    h = max(1, h)

    out_w = full_w if bordered else w

    # Expand the viewport to the full scroll height so that clip can reach
    # content below the original viewport fold. Without this, Playwright
    # silently clamps clip height to the current viewport height.
    page.set_viewport_size({"width": w, "height": h})

    if h <= max_tile_height:
        png = page.screenshot(
            clip={"x": 0, "y": 0, "width": out_w, "height": h},
            timeout=timeout_ms,
            animations="disabled",
        )
        im = Image.open(io.BytesIO(png)).convert("RGBA")
        if bordered:
            _draw_viewport_border(
                im,
                viewport_width=w,
                viewport_height=vh,
                border_rgba=border_rgba,
                border_width_px=border_width_px,
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(output_path)
        return

    tiles = []
    n = int(math.ceil(h / max_tile_height))
    for i in range(n):
        y = i * max_tile_height
        tile_h = min(max_tile_height, h - y)
        png = page.screenshot(
            clip={"x": 0, "y": y, "width": out_w, "height": tile_h},
            timeout=timeout_ms,
            animations="disabled",
        )
        tiles.append(Image.open(io.BytesIO(png)).convert("RGBA"))

    stitched = Image.new("RGBA", (out_w, h))
    oy = 0
    for im in tiles:
        stitched.paste(im, (0, oy))
        oy += im.height

    if bordered:
        _draw_viewport_border(
            stitched,
            viewport_width=w,
            viewport_height=vh,
            border_rgba=border_rgba,
            border_width_px=border_width_px,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stitched.save(output_path)


def _draw_viewport_border(
    im: Image.Image,
    *,
    viewport_width: int,
    viewport_height: int,
    border_rgba: tuple[int, int, int, int],
    border_width_px: int,
) -> None:
    w = max(1, int(viewport_width))
    vh = max(1, int(viewport_height))
    bw = max(1, int(border_width_px))

    draw = ImageDraw.Draw(im, "RGBA")
    x1 = min(w - 1, im.width - 1)
    y1 = min(vh - 1, im.height - 1)
    if x1 >= 0 and y1 >= 0:
        draw.rectangle((0, 0, x1, y1), outline=border_rgba, width=bw)
