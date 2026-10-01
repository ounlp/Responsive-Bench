"""
Responsive media evaluation metric using Playwright.

Checks if images and videos use responsive sizing (percentage or max-width, not fixed pixels).
"""

import logging
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger("responsive.eval.metrics.responsive_media")

try:
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page
except ImportError:
    async_playwright = None
    Browser = None
    BrowserContext = None
    Page = None

from .base import Metric
from .html_constants import VIEWPORTS


class ResponsiveMediaMetric(Metric):
    """Metric that checks if media elements use responsive sizing."""

    @property
    def name(self) -> str:
        """Return metric name."""
        return "responsive_media"

    @property
    def requires_browser(self) -> bool:
        """Responsive media check requires a browser."""
        return True

    async def run(
        self, sample: Any, generated_html_path: Path, context: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Path]]:
        """
        Run responsive media evaluation.

        Args:
            sample: Sample object (unused)
            generated_html_path: Path to generated HTML file
            context: Context dict with viewport, playwright config, and artifacts_dir

        Returns:
            Tuple of (result_dict, artifacts_dict)

        Raises:
            ImportError: If playwright is not installed
            FileNotFoundError: If generated_html_path does not exist
        """
        if async_playwright is None:
            raise ImportError(
                "playwright not installed. Install with: pip install playwright && playwright install"
            )

        if not generated_html_path.exists():
            raise FileNotFoundError(f"Generated HTML not found: {generated_html_path}")

        playwright_config = context["playwright"]
        artifacts_dir: Path = context["artifacts_dir"]

        # Ensure artifacts directory exists
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Convert to absolute path and file:// URL
        abs_path = generated_html_path.resolve()
        path_str = str(abs_path).replace("\\", "/")
        if path_str[1] == ":":
            path_str = "/" + path_str
        file_url = f"file://{path_str}"

        browser: Browser | None = None
        context_obj: BrowserContext | None = None
        page: Page | None = None

        try:
            logger.debug(f"Starting responsive media evaluation for: {generated_html_path}")
            async with async_playwright() as pw:
                # Launch browser
                browser = await pw.chromium.launch(
                    headless=playwright_config.get("headless", True)
                )

                # Use mobile viewport for testing
                context = await browser.new_context(
                    viewport={"width": VIEWPORTS["mobile"]["width"], "height": 688}
                )
                page = await context.new_page()

                # Set timeout
                context.set_default_timeout(playwright_config.get("timeout_ms", 30000))

                # Load HTML file
                await page.goto(file_url)
                await page.wait_for_load_state("networkidle")

                # Check all img and video elements for responsive sizing
                media_elements = await page.evaluate(
                    """
                    () => {
                        const elements = [];
                        const mediaSelectors = ['img', 'video'];

                        mediaSelectors.forEach(selector => {
                            const items = document.querySelectorAll(selector);
                            items.forEach(el => {
                                const computedStyle = window.getComputedStyle(el);
                                const inlineStyle = el.getAttribute('style') || '';
                                const widthAttr = el.getAttribute('width');
                                const heightAttr = el.getAttribute('height');

                                // Get width and max-width values
                                const width = computedStyle.width;
                                const maxWidth = computedStyle.maxWidth;
                                const inlineWidth = inlineStyle.match(/width\\s*:\\s*([^;]+)/i);
                                const inlineMaxWidth = inlineStyle.match(/max-width\\s*:\\s*([^;]+)/i);

                                // Check if using fixed pixel values
                                const hasFixedPixelWidth = (
                                    (widthAttr && !widthAttr.includes('%')) ||
                                    (inlineWidth && inlineWidth[1].trim().match(/^\\d+px$/))
                                );

                                const hasPercentageOrMaxWidth = (
                                    (inlineWidth && inlineWidth[1].includes('%')) ||
                                    (inlineMaxWidth && inlineMaxWidth[1].includes('%')) ||
                                    maxWidth !== 'none'
                                );

                                // Determine if responsive
                                const isResponsive = !hasFixedPixelWidth || hasPercentageOrMaxWidth;

                                elements.push({
                                    tag: selector,
                                    src: el.src || el.getAttribute('src') || '(no src)',
                                    width: width,
                                    maxWidth: maxWidth,
                                    inlineWidth: inlineWidth ? inlineWidth[1].trim() : null,
                                    inlineMaxWidth: inlineMaxWidth ? inlineMaxWidth[1].trim() : null,
                                    widthAttr: widthAttr,
                                    heightAttr: heightAttr,
                                    hasFixedPixelWidth: hasFixedPixelWidth,
                                    isResponsive: isResponsive,
                                    outerHTML: el.outerHTML.substring(0, 200)
                                });
                            });
                        });

                        return elements;
                    }
                    """
                )

                logger.debug(f"Found {len(media_elements)} media elements")

        except Exception as e:
            raise RuntimeError(
                f"Responsive media metric failed for {generated_html_path}: {e}\n"
                f"Traceback: {traceback.format_exc()}"
            ) from e

        finally:
            # Cleanup browser resources
            if page:
                try:
                    await page.close()
                except Exception:
                    pass
            if context_obj:
                try:
                    await context_obj.close()
                except Exception:
                    pass
            if browser:
                try:
                    await browser.close()
                except Exception:
                    pass

        # Analyze results
        total_media = len(media_elements)
        non_responsive = [el for el in media_elements if not el["isResponsive"]]
        violations_count = len(non_responsive)

        logger.info(
            f"Responsive media: {total_media} elements, {violations_count} non-responsive"
        )

        # Build result dict
        result = {
            "status": "success",
            "html_file": generated_html_path.name,
            "total_media_elements": total_media,
            "responsive_count": total_media - violations_count,
            "violations_count": violations_count,
            "violations": non_responsive,
            "passed": violations_count == 0,
        }

        # Build artifacts dict (no artifacts for this metric)
        artifacts: dict[str, Path] = {}

        return result, artifacts
