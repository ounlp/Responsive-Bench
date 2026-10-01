"""
Horizontal scroll detection metric using Playwright.

Detects whether a generated HTML page requires horizontal scrolling at a given viewport size,
counts overflowing elements, and captures screenshots.
"""

import logging
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger("responsive.eval.metrics.horizontal_scroll")

try:
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page
except ImportError:
    async_playwright = None
    Browser = None
    BrowserContext = None
    Page = None

from .base import Metric
from ..playwright_screenshots import screenshot_full_height_clipped


class HorizontalScrollMetric(Metric):
    """Metric that detects horizontal scrolling in generated HTML pages."""

    @property
    def name(self) -> str:
        """Return metric name."""
        return "horizontal_scroll"

    @property
    def requires_browser(self) -> bool:
        """Horizontal scroll detection requires a browser."""
        return True

    async def run(
        self, sample: Any, generated_html_path: Path, context: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Path]]:
        """
        Run horizontal scroll detection.

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

        viewport = context["viewport"]
        playwright_config = context["playwright"]
        artifacts_dir: Path = context["artifacts_dir"]

        # Ensure artifacts directory exists
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Convert to absolute path and file:// URL
        # Use three slashes for cross-platform compatibility (Windows needs file:///)
        abs_path = generated_html_path.resolve()
        # Convert Windows path separators to forward slashes for URL
        path_str = str(abs_path).replace("\\", "/")
        # Add leading slash if on Windows (C:/ -> /C:/)
        if path_str[1] == ":":
            path_str = "/" + path_str
        file_url = f"file://{path_str}"

        # Initialize browser
        browser: Browser | None = None
        context_obj: BrowserContext | None = None
        page: Page | None = None

        try:
            logger.debug(f"Starting horizontal scroll detection for: {generated_html_path}")
            async with async_playwright() as p:
                # Launch browser
                browser = await p.chromium.launch(headless=playwright_config.get("headless", True))
                logger.debug("Browser launched")

                # Create context with viewport
                context_obj = await browser.new_context(
                    viewport={
                        "width": viewport["width"],
                        "height": viewport["height"],
                    }
                )
                logger.debug(f"Created browser context with viewport: {viewport['width']}x{viewport['height']}")

                # Set timeout
                context_obj.set_default_timeout(playwright_config.get("timeout_ms", 30000))

                # Create page
                page = await context_obj.new_page()

                # Navigate to HTML file
                logger.debug(f"Navigating to: {file_url}")
                await page.goto(file_url, wait_until="networkidle")

                # Wait a bit for any dynamic content
                await page.wait_for_timeout(500)

                # Capture initial screenshot
                before_screenshot_path = artifacts_dir / "hscroll_before.png"
                await screenshot_full_height_clipped(
                    page,
                    viewport_width=viewport["width"],
                    output_path=before_screenshot_path,
                    timeout_ms=playwright_config.get("timeout_ms", 30000),
                    animations="disabled",
                )

                # Get page dimensions
                page_width = await page.evaluate("document.documentElement.scrollWidth")
                viewport_width = await page.evaluate("window.innerWidth")
                horizontal_scroll_needed = page_width > viewport_width
                logger.debug(f"Page dimensions: width={page_width}, viewport={viewport_width}, scroll_needed={horizontal_scroll_needed}")

                # Count overflowing elements
                overflowing_elements = await page.evaluate(
                    """
                    () => {
                        let count = 0;
                        const allElements = document.querySelectorAll('*');
                        for (const el of allElements) {
                            if (el.scrollWidth > el.clientWidth) {
                                count++;
                            }
                        }
                        return count;
                    }
                    """
                )
                logger.debug(f"Found {overflowing_elements} overflowing elements")

                scroll_successful = None
                after_screenshot_path = None

                if horizontal_scroll_needed:
                    # Try to scroll horizontally
                    logger.debug("Attempting horizontal scroll")
                    try:
                        await page.evaluate("window.scrollTo(document.documentElement.scrollWidth, 0)")
                        await page.wait_for_timeout(300)  # Wait for scroll animation

                        # Check if scroll position changed
                        scroll_x = await page.evaluate("window.scrollX")
                        scroll_successful = scroll_x > 0
                        logger.debug(f"Scroll attempted: scroll_x={scroll_x}, successful={scroll_successful}")

                        # Capture after-scroll screenshot
                        after_screenshot_path = artifacts_dir / "hscroll_after.png"
                        await screenshot_full_height_clipped(
                            page,
                            viewport_width=viewport["width"],
                            output_path=after_screenshot_path,
                            timeout_ms=playwright_config.get("timeout_ms", 30000),
                            animations="disabled",
                        )
                        logger.debug(f"Captured after-scroll screenshot: {after_screenshot_path}")
                    except Exception as e:
                        # Scroll failed, but we still have the before screenshot
                        logger.warning(f"Scroll attempt failed: {e}")
                        scroll_successful = False
                else:
                    # No scroll needed, capture a "no scroll" screenshot
                    logger.debug("No horizontal scroll needed")
                    after_screenshot_path = artifacts_dir / "hscroll_no_scroll.png"
                    await screenshot_full_height_clipped(
                        page,
                        viewport_width=viewport["width"],
                        output_path=after_screenshot_path,
                        timeout_ms=playwright_config.get("timeout_ms", 30000),
                        animations="disabled",
                    )
                    logger.debug(f"Captured no-scroll screenshot: {after_screenshot_path}")

                # Build result dict
                result = {
                    "horizontal_scroll_needed": bool(horizontal_scroll_needed),
                    "page_width": int(page_width),
                    "viewport_width": int(viewport_width),
                    "overflowing_elements": int(overflowing_elements),
                    "scroll_successful": scroll_successful,
                }

                # Build artifacts dict
                artifacts = {
                    "before": before_screenshot_path,
                }
                if after_screenshot_path:
                    artifacts["after"] = after_screenshot_path

                return result, artifacts

        except Exception as e:
            # Re-raise with more context
            raise RuntimeError(
                f"Horizontal scroll metric failed for {generated_html_path}: {e}\n"
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
