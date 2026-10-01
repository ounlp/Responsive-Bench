"""
Text enlargement evaluation metric using Playwright.

Uses CSS zoom property to simulate browser zoom (Cmd/Ctrl + Plus) and checks
if it causes horizontal scroll or text cutoff.
"""

import logging
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger("responsive.eval.metrics.text_enlargement")

try:
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page
except ImportError:
    async_playwright = None
    Browser = None
    BrowserContext = None
    Page = None

from .base import Metric
from ..playwright_screenshots import screenshot_full_height_clipped


class TextEnlargementMetric(Metric):
    """Metric that checks if enlarging text (via zoom) causes horizontal scroll or cutoff."""

    # Default zoom factor (1.5 = 150%, equivalent to pressing Cmd/Ctrl + Plus a few times)
    DEFAULT_ZOOM_FACTOR = 1.5

    @property
    def name(self) -> str:
        """Return metric name."""
        return "text_enlargement"

    @property
    def requires_browser(self) -> bool:
        """Text enlargement check requires a browser."""
        return True

    async def run(
        self, sample: Any, generated_html_path: Path, context: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Path]]:
        """
        Run text enlargement evaluation using CSS zoom.

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

        viewport = context.get("viewport", {"width": 375, "height": 800})
        playwright_config = context["playwright"]
        artifacts_dir: Path = context["artifacts_dir"]

        # Get zoom factor from context (default 1.5x = 150%)
        zoom_factor = context.get("text_enlargement_factor", self.DEFAULT_ZOOM_FACTOR)

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
            logger.debug(
                f"Starting text enlargement evaluation for: {generated_html_path} "
                f"(zoom factor: {zoom_factor}x)"
            )
            async with async_playwright() as p:
                # Launch browser
                browser = await p.chromium.launch(
                    headless=playwright_config.get("headless", True)
                )

                # Create context with viewport
                context_obj = await browser.new_context(
                    viewport={"width": viewport["width"], "height": viewport["height"]}
                )
                page = await context_obj.new_page()

                # Set timeout
                context_obj.set_default_timeout(playwright_config.get("timeout_ms", 30000))

                # Navigate to HTML file
                logger.debug(f"Navigating to: {file_url}")
                await page.goto(file_url, wait_until="networkidle")
                await page.wait_for_timeout(500)

                # Capture baseline screenshot (before zoom)
                before_screenshot_path = artifacts_dir / "text_enlargement_before.png"
                await screenshot_full_height_clipped(
                    page,
                    viewport_width=viewport["width"],
                    output_path=before_screenshot_path,
                    timeout_ms=playwright_config.get("timeout_ms", 30000),
                    animations="disabled",
                )
                logger.debug(f"Captured baseline screenshot: {before_screenshot_path}")

                # Get baseline measurements
                baseline_page_width = await page.evaluate("document.documentElement.scrollWidth")
                baseline_viewport_width = await page.evaluate("window.innerWidth")
                baseline_horizontal_scroll = baseline_page_width > baseline_viewport_width

                logger.debug(
                    f"Baseline: page_width={baseline_page_width}, "
                    f"viewport={baseline_viewport_width}, "
                    f"scroll={baseline_horizontal_scroll}"
                )

                # Apply CSS zoom to simulate browser zoom (Cmd/Ctrl + Plus)
                await page.evaluate(
                    f"""
                    () => {{
                        // Apply zoom to body (affects entire page, like browser zoom)
                        document.body.style.zoom = '{zoom_factor}';
                        // Force reflow to apply changes
                        document.body.offsetHeight;
                    }}
                    """
                )

                # Wait for layout to update
                await page.wait_for_timeout(500)

                # Capture after zoom screenshot
                after_screenshot_path = artifacts_dir / "text_enlargement_after.png"
                await screenshot_full_height_clipped(
                    page,
                    viewport_width=viewport["width"],
                    output_path=after_screenshot_path,
                    timeout_ms=playwright_config.get("timeout_ms", 30000),
                    animations="disabled",
                )
                logger.debug(f"Captured after-zoom screenshot: {after_screenshot_path}")

                # Check for horizontal scroll after zoom
                zoomed_page_width = await page.evaluate("document.documentElement.scrollWidth")
                zoomed_viewport_width = await page.evaluate("window.innerWidth")
                zoomed_horizontal_scroll = zoomed_page_width > zoomed_viewport_width

                logger.debug(
                    f"After zoom: page_width={zoomed_page_width}, "
                    f"viewport={zoomed_viewport_width}, "
                    f"scroll={zoomed_horizontal_scroll}"
                )

                # Count overflowing elements after zoom
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
                logger.debug(f"Found {overflowing_elements} overflowing elements after zoom")

                # Check for text cutoff after zoom
                cutoff_violations = await page.evaluate(
                    """
                    () => {
                        const violations = [];
                        const allElements = document.querySelectorAll('*');
                        
                        function hasDirectTextContent(el) {
                            for (let child of el.childNodes) {
                                if (child.nodeType === Node.TEXT_NODE && child.textContent.trim()) {
                                    return true;
                                }
                            }
                            return false;
                        }
                        
                        function isVisible(el) {
                            const style = getComputedStyle(el);
                            return style.display !== 'none' && 
                                   style.visibility !== 'hidden' && 
                                   style.opacity !== '0' &&
                                   el.offsetWidth > 0 && 
                                   el.offsetHeight > 0;
                        }
                        
                        for (const el of allElements) {
                            if (!isVisible(el)) continue;
                            if (!hasDirectTextContent(el)) continue;
                            
                            const rect = el.getBoundingClientRect();
                            const style = getComputedStyle(el);
                            const parent = el.parentElement;
                            const parentRect = parent ? parent.getBoundingClientRect() : null;
                            
                            // Check for horizontal overflow
                            if (el.scrollWidth > el.clientWidth) {
                                violations.push({
                                    selector: el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') + (el.className ? '.' + el.className.split(' ')[0] : ''),
                                    type: 'horizontal_overflow',
                                    scrollWidth: el.scrollWidth,
                                    clientWidth: el.clientWidth,
                                    overflowAmount: el.scrollWidth - el.clientWidth,
                                    text: el.textContent.trim().substring(0, 100)
                                });
                            }
                            
                            // Check for CSS clipping (only flag if there's actual overflow)
                            if (style.overflow === 'hidden' || style.overflow === 'clip') {
                                // Only flag if there's actual overflow being clipped
                                const hasHorizontalOverflow = el.scrollWidth > el.clientWidth;
                                const hasVerticalOverflow = el.scrollHeight > el.clientHeight;
                                if (hasHorizontalOverflow || hasVerticalOverflow) {
                                    violations.push({
                                        selector: el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') + (el.className ? '.' + el.className.split(' ')[0] : ''),
                                        type: 'css_clipped',
                                        overflow: style.overflow,
                                        horizontalOverflow: hasHorizontalOverflow,
                                        verticalOverflow: hasVerticalOverflow,
                                        text: el.textContent.trim().substring(0, 100)
                                    });
                                }
                            }
                            
                            // Check if element goes out of parent bounds
                            if (parentRect && (rect.x < parentRect.x || 
                                rect.x + rect.width > parentRect.x + parentRect.width ||
                                rect.y < parentRect.y ||
                                rect.y + rect.height > parentRect.y + parentRect.height)) {
                                violations.push({
                                    selector: el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') + (el.className ? '.' + el.className.split(' ')[0] : ''),
                                    type: 'out_of_bounds',
                                    elementBounds: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
                                    parentBounds: { x: parentRect.x, y: parentRect.y, width: parentRect.width, height: parentRect.height },
                                    text: el.textContent.trim().substring(0, 100)
                                });
                            }
                            
                            // Check for text truncation (ellipsis)
                            if (style.textOverflow === 'ellipsis' && el.scrollWidth > el.clientWidth) {
                                violations.push({
                                    selector: el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') + (el.className ? '.' + el.className.split(' ')[0] : ''),
                                    type: 'text_truncated',
                                    text: el.textContent.trim().substring(0, 100)
                                });
                            }
                        }
                        
                        return violations;
                    }
                    """
                )

                logger.debug(f"Found {len(cutoff_violations)} cutoff violations after zoom")

                # Determine if test passed
                # Pass if: no horizontal scroll introduced AND no cutoff violations
                passed = not zoomed_horizontal_scroll and len(cutoff_violations) == 0

                # Build result dict
                result = {
                    "status": "success",
                    "html_file": generated_html_path.name,
                    "viewport": f"{viewport['width']}x{viewport['height']}",
                    "zoom_factor": zoom_factor,
                    "baseline": {
                        "page_width": int(baseline_page_width),
                        "viewport_width": int(baseline_viewport_width),
                        "horizontal_scroll": bool(baseline_horizontal_scroll),
                    },
                    "after_zoom": {
                        "page_width": int(zoomed_page_width),
                        "viewport_width": int(zoomed_viewport_width),
                        "horizontal_scroll": bool(zoomed_horizontal_scroll),
                        "overflowing_elements": int(overflowing_elements),
                        "cutoff_violations": cutoff_violations,
                        "cutoff_violations_count": len(cutoff_violations),
                    },
                    "horizontal_scroll_introduced": bool(
                        zoomed_horizontal_scroll and not baseline_horizontal_scroll
                    ),
                    "passed": passed,
                }

                logger.info(
                    f"Text enlargement: {'passed' if passed else 'failed'} "
                    f"(scroll={zoomed_horizontal_scroll}, cutoff_violations={len(cutoff_violations)})"
                )

                # Build artifacts dict
                artifacts = {
                    "before": before_screenshot_path,
                    "after": after_screenshot_path,
                }

                return result, artifacts

        except Exception as e:
            raise RuntimeError(
                f"Text enlargement metric failed for {generated_html_path}: {e}\n"
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
