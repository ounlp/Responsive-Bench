"""
Line spacing evaluation metric using Playwright.

Checks that line-height is at least 1.5x the font-size for all text elements.
"""

import logging
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger("responsive.eval.metrics.line_spacing")

try:
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page
except ImportError:
    async_playwright = None
    Browser = None
    BrowserContext = None
    Page = None

from .base import Metric
from .html_constants import MIN_LINE_HEIGHT_RATIO, VIEWPORTS


class LineSpacingMetric(Metric):
    """Metric that checks line-height ratios meet readability requirements."""

    @property
    def name(self) -> str:
        """Return metric name."""
        return "line_spacing"

    @property
    def requires_browser(self) -> bool:
        """Line spacing check requires a browser."""
        return True

    async def run(
        self, sample: Any, generated_html_path: Path, context: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Path]]:
        """
        Run line spacing evaluation.

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

        viewport = context.get("viewport", VIEWPORTS["mobile"])
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
            logger.debug(f"Starting line spacing evaluation for: {generated_html_path}")
            async with async_playwright() as p:
                # Launch browser
                browser = await p.chromium.launch(
                    headless=playwright_config.get("headless", True)
                )

                # Create context with viewport
                context_obj = await browser.new_context(
                    viewport={"width": viewport["width"], "height": 688}
                )
                page = await context_obj.new_page()

                # Set timeout
                context_obj.set_default_timeout(playwright_config.get("timeout_ms", 30000))

                # Load HTML file
                await page.goto(file_url)
                await page.wait_for_load_state("networkidle")

                # Evaluate line spacing in browser context
                results = await page.evaluate(
                    """(minRatio) => {
                        function isVisible(el) {
                            const style = getComputedStyle(el);
                            if (style.display === 'none' || style.visibility === 'hidden') return false;
                            const rect = el.getBoundingClientRect();
                            return rect.width > 0 && rect.height > 0;
                        }

                        function cssPath(el) {
                            const path = [];
                            let current = el;
                            while (current && current.nodeType === 1 && current !== document.body) {
                                let selector = current.nodeName.toLowerCase();
                                if (current.id) {
                                    selector += '#' + current.id;
                                    path.unshift(selector);
                                    break;
                                } else {
                                    let sib = current;
                                    let nth = 1;
                                    while ((sib = sib.previousElementSibling)) {
                                        if (sib.nodeName === current.nodeName) nth++;
                                    }
                                    selector += `:nth-of-type(${nth})`;
                                }
                                path.unshift(selector);
                                current = current.parentElement;
                            }
                            return path.join(' > ');
                        }

                        const results = [];
                        const walker = document.createTreeWalker(
                            document.body,
                            NodeFilter.SHOW_ELEMENT,
                            null
                        );

                        let node;
                        while ((node = walker.nextNode())) {
                            if (!isVisible(node)) continue;

                            const text = node.innerText ? node.innerText.trim() : "";
                            if (!text) continue; // Skip elements without text

                            const style = getComputedStyle(node);
                            const fontSize = parseFloat(style.fontSize);
                            const lineHeight = parseFloat(style.lineHeight);

                            // Skip if lineHeight is NaN or not properly computed
                            if (isNaN(fontSize) || isNaN(lineHeight)) continue;

                            const ratio = lineHeight / fontSize;
                            const passes = ratio >= minRatio;

                            results.push({
                                tag: node.tagName.toLowerCase(),
                                selector: cssPath(node),
                                text: text.slice(0, 100),
                                fontSize: fontSize,
                                lineHeight: lineHeight,
                                ratio: ratio,
                                passes: passes
                            });
                        }

                        return results;
                    }""",
                    MIN_LINE_HEIGHT_RATIO,
                )

                logger.debug(f"Found {len(results)} text elements")

        except Exception as e:
            raise RuntimeError(
                f"Line spacing metric failed for {generated_html_path}: {e}\n"
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

        # Filter violations
        violations = [r for r in results if not r["passes"]]

        logger.info(
            f"Line spacing: {len(results)} text elements checked, {len(violations)} violations"
        )

        # Build result dict
        result = {
            "status": "success",
            "html_file": generated_html_path.name,
            "viewport": f"{viewport['width']}px",
            "total_text_elements": len(results),
            "violations": violations,
            "violations_count": len(violations),
            "min_ratio": MIN_LINE_HEIGHT_RATIO,
            "passed": len(violations) == 0,
        }

        # Build artifacts dict (no artifacts for this metric)
        artifacts: dict[str, Path] = {}

        return result, artifacts
