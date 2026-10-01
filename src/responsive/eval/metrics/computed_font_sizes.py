"""
Computed font sizes evaluation metric using Playwright.

Checks font sizes across multiple viewports to ensure minimum font size (12px) is met.
"""

import logging
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger("responsive.eval.metrics.computed_font_sizes")

try:
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page
except ImportError:
    async_playwright = None
    Browser = None
    BrowserContext = None
    Page = None

from .base import Metric
from .html_constants import MIN_FONT_SIZE, VIEWPORTS


class ComputedFontSizesMetric(Metric):
    """Metric that checks computed font sizes meet minimum requirements across viewports."""

    @property
    def name(self) -> str:
        """Return metric name."""
        return "computed_font_sizes"

    @property
    def requires_browser(self) -> bool:
        """Computed font sizes check requires a browser."""
        return True

    async def run(
        self, sample: Any, generated_html_path: Path, context: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Path]]:
        """
        Run computed font sizes evaluation across all viewports.

        Args:
            sample: Sample object (unused)
            generated_html_path: Path to generated HTML file
            context: Context dict with playwright config and artifacts_dir

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
        per_viewport_results: dict[str, dict[str, Any]] = {}

        try:
            logger.debug(
                f"Starting computed font sizes evaluation for: {generated_html_path}"
            )
            async with async_playwright() as pw:
                # Launch browser
                browser = await pw.chromium.launch(
                    headless=playwright_config.get("headless", True)
                )

                # Test across all viewports
                for viewport_name, viewport_config in VIEWPORTS.items():
                    logger.debug(f"Testing viewport: {viewport_name}")

                    # Create a new context for each viewport
                    context_obj = await browser.new_context(
                        viewport={"width": viewport_config["width"], "height": 688}
                    )
                    page = await context_obj.new_page()

                    # Set timeout
                    context_obj.set_default_timeout(
                        playwright_config.get("timeout_ms", 30000)
                    )

                    # Load HTML file
                    await page.goto(file_url)
                    await page.wait_for_load_state("networkidle")

                    # Evaluate in-page: gather elements that directly contain text nodes
                    results = await page.evaluate(
                        """(minPx) => {
                            function isVisible(el) {
                                const style = getComputedStyle(el);
                                if (style.display === 'none' || style.visibility === 'hidden' || parseFloat(style.opacity) === 0) return false;
                                const rect = el.getBoundingClientRect();
                                return rect.width > 0 && rect.height > 0;
                            }

                            function cssPath(el) {
                                const path = [];
                                while (el && el.nodeType === 1 && el !== document.body) {
                                    let selector = el.nodeName.toLowerCase();
                                    if (el.id) {
                                        selector += '#' + el.id;
                                        path.unshift(selector);
                                        break;
                                    } else {
                                        let sib = el, nth = 1;
                                        while (sib = sib.previousElementSibling) {
                                            if (sib.nodeName === el.nodeName) nth++;
                                        }
                                        selector += `:nth-of-type(${nth})`;
                                    }
                                    path.unshift(selector);
                                    el = el.parentElement;
                                }
                                return path.join(' > ');
                            }

                            function hasDirectTextContent(el) {
                                // Check if element has text nodes as direct children (not nested in other elements)
                                for (let child of el.childNodes) {
                                    if (child.nodeType === Node.TEXT_NODE && child.textContent.trim()) {
                                        return true;
                                    }
                                }
                                return false;
                            }

                            const out = [];
                            const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT, null);
                            let node;

                            while ((node = walker.nextNode())) {
                                if (!isVisible(node)) continue;

                                // Only check elements that directly contain text, not just parent containers
                                if (!hasDirectTextContent(node)) continue;

                                // Get direct text content only (not from child elements)
                                let directText = '';
                                for (let child of node.childNodes) {
                                    if (child.nodeType === Node.TEXT_NODE) {
                                        directText += child.textContent;
                                    }
                                }
                                directText = directText.trim();

                                if (!directText) continue; // Skip if no direct text

                                const s = getComputedStyle(node);
                                const fs = s.fontSize;
                                const px = parseFloat(fs);

                                out.push({
                                    tag: node.tagName.toLowerCase(),
                                    selector: cssPath(node),
                                    text: directText.slice(0, 120),
                                    font_size: fs,
                                    font_size_px: px,
                                    ok: !Number.isNaN(px) && px >= minPx
                                });
                            }
                            return out;
                        }""",
                        MIN_FONT_SIZE,
                    )

                    # Close context
                    await context_obj.close()

                    total_text_elements = len(results)
                    violations = [r for r in results if not r.get("ok", False)]

                    per_viewport_results[viewport_name] = {
                        "total_text_elements": total_text_elements,
                        "violations_count": len(violations),
                        "violations": violations,
                    }

                    logger.debug(
                        f"Viewport {viewport_name}: {total_text_elements} elements, "
                        f"{len(violations)} violations"
                    )

        except Exception as e:
            raise RuntimeError(
                f"Computed font sizes metric failed for {generated_html_path}: {e}\n"
                f"Traceback: {traceback.format_exc()}"
            ) from e

        finally:
            # Cleanup browser resources
            if browser:
                try:
                    await browser.close()
                except Exception:
                    pass

        # Build summary across all viewports
        total_elems = sum(v["total_text_elements"] for v in per_viewport_results.values())
        total_violations = sum(
            v["violations_count"] for v in per_viewport_results.values()
        )

        logger.info(
            f"Computed font sizes: {total_elems} total elements across all viewports, "
            f"{total_violations} total violations"
        )

        # Build result dict
        result = {
            "status": "success",
            "html_file": generated_html_path.name,
            "min_font_size": MIN_FONT_SIZE,
            "total_text_elements": total_elems,
            "total_violations": total_violations,
            "by_viewport": per_viewport_results,
            "passed": total_violations == 0,
        }

        # Build artifacts dict (no artifacts for this metric)
        artifacts: dict[str, Path] = {}

        return result, artifacts
