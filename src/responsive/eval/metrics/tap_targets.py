"""
Tap target size evaluation metric using Playwright.

Checks that graphic buttons and tap targets meet minimum size requirements (48x48px).
"""

import logging
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger("responsive.eval.metrics.tap_targets")

try:
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page
except ImportError:
    async_playwright = None
    Browser = None
    BrowserContext = None
    Page = None

from .base import Metric
from .html_constants import MIN_TAP_TARGET_SIZE, VIEWPORTS


class TapTargetsMetric(Metric):
    """Metric that checks tap target sizes meet accessibility requirements."""

    @property
    def name(self) -> str:
        """Return metric name."""
        return "tap_targets"

    @property
    def requires_browser(self) -> bool:
        """Tap targets check requires a browser."""
        return True

    async def run(
        self, sample: Any, generated_html_path: Path, context: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Path]]:
        """
        Run tap target size evaluation.

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
            logger.debug(f"Starting tap target evaluation for: {generated_html_path}")
            async with async_playwright() as p:
                # Launch browser
                browser = await p.chromium.launch(
                    headless=playwright_config.get("headless", True)
                )

                # Create page with viewport
                page = await browser.new_page(
                    viewport={"width": viewport["width"], "height": 800}
                )

                # Set timeout
                page.set_default_timeout(playwright_config.get("timeout_ms", 30000))

                # Load HTML file
                await page.goto(file_url)
                await page.wait_for_load_state("networkidle")

                # Evaluate tap targets in browser context
                results = await page.evaluate(
                    """
                    () => {
                      // Candidate selectors for tap targets
                      const candidates = Array.from(document.querySelectorAll(
                        'button, a, input[type="button"], input[type="submit"], input[type="reset"]'
                      ));

                      function isGraphicButton(el) {
                        // Heuristics: has <img>, <svg>, or icon classes
                        if (el.querySelector('img, svg')) return true;

                        const className = el.className || '';
                        if (typeof className === 'string') {
                          const iconHints = ['icon', 'fa-', 'mdi-', 'material-icons'];
                          if (iconHints.some(h => className.includes(h))) return true;
                        }

                        // If it has no visible text but has aria-label -> likely graphic
                        const text = (el.textContent || '').trim();
                        if (!text && el.getAttribute('aria-label')) return true;

                        return false;
                      }

                      return candidates.map(el => {
                        const rect = el.getBoundingClientRect();
                        const isGraphic = isGraphicButton(el);
                        const passes = rect.width >= 48 && rect.height >= 48;

                        return {
                          tag: el.tagName.toLowerCase(),
                          text: (el.textContent || '').trim().slice(0, 40),
                          ariaLabel: el.getAttribute('aria-label') || '',
                          isGraphic,
                          width: rect.width,
                          height: rect.height,
                          passes,
                        };
                      });
                    }
                    """
                )

                logger.debug(f"Found {len(results)} tap target candidates")

        except Exception as e:
            raise RuntimeError(
                f"Tap targets metric failed for {generated_html_path}: {e}\n"
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

        # Filter graphic buttons and check for violations
        graphic_buttons = [r for r in results if r["isGraphic"]]
        violations = [r for r in graphic_buttons if not r["passes"]]

        logger.info(
            f"Tap targets: {len(graphic_buttons)} graphic buttons, {len(violations)} violations"
        )

        # Build result dict
        result = {
            "status": "success",
            "html_file": generated_html_path.name,
            "viewport": f"{viewport['width']}px",
            "total_tap_targets": len(results),
            "graphic_buttons": len(graphic_buttons),
            "violations": violations,
            "violations_count": len(violations),
            "min_tap_size": MIN_TAP_TARGET_SIZE,
            "passed": len(violations) == 0,
        }

        # Build artifacts dict (no artifacts for this metric)
        artifacts: dict[str, Path] = {}

        return result, artifacts
