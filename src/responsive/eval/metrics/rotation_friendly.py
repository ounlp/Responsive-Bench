"""
Rotation-friendly evaluation metric using Playwright.

Tests whether text elements and media sizes change appropriately when viewport
rotates from portrait (375x688) to landscape (688x375). Only checks containers
that contain text or media content.
"""

import logging
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger("responsive.eval.metrics.rotation_friendly")

try:
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page
except ImportError:
    async_playwright = None
    Browser = None
    BrowserContext = None
    Page = None

from .base import Metric
from ..playwright_screenshots import screenshot_full_height_clipped
from .html_constants import MIN_FONT_SIZE


class RotationFriendlyMetric(Metric):
    """Metric that checks if layout adapts properly to viewport rotation."""

    # Viewport configurations
    PORTRAIT = {"width": 375, "height": 688}
    LANDSCAPE = {"width": 688, "height": 375}

    # Thresholds
    MAX_WIDTH_RATIO_DEVIATION = 0.2  # Allow 20% deviation from expected ratio
    EXPECTED_RATIO = 688 / 375  # ≈ 1.834

    @property
    def name(self) -> str:
        """Return metric name."""
        return "rotation_friendly"

    @property
    def requires_browser(self) -> bool:
        """Rotation-friendly check requires a browser."""
        return True

    async def run(
        self, sample: Any, generated_html_path: Path, context: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Path]]:
        """
        Run rotation-friendly evaluation.

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
        page: Page | None = None

        try:
            logger.debug(
                f"Starting rotation-friendly evaluation for: {generated_html_path}"
            )
            async with async_playwright() as p:
                # Launch browser
                browser = await p.chromium.launch(
                    headless=playwright_config.get("headless", True)
                )

                # Create page with portrait viewport
                page = await browser.new_page(viewport=self.PORTRAIT)
                page.set_default_timeout(playwright_config.get("timeout_ms", 30000))

                # Load HTML file
                await page.goto(file_url)
                await page.wait_for_load_state("networkidle")
                await page.wait_for_timeout(500)  # Wait for layout stabilization

                logger.debug("Extracting elements at portrait orientation")
                # Extract elements at PORTRAIT
                portrait_elements = await page.evaluate(
                    """() => {
                        function isVisible(el) {
                            const style = getComputedStyle(el);
                            if (style.display === 'none' || style.visibility === 'hidden' || parseFloat(style.opacity) === 0) return false;
                            const rect = el.getBoundingClientRect();
                            return rect.width > 0 && rect.height > 0;
                        }

                        function hasContent(el) {
                            // Check for direct text content
                            for (let child of el.childNodes) {
                                if (child.nodeType === Node.TEXT_NODE && child.textContent.trim()) {
                                    return true;
                                }
                            }
                            // Check for media elements
                            if (el.querySelector('img, video')) return true;
                            // Check for text in descendants (but not nested containers)
                            const text = el.innerText.trim();
                            const hasNestedContainers = el.querySelector('div, section, article, aside, header, footer, main, nav');
                            return text && (!hasNestedContainers || el.querySelector('img, video'));
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

                        const elements = [];
                        const walker = document.createTreeWalker(
                            document.body,
                            NodeFilter.SHOW_ELEMENT,
                            null
                        );

                        let node;
                        while ((node = walker.nextNode())) {
                            if (!isVisible(node)) continue;
                            if (!hasContent(node)) continue;  // Skip empty containers

                            const rect = node.getBoundingClientRect();
                            const style = getComputedStyle(node);

                            // Extract text element data
                            const text = node.innerText.trim();
                            const fontSize = parseFloat(style.fontSize);
                            const lineHeight = parseFloat(style.lineHeight) || fontSize * 1.2;

                            // Calculate approximate line count (for text wrapping detection)
                            let lineCount = 1;
                            if (text && rect.height > 0 && lineHeight > 0) {
                                lineCount = Math.max(1, Math.round(rect.height / lineHeight));
                            }

                            // Check if text fits on one line (no wrapping)
                            const textFitsOnOneLine = node.scrollWidth <= node.clientWidth;

                            // Determine if element should be excluded from width scaling checks
                            const tagName = node.tagName.toLowerCase();
                            const isButton = tagName === 'button' || node.closest('button') !== null;
                            const isLink = tagName === 'a';
                            const isNav = tagName === 'nav' || node.closest('nav') !== null;
                            const isHeader = tagName === 'header' || node.closest('header') !== null;
                            const isFooter = tagName === 'footer' || node.closest('footer') !== null;

                            const shouldExcludeFromScaling = isButton || isLink || isNav || isHeader || isFooter;

                            // Extract media element data
                            const mediaElements = Array.from(node.querySelectorAll('img, video')).map(media => ({
                                tag: media.tagName.toLowerCase(),
                                src: media.src || media.getAttribute('src') || '',
                                width: media.getBoundingClientRect().width,
                                height: media.getBoundingClientRect().height,
                                naturalWidth: media.naturalWidth || 0,
                                naturalHeight: media.naturalHeight || 0
                            }));

                            elements.push({
                                selector: cssPath(node),
                                tag: tagName,
                                element_type: mediaElements.length > 0 ? 'media_container' :
                                             (text ? 'text_container' : 'container'),
                                // Container dimensions
                                width: rect.width,
                                height: rect.height,
                                // Text properties
                                fontSize: fontSize,
                                lineHeight: lineHeight,
                                hasText: !!text,
                                textPreview: text.slice(0, 50),
                                // Text wrapping detection
                                lineCount: lineCount,
                                textFitsOnOneLine: textFitsOnOneLine,
                                // Exclusion flags
                                shouldExcludeFromScaling: shouldExcludeFromScaling,
                                isButton: isButton,
                                isLink: isLink,
                                // Media properties
                                mediaCount: mediaElements.length,
                                mediaElements: mediaElements,
                                // Overflow check
                                scrollWidth: node.scrollWidth,
                                scrollHeight: node.scrollHeight,
                                clientWidth: node.clientWidth,
                                clientHeight: node.clientHeight
                            });
                        }

                        return elements;
                    }"""
                )

                logger.debug(f"Found {len(portrait_elements)} elements with content at portrait")

                # Take screenshot at portrait
                portrait_screenshot = artifacts_dir / "rotation_portrait.png"
                await screenshot_full_height_clipped(
                    page,
                    viewport_width=self.PORTRAIT["width"],
                    output_path=portrait_screenshot,
                    timeout_ms=playwright_config.get("timeout_ms", 30000),
                    animations="disabled",
                )
                logger.debug(f"Saved portrait screenshot: {portrait_screenshot}")

                # Rotate to LANDSCAPE
                logger.debug("Rotating to landscape orientation")
                await page.set_viewport_size(self.LANDSCAPE)
                await page.wait_for_timeout(500)  # Wait for reflow

                # Extract elements at LANDSCAPE
                logger.debug("Extracting elements at landscape orientation")
                landscape_elements = await page.evaluate(
                    """() => {
                        function isVisible(el) {
                            const style = getComputedStyle(el);
                            if (style.display === 'none' || style.visibility === 'hidden' || parseFloat(style.opacity) === 0) return false;
                            const rect = el.getBoundingClientRect();
                            return rect.width > 0 && rect.height > 0;
                        }

                        function hasContent(el) {
                            for (let child of el.childNodes) {
                                if (child.nodeType === Node.TEXT_NODE && child.textContent.trim()) {
                                    return true;
                                }
                            }
                            if (el.querySelector('img, video')) return true;
                            const text = el.innerText.trim();
                            const hasNestedContainers = el.querySelector('div, section, article, aside, header, footer, main, nav');
                            return text && (!hasNestedContainers || el.querySelector('img, video'));
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

                        const elements = [];
                        const walker = document.createTreeWalker(
                            document.body,
                            NodeFilter.SHOW_ELEMENT,
                            null
                        );

                        let node;
                        while ((node = walker.nextNode())) {
                            if (!isVisible(node)) continue;
                            if (!hasContent(node)) continue;

                            const rect = node.getBoundingClientRect();
                            const style = getComputedStyle(node);

                            const text = node.innerText.trim();
                            const fontSize = parseFloat(style.fontSize);
                            const lineHeight = parseFloat(style.lineHeight) || fontSize * 1.2;

                            // Calculate approximate line count (for text wrapping detection)
                            let lineCount = 1;
                            if (text && rect.height > 0 && lineHeight > 0) {
                                lineCount = Math.max(1, Math.round(rect.height / lineHeight));
                            }

                            // Check if text fits on one line (no wrapping)
                            const textFitsOnOneLine = node.scrollWidth <= node.clientWidth;

                            // Determine if element should be excluded from width scaling checks
                            const tagName = node.tagName.toLowerCase();
                            const isButton = tagName === 'button' || node.closest('button') !== null;
                            const isLink = tagName === 'a';
                            const isNav = tagName === 'nav' || node.closest('nav') !== null;
                            const isHeader = tagName === 'header' || node.closest('header') !== null;
                            const isFooter = tagName === 'footer' || node.closest('footer') !== null;

                            const shouldExcludeFromScaling = isButton || isLink || isNav || isHeader || isFooter;

                            const mediaElements = Array.from(node.querySelectorAll('img, video')).map(media => ({
                                tag: media.tagName.toLowerCase(),
                                src: media.src || media.getAttribute('src') || '',
                                width: media.getBoundingClientRect().width,
                                height: media.getBoundingClientRect().height,
                                naturalWidth: media.naturalWidth || 0,
                                naturalHeight: media.naturalHeight || 0
                            }));

                            elements.push({
                                selector: cssPath(node),
                                tag: tagName,
                                element_type: mediaElements.length > 0 ? 'media_container' :
                                             (text ? 'text_container' : 'container'),
                                width: rect.width,
                                height: rect.height,
                                fontSize: fontSize,
                                lineHeight: lineHeight,
                                hasText: !!text,
                                textPreview: text.slice(0, 50),
                                // Text wrapping detection
                                lineCount: lineCount,
                                textFitsOnOneLine: textFitsOnOneLine,
                                // Exclusion flags
                                shouldExcludeFromScaling: shouldExcludeFromScaling,
                                isButton: isButton,
                                isLink: isLink,
                                mediaCount: mediaElements.length,
                                mediaElements: mediaElements,
                                scrollWidth: node.scrollWidth,
                                scrollHeight: node.scrollHeight,
                                clientWidth: node.clientWidth,
                                clientHeight: node.clientHeight
                            });
                        }

                        return elements;
                    }"""
                )

                logger.debug(f"Found {len(landscape_elements)} elements with content at landscape")

                # Take screenshot at landscape
                landscape_screenshot = artifacts_dir / "rotation_landscape.png"
                await screenshot_full_height_clipped(
                    page,
                    viewport_width=self.LANDSCAPE["width"],
                    output_path=landscape_screenshot,
                    timeout_ms=playwright_config.get("timeout_ms", 30000),
                    animations="disabled",
                )
                logger.debug(f"Saved landscape screenshot: {landscape_screenshot}")

        except Exception as e:
            raise RuntimeError(
                f"Rotation-friendly metric failed for {generated_html_path}: {e}\n"
                f"Traceback: {traceback.format_exc()}"
            ) from e

        finally:
            # Cleanup browser resources
            if page:
                try:
                    await page.close()
                except Exception:
                    pass
            if browser:
                try:
                    await browser.close()
                except Exception:
                    pass

        # Detect violations
        violations = self._detect_violations(portrait_elements, landscape_elements)

        # Build violation summary
        violation_summary = {
            "font_too_small": sum(1 for v in violations if v["issue"] == "font_too_small"),
            "width_not_scaling": sum(1 for v in violations if v["issue"] == "width_not_scaling"),
            "horizontal_overflow": sum(1 for v in violations if v["issue"] == "horizontal_overflow"),
            "media_not_scaling": sum(1 for v in violations if v["issue"] == "media_not_scaling"),
            "element_disappeared": sum(1 for v in violations if v["issue"] == "element_disappeared"),
        }

        logger.info(
            f"Rotation-friendly: {len(portrait_elements)} elements checked, {len(violations)} violations"
        )

        # Build result dict
        result = {
            "status": "success",
            "html_file": generated_html_path.name,
            "portrait_viewport": self.PORTRAIT,
            "landscape_viewport": self.LANDSCAPE,
            "expected_ratio": self.EXPECTED_RATIO,
            "elements_checked": len(portrait_elements),
            "violations": violations,
            "violations_count": len(violations),
            "violation_summary": violation_summary,
            "passed": len(violations) == 0,
        }

        # Build artifacts dict
        artifacts = {
            "portrait_screenshot": portrait_screenshot,
            "landscape_screenshot": landscape_screenshot,
        }

        return result, artifacts

    def _detect_violations(
        self, portrait_elements: list[dict], landscape_elements: list[dict]
    ) -> list[dict]:
        """
        Compare elements between orientations and detect violations.

        Args:
            portrait_elements: List of element data from portrait orientation
            landscape_elements: List of element data from landscape orientation

        Returns:
            List of violation dictionaries
        """
        violations = []

        # Create lookup by selector
        landscape_lookup = {el["selector"]: el for el in landscape_elements}

        for portrait_el in portrait_elements:
            selector = portrait_el["selector"]
            landscape_el = landscape_lookup.get(selector)

            if not landscape_el:
                # Element disappeared in landscape (potential issue)
                violations.append(
                    {
                        "selector": selector,
                        "issue": "element_disappeared",
                        "element_type": portrait_el["element_type"],
                    }
                )
                continue

            # Check 1: Font size violation
            if portrait_el["hasText"]:
                landscape_font_size = landscape_el["fontSize"]
                if not (landscape_font_size is None or landscape_font_size == 0):
                    if landscape_font_size < MIN_FONT_SIZE:
                        violations.append(
                            {
                                "selector": selector,
                                "issue": "font_too_small",
                                "element_type": "text_container",
                                "portrait_font_size": portrait_el["fontSize"],
                                "landscape_font_size": landscape_font_size,
                                "min_required": MIN_FONT_SIZE,
                            }
                        )

            # Check 2: Container width scaling
            # Only check elements that should scale (exclude headings, buttons, links, etc.)
            # and only flag violations if text actually wraps or would overflow
            if portrait_el["width"] > 0 and not portrait_el.get("shouldExcludeFromScaling", False):
                # Check if text wrapping occurs or changes between orientations
                portrait_wraps = not portrait_el.get("textFitsOnOneLine", True)
                landscape_wraps = not landscape_el.get("textFitsOnOneLine", True)
                portrait_line_count = portrait_el.get("lineCount", 1)
                landscape_line_count = landscape_el.get("lineCount", 1)

                # Text should wrap in landscape if it needs more space
                # Or line count should increase, indicating wrapping occurred
                text_needs_wrapping = (
                    landscape_wraps or  # Text wraps in landscape
                    landscape_line_count > portrait_line_count or  # Line count increased
                    (landscape_el["scrollWidth"] > landscape_el["clientWidth"])  # Would overflow
                )

                # Only flag width scaling violation if text actually needs to wrap
                if text_needs_wrapping:
                    width_ratio = landscape_el["width"] / portrait_el["width"]
                    expected_ratio = self.EXPECTED_RATIO
                    deviation = abs(width_ratio - expected_ratio)
                    max_deviation = expected_ratio * self.MAX_WIDTH_RATIO_DEVIATION

                    if deviation > max_deviation:
                        violations.append(
                            {
                                "selector": selector,
                                "issue": "width_not_scaling",
                                "element_type": portrait_el["element_type"],
                                "portrait_width": portrait_el["width"],
                                "landscape_width": landscape_el["width"],
                                "actual_ratio": round(width_ratio, 3),
                                "expected_ratio": round(expected_ratio, 3),
                                "deviation": round(deviation, 3),
                                "portrait_wraps": portrait_wraps,
                                "landscape_wraps": landscape_wraps,
                                "portrait_line_count": portrait_line_count,
                                "landscape_line_count": landscape_line_count,
                            }
                        )

            # Check 3: Overflow detection
            if landscape_el["scrollWidth"] > landscape_el["clientWidth"]:
                violations.append(
                    {
                        "selector": selector,
                        "issue": "horizontal_overflow",
                        "element_type": portrait_el["element_type"],
                        "scroll_width": landscape_el["scrollWidth"],
                        "client_width": landscape_el["clientWidth"],
                        "overflow": landscape_el["scrollWidth"] - landscape_el["clientWidth"],
                    }
                )

            # Check 4: Media scaling
            if portrait_el["mediaCount"] > 0:
                portrait_media = portrait_el["mediaElements"]
                landscape_media = landscape_el["mediaElements"]

                # Match media elements by index (assuming same order)
                for i, (p_media, l_media) in enumerate(
                    zip(portrait_media, landscape_media)
                ):
                    if p_media["width"] > 0:
                        media_width_ratio = l_media["width"] / p_media["width"]
                        expected_ratio = self.EXPECTED_RATIO
                        deviation = abs(media_width_ratio - expected_ratio)
                        max_deviation = expected_ratio * self.MAX_WIDTH_RATIO_DEVIATION

                        if deviation > max_deviation:
                            violations.append(
                                {
                                    "selector": selector,
                                    "issue": "media_not_scaling",
                                    "media_index": i,
                                    "media_src": p_media["src"][:100],  # Truncate long URLs
                                    "portrait_width": p_media["width"],
                                    "landscape_width": l_media["width"],
                                    "actual_ratio": round(media_width_ratio, 3),
                                    "expected_ratio": round(expected_ratio, 3),
                                    "deviation": round(deviation, 3),
                                }
                            )

        return violations
