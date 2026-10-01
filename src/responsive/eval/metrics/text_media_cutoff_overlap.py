"""
Text and media cutoff/overlap evaluation metric using Playwright.

Detects if text elements and visible media are cut off or overlapping with other elements.
"""

import logging
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger("responsive.eval.metrics.text_media_cutoff_overlap")

try:
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page
except ImportError:
    async_playwright = None
    Browser = None
    BrowserContext = None
    Page = None

from .base import Metric
from ..playwright_screenshots import screenshot_full_height_clipped


class TextMediaCutoffOverlapMetric(Metric):
    """Metric that detects text and media cutoff/overlap issues."""

    @property
    def name(self) -> str:
        """Return metric name."""
        return "text_media_cutoff_overlap"

    @property
    def requires_browser(self) -> bool:
        """Text/media cutoff/overlap check requires a browser."""
        return True

    async def run(
        self, sample: Any, generated_html_path: Path, context: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Path]]:
        """
        Run text and media cutoff/overlap evaluation.

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
                f"Starting text/media cutoff/overlap evaluation for: {generated_html_path}"
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

                # Load HTML file
                await page.goto(file_url)
                await page.wait_for_load_state("networkidle")
                await page.wait_for_timeout(500)  # Wait for layout stabilization

                # Extract text and media elements
                logger.debug("Extracting text and media elements")
                elements_data = await page.evaluate(
                    """() => {
                        function isVisible(el) {
                            const style = getComputedStyle(el);
                            if (style.display === 'none' || style.visibility === 'hidden' || parseFloat(style.opacity) === 0) return false;
                            const rect = el.getBoundingClientRect();
                            return rect.width > 0 && rect.height > 0;
                        }

                        function hasDirectTextContent(el) {
                            for (let child of el.childNodes) {
                                if (child.nodeType === Node.TEXT_NODE && child.textContent.trim()) {
                                    return true;
                                }
                            }
                            return false;
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

                        const textElements = [];
                        const walker = document.createTreeWalker(
                            document.body,
                            NodeFilter.SHOW_ELEMENT,
                            null
                        );

                        let node;
                        while ((node = walker.nextNode())) {
                            if (!isVisible(node)) continue;

                            // Check for text content
                            if (hasDirectTextContent(node)) {
                                const rect = node.getBoundingClientRect();
                                const style = getComputedStyle(node);
                                const parent = node.parentElement;
                                const parentRect = parent ? parent.getBoundingClientRect() : rect;

                                textElements.push({
                                    selector: cssPath(node),
                                    tag: node.tagName.toLowerCase(),
                                    // Position and size
                                    x: rect.x,
                                    y: rect.y,
                                    width: rect.width,
                                    height: rect.height,
                                    // Container bounds
                                    parentX: parentRect.x,
                                    parentY: parentRect.y,
                                    parentWidth: parentRect.width,
                                    parentHeight: parentRect.height,
                                    // Overflow detection
                                    scrollWidth: node.scrollWidth,
                                    scrollHeight: node.scrollHeight,
                                    clientWidth: node.clientWidth,
                                    clientHeight: node.clientHeight,
                                    // CSS properties
                                    overflow: style.overflow,
                                    overflowX: style.overflowX,
                                    overflowY: style.overflowY,
                                    textOverflow: style.textOverflow,
                                    whiteSpace: style.whiteSpace,
                                    // Text content
                                    text: node.innerText.trim().slice(0, 100)
                                });
                            }
                        }

                        // Get all media elements
                        const mediaElements = [];
                        document.querySelectorAll('img, video').forEach(media => {
                            if (!isVisible(media)) return;

                            const rect = media.getBoundingClientRect();
                            const style = getComputedStyle(media);
                            const parent = media.parentElement;
                            const parentRect = parent ? parent.getBoundingClientRect() : rect;

                            mediaElements.push({
                                selector: cssPath(media),
                                tag: media.tagName.toLowerCase(),
                                src: media.src || media.getAttribute('src') || '',
                                // Position and size
                                x: rect.x,
                                y: rect.y,
                                width: rect.width,
                                height: rect.height,
                                naturalWidth: media.naturalWidth || 0,
                                naturalHeight: media.naturalHeight || 0,
                                // Container bounds
                                parentX: parentRect.x,
                                parentY: parentRect.y,
                                parentWidth: parentRect.width,
                                parentHeight: parentRect.height,
                                // Overflow detection
                                scrollWidth: media.scrollWidth,
                                scrollHeight: media.scrollHeight,
                                clientWidth: media.clientWidth,
                                clientHeight: media.clientHeight,
                                // CSS properties
                                overflow: style.overflow,
                                objectFit: style.objectFit
                            });
                        });

                        return { textElements, mediaElements };
                    }"""
                )

                logger.debug(
                    f"Found {len(elements_data['textElements'])} text elements and "
                    f"{len(elements_data['mediaElements'])} media elements"
                )

                # Take screenshot for artifacts
                screenshot_path = artifacts_dir / "cutoff_overlap_screenshot.png"
                await screenshot_full_height_clipped(
                    page,
                    viewport_width=viewport["width"],
                    output_path=screenshot_path,
                    timeout_ms=playwright_config.get("timeout_ms", 30000),
                    animations="disabled",
                )
                logger.debug(f"Saved screenshot: {screenshot_path}")

        except Exception as e:
            raise RuntimeError(
                f"Text/media cutoff/overlap metric failed for {generated_html_path}: {e}\n"
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

        # Detect cutoff violations
        cutoff_violations = self._detect_cutoff(elements_data)

        # Detect overlap violations
        overlap_violations = self._detect_overlap(elements_data)

        total_violations = len(cutoff_violations) + len(overlap_violations)

        logger.info(
            f"Text/media cutoff/overlap: {len(cutoff_violations)} cutoff violations, "
            f"{len(overlap_violations)} overlap violations"
        )

        # Build result dict
        result = {
            "status": "success",
            "html_file": generated_html_path.name,
            "viewport": f"{viewport['width']}x{viewport['height']}",
            "text_elements_checked": len(elements_data["textElements"]),
            "media_elements_checked": len(elements_data["mediaElements"]),
            "cutoff_violations": cutoff_violations,
            "overlap_violations": overlap_violations,
            "cutoff_violations_count": len(cutoff_violations),
            "overlap_violations_count": len(overlap_violations),
            "total_violations": total_violations,
            "passed": total_violations == 0,
        }

        # Build artifacts dict
        artifacts = {
            "screenshot": screenshot_path,
        }

        return result, artifacts

    def _detect_cutoff(self, elements_data: dict[str, list]) -> list[dict]:
        """
        Detect cutoff issues for text and media elements.

        Args:
            elements_data: Dict with 'textElements' and 'mediaElements' lists

        Returns:
            List of cutoff violation dictionaries
        """
        violations = []

        # Check text elements
        for text_el in elements_data["textElements"]:
            issues = []

            # 1. Overflow check (horizontal)
            if text_el["scrollWidth"] > text_el["clientWidth"]:
                issues.append(
                    {
                        "type": "horizontal_overflow",
                        "scroll_width": text_el["scrollWidth"],
                        "client_width": text_el["clientWidth"],
                        "overflow_amount": text_el["scrollWidth"] - text_el["clientWidth"],
                    }
                )

            # 2. Overflow check (vertical)
            if text_el["scrollHeight"] > text_el["clientHeight"]:
                issues.append(
                    {
                        "type": "vertical_overflow",
                        "scroll_height": text_el["scrollHeight"],
                        "client_height": text_el["clientHeight"],
                        "overflow_amount": text_el["scrollHeight"] - text_el["clientHeight"],
                    }
                )

            # 3. CSS clipping (only flag if there's actual overflow)
            if text_el["overflow"] in ["hidden", "clip"]:
                # Only flag if there's actual overflow being clipped
                has_horizontal_overflow = text_el["scrollWidth"] > text_el["clientWidth"]
                has_vertical_overflow = text_el["scrollHeight"] > text_el["clientHeight"]
                if has_horizontal_overflow or has_vertical_overflow:
                    issues.append(
                        {
                            "type": "css_clipped",
                            "overflow": text_el["overflow"],
                            "horizontal_overflow": has_horizontal_overflow,
                            "vertical_overflow": has_vertical_overflow,
                        }
                    )

            # 4. Text truncation (ellipsis) - only flag if text is actually truncated
            if (
                text_el["textOverflow"] == "ellipsis"
                and text_el["whiteSpace"] == "nowrap"
                and text_el["scrollWidth"] > text_el["clientWidth"]  # Only if actually truncated
            ):
                issues.append(
                    {
                        "type": "text_truncated",
                        "text_preview": text_el["text"],
                        "scroll_width": text_el["scrollWidth"],
                        "client_width": text_el["clientWidth"],
                    }
                )

            # 5. Bounds check (element extends beyond parent)
            element_right = text_el["x"] + text_el["width"]
            element_bottom = text_el["y"] + text_el["height"]
            parent_right = text_el["parentX"] + text_el["parentWidth"]
            parent_bottom = text_el["parentY"] + text_el["parentHeight"]

            if (
                text_el["x"] < text_el["parentX"]
                or element_right > parent_right
                or text_el["y"] < text_el["parentY"]
                or element_bottom > parent_bottom
            ):
                issues.append(
                    {
                        "type": "out_of_bounds",
                        "element_bounds": {
                            "x": text_el["x"],
                            "y": text_el["y"],
                            "width": text_el["width"],
                            "height": text_el["height"],
                        },
                        "parent_bounds": {
                            "x": text_el["parentX"],
                            "y": text_el["parentY"],
                            "width": text_el["parentWidth"],
                            "height": text_el["parentHeight"],
                        },
                    }
                )

            if issues:
                violations.append(
                    {
                        "selector": text_el["selector"],
                        "element_type": "text",
                        "tag": text_el["tag"],
                        "issues": issues,
                    }
                )

        # Check media elements
        for media_el in elements_data["mediaElements"]:
            issues = []

            # 1. Overflow check (horizontal)
            if media_el["scrollWidth"] > media_el["clientWidth"]:
                issues.append(
                    {
                        "type": "horizontal_overflow",
                        "scroll_width": media_el["scrollWidth"],
                        "client_width": media_el["clientWidth"],
                        "overflow_amount": media_el["scrollWidth"] - media_el["clientWidth"],
                    }
                )

            # 2. Overflow check (vertical)
            if media_el["scrollHeight"] > media_el["clientHeight"]:
                issues.append(
                    {
                        "type": "vertical_overflow",
                        "scroll_height": media_el["scrollHeight"],
                        "client_height": media_el["clientHeight"],
                        "overflow_amount": media_el["scrollHeight"] - media_el["clientHeight"],
                    }
                )

            # 3. CSS clipping (only flag if there's actual overflow)
            if media_el["overflow"] in ["hidden", "clip"]:
                # Only flag if there's actual overflow being clipped
                has_horizontal_overflow = media_el["scrollWidth"] > media_el["clientWidth"]
                has_vertical_overflow = media_el["scrollHeight"] > media_el["clientHeight"]
                if has_horizontal_overflow or has_vertical_overflow:
                    issues.append(
                        {
                            "type": "css_clipped",
                            "overflow": media_el["overflow"],
                            "horizontal_overflow": has_horizontal_overflow,
                            "vertical_overflow": has_vertical_overflow,
                        }
                    )

            # 4. Bounds check (element extends beyond parent)
            element_right = media_el["x"] + media_el["width"]
            element_bottom = media_el["y"] + media_el["height"]
            parent_right = media_el["parentX"] + media_el["parentWidth"]
            parent_bottom = media_el["parentY"] + media_el["parentHeight"]

            if (
                media_el["x"] < media_el["parentX"]
                or element_right > parent_right
                or media_el["y"] < media_el["parentY"]
                or element_bottom > parent_bottom
            ):
                issues.append(
                    {
                        "type": "out_of_bounds",
                        "element_bounds": {
                            "x": media_el["x"],
                            "y": media_el["y"],
                            "width": media_el["width"],
                            "height": media_el["height"],
                        },
                        "parent_bounds": {
                            "x": media_el["parentX"],
                            "y": media_el["parentY"],
                            "width": media_el["parentWidth"],
                            "height": media_el["parentHeight"],
                        },
                    }
                )

            if issues:
                violations.append(
                    {
                        "selector": media_el["selector"],
                        "element_type": "media",
                        "tag": media_el["tag"],
                        "src": media_el["src"][:100],  # Truncate long URLs
                        "issues": issues,
                    }
                )

        return violations

    def _detect_overlap(self, elements_data: dict[str, list]) -> list[dict]:
        """
        Detect overlapping text and media elements.

        Filters out parent-child relationships and sibling inline elements to avoid false positives.

        Args:
            elements_data: Dict with 'textElements' and 'mediaElements' lists

        Returns:
            List of overlap violation dictionaries
        """
        violations = []

        # Combine all elements for overlap checking
        all_elements = elements_data["textElements"] + elements_data["mediaElements"]

        # Compare all pairs of elements
        for i, el1 in enumerate(all_elements):
            for el2 in all_elements[i + 1 :]:
                # Skip if one element is a parent/child of the other
                # Check if el2's selector starts with el1's selector (el2 is child of el1)
                if el2["selector"].startswith(el1["selector"] + " > "):
                    continue
                # Check if el1's selector starts with el2's selector (el1 is child of el2)
                if el1["selector"].startswith(el2["selector"] + " > "):
                    continue

                # Skip if elements are siblings (share the same parent in their selector path)
                # Extract parent path by removing the last segment
                el1_parts = el1["selector"].split(" > ")
                el2_parts = el2["selector"].split(" > ")

                # If they have the same parent path (all parts except the last), they're siblings
                if len(el1_parts) > 1 and len(el2_parts) > 1:
                    el1_parent = " > ".join(el1_parts[:-1])
                    el2_parent = " > ".join(el2_parts[:-1])
                    if el1_parent == el2_parent:
                        # They're siblings - skip overlap check
                        continue

                # Check if bounding boxes intersect
                if self._boxes_intersect(el1, el2):
                    # Calculate overlap area
                    overlap_area = self._calculate_overlap_area(el1, el2)

                    # Only report if overlap is significant (more than 10% of smaller element)
                    min_area = min(el1["width"] * el1["height"], el2["width"] * el2["height"])
                    if overlap_area > (min_area * 0.1):
                        violations.append(
                            {
                                "element1": {
                                    "selector": el1["selector"],
                                    "type": el1.get("tag", "text"),
                                    "bounds": {
                                        "x": el1["x"],
                                        "y": el1["y"],
                                        "width": el1["width"],
                                        "height": el1["height"],
                                    },
                                },
                                "element2": {
                                    "selector": el2["selector"],
                                    "type": el2.get("tag", "text"),
                                    "bounds": {
                                        "x": el2["x"],
                                        "y": el2["y"],
                                        "width": el2["width"],
                                        "height": el2["height"],
                                    },
                                },
                                "overlap_area": round(overlap_area, 2),
                                "overlap_percentage": round(
                                    (overlap_area / min_area) * 100, 2
                                ),
                            }
                        )

        return violations

    def _boxes_intersect(self, el1: dict, el2: dict) -> bool:
        """
        Check if two element bounding boxes intersect.

        Args:
            el1: First element dict with x, y, width, height
            el2: Second element dict with x, y, width, height

        Returns:
            True if boxes intersect, False otherwise
        """
        return not (
            el1["x"] + el1["width"] < el2["x"]
            or el2["x"] + el2["width"] < el1["x"]
            or el1["y"] + el1["height"] < el2["y"]
            or el2["y"] + el2["height"] < el1["y"]
        )

    def _calculate_overlap_area(self, el1: dict, el2: dict) -> float:
        """
        Calculate the area of overlap between two bounding boxes.

        Args:
            el1: First element dict with x, y, width, height
            el2: Second element dict with x, y, width, height

        Returns:
            Overlap area in pixels
        """
        # Calculate intersection rectangle
        left = max(el1["x"], el2["x"])
        right = min(el1["x"] + el1["width"], el2["x"] + el2["width"])
        top = max(el1["y"], el2["y"])
        bottom = min(el1["y"] + el1["height"], el2["y"] + el2["height"])

        if right <= left or bottom <= top:
            return 0.0

        return (right - left) * (bottom - top)
