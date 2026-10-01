"""
Relative units evaluation metric using Playwright.

Checks if containers use relative units (%, em, rem, vw, vh) instead of absolute units (px).
"""

import logging
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger("responsive.eval.metrics.relative_units")

try:
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page
except ImportError:
    async_playwright = None
    Browser = None
    BrowserContext = None
    Page = None

from .base import Metric
from .html_constants import VIEWPORTS


class RelativeUnitsMetric(Metric):
    """Metric that checks if containers use relative units for responsive design."""

    @property
    def name(self) -> str:
        """Return metric name."""
        return "relative_units"

    @property
    def requires_browser(self) -> bool:
        """Relative units check requires a browser."""
        return True

    async def run(
        self, sample: Any, generated_html_path: Path, context: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Path]]:
        """
        Run relative units evaluation.

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
            logger.debug(f"Starting relative units evaluation for: {generated_html_path}")
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

                # Check all container elements for absolute units
                results = await page.evaluate(
                    """
                    () => {
                        // Properties to check for absolute units
                        // Note: padding/margin are excluded since px is acceptable for spacing
                        const propertiesToCheck = [
                            'width', 'maxWidth', 'minWidth',
                            'height', 'maxHeight', 'minHeight',
                            'fontSize', 'lineHeight'
                        ];

                        // Relative units that are good for responsive design
                        const relativeUnits = ['%', 'em', 'rem', 'vw', 'vh', 'vmin', 'vmax', 'ch', 'ex'];

                        // Absolute units to avoid
                        const absoluteUnits = ['px', 'pt', 'cm', 'mm', 'in', 'pc'];

                        function isVisible(el) {
                            const style = getComputedStyle(el);
                            if (style.display === 'none' || style.visibility === 'hidden') return false;
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

                        function checkUnit(value) {
                            if (!value || value === 'auto' || value === 'none' || value === 'normal' || value === '0px') {
                                return { type: 'auto', isAbsolute: false };
                            }

                            // Check for absolute units
                            for (const unit of absoluteUnits) {
                                if (value.includes(unit)) {
                                    return { type: 'absolute', unit: unit, value: value, isAbsolute: true };
                                }
                            }

                            // Check for relative units
                            for (const unit of relativeUnits) {
                                if (value.includes(unit)) {
                                    return { type: 'relative', unit: unit, value: value, isAbsolute: false };
                                }
                            }

                            // Unitless values (like line-height: 1.5)
                            if (!isNaN(parseFloat(value)) && value === parseFloat(value).toString()) {
                                return { type: 'unitless', value: value, isAbsolute: false };
                            }

                            return { type: 'unknown', value: value, isAbsolute: false };
                        }

                        function getInlineStyleValue(el, property) {
                            const inlineStyle = el.getAttribute('style') || '';
                            const regex = new RegExp(property + '\\\\s*:\\\\s*([^;]+)', 'i');
                            const match = inlineStyle.match(regex);
                            return match ? match[1].trim() : null;
                        }

                        function getStylesheetValue(el, property) {
                            // Get CSS property value from stylesheets (not computed style)
                            // This checks all stylesheets including <style> tags
                            try {
                                const styleSheets = Array.from(document.styleSheets);
                                const propertyName = property.replace(/([A-Z])/g, '-$1').toLowerCase();
                                
                                // Collect all matching rules (to handle CSS cascade)
                                const matchingRules = [];
                                
                                // Try to match element against all CSS rules
                                for (const sheet of styleSheets) {
                                    try {
                                        const rules = Array.from(sheet.cssRules || sheet.rules || []);
                                        for (const rule of rules) {
                                            if (rule.style && rule.selectorText) {
                                                try {
                                                    // Check if selector matches this element
                                                    if (el.matches && el.matches(rule.selectorText)) {
                                                        const value = rule.style[property] || rule.style.getPropertyValue(propertyName);
                                                        if (value && value.trim() && value !== 'initial' && value !== 'inherit' && value !== 'unset') {
                                                            matchingRules.push({
                                                                value: value.trim(),
                                                                selector: rule.selectorText,
                                                                sheetIndex: styleSheets.indexOf(sheet),
                                                                ruleIndex: rules.indexOf(rule)
                                                            });
                                                        }
                                                    }
                                                } catch (e) {
                                                    // Skip invalid selectors
                                                    continue;
                                                }
                                            }
                                        }
                                    } catch (e) {
                                        // Skip stylesheets that can't be accessed (CORS, etc.)
                                        continue;
                                    }
                                }
                                
                                // Return the last matching rule (CSS cascade: later rules override earlier ones)
                                if (matchingRules.length > 0) {
                                    return matchingRules[matchingRules.length - 1].value;
                                }
                            } catch (e) {
                                // Fallback: stylesheet access failed
                            }
                            return null;
                        }

                        function getCSSValue(el, property) {
                            // Check inline style first (higher specificity)
                            const inlineValue = getInlineStyleValue(el, property);
                            if (inlineValue) {
                                return { value: inlineValue, source: 'inline' };
                            }
                            
                            // Then check stylesheet
                            const stylesheetValue = getStylesheetValue(el, property);
                            if (stylesheetValue) {
                                return { value: stylesheetValue, source: 'stylesheet' };
                            }
                            
                            return null;
                        }

                        function isJustWrapperParent(el) {
                            // Check if container is just a wrapper (only contains other containers, no direct content)
                            // Get direct children that are container elements
                            const directContainerChildren = Array.from(el.children).filter(child => {
                                const tag = child.tagName.toLowerCase();
                                return ['div', 'section', 'article', 'aside', 'header', 'footer', 'main', 'nav'].includes(tag);
                            });
                            
                            // Check for direct text content (not in nested containers)
                            let hasDirectText = false;
                            for (let child of el.childNodes) {
                                if (child.nodeType === Node.TEXT_NODE && child.textContent.trim()) {
                                    hasDirectText = true;
                                    break;
                                }
                            }
                            
                            // Check for direct media elements (not in nested containers)
                            const directMedia = el.querySelector(':scope > img, :scope > video');
                            
                            // If it has direct container children but no direct text/media, it's likely just a wrapper
                            if (directContainerChildren.length > 0 && !hasDirectText && !directMedia) {
                                return true;
                            }
                            
                            return false;
                        }

                        const violations = [];

                        // Check all container elements (divs, sections, articles, etc.)
                        const containerSelectors = ['div', 'section', 'article', 'aside', 'header', 'footer', 'main', 'nav'];
                        const containers = document.querySelectorAll(containerSelectors.join(','));

                        containers.forEach(el => {
                            if (!isVisible(el)) return;
                            if (!hasContent(el)) return;  // Skip empty containers
                            if (isJustWrapperParent(el)) return;  // Skip parent containers that are just wrappers

                            const elementViolations = [];

                            // Check both inline styles and stylesheet rules
                            // Computed styles are always in px, so we check original CSS source
                            propertiesToCheck.forEach(prop => {
                                const cssValue = getCSSValue(el, prop);
                                if (cssValue) {
                                    const unitCheck = checkUnit(cssValue.value);
                                    if (unitCheck.isAbsolute) {
                                        elementViolations.push({
                                            property: prop,
                                            value: cssValue.value,
                                            unit: unitCheck.unit,
                                            source: cssValue.source
                                        });
                                    }
                                }
                            });

                            if (elementViolations.length > 0) {
                                const text = el.innerText ? el.innerText.trim().slice(0, 80) : '';
                                violations.push({
                                    tag: el.tagName.toLowerCase(),
                                    selector: cssPath(el),
                                    violations: elementViolations,
                                    text: text,
                                    className: el.className || '',
                                    id: el.id || ''
                                });
                            }
                        });

                        return violations;
                    }
                    """
                )

                logger.debug(f"Found {len(results)} containers with violations")

        except Exception as e:
            raise RuntimeError(
                f"Relative units metric failed for {generated_html_path}: {e}\n"
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
        total_containers_checked = len(results)
        containers_with_violations = len(results)
        total_violations = sum(len(r["violations"]) for r in results)

        logger.info(
            f"Relative units: {total_containers_checked} containers checked, "
            f"{total_violations} violations"
        )

        # Build result dict
        result = {
            "status": "success",
            "html_file": generated_html_path.name,
            "total_containers_checked": total_containers_checked,
            "containers_with_violations": containers_with_violations,
            "total_violations": total_violations,
            "violations": results,
            "passed": total_violations == 0,
            "recommendation": "Use relative units (%, em, rem, vw, vh) instead of absolute units (px, pt) for responsive design",
        }

        # Build artifacts dict (no artifacts for this metric)
        artifacts: dict[str, Path] = {}

        return result, artifacts
