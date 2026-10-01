from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import Any

try:
    from playwright.async_api import async_playwright, Browser
except ImportError:  # pragma: no cover
    async_playwright = None
    Browser = None

from .a11y_helpers import VIEWPORTS, prefer_default_playwright_browsers_path, to_file_url
from .base import Metric

logger = logging.getLogger("responsive.eval.metrics.accessible_form_labels")


class AccessibleFormLabelsMetric(Metric):
    @property
    def name(self) -> str:
        return "accessible_form_labels"

    @property
    def requires_browser(self) -> bool:
        return True

    async def run(
        self, sample: Any, generated_html_path: Path, context: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Path]]:
        if async_playwright is None:
            raise ImportError(
                "playwright not installed. Install with: pip install playwright && playwright install"
            )
        if not generated_html_path.exists():
            raise FileNotFoundError(f"Generated HTML not found: {generated_html_path}")

        artifacts_dir: Path = context["artifacts_dir"]
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        playwright_config: dict[str, Any] = context.get("playwright") or {}

        artifacts: dict[str, Path] = {}

        try:
            file_url = to_file_url(generated_html_path)
            prefer_default_playwright_browsers_path()
            async with async_playwright() as p:
                browser: Browser = await p.chromium.launch(
                    headless=playwright_config.get("headless", True)
                )
                try:
                    page = await browser.new_page(
                        viewport={"width": VIEWPORTS["mobile"]["width"], "height": VIEWPORTS["mobile"]["height"]}
                    )
                    page.set_default_timeout(playwright_config.get("timeout_ms", 30000))
                    await page.goto(file_url)
                    await page.wait_for_load_state("networkidle")
                    await page.wait_for_timeout(150)

                    checks = await page.evaluate(
                            """() => {
                              const controls = Array.from(document.querySelectorAll('input, textarea, select'))
                                .filter(el => {
                                  const tag = el.tagName.toLowerCase();
                                  if (tag === 'input') {
                                    const t = (el.getAttribute('type') || 'text').toLowerCase();
                                    return !['hidden','submit','button','reset','image','file'].includes(t);
                                  }
                                  return true;
                                });

                              if (controls.length === 0) return { _null: true, reason: 'no_form_controls_found' };

                              const textOf = (el) => (el && el.textContent ? el.textContent : '').trim();
                              const byIdText = (id) => {
                                const el = id ? document.getElementById(id) : null;
                                return textOf(el);
                              };

                              const labelFor = (el) => {
                                if (!el) return null;
                                if (el.id) {
                                  const l = document.querySelector('label[for=\"' + CSS.escape(el.id) + '\"]');
                                  if (l) return l;
                                }
                                return el.closest('label');
                              };

                              const accessibleName = (el) => {
                                const l = labelFor(el);
                                if (l) return textOf(l);
                                const ariaLabel = (el.getAttribute('aria-label') || '').trim();
                                if (ariaLabel) return ariaLabel;
                                const labelledby = (el.getAttribute('aria-labelledby') || '').trim();
                                if (labelledby) {
                                  const parts = labelledby.split(/\\s+/).map(byIdText).filter(Boolean);
                                  if (parts.length) return parts.join(' ');
                                }
                                return '';
                              };

                              const hasHelperTextNearby = (el) => {
                                if (!el || !el.parentElement) return false;
                                const scope = el.closest('.field, .form-field, .input, .control') || el.parentElement;
                                const helpers = scope.querySelectorAll('[role=\"status\"], .help, .helper, .hint, .error, .form-text, small');
                                for (const h of helpers) {
                                  const t = textOf(h);
                                  if (t && t.length >= 3) return true;
                                }
                                return false;
                              };

                              const describedbyOk = (el) => {
                                const d = (el.getAttribute('aria-describedby') || '').trim();
                                if (!d) return !hasHelperTextNearby(el);
                                const ids = d.split(/\\s+/).filter(Boolean);
                                if (!ids.length) return !hasHelperTextNearby(el);
                                return ids.every(id => !!document.getElementById(id));
                              };

                              const ariaInvalidPreset = (el) => (el.getAttribute('aria-invalid') || '').toLowerCase() === 'true';

                              const isRequiredSemantics = (el) => el.hasAttribute('required') || (el.getAttribute('aria-required') || '').toLowerCase() === 'true';

                              const labelIndicatesRequired = (el) => {
                                const l = labelFor(el);
                                const t = textOf(l);
                                return t.includes('*');
                              };

                              const groupLabelOk = (el) => {
                                const type = (el.getAttribute('type') || '').toLowerCase();
                                if (type !== 'radio' && type !== 'checkbox') return true;
                                const fs = el.closest('fieldset');
                                if (fs) {
                                  const legend = fs.querySelector('legend');
                                  if (legend && textOf(legend)) return true;
                                }
                                const grp = el.closest('[role=\"group\"]');
                                if (grp) {
                                  const lb = (grp.getAttribute('aria-labelledby') || '').trim();
                                  if (lb) {
                                    const parts = lb.split(/\\s+/).map(byIdText).filter(Boolean);
                                    if (parts.length) return true;
                                  }
                                }
                                return false;
                              };

                              let unlabeled = 0;
                              let placeholderOnly = 0;
                              let requiredMarkerMissing = 0;
                              let describedbyMissing = 0;
                              let groupMissing = 0;
                              let ariaInvalidOnLoad = 0;
                              let passedCount = 0;

                              const examples = {
                                unlabeled: [],
                                placeholder_only: [],
                                required_marker_missing: [],
                                describedby_missing: [],
                                group_missing: [],
                                aria_invalid_on_load: [],
                              };

                              controls.forEach((el) => {
                                const tag = el.tagName.toLowerCase();
                                const type = tag === 'input' ? ((el.getAttribute('type') || 'text').toLowerCase()) : tag;
                                const name = accessibleName(el);
                                const ph = (el.getAttribute('placeholder') || '').trim();

                                const sig = tag + (el.id ? ('#' + el.id) : '') + (el.getAttribute('name') ? ('[name=' + el.getAttribute('name') + ']') : '') + (type ? (':' + type) : '');

                                let controlPasses = true;

                                if (!name) {
                                  controlPasses = false;
                                  unlabeled += 1;
                                  if (examples.unlabeled.length < 3) examples.unlabeled.push(sig);
                                  if (ph) {
                                    placeholderOnly += 1;
                                    if (examples.placeholder_only.length < 3) examples.placeholder_only.push(sig);
                                  }
                                }

                                if (labelIndicatesRequired(el) && !isRequiredSemantics(el)) {
                                  controlPasses = false;
                                  requiredMarkerMissing += 1;
                                  if (examples.required_marker_missing.length < 3) examples.required_marker_missing.push(sig);
                                }

                                if (!describedbyOk(el)) {
                                  controlPasses = false;
                                  describedbyMissing += 1;
                                  if (examples.describedby_missing.length < 3) examples.describedby_missing.push(sig);
                                }

                                if (!groupLabelOk(el)) {
                                  controlPasses = false;
                                  groupMissing += 1;
                                  if (examples.group_missing.length < 3) examples.group_missing.push(sig);
                                }

                                if (ariaInvalidPreset(el)) {
                                  controlPasses = false;
                                  ariaInvalidOnLoad += 1;
                                  if (examples.aria_invalid_on_load.length < 3) examples.aria_invalid_on_load.push(sig);
                                }

                                if (controlPasses) passedCount += 1;
                              });

                              return {
                                control_count: controls.length,
                                passed_count: passedCount,
                                unlabeled_count: unlabeled,
                                placeholder_only_count: placeholderOnly,
                                required_marker_missing_count: requiredMarkerMissing,
                                describedby_missing_count: describedbyMissing,
                                group_label_missing_count: groupMissing,
                                aria_invalid_on_load_count: ariaInvalidOnLoad,
                                examples,
                              };
                            }"""
                        )

                    if checks.get("_null"):
                        mobile_result: dict[str, Any] = {"score": None, "reason": checks.get("reason")}
                    else:
                        total = int(checks.get("control_count") or 0)
                        passed = int(checks.get("passed_count") or 0)
                        score = round(passed / total, 4) if total > 0 else None

                        failures: list[str] = []
                        if int(checks.get("unlabeled_count") or 0) > 0:
                            failures.append("inputs_missing_accessible_label")
                        if int(checks.get("placeholder_only_count") or 0) > 0:
                            failures.append("placeholder_used_as_only_label")
                        if int(checks.get("required_marker_missing_count") or 0) > 0:
                            failures.append("required_fields_not_marked_programmatically")
                        if int(checks.get("describedby_missing_count") or 0) > 0:
                            failures.append("helper_text_not_linked_via_aria_describedby")
                        if int(checks.get("group_label_missing_count") or 0) > 0:
                            failures.append("radio_checkbox_group_missing_group_label")
                        if int(checks.get("aria_invalid_on_load_count") or 0) > 0:
                            failures.append("aria_invalid_true_on_load")

                        mobile_result = {
                            "score": score,
                            "passed": passed,
                            "checked": total,
                            "checks": checks,
                            "failures": failures,
                        }

                    result = {"status": "success", "metric": self.name, "viewport_results": {"mobile": mobile_result}}
                    out_path = artifacts_dir / "accessible_form_labels.json"
                    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
                    artifacts["accessible_form_labels"] = out_path
                    return result, artifacts
                finally:
                    await browser.close()
        except Exception as e:
            logger.exception("accessible_form_labels failed")
            return (
                {
                    "status": "error",
                    "metric": self.name,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
                artifacts,
            )

