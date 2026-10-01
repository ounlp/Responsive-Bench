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

logger = logging.getLogger("responsive.eval.metrics.accessible_no_keyboard_trap")


class AccessibleNoKeyboardTrapMetric(Metric):
    @property
    def name(self) -> str:
        return "accessible_no_keyboard_trap"

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
                    vp = VIEWPORTS["desktop"]
                    page = await browser.new_page(
                        viewport={"width": vp["width"], "height": vp["height"]}
                    )
                    page.set_default_timeout(playwright_config.get("timeout_ms", 30000))
                    await page.goto(file_url)
                    await page.wait_for_load_state("networkidle")
                    await page.wait_for_timeout(200)

                    interactive_count = await page.locator(
                        "a[href], button:not([disabled]), input, select, textarea, [role='button'], [role='link']"
                    ).count()

                    if interactive_count < 3:
                        viewport_results: dict[str, Any] = {
                            "desktop": {
                                "score": None,
                                "reason": f"only_{interactive_count}_interactive_elements",
                            }
                        }
                    else:
                        positive = await page.evaluate(
                            """() => [...document.querySelectorAll('[tabindex]')]
                              .filter(el => {
                                const v = parseInt(el.getAttribute('tabindex') || '', 10);
                                return Number.isFinite(v) && v > 0;
                              })
                              .map(el => (el.tagName.toLowerCase()) + (el.id ? ('#' + el.id) : ''))"""
                        )
                        positive = positive or []

                        max_tabs = min(max(int(interactive_count) * 2, 40), 50)
                        seq: list[str] = []
                        await page.keyboard.press("Tab")
                        for _ in range(max_tabs):
                            sig = await page.evaluate(
                                """() => {
                                  const el = document.activeElement;
                                  if (!el) return 'null';
                                  const id = el.id ? ('#' + el.id) : '';
                                  const role = el.getAttribute && el.getAttribute('role') ? ('[role=' + el.getAttribute('role') + ']') : '';
                                  return el.tagName.toLowerCase() + id + role;
                                }"""
                            )
                            seq.append(str(sig))
                            await page.keyboard.press("Tab")

                        trap_detected = False
                        run = 1
                        for i in range(1, len(seq)):
                            if seq[i] == seq[i - 1]:
                                run += 1
                                if run >= 3:
                                    trap_detected = True
                                    break
                            else:
                                run = 1

                        before_shift = await page.evaluate(
                            """() => {
                              const el = document.activeElement;
                              if (!el) return 'null';
                              const id = el.id ? ('#' + el.id) : '';
                              return el.tagName.toLowerCase() + id;
                            }"""
                        )
                        await page.keyboard.press("Shift+Tab")
                        await page.wait_for_timeout(50)
                        after_shift = await page.evaluate(
                            """() => {
                              const el = document.activeElement;
                              if (!el) return 'null';
                              const id = el.id ? ('#' + el.id) : '';
                              return el.tagName.toLowerCase() + id;
                            }"""
                        )
                        shift_tab_moves = str(after_shift) != str(before_shift)

                        # Escape check: dropdown trigger (desktop pattern)
                        escape_checks: dict[str, Any] = {}
                        trigger = page.locator("button[aria-haspopup][aria-expanded]").first
                        if await trigger.count() > 0:
                            trig_id = await trigger.get_attribute("id")
                            was_open = (await trigger.get_attribute("aria-expanded")) == "true"
                            if not was_open:
                                await trigger.click()
                                await page.wait_for_timeout(150)
                            await page.keyboard.press("Escape")
                            await page.wait_for_timeout(150)
                            expanded_after = await trigger.get_attribute("aria-expanded")
                            focused_id = await page.evaluate(
                                "document.activeElement && document.activeElement.id"
                            )
                            escape_checks = {
                                "escape_closes_popover": expanded_after == "false",
                                "focus_returns_to_trigger": (focused_id == trig_id) if trig_id else True,
                            }

                        last_is_body = await page.evaluate(
                            "document.activeElement === document.body || document.activeElement === null"
                        )

                        checks: dict[str, Any] = {
                            "no_positive_tabindex": len(positive) == 0,
                            "no_tab_trap_detected": not trap_detected,
                            "tab_exits_to_body_or_null": bool(last_is_body),
                            "shift_tab_moves_focus": bool(shift_tab_moves),
                            **escape_checks,
                        }

                        failures = [k for k, v in checks.items() if v is False]
                        if not failures:
                            score = 5
                        elif trap_detected:
                            score = 1
                        elif not checks["no_positive_tabindex"]:
                            score = 2
                        elif any(k.startswith("escape_") for k in failures) or any(
                            k.startswith("focus_returns_") for k in failures
                        ):
                            score = 3
                        elif len(failures) <= 2:
                            score = 3
                        else:
                            score = 2

                        viewport_results = {
                            "desktop": {
                                "score": score,
                                "checks": checks,
                                "failures": failures,
                                "positive_tabindex_elements": positive,
                                "tab_sequence_sample": seq[:20],
                            }
                        }

                    result = {"status": "success", "metric": self.name, "viewport_results": viewport_results}
                    out_path = artifacts_dir / "accessible_no_keyboard_trap.json"
                    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
                    artifacts["accessible_no_keyboard_trap"] = out_path
                    return result, artifacts
                finally:
                    await browser.close()
        except Exception as e:
            logger.exception("accessible_no_keyboard_trap failed")
            return (
                {
                    "status": "error",
                    "metric": self.name,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
                artifacts,
            )

