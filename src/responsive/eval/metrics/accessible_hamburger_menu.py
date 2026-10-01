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

from .a11y_helpers import (
    VIEWPORTS,
    aria_snapshot_yaml,
    prefer_default_playwright_browsers_path,
    to_file_url,
)
from .base import Metric
from ..judge_client import call_judge, extract_json_object
from ..prompts.loader import render_judge_prompt

logger = logging.getLogger("responsive.eval.metrics.accessible_hamburger_menu")


class AccessibleHamburgerMenuMetric(Metric):
    @property
    def name(self) -> str:
        return "accessible_hamburger_menu"

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
        judge_cfg: dict[str, Any] = context.get("judge") or {}

        artifacts: dict[str, Path] = {}

        try:
            html_source = generated_html_path.read_text(encoding="utf-8", errors="replace")

            # --- Phase 1: LLM pre-check — does a hamburger menu exist? ---
            precheck_result: dict[str, Any] | None = None
            applicable = False
            selector: str | None = None
            precheck_error: str | None = None

            if judge_cfg.get("provider") and judge_cfg.get("model"):
                precheck_prompt = _build_precheck_prompt(html_source)

                precheck_prompt_path = artifacts_dir / "hamburger_precheck_prompt.txt"
                precheck_prompt_path.write_text(precheck_prompt, encoding="utf-8")
                artifacts["hamburger_precheck_prompt"] = precheck_prompt_path

                precheck_call = call_judge(
                    prompt_text=precheck_prompt, image_paths=[], judge_cfg=judge_cfg
                )
                precheck_text = (precheck_call.get("text") or "").strip()

                precheck_response_path = artifacts_dir / "hamburger_precheck_response.json"
                precheck_response_path.write_text(
                    json.dumps(
                        {
                            "provider": precheck_call.get("provider"),
                            "model": precheck_call.get("model"),
                            "latency_ms": precheck_call.get("latency_ms"),
                            "usage": precheck_call.get("usage"),
                            "response_text": precheck_text,
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                artifacts["hamburger_precheck_response"] = precheck_response_path

                try:
                    precheck_json = json.loads(extract_json_object(precheck_text))
                    if isinstance(precheck_json, dict):
                        precheck_result = precheck_json
                        applicable = bool(precheck_json.get("applicable", False))
                        selector = precheck_json.get("selector") or None
                except Exception as exc:
                    precheck_error = f"precheck_returned_non_json: {exc}"
            else:
                precheck_error = "judge_not_configured"

            # Not applicable — return early without running Playwright.
            if not applicable:
                result: dict[str, Any] = {
                    "status": "success",
                    "metric": self.name,
                    "score": None,
                    "rationale": None,
                    "reason": "not_applicable",
                    "precheck": precheck_result,
                    "precheck_error": precheck_error,
                }
                out_path = artifacts_dir / "accessible_hamburger_menu.json"
                out_path.write_text(
                    json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                artifacts["accessible_hamburger_menu"] = out_path
                return result, artifacts

            # --- Phase 2: Playwright — capture accessibility trees before and after click ---
            file_url = to_file_url(generated_html_path)
            prefer_default_playwright_browsers_path()

            tree_closed = ""
            tree_open = ""
            click_error: str | None = None

            async with async_playwright() as p:
                browser: Browser = await p.chromium.launch(
                    headless=playwright_config.get("headless", True)
                )
                try:
                    page = await browser.new_page(
                        viewport={
                            "width": VIEWPORTS["mobile"]["width"],
                            "height": VIEWPORTS["mobile"]["height"],
                        }
                    )
                    page.set_default_timeout(playwright_config.get("timeout_ms", 30000))
                    await page.goto(file_url)
                    await page.wait_for_load_state("networkidle")
                    await page.wait_for_timeout(150)

                    tree_closed = await aria_snapshot_yaml(page)

                    btn = page.locator(selector).first  # type: ignore[arg-type]
                    if await btn.count() == 0:
                        click_error = f"selector_not_found: {selector}"
                    else:
                        await btn.click()
                        await page.wait_for_timeout(300)
                        tree_open = await aria_snapshot_yaml(page)
                finally:
                    await browser.close()

            # --- Phase 3: LLM judge — score the accessibility of the interaction ---
            judge_result: dict[str, Any] | None = None
            judge_score: int | None = None
            judge_rationale: str | None = None
            judge_error: str | None = None

            if click_error:
                judge_error = click_error
            elif judge_cfg.get("provider") and judge_cfg.get("model"):
                judge_prompt = _build_judge_prompt(
                    html_source=html_source,
                    tree_closed=tree_closed,
                    tree_open=tree_open,
                )

                judge_prompt_path = artifacts_dir / "hamburger_judge_prompt.txt"
                judge_prompt_path.write_text(judge_prompt, encoding="utf-8")
                artifacts["hamburger_judge_prompt"] = judge_prompt_path

                judge_call = call_judge(
                    prompt_text=judge_prompt, image_paths=[], judge_cfg=judge_cfg
                )
                response_text = (judge_call.get("text") or "").strip()

                judge_response_path = artifacts_dir / "hamburger_judge_response.json"
                judge_response_path.write_text(
                    json.dumps(
                        {
                            "provider": judge_call.get("provider"),
                            "model": judge_call.get("model"),
                            "latency_ms": judge_call.get("latency_ms"),
                            "usage": judge_call.get("usage"),
                            "response_text": response_text,
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                artifacts["hamburger_judge_response"] = judge_response_path

                try:
                    judge_json = json.loads(extract_json_object(response_text))
                    if isinstance(judge_json, dict):
                        judge_result = judge_json
                        raw_score = judge_result.get("score")
                        if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool):
                            judge_score = int(raw_score)
                        judge_rationale = (
                            str(judge_result.get("rationale"))
                            if judge_result.get("rationale") is not None
                            else None
                        )
                except Exception as exc:
                    judge_error = f"judge_returned_non_json: {exc}"
            else:
                judge_error = "judge_not_configured"

            result = {
                "status": "success",
                "metric": self.name,
                "score": judge_score,
                "rationale": judge_rationale,
                "judge_error": judge_error,
                "judge_raw": judge_result,
                "precheck": precheck_result,
                "precheck_error": precheck_error,
                "selector_used": selector,
            }
            out_path = artifacts_dir / "accessible_hamburger_menu.json"
            out_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            artifacts["accessible_hamburger_menu"] = out_path
            return result, artifacts

        except Exception as e:
            logger.exception("accessible_hamburger_menu failed")
            return (
                {
                    "status": "error",
                    "metric": self.name,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
                artifacts,
            )


def _build_precheck_prompt(html_source: str) -> str:
    return render_judge_prompt(
        "accessible/hamburger_menu_precheck_v1",
        {"html_source": html_source},
    )


def _build_judge_prompt(*, html_source: str, tree_closed: str, tree_open: str) -> str:
    return render_judge_prompt(
        "accessible/hamburger_menu_judge_v1",
        {"html_source": html_source, "tree_closed": tree_closed, "tree_open": tree_open},
    )


