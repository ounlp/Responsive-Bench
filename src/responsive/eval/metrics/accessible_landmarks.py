from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import Any

try:
    from playwright.async_api import async_playwright, Browser  # type: ignore[import-not-found]
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

logger = logging.getLogger("responsive.eval.metrics.accessible_landmarks")


class AccessibleLandmarksMetric(Metric):
    @property
    def name(self) -> str:
        return "accessible_landmarks"

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

                    html_source = generated_html_path.read_text(encoding="utf-8", errors="replace")

                    raw_trees: dict[str, str] = {}
                    for vp_name, vp in VIEWPORTS.items():
                        await page.set_viewport_size(vp)
                        await page.wait_for_timeout(150)
                        raw_trees[vp_name] = await aria_snapshot_yaml(page)

                    # LLM judge: use HTML + three accessibility trees to produce a single 1–5 score.
                    judge_result: dict[str, Any] | None = None
                    judge_score: int | None = None
                    judge_rationale: str | None = None
                    judge_error: str | None = None

                    if judge_cfg.get("provider") and judge_cfg.get("model"):
                        prompt_text = _build_landmarks_judge_prompt(
                            html_source=html_source,
                            tree_mobile=raw_trees.get("mobile", ""),
                            tree_tablet=raw_trees.get("tablet", ""),
                            tree_desktop=raw_trees.get("desktop", ""),
                        )

                        prompt_path = artifacts_dir / "accessible_landmarks_judge_prompt.txt"
                        response_path = artifacts_dir / "accessible_landmarks_judge_response.json"
                        prompt_path.write_text(prompt_text, encoding="utf-8")
                        artifacts["accessible_landmarks_judge_prompt"] = prompt_path

                        judge_call = call_judge(prompt_text=prompt_text, image_paths=[], judge_cfg=judge_cfg)
                        response_text = (judge_call.get("text") or "").strip()
                        response_payload = {
                            "provider": judge_call.get("provider"),
                            "model": judge_call.get("model"),
                            "latency_ms": judge_call.get("latency_ms"),
                            "usage": judge_call.get("usage"),
                            "raw": judge_call.get("raw"),
                            "response_text": response_text,
                        }
                        response_path.write_text(json.dumps(response_payload, indent=2, ensure_ascii=False), encoding="utf-8")
                        artifacts["accessible_landmarks_judge_response"] = response_path

                        try:
                            judge_json = json.loads(extract_json_object(response_text))
                            judge_result = judge_json if isinstance(judge_json, dict) else None
                            if isinstance(judge_result, dict):
                                raw_score = judge_result.get("score")
                                if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool):
                                    judge_score = int(raw_score)
                                judge_rationale = (
                                    str(judge_result.get("rationale"))
                                    if judge_result.get("rationale") is not None
                                    else None
                                )
                        except Exception as e:
                            judge_error = f"judge_returned_non_json_or_invalid: {e}"
                    else:
                        judge_error = "judge_not_configured"

                    result = {
                        "status": "success",
                        "metric": self.name,
                        "score": judge_score,
                        "rationale": judge_rationale,
                        "judge_error": judge_error,
                        "judge_raw": judge_result,
                    }

                    out_path = artifacts_dir / "accessible_landmarks.json"
                    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
                    artifacts["accessible_landmarks"] = out_path

                    return result, artifacts
                finally:
                    await browser.close()
        except Exception as e:
            logger.exception("accessible_landmarks failed")
            return (
                {
                    "status": "error",
                    "metric": self.name,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
                artifacts,
            )


def _build_landmarks_judge_prompt(
    *,
    html_source: str,
    tree_mobile: str,
    tree_tablet: str,
    tree_desktop: str,
) -> str:
    return render_judge_prompt(
        "accessible/landmarks_v1",
        {
            "html_source": html_source,
            "tree_mobile": tree_mobile,
            "tree_tablet": tree_tablet,
            "tree_desktop": tree_desktop,
        },
    )



