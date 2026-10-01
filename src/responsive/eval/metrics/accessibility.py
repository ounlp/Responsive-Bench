"""
Accessibility evaluation metric using axe-core (Deque) via Playwright.

Loads a generated HTML page at mobile viewport (375×688) and runs axe-core.
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger("responsive.eval.metrics.accessibility")

try:
    from playwright.async_api import async_playwright, Browser
except ImportError:
    async_playwright = None
    Browser = None

from .base import Metric


def _to_file_url(path: Path) -> str:
    abs_path = path.resolve()
    path_str = str(abs_path).replace("\\", "/")
    # Windows drive letter: C:/... -> /C:/...
    if len(path_str) > 1 and path_str[1] == ":":
        path_str = "/" + path_str
    return f"file://{path_str}"


_MOBILE_VIEWPORT = {"width": 375, "height": 688}


def _resolve_axe_script_path(repo_root: Path) -> Path:
    # We inject axe-core from local node_modules to avoid network requests.
    axe_path = repo_root / "node_modules" / "axe-core" / "axe.min.js"
    if not axe_path.exists():
        raise FileNotFoundError(
            f"axe-core script not found at: {axe_path}. "
            "Install dependencies with: npm install"
        )
    return axe_path


def _count_by_impact(violations: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in violations:
        impact = v.get("impact") or "unknown"
        counts[str(impact)] = counts.get(str(impact), 0) + 1
    return counts


class AccessibilityMetric(Metric):
    @property
    def name(self) -> str:
        return "accessibility"

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
        playwright_config: dict[str, Any] = context["playwright"]
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # accessibility.py -> metrics/ -> eval/ -> responsive/ -> src/ -> repo_root
        repo_root = Path(__file__).resolve().parents[4]
        axe_script_path = _resolve_axe_script_path(repo_root)
        file_url = _to_file_url(generated_html_path)

        vp = _MOBILE_VIEWPORT
        artifacts: dict[str, Path] = {}

        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(
                    headless=playwright_config.get("headless", True)
                )
                try:
                    context_obj = await browser.new_context(
                        viewport={"width": vp["width"], "height": vp["height"]}
                    )
                    context_obj.set_default_timeout(
                        playwright_config.get("timeout_ms", 30000)
                    )
                    page = await context_obj.new_page()
                    await page.goto(file_url, wait_until="networkidle")
                    await page.wait_for_timeout(200)

                    await page.add_script_tag(path=str(axe_script_path))
                    axe_results = await page.evaluate(
                        """
                        async () => {
                          if (!globalThis.axe) {
                            throw new Error("axe not found on page (script injection failed)");
                          }
                          return await globalThis.axe.run(document);
                        }
                        """
                    )
                finally:
                    await browser.close()

        except Exception as e:
            raise RuntimeError(
                f"Accessibility metric failed for {generated_html_path}: {e}\n"
                f"Traceback: {traceback.format_exc()}"
            ) from e

        artifact_path = artifacts_dir / "axe_mobile.json"
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(axe_results, f, indent=2)
        artifacts["mobile"] = artifact_path

        violations = axe_results.get("violations", []) or []
        violations_count = len(violations)
        passed = violations_count == 0

        result: dict[str, Any] = {
            "status": "success",
            "html_file": generated_html_path.name,
            "passed": bool(passed),
            "total_violations_count": int(violations_count),
            "viewports": {
                "mobile": {
                    "viewport": {"width": vp["width"], "height": vp["height"]},
                    "violations": violations,
                    "violations_count": int(violations_count),
                    "violations_by_impact": _count_by_impact(violations),
                }
            },
        }

        return result, artifacts

