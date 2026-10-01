"""
LLM-as-a-Judge evaluation metric using screenshots.

This metric uses an LLM judge (OpenAI/Claude) to evaluate qualitative responsiveness
and layout fidelity by analyzing rendered screenshots at multiple viewports.
"""

import asyncio
import json
import logging
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger("responsive.eval.metrics.llm_judge_screenshots")

try:
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page
except ImportError:
    async_playwright = None
    Browser = None
    BrowserContext = None
    Page = None

try:
    from PIL import Image
except ImportError:
    Image = None

from .base import Metric
from ..playwright_screenshots import (
    screenshot_full_height_clipped,
    screenshot_full_page_bordered,
)
from .llm_judge_prompts import (
    JUDGE_DIMENSIONS,
    JUDGE_DIMENSION_KEYS,
    SCHEMA_SCORE_1_5,
    SCHEMA_MULTI_SCORE_1_5,
    _format_unified_rubric_guidance_markdown,
)
from ..judge_client import call_judge, extract_json_object
from ..prompts.loader import render_judge_prompt


class LLMJudgeScreenshotsMetric(Metric):
    """Metric that uses an LLM judge to evaluate screenshots."""

    # Viewport configurations for mobile, tablet, desktop
    VIEWPORTS = {
        "mobile": {"width": 375, "height": 800},
        "tablet": {"width": 768, "height": 1024},
        "desktop": {"width": 1280, "height": 720},
    }

    @property
    def name(self) -> str:
        """Return metric name."""
        return "llm_judge_screenshots"

    @property
    def requires_browser(self) -> bool:
        """LLM judge screenshots requires a browser to render screenshots."""
        return True

    def _flatten_sketch_for_judge(self, src: Path, dst: Path) -> Path:
        """
        Sketch PNGs in this repo are often black strokes on a transparent background.
        Flatten them onto white so the judge model can perceive the lines.
        """
        if Image is None:
            raise ImportError("Pillow is required to preprocess sketch images for the judge")

        img = Image.open(src).convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.alpha_composite(img)
        out = bg.convert("RGB")
        dst.parent.mkdir(parents=True, exist_ok=True)
        out.save(dst, format="PNG")
        return dst

    def _validate_judge_config(self, judge_config: dict[str, Any]) -> tuple[str | None, str | None]:
        provider = judge_config.get("provider")
        model = judge_config.get("model")
        if not provider or not model:
            return None, None
        from ...providers.registry import _PROVIDER_CLASSES
        if provider not in _PROVIDER_CLASSES:
            available = ", ".join(sorted(_PROVIDER_CLASSES.keys()))
            raise ValueError(f"Unsupported judge provider: '{provider}'. Available: {available}")
        return str(provider), str(model)

    async def run(
        self, sample: Any, generated_html_path: Path, context: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Path]]:
        """
        Run LLM judge evaluation.

        Args:
            sample: Sample object from dataset (may contain sketch paths)
            generated_html_path: Path to generated HTML file
            context: Context dict with viewport, playwright config, artifacts_dir, and judge config

        Returns:
            Tuple of (result_dict, artifacts_dict)

        Raises:
            ImportError: If playwright or openai is not installed
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
        judge_config = context.get("judge", {})
        sketches = context.get("sketches") or {}

        # Ensure artifacts directory exists
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Convert to absolute path and file:// URL
        abs_path = generated_html_path.resolve()
        # Use three slashes for cross-platform compatibility (Windows needs file:///)
        path_str = str(abs_path).replace("\\", "/")
        # Add leading slash if on Windows (C:/ -> /C:/)
        if path_str[1] == ":":
            path_str = "/" + path_str
        file_url = f"file://{path_str}"

        # Define screenshot paths
        screenshot_paths = {
            "mobile": artifacts_dir / "render_mobile.png",
            "tablet": artifacts_dir / "render_tablet.png",
            "desktop": artifacts_dir / "render_desktop.png",
        }

        # Render screenshots if they don't exist
        screenshots_to_render = {
            name: path
            for name, path in screenshot_paths.items()
            if not path.exists()
        }

        if screenshots_to_render:
            logger.info(f"Rendering {len(screenshots_to_render)} missing screenshots: {list(screenshots_to_render.keys())}")
        else:
            logger.debug("All screenshots already exist, skipping rendering")

        browser: Browser | None = None
        context_obj: BrowserContext | None = None
        page: Page | None = None

        try:
            if screenshots_to_render:
                logger.debug(f"Starting browser for screenshot rendering")
                async with async_playwright() as p:
                    # Launch browser
                    browser = await p.chromium.launch(
                        headless=playwright_config.get("headless", True)
                    )

                    # Create context (viewport will be set per page)
                    context_obj = await browser.new_context()
                    context_obj.set_default_timeout(
                        playwright_config.get("timeout_ms", 60000)
                    )

                    # Render each missing screenshot
                    for viewport_name, screenshot_path in screenshots_to_render.items():
                        viewport_size = self.VIEWPORTS[viewport_name]
                        logger.debug(f"Rendering {viewport_name} screenshot ({viewport_size['width']}x{viewport_size['height']})")

                        # Create new page with specific viewport
                        page = await context_obj.new_page()
                        await page.set_viewport_size({
                            "width": viewport_size["width"],
                            "height": viewport_size["height"]
                        })

                        # Navigate to HTML file
                        await page.goto(file_url, wait_until="networkidle")

                        # Wait a bit for any dynamic content
                        await page.wait_for_timeout(500)

                        await screenshot_full_page_bordered(
                            page,
                            viewport_width=viewport_size["width"],
                            output_path=screenshot_path,
                            timeout_ms=playwright_config.get("timeout_ms", 60000),
                            animations="disabled",
                            border_rgba=(255, 0, 0, 128),
                            border_width_px=4,
                        )
                        logger.debug(f"Saved {viewport_name} screenshot: {screenshot_path}")

                        # Close page
                        await page.close()
                        page = None

        except Exception as e:
            # Re-raise with more context
            raise RuntimeError(
                f"LLM judge screenshots metric failed for {generated_html_path}: {e}\n"
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

        # Get sample directory (parent of artifacts_dir)
        sample_dir = artifacts_dir.parent

        # Validate judge config (or run in stub mode if not configured)
        try:
            judge_provider, judge_model = self._validate_judge_config(judge_config)
        except Exception as e:
            logger.error(f"Invalid judge config: {e}")
            return self._create_stub_result(sample_dir, screenshot_paths, str(e))

        if judge_provider is None or judge_model is None:
            logger.info("Running in stub mode (judge provider/model not configured)")
            return self._create_stub_result(sample_dir, screenshot_paths, None)

        # Provider callers receive images separately; we rely on prompt text to describe
        # which screenshots are provided and in what order.

        scores: dict[str, Any] = {}
        rationale: dict[str, str] = {}
        score_normalized: dict[str, float] = {}
        artifacts: dict[str, Path] = {
            "mobile_screenshot": screenshot_paths["mobile"],
            "tablet_screenshot": screenshot_paths["tablet"],
            "desktop_screenshot": screenshot_paths["desktop"],
        }
        # Sketch artifacts (copied into eval sample by orchestrator when available)
        if isinstance(sketches, dict):
            desktop_sketch = sketches.get("desktop")
            mobile_sketch = sketches.get("mobile")
            if isinstance(desktop_sketch, Path):
                artifacts["sketch_desktop"] = desktop_sketch
            if isinstance(mobile_sketch, Path):
                artifacts["sketch_mobile"] = mobile_sketch
        last_response_path: Path | None = None
        score_overall: float | None = None

        include_sketches: bool = judge_config.get("include_sketches", True)
        template_id = judge_config.get("prompt", "llm_judge/unified_responsive_v1")
        user_prompt_vars = judge_config.get("prompt_vars") or {}
        default_prompt_vars: dict[str, Any] = {
            "rubric_guidance": _format_unified_rubric_guidance_markdown(),
        }
        prompt_vars = {**default_prompt_vars, **user_prompt_vars}

        try:
            for dim in JUDGE_DIMENSIONS:
                dim_key = dim["key"]
                prompt_text = render_judge_prompt(template_id, prompt_vars)
                if judge_provider == "claude":
                    prompt_text = (
                        prompt_text
                        + "\n\nCRITICAL OUTPUT FORMAT: Output the JSON object only. Do not wrap it in ``` fences. Do not include any extra text."
                    )
                prompt_path = sample_dir / f"llm_judge_{dim_key}_prompt.txt"
                response_path = sample_dir / f"llm_judge_{dim_key}_response.json"

                logger.info(
                    f"Calling judge for dimension '{dim_key}' with provider={judge_provider}, model={judge_model}"
                )

                if include_sketches:
                    if not isinstance(sketches, dict):
                        raise ValueError("Missing sketches in metric context (expected context['sketches'] dict)")
                    desktop_sketch = sketches.get("desktop")
                    mobile_sketch = sketches.get("mobile")
                    if not isinstance(desktop_sketch, Path) or not desktop_sketch.exists():
                        raise ValueError("Desktop sketch not found for judge input")
                    if not isinstance(mobile_sketch, Path) or not mobile_sketch.exists():
                        raise ValueError("Mobile sketch not found for judge input")

                    desktop_sketch_for_judge = self._flatten_sketch_for_judge(
                        desktop_sketch, artifacts_dir / "sketch_desktop_for_judge.png"
                    )
                    mobile_sketch_for_judge = self._flatten_sketch_for_judge(
                        mobile_sketch, artifacts_dir / "sketch_mobile_for_judge.png"
                    )
                    artifacts["sketch_desktop_for_judge"] = desktop_sketch_for_judge
                    artifacts["sketch_mobile_for_judge"] = mobile_sketch_for_judge

                    prompt_text = (
                        prompt_text
                        + "\n\nIMAGE ORDER AND FILENAMES (critical):\n"
                        + f"1) Desktop sketch: {desktop_sketch_for_judge.name}\n"
                        + f"2) Mobile sketch: {mobile_sketch_for_judge.name}\n"
                        + f"3) Mobile screenshot: {screenshot_paths['mobile'].name}\n"
                        + f"4) Tablet screenshot: {screenshot_paths['tablet'].name}\n"
                        + f"5) Desktop screenshot: {screenshot_paths['desktop'].name}\n"
                        + "When you refer to a specific image in your rationale, use these labels "
                        + "(e.g., 'mobile sketch', 'tablet screenshot') so it is unambiguous.\n"
                    )
                    judge_images = [
                        desktop_sketch_for_judge,
                        mobile_sketch_for_judge,
                        screenshot_paths["mobile"],
                        screenshot_paths["tablet"],
                        screenshot_paths["desktop"],
                    ]
                else:
                    prompt_text = (
                        prompt_text
                        + "\n\nIMAGE ORDER AND FILENAMES (critical):\n"
                        + f"1) Mobile screenshot: {screenshot_paths['mobile'].name}\n"
                        + f"2) Tablet screenshot: {screenshot_paths['tablet'].name}\n"
                        + f"3) Desktop screenshot: {screenshot_paths['desktop'].name}\n"
                        + "When you refer to a specific image in your rationale, use these labels "
                        + "(e.g., 'tablet screenshot', 'desktop screenshot') so it is unambiguous.\n"
                    )
                    judge_images = [
                        screenshot_paths["mobile"],
                        screenshot_paths["tablet"],
                        screenshot_paths["desktop"],
                    ]

                with open(prompt_path, "w", encoding="utf-8") as f:
                    f.write(prompt_text)
                logger.debug(f"Saved prompt file: {prompt_path}")

                max_retries: int = judge_config.get("max_retries", 3)
                last_exc: Exception | None = None
                last_response_text: str = ""
                judge_result: dict | None = None
                for attempt in range(1, max_retries + 1):
                    try:
                        judge_call = call_judge(
                            prompt_text=prompt_text,
                            image_paths=judge_images,
                            judge_cfg=judge_config,
                        )
                        response_text = (judge_call.get("text") or "").strip()
                        last_response_text = response_text

                        raw_response_data = {
                            "provider": judge_call.get("provider"),
                            "model": judge_call.get("model"),
                            "latency_ms": judge_call.get("latency_ms"),
                            "usage": judge_call.get("usage"),
                            "raw": judge_call.get("raw"),
                            "response_text": response_text,
                        }
                        with open(response_path, "w", encoding="utf-8") as f:
                            json.dump(raw_response_data, f, indent=2)

                        judge_result = json.loads(extract_json_object(response_text))
                        break  # success
                    except (json.JSONDecodeError, ValueError, Exception) as e:
                        last_exc = e
                        if attempt < max_retries:
                            wait = 2 ** (attempt - 1)
                            logger.warning(
                                f"Dimension '{dim_key}': attempt {attempt}/{max_retries} failed "
                                f"({type(e).__name__}: {e}). Retrying in {wait}s…"
                            )
                            await asyncio.sleep(wait)
                        else:
                            logger.error(
                                f"Dimension '{dim_key}': all {max_retries} attempts failed. "
                                f"Last error: {last_exc}"
                            )

                if judge_result is None:
                    raise ValueError(
                        f"Dimension '{dim_key}': judge returned non-JSON output after "
                        f"{max_retries} attempt(s): {last_exc}. "
                        f"Raw response text from last attempt ({len(last_response_text)} chars): "
                        f"{last_response_text!r}"
                    )

                last_response_path = response_path
                artifacts[f"{dim_key}_prompt"] = prompt_path
                artifacts[f"{dim_key}_response"] = response_path

                output_schema = dim.get("output_schema")

                if output_schema == SCHEMA_MULTI_SCORE_1_5:
                    # Unified multi-dimension response: fill all per-dimension scores/rationales at once
                    (
                        multi_scores,
                        multi_rationale,
                        multi_normalized,
                        overall_score,
                        overall_rationale,
                    ) = self._parse_multi_score_response(judge_result)

                    scores.update(multi_scores)
                    rationale.update(multi_rationale)
                    score_normalized.update(multi_normalized)
                    # Prefer model-provided overall_score (expected by the unified schema).
                    score_overall = float(overall_score) if overall_score is not None else None

                    # Attach overall prompt/response paths for debugging
                    artifacts["responsive_overall_prompt"] = prompt_path
                    artifacts["responsive_overall_response"] = response_path

                else:
                    # Legacy per-dimension schema
                    score_val, rationale_val, normalized = self._parse_dimension_response(
                        dim_key, output_schema or "", judge_result
                    )
                    scores[dim_key] = score_val
                    rationale[dim_key] = rationale_val
                    if normalized is not None:
                        score_normalized[dim_key] = normalized

            # If no explicit overall score was provided, fall back to mean of normalized scores
            if score_overall is None:
                valid_normalized = [v for v in score_normalized.values() if v is not None]
                score_overall = (
                    sum(valid_normalized) / len(valid_normalized) if valid_normalized else None
                )

            result = {
                "status": "success",
                "score_overall": score_overall,
                "scores": scores,
                "rationale": rationale,
                "score_normalized": score_normalized if score_normalized else None,
            }
            logger.info("LLM judge evaluation completed successfully")
            return result, artifacts

        except Exception as e:
            logger.error(f"LLM judge API call failed: {e}", exc_info=True)
            return self._create_error_result(
                sample_dir, screenshot_paths, last_response_path, str(e)
            )

    def _parse_dimension_response(
        self, dim_key: str, output_schema: str, judge_result: dict[str, Any]
    ) -> tuple[Any, str, float | None]:
        """
        Parse and validate a single dimension's judge response.

        Args:
            dim_key: Dimension key (for logging)
            output_schema: Schema identifier (e.g. SCHEMA_SCORE_1_5)
            judge_result: Parsed JSON response for this dimension

        Returns:
            Tuple of (raw_score, rationale_str, normalized_0_1 or None)
        """
        if not isinstance(judge_result, dict):
            logger.warning(f"Dimension '{dim_key}': response is not a JSON object")
            return None, "", None

        if output_schema == SCHEMA_SCORE_1_5:
            raw = judge_result.get("score")
            rationale_str = judge_result.get("rationale", "")
            if rationale_str is not None and not isinstance(rationale_str, str):
                rationale_str = str(rationale_str)
            try:
                score_int = int(raw) if raw is not None else None
            except (TypeError, ValueError):
                score_int = None
            if score_int is None:
                logger.warning(f"Dimension '{dim_key}': missing or invalid 'score'")
                return None, rationale_str or "", None
            score_clamped = max(1, min(5, score_int))
            normalized = (score_clamped - 1) / 4.0
            return score_clamped, rationale_str or "", normalized

        logger.warning(f"Dimension '{dim_key}': unknown output_schema '{output_schema}'")
        return None, "", None

    def _parse_multi_score_response(
        self, judge_result: dict[str, Any]
    ) -> tuple[
        dict[str, int | None],
        dict[str, str],
        dict[str, float],
        int | None,
        str,
    ]:
        """
        Parse unified multi-dimension judge response.

        Supported schemas:

        Legacy:
        {
          "scores": {
            "<dim_key>": { "score": 1-5, "rationale": "<text>" },
            ...
          },
          "overall_score": 1-5,
          "overall_rationale": "<text>"
        }
        """
        if not isinstance(judge_result, dict):
            logger.warning("Unified judge response is not a JSON object")
            return {}, {}, {}, None, ""

        raw_scores = judge_result.get("scores") or {}
        if not isinstance(raw_scores, dict):
            logger.warning("Unified judge response 'scores' field is not an object")
            raw_scores = {}

        per_dim_scores: dict[str, int | None] = {}
        per_dim_rationale: dict[str, str] = {}
        per_dim_normalized: dict[str, float] = {}

        for dim_key in JUDGE_DIMENSION_KEYS:
            entry = raw_scores.get(dim_key) or {}
            if not isinstance(entry, dict):
                logger.warning(f"Unified judge: dimension '{dim_key}' entry is not an object")
                per_dim_scores[dim_key] = None
                per_dim_rationale[dim_key] = ""
                continue

            raw_score = entry.get("score")
            rationale_str = entry.get("rationale", "")
            if rationale_str is not None and not isinstance(rationale_str, str):
                rationale_str = str(rationale_str)

            try:
                score_int = int(raw_score) if raw_score is not None else None
            except (TypeError, ValueError):
                score_int = None

            if score_int is None:
                logger.warning(f"Unified judge: missing or invalid 'score' for dimension '{dim_key}'")
                per_dim_scores[dim_key] = None
                per_dim_rationale[dim_key] = rationale_str or ""
                continue

            score_clamped = max(1, min(5, score_int))
            normalized = (score_clamped - 1) / 4.0
            per_dim_scores[dim_key] = score_clamped
            per_dim_rationale[dim_key] = rationale_str or ""
            per_dim_normalized[dim_key] = normalized

        overall_raw = judge_result.get("overall_score")
        overall_rationale = judge_result.get("overall_rationale", "")
        if overall_rationale is not None and not isinstance(overall_rationale, str):
            overall_rationale = str(overall_rationale)
        try:
            overall_score = int(overall_raw) if overall_raw is not None else None
        except (TypeError, ValueError):
            overall_score = None
        if overall_score is not None:
            overall_score = max(1, min(5, overall_score))

        # Fallback: if overall_score is missing/invalid, compute from final per-dimension scores.
        if overall_score is None:
            vals = [s for s in per_dim_scores.values() if isinstance(s, int)]
            if vals:
                overall_score = int(round(sum(vals) / len(vals)))
                overall_score = max(1, min(5, overall_score))

        return (
            per_dim_scores,
            per_dim_rationale,
            per_dim_normalized,
            overall_score,
            overall_rationale,
        )

    def _create_stub_result(
        self,
        sample_dir: Path,
        screenshot_paths: dict[str, Path],
        error_msg: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Path]]:
        """
        Create stub result when judge is not configured.

        Args:
            sample_dir: Sample directory (parent of artifacts_dir)
            screenshot_paths: Dict of screenshot paths
            error_msg: Optional error message

        Returns:
            Tuple of (result_dict, artifacts_dict)
        """
        dim_keys = list(JUDGE_DIMENSION_KEYS)
        result = {
            "status": "stub",
            "score_overall": None,
            "scores": {k: None for k in dim_keys},
            "rationale": {k: None for k in dim_keys},
            "notes": error_msg
            or "LLM judge not configured. Set judge.provider and judge.model in eval config.",
        }
        artifacts = {
            "mobile_screenshot": screenshot_paths["mobile"],
            "tablet_screenshot": screenshot_paths["tablet"],
            "desktop_screenshot": screenshot_paths["desktop"],
        }
        return result, artifacts

    def _create_error_result(
        self,
        sample_dir: Path,
        screenshot_paths: dict[str, Path],
        response_path: Path | None,
        error_msg: str,
    ) -> tuple[dict[str, Any], dict[str, Path]]:
        """
        Create error result when judge API call fails.

        Args:
            sample_dir: Sample directory (parent of artifacts_dir)
            screenshot_paths: Dict of screenshot paths
            response_path: Path to response file (if exists)
            error_msg: Error message

        Returns:
            Tuple of (result_dict, artifacts_dict)
        """
        dim_keys = list(JUDGE_DIMENSION_KEYS)
        result = {
            "status": "error",
            "error": error_msg,
            "score_overall": None,
            "scores": {k: None for k in dim_keys},
            "rationale": {k: None for k in dim_keys},
        }
        artifacts: dict[str, Path] = {
            "mobile_screenshot": screenshot_paths["mobile"],
            "tablet_screenshot": screenshot_paths["tablet"],
            "desktop_screenshot": screenshot_paths["desktop"],
        }
        if response_path and response_path.exists():
            artifacts["response"] = response_path
        return result, artifacts
