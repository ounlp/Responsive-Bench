"""
Generation runner that executes runs from config files.

Handles deterministic output layout, run hashing, and error isolation.
Samples are processed in parallel using ThreadPoolExecutor (I/O-bound inference calls).
"""

import functools
import hashlib
import json
import logging
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from ..dataset.loader import ResponsiveDataset
from ..prompts.builder import PromptBuilder
from ..providers.registry import get_provider

from .logging_config import setup_run_logging
from .screenshots import capture_screenshots, capture_single_screenshot

logger = logging.getLogger("responsive.runner.generate")


def _ensure_dataset_output_subdir(dataset: ResponsiveDataset, subdir_name: str) -> Path:
    """
    Return dataset_root/<subdir_name>, creating it if needed.

    Raises:
        RuntimeError: If the path exists but is not a directory.
    """
    out = Path(dataset.root) / subdir_name
    if out.exists() and not out.is_dir():
        raise RuntimeError(
            f"Dataset output path is not a directory (remove or rename it): {out}"
        )
    out.mkdir(parents=True, exist_ok=True)
    return out


def normalize_html(text: str) -> str:
    """
    Normalize HTML output by stripping markdown fences and validating structure.

    Args:
        text: Raw model output text

    Returns:
        Normalized HTML string
    """
    text = text.strip()

    # Strip reasoning preambles from Qwen3 and similar reasoning models. Some serving
    # stacks inject the opening <think> tag as part of the prompt template rather than
    # the completion, so the opening tag may be absent even though </think> is present.
    # Match on the closing tag alone: keep only what follows it.
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()

    # Extract HTML from a ```html ... ``` fence block anywhere in the text.
    # This handles models that wrap the HTML in a code fence after prose analysis.
    fence_match = re.search(r"```(?:html)?\s*\n(.*?)\n```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()

    # Remove ```html or ``` at start/end (legacy single-fence wrapping).
    text = re.sub(r"^```(?:html)?\s*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n```\s*$", "", text, flags=re.MULTILINE)

    # Seek to <!DOCTYPE or <html — never land on an incidental < inside prose.
    for marker in ("<!DOCTYPE", "<!doctype", "<html", "<HTML"):
        idx = text.find(marker)
        if idx >= 0:
            return text[idx:].strip()

    # Fallback: find first < tag.
    if not text.startswith("<"):
        first_tag = text.find("<")
        if first_tag > 0:
            text = text[first_tag:]

    return text.strip()


def normalize_json_response(text: str) -> str:
    """
    Extract a single JSON object from model output (strip fences / prose).

    Args:
        text: Raw model output

    Returns:
        JSON string suitable for json.loads

    Raises:
        ValueError: If no valid JSON object can be parsed
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()

    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in model response")

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start : i + 1]
                json.loads(chunk)
                return chunk

    raise ValueError("Unbalanced or invalid JSON object in model response")


def compute_run_hash(config_path: Path, *template_paths: Path) -> str:
    """
    Compute SHA256 hash of config and one or more prompt template files.

    Args:
        config_path: Path to run config JSON file
        template_paths: Paths to prompt template files (all hashed in order)

    Returns:
        Hex digest of combined hash
    """
    hasher = hashlib.sha256()

    if config_path.exists():
        hasher.update(config_path.read_bytes())

    for template_path in template_paths:
        if template_path.exists():
            hasher.update(template_path.read_bytes())

    return hasher.hexdigest()


def is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error is retryable (transient) or permanent.

    Args:
        error: Exception to check

    Returns:
        True if error is retryable, False otherwise
    """
    error_str = str(error).lower()
    error_type = type(error).__name__

    # Check for rate limit errors
    if "rate limit" in error_str or "429" in error_str:
        return True

    # Check for timeout errors
    if "timeout" in error_str or "timed out" in error_str:
        return True

    # Check for connection errors
    if "connection" in error_str or "network" in error_str:
        return True

    # Check for server errors (5xx)
    if "500" in error_str or "502" in error_str or "503" in error_str or "504" in error_str:
        return True

    # Check for specific exception types that are typically retryable
    retryable_types = [
        "TimeoutError",
        "ConnectionError",
        "ConnectionTimeout",
        "ReadTimeout",
        "APIConnectionError",
        "RateLimitError",
    ]
    if error_type in retryable_types:
        return True

    # Check for OpenAI-specific retryable errors
    if hasattr(error, "status_code"):
        status_code = getattr(error, "status_code", None)
        if status_code in [429, 500, 502, 503, 504]:
            return True

    # Check for Anthropic-specific retryable errors
    if hasattr(error, "status"):
        status = getattr(error, "status", None)
        if status in [429, 500, 502, 503, 504]:
            return True

    # Default: non-retryable (authentication, validation, etc.)
    return False


def exponential_backoff(attempt: int, initial: float, max_delay: float, multiplier: float) -> float:
    """
    Calculate exponential backoff delay for a retry attempt.

    Args:
        attempt: Attempt number (0-indexed, so first retry is attempt 1)
        initial: Initial backoff delay in seconds
        max_delay: Maximum backoff delay in seconds
        multiplier: Backoff multiplier

    Returns:
        Delay in seconds
    """
    delay = initial * (multiplier ** attempt)
    return min(delay, max_delay)


def _run_one_sample(
    sample_id: str,
    *,
    dataset: Any,
    prompt_builder: Any,
    provider: Any,
    mode: str,
    samples_dir: Path,
    manifest_path: Path,
    manifest_lock: threading.Lock,
    rick_jpg_source: Path,
    overwrite: bool,
    generation_params: dict[str, Any],
    max_attempts: int,
    initial_backoff: float,
    max_backoff: float,
    backoff_multiplier: float,
    dataset_output_subdir: str = "",
    prompt_template_id: str = "generate_a11y",
) -> str:
    """
    Process a single sample end-to-end (all modes, with retries).

    Returns:
        "success", "skipped", or "failed"
    """
    sample_dir = samples_dir / sample_id
    sample_dir.mkdir(exist_ok=True)

    sample_assets_dir = sample_dir / "assets"
    sample_assets_dir.mkdir(exist_ok=True)

    if rick_jpg_source.exists():
        rick_jpg_dest = sample_dir / "rick.jpg"
        if not rick_jpg_dest.exists():
            shutil.copy2(rick_jpg_source, rick_jpg_dest)
            logger.debug(f"Sample {sample_id}: Copied rick.jpg to sample folder")

    output_html_path = sample_dir / "output.html"
    error_path = sample_dir / "error.txt"
    dataset_root_resolved = Path(dataset.root)

    _AUGMENTING_MODES = frozenset(("augment_from_sketch", "augment_from_screenshot",
                                   "reconstruct_from_screenshot"))

    # --- Skip check ---
    if mode in _AUGMENTING_MODES:
        dataset_output_path = dataset_root_resolved / dataset_output_subdir / f"{sample_id}.html"
        if dataset_output_path.exists() and not overwrite:
            logger.info(
                f"Skipping {sample_id} ({mode} output exists at "
                f"{dataset_output_subdir}/{sample_id}.html, overwrite=False)"
            )
            entry = {"id": sample_id, "status": "skipped", "reason": f"{mode}_output_exists"}
            with manifest_lock:
                with open(manifest_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            return "skipped"
    elif output_html_path.exists() and not overwrite:
        logger.info(f"Skipping {sample_id} (output exists, overwrite=False)")
        entry = {"id": sample_id, "status": "skipped", "reason": "output_exists"}
        with manifest_lock:
            with open(manifest_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        return "skipped"

    # --- Retry loop ---
    last_error = None
    retry_attempts: list[dict[str, Any]] = []
    success = False

    logger.debug(f"Sample {sample_id}: Starting generation (max_attempts={max_attempts})")

    for attempt in range(max_attempts):
        try:
            logger.debug(f"Sample {sample_id}: Attempt {attempt + 1}/{max_attempts}")

            sample = dataset.get(sample_id)
            logger.debug(f"Sample {sample_id}: Loaded sample from dataset")

            # Augmenting modes start from the original Sketch2Code page, not the
            # already-augmented benchmark reference.
            source_html = sample.source_html or sample.reference_html

            # Capture reference screenshot once on the first attempt.
            if attempt == 0:
                try:
                    reference_staging = sample_dir / "reference.html"
                    shutil.copy2(sample.reference_html, reference_staging)
                    reference_png = sample_assets_dir / "reference.png"
                    capture_single_screenshot(
                        reference_staging,
                        reference_png,
                        width=375,
                        height=688,
                    )
                    logger.debug(f"Sample {sample_id}: Saved reference screenshot to {reference_png}")
                except Exception as ref_err:
                    logger.warning(f"Sample {sample_id}: Failed to capture reference screenshot: {ref_err}")

            if mode == "generate_a11y":
                # --- accessible_js mode: single generation with desktop-first sketches ---
                prompt_package = prompt_builder.build_accessible_js(sample)
                logger.debug(
                    f"Sample {sample_id}: Built accessible_js prompt package with {len(prompt_package.images)} images"
                )

                if attempt == 0:
                    (sample_dir / "prompt.txt").write_text(prompt_package.text, encoding="utf-8")
                    with open(sample_dir / "prompt_meta.json", "w") as f:
                        json.dump(prompt_package.meta, f, indent=2)
                    with open(sample_dir / "sketch_paths.json", "w") as f:
                        _ordering = prompt_package.meta.get("image_ordering") or []
                        _sketch_paths: dict | list = (
                            {dev: str(prompt_package.images[i]) for i, dev in enumerate(_ordering) if i < len(prompt_package.images)}
                            if len(_ordering) == len(prompt_package.images)
                            else [str(p) for p in prompt_package.images]
                        )
                        json.dump(_sketch_paths, f, indent=2)
                    logger.debug(f"Sample {sample_id}: Saved prompt artifacts")

                logger.debug(f"Sample {sample_id}: Calling provider.generate() (generate_a11y)")
                generate_start = time.time()
                result = provider.generate(
                    prompt_package.text, prompt_package.images, generation_params
                )
                generate_elapsed = time.time() - generate_start
                logger.debug(f"Sample {sample_id}: Provider returned in {generate_elapsed:.2f}s")
                logger.debug(
                    f"Sample {sample_id}: Raw LLM output (length={len(result.text)}):\n{result.text}"
                )

                normalized_html = normalize_html(result.text)
                logger.debug(
                    f"Sample {sample_id}: Normalized HTML (length: {len(normalized_html)} chars)"
                )
                output_html_path.write_text(normalized_html, encoding="utf-8")
                logger.debug(f"Sample {sample_id}: Saved output.html")

                try:
                    generated_png = sample_assets_dir / "generated.png"
                    capture_single_screenshot(output_html_path, generated_png, width=375, height=688)
                    logger.debug(f"Sample {sample_id}: Saved generated screenshot to {generated_png}")
                except Exception as gen_err:
                    logger.warning(
                        f"Sample {sample_id}: Failed to capture generated screenshot: {gen_err}"
                    )

                raw_response_data = {
                    "text": result.text[:500] + "..." if len(result.text) > 500 else result.text,
                    "raw": result.raw,
                }
                with open(sample_dir / "raw_response.json", "w") as f:
                    json.dump(raw_response_data, f, indent=2)
                (sample_dir / "raw_response.txt").write_text(result.text, encoding="utf-8")

                provider_meta = {
                    "model": result.model,
                    "provider": result.provider,
                    "latency_ms": result.latency_ms,
                    "usage": result.usage,
                    "mode": "generate_a11y",
                }
                with open(sample_dir / "provider_meta.json", "w") as f:
                    json.dump(provider_meta, f, indent=2)
                logger.debug(f"Sample {sample_id}: Saved provider artifacts")

                manifest_entry = {
                    "id": sample_id,
                    "status": "success",
                    "mode": "generate_a11y",
                    "latency_ms": result.latency_ms,
                    "usage": result.usage,
                }
                if retry_attempts:
                    manifest_entry["retry_attempts"] = len(retry_attempts)
                with manifest_lock:
                    with open(manifest_path, "a") as f:
                        f.write(json.dumps(manifest_entry) + "\n")

                if attempt > 0:
                    logger.info(f"Sample {sample_id}: Succeeded after {attempt} retries")
                else:
                    logger.info(f"Sample {sample_id}: Completed successfully (generate_a11y)")

                success = True
                break

            elif mode == "augment_from_sketch":
                # --- Sketch analyzer JSON + reference HTML augment (two provider calls) ---
                augmented_dir = _ensure_dataset_output_subdir(dataset, dataset_output_subdir)
                augmented_html_path = augmented_dir / f"{sample_id}.html"
                augmented_analysis_path = augmented_dir / "analysis" / f"{sample_id}.json"
                augmented_analysis_path.parent.mkdir(exist_ok=True)

                reference_html_text = source_html.read_text(encoding="utf-8")

                gen_analyze = dict(generation_params)
                gen_analyze["_responsive_step"] = "sketch_analyze"
                gen_augment = dict(generation_params)
                gen_augment["_responsive_step"] = "sketch_augment_html"

                analyzer_package = prompt_builder.build_sketch_analyzer(
                    sample, reference_html_text
                )
                logger.debug(
                    f"Sample {sample_id}: Built sketch analyzer package "
                    f"with {len(analyzer_package.images)} images"
                )

                if attempt == 0:
                    (sample_dir / "prompt_augment_from_sketch_analyze.txt").write_text(
                        analyzer_package.text, encoding="utf-8"
                    )
                    with open(sample_dir / "prompt_augment_from_sketch_analyze_meta.json", "w") as f:
                        json.dump(analyzer_package.meta, f, indent=2)
                    with open(sample_dir / "sketch_paths.json", "w") as f:
                        _ordering = analyzer_package.meta.get("image_ordering") or []
                        _sketch_paths: dict | list = (
                            {dev: str(analyzer_package.images[i]) for i, dev in enumerate(_ordering) if i < len(analyzer_package.images)}
                            if len(_ordering) == len(analyzer_package.images)
                            else [str(p) for p in analyzer_package.images]
                        )
                        json.dump(_sketch_paths, f, indent=2)
                    logger.debug(f"Sample {sample_id}: Saved sketch analyzer prompt artifacts")

                logger.debug(f"Sample {sample_id}: Calling provider for sketch analyzer (step 1)")
                analyzer_start = time.time()
                analyzer_result = provider.generate(
                    analyzer_package.text, analyzer_package.images, gen_analyze
                )
                analyzer_elapsed = time.time() - analyzer_start
                logger.debug(
                    f"Sample {sample_id}: Sketch analyzer returned in {analyzer_elapsed:.2f}s"
                )

                analysis_raw = normalize_json_response(analyzer_result.text)
                analysis_obj = json.loads(analysis_raw)
                analysis_pretty = json.dumps(analysis_obj, indent=2, ensure_ascii=False)

                (sample_dir / "sketch_analysis.json").write_text(analysis_pretty, encoding="utf-8")
                _ensure_dataset_output_subdir(dataset, dataset_output_subdir)
                augmented_analysis_path.write_text(analysis_pretty, encoding="utf-8")
                logger.debug(f"Sample {sample_id}: Saved sketch analysis JSON")

                augment_package = prompt_builder.build_sketch_html_augment(
                    sample, reference_html_text, analysis_pretty
                )
                if attempt == 0:
                    (sample_dir / "prompt_sketch_augment.txt").write_text(
                        augment_package.text, encoding="utf-8"
                    )

                logger.debug(f"Sample {sample_id}: Calling provider for HTML augment (step 2)")
                augment_start = time.time()
                augment_result = provider.generate(
                    augment_package.text, augment_package.images, gen_augment
                )
                augment_elapsed = time.time() - augment_start
                logger.debug(
                    f"Sample {sample_id}: HTML augment returned in {augment_elapsed:.2f}s"
                )
                logger.debug(
                    f"Sample {sample_id}: Raw LLM augment output (length={len(augment_result.text)}):\n{augment_result.text}"
                )

                final_html = normalize_html(augment_result.text)
                _ensure_dataset_output_subdir(dataset, dataset_output_subdir)
                augmented_html_path.write_text(final_html, encoding="utf-8")
                output_html_path.write_text(final_html, encoding="utf-8")
                logger.debug(
                    f"Sample {sample_id}: Saved augmented HTML to {augmented_html_path} "
                    f"and run output.html"
                )

                try:
                    generated_png = sample_assets_dir / "generated.png"
                    capture_single_screenshot(output_html_path, generated_png, width=375, height=688)
                    logger.debug(f"Sample {sample_id}: Saved generated screenshot to {generated_png}")
                except Exception as gen_err:
                    logger.warning(
                        f"Sample {sample_id}: Failed to capture generated screenshot: {gen_err}"
                    )

                ar_txt = analyzer_result.text
                with open(sample_dir / "raw_response_augment_from_sketch_analyze.json", "w") as f:
                    json.dump(
                        {"text": ar_txt[:500] + "..." if len(ar_txt) > 500 else ar_txt, "raw": analyzer_result.raw},
                        f, indent=2,
                    )
                (sample_dir / "raw_response_augment_from_sketch_analyze.txt").write_text(
                    ar_txt, encoding="utf-8"
                )
                aug_txt = augment_result.text
                with open(sample_dir / "raw_response_sketch_augment.json", "w") as f:
                    json.dump(
                        {"text": aug_txt[:500] + "..." if len(aug_txt) > 500 else aug_txt, "raw": augment_result.raw},
                        f, indent=2,
                    )
                (sample_dir / "raw_response_sketch_augment.txt").write_text(aug_txt, encoding="utf-8")

                provider_meta = {
                    "model": augment_result.model,
                    "provider": augment_result.provider,
                    "mode": "augment_from_sketch",
                    "augment_from_sketch_analyze": {
                        "latency_ms": analyzer_result.latency_ms,
                        "usage": analyzer_result.usage,
                    },
                    "sketch_augment": {
                        "latency_ms": augment_result.latency_ms,
                        "usage": augment_result.usage,
                    },
                }
                with open(sample_dir / "provider_meta.json", "w") as f:
                    json.dump(provider_meta, f, indent=2)

                manifest_entry = {
                    "id": sample_id,
                    "status": "success",
                    "mode": "augment_from_sketch",
                    "latency_ms": augment_result.latency_ms,
                    "usage": augment_result.usage,
                    "augment_from_sketch_analyze_latency_ms": analyzer_result.latency_ms,
                }
                if retry_attempts:
                    manifest_entry["retry_attempts"] = len(retry_attempts)
                with manifest_lock:
                    with open(manifest_path, "a") as f:
                        f.write(json.dumps(manifest_entry) + "\n")

                if attempt > 0:
                    logger.info(f"Sample {sample_id}: Succeeded after {attempt} retries")
                else:
                    logger.info(f"Sample {sample_id}: Completed successfully (augment_from_sketch)")
                success = True
                break

            elif mode == "augment_from_screenshot":
                # --- Single-step: capture reference screenshots → augment HTML ---
                augmented_screenshot_dir = _ensure_dataset_output_subdir(dataset, dataset_output_subdir)
                augmented_screenshot_html_path = augmented_screenshot_dir / f"{sample_id}.html"

                reference_html_text = source_html.read_text(encoding="utf-8")

                # Capture reference screenshots at desktop/tablet/mobile (once per run)
                ref_screenshot_dir = sample_assets_dir / "reference_screenshots"
                if attempt == 0:
                    try:
                        ref_screenshot_paths = capture_screenshots(
                            source_html,
                            ref_screenshot_dir,
                            mode="default",
                        )
                        logger.debug(
                            f"Sample {sample_id}: Captured reference screenshots at "
                            f"{list(ref_screenshot_paths.keys())}"
                        )
                    except Exception as ss_err:
                        logger.warning(
                            f"Sample {sample_id}: Failed to capture reference screenshots: {ss_err}"
                        )
                        raise

                desktop_png = ref_screenshot_dir / "render_desktop.png"
                tablet_png  = ref_screenshot_dir / "render_tablet.png"
                mobile_png  = ref_screenshot_dir / "render_mobile.png"

                augment_package = prompt_builder.build_augment_reference_screenshot(
                    sample, reference_html_text, desktop_png, tablet_png, mobile_png
                )
                logger.debug(
                    f"Sample {sample_id}: Built screenshot augment package "
                    f"with {len(augment_package.images)} images"
                )

                if attempt == 0:
                    (sample_dir / "prompt_screenshot_augment.txt").write_text(
                        augment_package.text, encoding="utf-8"
                    )
                    with open(sample_dir / "prompt_screenshot_augment_meta.json", "w") as f:
                        json.dump(augment_package.meta, f, indent=2)

                gen_augment = dict(generation_params)
                gen_augment["_responsive_step"] = "screenshot_augment_html"

                logger.debug(f"Sample {sample_id}: Calling provider for screenshot augment")
                augment_start = time.time()
                augment_result = provider.generate(
                    augment_package.text, augment_package.images, gen_augment
                )
                augment_elapsed = time.time() - augment_start
                logger.debug(
                    f"Sample {sample_id}: Screenshot augment returned in {augment_elapsed:.2f}s"
                )

                final_html = normalize_html(augment_result.text)
                _ensure_dataset_output_subdir(dataset, dataset_output_subdir)
                augmented_screenshot_html_path.write_text(final_html, encoding="utf-8")
                output_html_path.write_text(final_html, encoding="utf-8")
                logger.debug(
                    f"Sample {sample_id}: Saved augmented HTML to "
                    f"{augmented_screenshot_html_path} and run output.html"
                )

                try:
                    generated_png = sample_assets_dir / "generated.png"
                    capture_single_screenshot(output_html_path, generated_png, width=375, height=688)
                    logger.debug(f"Sample {sample_id}: Saved generated screenshot to {generated_png}")
                except Exception as gen_err:
                    logger.warning(
                        f"Sample {sample_id}: Failed to capture generated screenshot: {gen_err}"
                    )

                aug_txt = augment_result.text
                with open(sample_dir / "raw_response_screenshot_augment.json", "w") as f:
                    json.dump(
                        {"text": aug_txt[:500] + "..." if len(aug_txt) > 500 else aug_txt, "raw": augment_result.raw},
                        f, indent=2,
                    )
                (sample_dir / "raw_response_screenshot_augment.txt").write_text(aug_txt, encoding="utf-8")

                provider_meta = {
                    "model": augment_result.model,
                    "provider": augment_result.provider,
                    "mode": "augment_from_screenshot",
                    "screenshot_augment": {
                        "latency_ms": augment_result.latency_ms,
                        "usage": augment_result.usage,
                    },
                }
                with open(sample_dir / "provider_meta.json", "w") as f:
                    json.dump(provider_meta, f, indent=2)

                manifest_entry = {
                    "id": sample_id,
                    "status": "success",
                    "mode": "augment_from_screenshot",
                    "latency_ms": augment_result.latency_ms,
                    "usage": augment_result.usage,
                }
                if retry_attempts:
                    manifest_entry["retry_attempts"] = len(retry_attempts)
                with manifest_lock:
                    with open(manifest_path, "a") as f:
                        f.write(json.dumps(manifest_entry) + "\n")

                if attempt > 0:
                    logger.info(f"Sample {sample_id}: Succeeded after {attempt} retries")
                else:
                    logger.info(f"Sample {sample_id}: Completed successfully (augment_from_screenshot)")
                success = True
                break

            elif mode == "reconstruct_from_screenshot":
                # --- Single-step: capture reference screenshots → reconstruct HTML from vision only ---
                augmented_just_screenshot_dir = _ensure_dataset_output_subdir(dataset, dataset_output_subdir)
                augmented_just_screenshot_html_path = augmented_just_screenshot_dir / f"{sample_id}.html"

                # Capture reference screenshots at desktop/tablet/mobile (once per run)
                ref_screenshot_dir = sample_assets_dir / "reference_screenshots"
                if attempt == 0:
                    try:
                        ref_screenshot_paths = capture_screenshots(
                            source_html,
                            ref_screenshot_dir,
                            mode="default",
                        )
                        logger.debug(
                            f"Sample {sample_id}: Captured reference screenshots at "
                            f"{list(ref_screenshot_paths.keys())}"
                        )
                    except Exception as ss_err:
                        logger.warning(
                            f"Sample {sample_id}: Failed to capture reference screenshots: {ss_err}"
                        )
                        raise

                desktop_png = ref_screenshot_dir / "render_desktop.png"
                tablet_png  = ref_screenshot_dir / "render_tablet.png"
                mobile_png  = ref_screenshot_dir / "render_mobile.png"

                augment_package = prompt_builder.build_screenshot_only_augment(
                    sample, desktop_png, tablet_png, mobile_png
                )
                logger.debug(
                    f"Sample {sample_id}: Built screenshot-only augment package "
                    f"with {len(augment_package.images)} images"
                )

                if attempt == 0:
                    (sample_dir / "prompt_reconstruct_from_screenshot.txt").write_text(
                        augment_package.text, encoding="utf-8"
                    )
                    with open(sample_dir / "prompt_reconstruct_from_screenshot_meta.json", "w") as f:
                        json.dump(augment_package.meta, f, indent=2)

                gen_augment = dict(generation_params)
                gen_augment["_responsive_step"] = "reconstruct_from_screenshot_html"

                logger.debug(f"Sample {sample_id}: Calling provider for screenshot-only augment")
                augment_start = time.time()
                augment_result = provider.generate(
                    augment_package.text, augment_package.images, gen_augment
                )
                augment_elapsed = time.time() - augment_start
                logger.debug(
                    f"Sample {sample_id}: Screenshot-only augment returned in {augment_elapsed:.2f}s"
                )

                final_html = normalize_html(augment_result.text)
                _ensure_dataset_output_subdir(dataset, dataset_output_subdir)
                augmented_just_screenshot_html_path.write_text(final_html, encoding="utf-8")
                output_html_path.write_text(final_html, encoding="utf-8")
                logger.debug(
                    f"Sample {sample_id}: Saved augmented HTML to "
                    f"{augmented_just_screenshot_html_path} and run output.html"
                )

                try:
                    generated_png = sample_assets_dir / "generated.png"
                    capture_single_screenshot(output_html_path, generated_png, width=375, height=688)
                    logger.debug(f"Sample {sample_id}: Saved generated screenshot to {generated_png}")
                except Exception as gen_err:
                    logger.warning(
                        f"Sample {sample_id}: Failed to capture generated screenshot: {gen_err}"
                    )

                aug_txt = augment_result.text
                with open(sample_dir / "raw_response_reconstruct_from_screenshot.json", "w") as f:
                    json.dump(
                        {"text": aug_txt[:500] + "..." if len(aug_txt) > 500 else aug_txt, "raw": augment_result.raw},
                        f, indent=2,
                    )
                (sample_dir / "raw_response_reconstruct_from_screenshot.txt").write_text(
                    aug_txt, encoding="utf-8"
                )

                provider_meta = {
                    "model": augment_result.model,
                    "provider": augment_result.provider,
                    "mode": "reconstruct_from_screenshot",
                    "reconstruct_from_screenshot": {
                        "latency_ms": augment_result.latency_ms,
                        "usage": augment_result.usage,
                    },
                }
                with open(sample_dir / "provider_meta.json", "w") as f:
                    json.dump(provider_meta, f, indent=2)

                manifest_entry = {
                    "id": sample_id,
                    "status": "success",
                    "mode": "reconstruct_from_screenshot",
                    "latency_ms": augment_result.latency_ms,
                    "usage": augment_result.usage,
                }
                if retry_attempts:
                    manifest_entry["retry_attempts"] = len(retry_attempts)
                with manifest_lock:
                    with open(manifest_path, "a") as f:
                        f.write(json.dumps(manifest_entry) + "\n")

                if attempt > 0:
                    logger.info(f"Sample {sample_id}: Succeeded after {attempt} retries")
                else:
                    logger.info(f"Sample {sample_id}: Completed successfully (reconstruct_from_screenshot)")
                success = True
                break

            else:
                raise ValueError(f"Unsupported mode: {mode}")

        except Exception as e:
            last_error = e
            retry_attempts.append({
                "attempt": attempt + 1,
                "error": str(e),
                "error_type": type(e).__name__,
            })

            is_retryable = is_retryable_error(e)
            logger.debug(
                f"Sample {sample_id}: Attempt {attempt + 1} failed: {e} (retryable={is_retryable})"
            )

            if not is_retryable:
                logger.warning(f"Sample {sample_id}: Non-retryable error, stopping attempts: {e}")
                break

            if attempt < max_attempts - 1:
                delay = exponential_backoff(attempt, initial_backoff, max_backoff, backoff_multiplier)
                logger.warning(
                    f"Sample {sample_id}: Retrying in {delay:.1f}s "
                    f"(attempt {attempt + 1}/{max_attempts}): {e}"
                )
                time.sleep(delay)
            else:
                logger.warning(f"Sample {sample_id}: All {max_attempts} attempts exhausted: {e}")

    if not success:
        error_details = {
            "error": str(last_error),
            "error_type": type(last_error).__name__,
            "retry_attempts": retry_attempts,
            "max_attempts": max_attempts,
        }
        error_path.write_text(json.dumps(error_details, indent=2), encoding="utf-8")
        logger.debug(f"Sample {sample_id}: Wrote error details to {error_path}")

        manifest_entry = {
            "id": sample_id,
            "status": "failed",
            "error": str(last_error),
            "error_type": type(last_error).__name__ if last_error else None,
            "retry_attempts": len(retry_attempts),
            "max_attempts": max_attempts,
        }
        with manifest_lock:
            with open(manifest_path, "a") as f:
                f.write(json.dumps(manifest_entry) + "\n")

        logger.error(f"Sample {sample_id}: Failed after {len(retry_attempts)} attempts: {last_error}")
        return "failed"

    return "success"


def run_from_config(config_path: Path, mode_override: str | None = None) -> None:
    """
    Execute a generation run from a config file, processing samples in parallel.

    Args:
        config_path: Path to run config JSON file
        mode_override: If set, override config "mode" (e.g. from CLI --mode).

    Raises:
        FileNotFoundError: If config file does not exist
        ValueError: If config is invalid
    """
    config_path = Path(config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.debug(f"Loading run config from: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    if mode_override is not None:
        config["mode"] = mode_override
    mode = config.get("mode", "generate_a11y")
    _VALID_MODES = ("generate_a11y", "augment_from_sketch", "augment_from_screenshot", "reconstruct_from_screenshot")
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Invalid config mode: {mode}. Must be one of: {', '.join(_VALID_MODES)}."
        )

    required_fields = ["run_id", "provider", "model", "prompt_template"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")

    dataset_root = config.get("dataset_root", "dataset")
    output_root = config.get("output", {}).get("root", "runs")
    overwrite = config.get("output", {}).get("overwrite", False)
    prompt_template_id = config["prompt_template"]
    concurrency = int(config.get("concurrency", 8))
    config.setdefault("logging", {"level": "INFO", "console": True, "file": True})

    retry_config = config.get("retry", {})
    max_attempts = retry_config.get("max_attempts", 3)
    initial_backoff = retry_config.get("initial_backoff_seconds", 1.0)
    max_backoff = retry_config.get("max_backoff_seconds", 60.0)
    backoff_multiplier = retry_config.get("backoff_multiplier", 2.0)

    logger.debug(f"Retry config: max_attempts={max_attempts}, initial_backoff={initial_backoff}s")
    logger.debug(f"Concurrency: {concurrency} workers")

    dataset = ResponsiveDataset(dataset_root)
    logger.debug(f"Initialized dataset from: {dataset_root}")

    if "ids" in config and config["ids"]:
        sample_ids = sorted(config["ids"])
        logger.debug(f"Using explicit sample IDs from config: {len(sample_ids)} samples")
    elif "index_path" in config and config["index_path"]:
        index_path = Path(config["index_path"])
        sample_ids = dataset.load_index(index_path)
        logger.debug(f"Loaded sample IDs from index: {len(sample_ids)} samples")
    else:
        sample_ids = sorted(dataset.ids())
        logger.debug(f"Discovered {len(sample_ids)} samples from dataset")

    prompt_builder = PromptBuilder()
    logger.debug(f"Initialized prompt builder with template: {prompt_template_id}")

    provider = get_provider(config["provider"], config["model"])
    logger.debug(f"Initialized provider: {config['provider']} with model: {config['model']}")

    run_id = config["run_id"]
    run_dir = Path(output_root) / run_id
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)


    # All augmenting modes write to a configurable dataset subdirectory.
    # "dataset_output_dir" in config overrides the mode default.
    _AUGMENTING_MODE_DEFAULTS: dict[str, str] = {
        "augment_from_sketch": "references",
        "augment_from_screenshot": "layout_similarity_study/html_and_screenshots",
        "reconstruct_from_screenshot": "layout_similarity_study/screenshots_only",
    }
    dataset_output_subdir: str = (
        config.get("dataset_output_dir")
        or _AUGMENTING_MODE_DEFAULTS.get(mode, "")
    )

    setup_run_logging(config, run_dir)
    logger.info(
        f"Generation run started: run_id={run_id}, provider={config['provider']}, "
        f"model={config['model']}, mode={mode}, concurrency={concurrency}"
    )
    logger.info(f"Processing {len(sample_ids)} samples")

    config_copy_path = run_dir / "config.json"
    with open(config_copy_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.debug(f"Saved config copy to: {config_copy_path}")

    prompt_template_path = prompt_builder.templates_dir / f"{prompt_template_id}.j2"
    augment_from_sketch_analyze_path = prompt_builder.templates_dir / "augment_from_sketch_analyze.j2"
    augment_from_sketch_html_path = prompt_builder.templates_dir / "augment_from_sketch_html.j2"
    augment_from_screenshot_path = prompt_builder.templates_dir / "augment_from_screenshot.j2"
    reconstruct_from_screenshot_path = prompt_builder.templates_dir / "reconstruct_from_screenshot.j2"
    if mode == "augment_from_sketch":
        run_hash = compute_run_hash(config_path, augment_from_sketch_analyze_path, augment_from_sketch_html_path)
    elif mode == "augment_from_screenshot":
        run_hash = compute_run_hash(config_path, augment_from_screenshot_path)
    elif mode == "reconstruct_from_screenshot":
        run_hash = compute_run_hash(config_path, reconstruct_from_screenshot_path)
    else:
        run_hash = compute_run_hash(config_path, prompt_template_path)
    logger.debug(f"Computed run hash: {run_hash[:16]}...")

    run_meta: dict[str, Any] = {
        "run_id": run_id,
        "run_hash": run_hash,
        "config_path": str(config_path),
        "prompt_template": prompt_template_id,
        "mode": mode,
        "concurrency": concurrency,
    }
    if mode == "augment_from_sketch":
        run_meta["sketch_pipeline_templates"] = [
            "augment_from_sketch_analyze.j2",
            "augment_from_sketch_html.j2",
        ]
    elif mode == "augment_from_screenshot":
        run_meta["screenshot_pipeline_template"] = "augment_from_screenshot.j2"
    elif mode == "reconstruct_from_screenshot":
        run_meta["screenshot_pipeline_template"] = "reconstruct_from_screenshot.j2"
    if mode in _AUGMENTING_MODE_DEFAULTS:
        run_meta["dataset_output_subdir"] = dataset_output_subdir
    if "seed" in config:
        run_meta["seed"] = config["seed"]
    if "created_by" in config:
        run_meta["created_by"] = config["created_by"]

    run_meta_path = run_dir / "run_meta.json"
    with open(run_meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)
    logger.debug(f"Saved run metadata to: {run_meta_path}")

    manifest_path = run_dir / "manifest.jsonl"

    config_dir = config_path.parent
    project_root = config_dir if (config_dir / "assets").exists() else Path.cwd()
    assets_dir = project_root / "assets"
    rick_jpg_source = assets_dir / "rick.jpg"
    if rick_jpg_source.exists():
        logger.debug(f"Found assets/rick.jpg at {rick_jpg_source}, will copy to each sample folder")
    else:
        logger.debug(f"assets/rick.jpg not found at {rick_jpg_source}, skipping asset copy")

    generation_params = config.get("generation", {})
    logger.debug(f"Generation parameters: {generation_params}")

    # Shared lock for thread-safe manifest writes.
    manifest_lock = threading.Lock()

    # Counters for the progress bar postfix (protected by their own lock).
    counts: dict[str, int] = {"ok": 0, "skip": 0, "fail": 0}
    counts_lock = threading.Lock()

    worker = functools.partial(
        _run_one_sample,
        dataset=dataset,
        prompt_builder=prompt_builder,
        provider=provider,
        mode=mode,
        samples_dir=samples_dir,
        manifest_path=manifest_path,
        manifest_lock=manifest_lock,
        rick_jpg_source=rick_jpg_source,
        overwrite=overwrite,
        generation_params=generation_params,
        max_attempts=max_attempts,
        initial_backoff=initial_backoff,
        max_backoff=max_backoff,
        backoff_multiplier=backoff_multiplier,
        dataset_output_subdir=dataset_output_subdir,
        prompt_template_id=prompt_template_id,
    )

    logger.info(f"Dispatching {len(sample_ids)} samples across {concurrency} workers")

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(worker, sid): sid for sid in sample_ids}

        with logging_redirect_tqdm():
            with tqdm(
                as_completed(futures),
                total=len(sample_ids),
                desc="Generating",
                unit="sample",
                dynamic_ncols=True,
                colour="green",
                postfix=counts,
            ) as pbar:
                for future in pbar:
                    sid = futures[future]
                    try:
                        status = future.result()
                    except Exception as exc:
                        logger.error(f"Sample {sid}: Unhandled worker exception: {exc}")
                        status = "failed"

                    with counts_lock:
                        if status == "success":
                            counts["ok"] += 1
                        elif status == "skipped":
                            counts["skip"] += 1
                        else:
                            counts["fail"] += 1
                        pbar.set_postfix(counts, refresh=False)

    logger.info(
        f"Generation run complete: {run_dir} "
        f"(ok={counts['ok']}, skip={counts['skip']}, fail={counts['fail']})"
    )
