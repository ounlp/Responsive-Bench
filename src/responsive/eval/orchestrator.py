"""
Evaluation orchestrator for responsive webpage generation outputs.

Reads generated HTML files from run directories and evaluates them using configurable metrics.
Samples are evaluated in parallel using asyncio.Semaphore (metrics are async / Playwright-based).
"""

import asyncio
import json
import logging
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from ..dataset.loader import ResponsiveDataset, Sample

from .logging_config import setup_eval_logging
from .metrics.registry import get_metric

logger = logging.getLogger("responsive.eval.orchestrator")


def load_eval_config(config_path: Path) -> dict[str, Any]:
    """
    Load and validate evaluation config.

    Args:
        config_path: Path to evaluation config JSON file

    Returns:
        Config dictionary

    Raises:
        FileNotFoundError: If config file does not exist
        ValueError: If config is invalid
    """
    config_path = Path(config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Evaluation config not found: {config_path}")

    logger.debug(f"Loading evaluation config from: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    # Validate required fields
    required_fields = ["eval_id", "run_id", "metrics"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")

    # Set defaults
    config.setdefault("dataset_root", "dataset")
    config.setdefault("viewport", {"width": 480, "height": 800})
    config.setdefault("playwright", {"headless": True, "timeout_ms": 30000})
    config.setdefault(
        "judge",
        {
            "provider": "openai",
            "model": "gpt-4o",
            "temperature": 0.0,
            "max_tokens": 5000,
        },
    )
    config.setdefault("output", {"overwrite": False})
    config.setdefault("concurrency", {"max_workers": 4})
    config.setdefault("caching", {"skip_if_metrics_exist": True})
    config.setdefault("logging", {"level": "INFO", "console": True, "file": True})

    logger.debug(f"Config loaded: eval_id={config['eval_id']}, metrics={config['metrics']}")

    return config


def resolve_run_root(run_id_or_path: str, output_root: str = "runs") -> Path:
    """
    Resolve run directory from run_id or path.

    Args:
        run_id_or_path: Either a run_id string or a path to run directory
        output_root: Root directory for runs (default "runs")

    Returns:
        Absolute path to run directory

    Raises:
        FileNotFoundError: If run directory does not exist
    """
    run_path = Path(run_id_or_path)

    if run_path.is_absolute() or run_path.exists():
        run_dir = run_path.resolve()
    else:
        run_dir = Path(output_root) / run_id_or_path

    run_dir = run_dir.resolve()

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    logger.debug(f"Resolved run directory: {run_dir}")

    return run_dir


def resolve_ids(config: dict[str, Any], run_dir: Path) -> list[str]:
    """
    Resolve sample IDs from config or by scanning run directory.

    Args:
        config: Evaluation config dict
        run_dir: Path to run directory

    Returns:
        Sorted list of sample IDs
    """
    if "ids" in config and config["ids"]:
        sample_ids = sorted(config["ids"])
        logger.debug(f"Using explicit sample IDs from config: {len(sample_ids)} samples")
        return sample_ids

    samples_dir = run_dir / "samples"
    if not samples_dir.exists():
        logger.warning(f"Samples directory not found: {samples_dir}")
        return []

    sample_ids = sorted(item.name for item in samples_dir.iterdir() if item.is_dir())
    logger.debug(f"Scanned samples directory: found {len(sample_ids)} sample IDs")

    return sample_ids


async def _eval_one_sample(
    *,
    sample_id: str,
    dataset: Any,
    metrics: list[Any],
    run_dir: Path,
    eval_samples_dir: Path,
    manifest_path: Path,
    results_path: Path,
    manifest_lock: asyncio.Lock,
    results_lock: asyncio.Lock,
    viewport: dict[str, Any],
    playwright_config: dict[str, Any],
    judge_config: dict[str, Any],
    layout_block_f1_config: dict[str, Any],
    overwrite: bool,
    skip_if_exists: bool,
) -> tuple[str, str]:
    """
    Evaluate all metrics for one sample.

    Returns:
        (sample_id, status) where status is "success", "skipped", or "failed".
    """
    logger.info(f"Processing sample: {sample_id}")

    sample_dir = eval_samples_dir / sample_id
    sample_dir.mkdir(exist_ok=True)
    artifacts_dir = sample_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    metrics_json_path = sample_dir / "metrics.json"
    errors_json_path = sample_dir / "errors.json"

    # --- Load any existing per-metric results for this sample ---
    existing_metric_results: dict[str, Any] = {}
    existing_metric_artifacts: dict[str, Any] = {}
    if metrics_json_path.exists():
        try:
            with open(metrics_json_path, "r") as f:
                existing_data = json.load(f)
            existing_metric_results = existing_data.get("metrics", {})
            existing_metric_artifacts = existing_data.get("artifacts", {})
        except Exception as e:
            logger.debug(f"Sample {sample_id}: Could not read existing metrics.json: {e}")

    def _metric_already_done(metric_name: str) -> bool:
        if not skip_if_exists or overwrite:
            return False
        result = existing_metric_results.get(metric_name)
        if result is None or not isinstance(result, dict):
            return False
        # A metric is "done" if it ran to completion (any shape of success result).
        # Only an explicit status="error" means it needs to be retried.
        return result.get("status") != "error"

    # --- Caching check (per requested metric, not per sample) ---
    metrics_to_run = [m for m in metrics if not _metric_already_done(m.name)]

    if not metrics_to_run:
        logger.info(f"Skipping {sample_id} (all {len(metrics)} requested metrics already computed)")
        async with manifest_lock:
            with open(manifest_path, "a") as f:
                f.write(
                    json.dumps({"id": sample_id, "status": "skipped", "reason": "all_metrics_exist"})
                    + "\n"
                )
        return sample_id, "skipped"

    if len(metrics_to_run) < len(metrics):
        already_done = [m.name for m in metrics if m not in metrics_to_run]
        logger.info(
            f"Sample {sample_id}: {len(already_done)} metric(s) already done ({already_done}), "
            f"running {len(metrics_to_run)} remaining: {[m.name for m in metrics_to_run]}"
        )

    # --- Locate generated HTML ---
    generated_html_path = run_dir / "samples" / sample_id / "output.html"
    if not generated_html_path.exists():
        logger.error(f"Sample {sample_id}: Generated HTML not found at {generated_html_path}")
        with open(errors_json_path, "w") as f:
            json.dump(
                {"sample_id": sample_id, "error": "Generated HTML not found", "path": str(generated_html_path)},
                f, indent=2,
            )
        async with manifest_lock:
            with open(manifest_path, "a") as f:
                f.write(
                    json.dumps({"id": sample_id, "status": "failed", "error": "Generated HTML not found"})
                    + "\n"
                )
        return sample_id, "failed"

    logger.debug(f"Sample {sample_id}: Found generated HTML at {generated_html_path}")

    # --- Load sample object ---
    sample: Sample | None = None
    try:
        sample = dataset.get(sample_id)
        logger.debug(f"Sample {sample_id}: Loaded sample object from dataset")
    except Exception as e:
        logger.debug(
            f"Sample {sample_id}: Not found in dataset ({e}), continuing without sample object"
        )

    # --- Resolve sketch paths ---
    sketches_context: dict[str, Path] = {}
    run_sample_dir = run_dir / "samples" / sample_id
    sketch_paths_file = run_sample_dir / "sketch_paths.json"
    mobile_sketch_src: Path | None = None
    desktop_sketch_src: Path | None = None

    if sketch_paths_file.exists():
        try:
            with open(sketch_paths_file, "r", encoding="utf-8") as f:
                paths = json.load(f)

            def _resolve(raw: str) -> Path:
                p = Path(raw)
                return p if p.is_absolute() else (run_sample_dir / p).resolve()

            if isinstance(paths, dict):
                if "mobile" in paths:
                    mobile_sketch_src = _resolve(paths["mobile"])
                if "desktop" in paths:
                    desktop_sketch_src = _resolve(paths["desktop"])
            elif isinstance(paths, list) and len(paths) >= 2:
                # Legacy list format — use image_ordering from prompt_meta.json to determine which index is which
                ordering: list[str] = []
                meta_file = run_sample_dir / "prompt_meta.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as mf:
                            ordering = json.load(mf).get("image_ordering") or []
                    except Exception:
                        pass
                if not ordering:
                    ordering = ["mobile", "desktop"]  # legacy default
                for idx, device in enumerate(ordering):
                    if idx < len(paths):
                        if device == "mobile":
                            mobile_sketch_src = _resolve(paths[idx])
                        elif device == "desktop":
                            desktop_sketch_src = _resolve(paths[idx])
        except Exception as e:
            logger.debug(f"Sample {sample_id}: Failed to read sketch_paths.json: {e}")

    if (mobile_sketch_src is None or desktop_sketch_src is None) and sample is not None:
        try:
            mobile_sketch_src = sample.sketches.get("mobile")
            desktop_sketch_src = sample.sketches.get("desktop")
        except Exception:
            pass

    if mobile_sketch_src and mobile_sketch_src.exists():
        shutil.copy2(mobile_sketch_src, artifacts_dir / "sketch_mobile.png")
        sketches_context["mobile"] = artifacts_dir / "sketch_mobile.png"
    else:
        logger.debug(f"Sample {sample_id}: Mobile sketch not found (src={mobile_sketch_src})")

    if desktop_sketch_src and desktop_sketch_src.exists():
        shutil.copy2(desktop_sketch_src, artifacts_dir / "sketch_desktop.png")
        sketches_context["desktop"] = artifacts_dir / "sketch_desktop.png"
    else:
        logger.debug(f"Sample {sample_id}: Desktop sketch not found (src={desktop_sketch_src})")

    # --- Run metrics sequentially within this sample ---
    start_time = time.time()
    metric_results: dict[str, Any] = {}
    metric_artifacts: dict[str, Any] = {}
    sample_errors: list[dict[str, Any]] = []

    logger.debug(f"Sample {sample_id}: Running {len(metrics_to_run)} metrics")

    for metric in metrics_to_run:
        metric_name = metric.name
        logger.debug(f"Sample {sample_id}: Running metric '{metric_name}'")
        metric_start = time.time()

        try:
            metric_context = {
                "viewport": viewport,
                "playwright": playwright_config,
                "artifacts_dir": artifacts_dir,
                "judge": judge_config,
                "layout_block_f1": layout_block_f1_config,
                "sketches": sketches_context,
            }
            result, artifacts = await metric.run(sample, generated_html_path, metric_context)
            metric_elapsed = time.time() - metric_start
            logger.debug(
                f"Sample {sample_id}: Metric '{metric_name}' completed in {metric_elapsed:.2f}s"
            )

            metric_results[metric_name] = result
            metric_artifacts[metric_name] = {name: str(path) for name, path in artifacts.items()}

            if isinstance(result, dict) and result.get("status") == "error":
                raise RuntimeError(str(result.get("error") or "Metric returned status=error"))

        except Exception as e:
            metric_elapsed = time.time() - metric_start
            logger.warning(
                f"Sample {sample_id}: Metric '{metric_name}' failed after {metric_elapsed:.2f}s: {e}",
                exc_info=True,
            )
            sample_errors.append({
                "metric": metric_name,
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
            })
            metric_results[metric_name] = {"status": "error", "error": str(e)}

    elapsed_time = time.time() - start_time

    # --- Write errors (this session's failures only; clear stale file if none) ---
    if sample_errors:
        logger.warning(
            f"Sample {sample_id}: Completed with {len(sample_errors)} metric error(s)"
        )
        with open(errors_json_path, "w") as f:
            json.dump(sample_errors, f, indent=2)
        logger.debug(f"Sample {sample_id}: Wrote errors to {errors_json_path}")
    elif errors_json_path.exists():
        errors_json_path.unlink()

    # --- Merge this session's results with any previously-computed metrics for ---
    # --- this sample so metrics outside the current config's list are preserved ---
    merged_metric_results = {**existing_metric_results, **metric_results}
    merged_metric_artifacts = {**existing_metric_artifacts, **metric_artifacts}

    # --- Atomic write of metrics.json (no lock needed — path is unique per sample) ---
    metrics_data = {
        "sample_id": sample_id,
        "metrics": merged_metric_results,
        "artifacts": merged_metric_artifacts,
        "elapsed_time_seconds": elapsed_time,
    }
    temp_path = metrics_json_path.with_suffix(".json.tmp")
    with open(temp_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    temp_path.replace(metrics_json_path)
    logger.debug(f"Sample {sample_id}: Wrote metrics.json atomically")

    # --- Append to results.jsonl (lock: shared append-only file) ---
    async with results_lock:
        with open(results_path, "a") as f:
            f.write(json.dumps({"id": sample_id, **merged_metric_results}) + "\n")

    # --- Append to manifest.jsonl (lock: shared append-only file) ---
    has_errors = bool(sample_errors)
    status = "failed" if has_errors else "success"
    manifest_entry: dict[str, Any] = {
        "id": sample_id,
        "status": status,
        "elapsed_time_seconds": elapsed_time,
        "metrics_run": list(metric_results.keys()),
    }
    if sample_errors:
        manifest_entry["error_count"] = len(sample_errors)

    async with manifest_lock:
        with open(manifest_path, "a") as f:
            f.write(json.dumps(manifest_entry) + "\n")

    if has_errors:
        logger.warning(
            f"Sample {sample_id}: Completed with {len(sample_errors)} error(s) in {elapsed_time:.2f}s"
        )
    else:
        logger.info(f"Sample {sample_id}: Completed successfully in {elapsed_time:.2f}s")

    return sample_id, status


async def run_evaluation(config_path: Path) -> None:
    """
    Run evaluation on generated outputs, processing samples in parallel.

    Args:
        config_path: Path to evaluation config JSON file

    Raises:
        FileNotFoundError: If config or run directory does not exist
        ValueError: If config is invalid
    """
    config = load_eval_config(config_path)

    output_root = config.get("output", {}).get("root", "runs")
    run_dir = resolve_run_root(config["run_id"], output_root)

    sample_ids = resolve_ids(config, run_dir)
    if not sample_ids:
        logger.warning("No samples found to evaluate")
        return

    dataset_root = config.get("dataset_root", "dataset")
    dataset = ResponsiveDataset(dataset_root)
    logger.debug(f"Initialized dataset from: {dataset_root}")

    eval_id = config["eval_id"]
    eval_dir = run_dir / "eval" / eval_id
    eval_samples_dir = eval_dir / "samples"
    eval_samples_dir.mkdir(parents=True, exist_ok=True)

    setup_eval_logging(config, eval_dir)

    max_workers = int(config.get("concurrency", {}).get("max_workers", 4))
    metric_names = config["metrics"]

    logger.info(
        f"Evaluation started: eval_id={eval_id}, run_id={config['run_id']}, "
        f"concurrency={max_workers}"
    )
    logger.info(f"Evaluating {len(sample_ids)} samples with metrics: {metric_names}")

    config_copy_path = eval_dir / "config.json"
    with open(config_copy_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.debug(f"Saved config copy to: {config_copy_path}")

    metrics = []
    for metric_name in metric_names:
        try:
            metric = get_metric(metric_name)
            metrics.append(metric)
            logger.debug(
                f"Initialized metric: {metric_name} (requires_browser={metric.requires_browser})"
            )
        except ValueError as e:
            logger.warning(f"{e}, skipping metric")

    if not metrics:
        raise ValueError("No valid metrics found")

    logger.info(f"Initialized {len(metrics)} metrics: {[m.name for m in metrics]}")

    manifest_path = eval_dir / "manifest.jsonl"
    results_path = eval_dir / "results.jsonl"

    viewport = config["viewport"]
    playwright_config = config["playwright"]
    judge_config = config.get("judge", {})
    layout_block_f1_config = config.get("layout_block_f1", {})
    overwrite = config.get("output", {}).get("overwrite", False)
    skip_if_exists = config.get("caching", {}).get("skip_if_metrics_exist", True)

    logger.debug(
        f"Processing config: overwrite={overwrite}, skip_if_exists={skip_if_exists}, "
        f"max_workers={max_workers}"
    )

    manifest_lock = asyncio.Lock()
    results_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max_workers)

    async def bounded(sample_id: str) -> tuple[str, str]:
        async with semaphore:
            return await _eval_one_sample(
                sample_id=sample_id,
                dataset=dataset,
                metrics=metrics,
                run_dir=run_dir,
                eval_samples_dir=eval_samples_dir,
                manifest_path=manifest_path,
                results_path=results_path,
                manifest_lock=manifest_lock,
                results_lock=results_lock,
                viewport=viewport,
                playwright_config=playwright_config,
                judge_config=judge_config,
                layout_block_f1_config=layout_block_f1_config,
                overwrite=overwrite,
                skip_if_exists=skip_if_exists,
            )

    counts: dict[str, int] = {"ok": 0, "skip": 0, "fail": 0}
    tasks = [bounded(sid) for sid in sample_ids]

    logger.info(f"Dispatching {len(sample_ids)} samples across {max_workers} concurrent workers")

    with logging_redirect_tqdm():
        with tqdm(
            total=len(sample_ids),
            desc="Evaluating",
            unit="sample",
            dynamic_ncols=True,
            colour="cyan",
            postfix=counts,
        ) as pbar:
            for coro in asyncio.as_completed(tasks):
                try:
                    sid, status = await coro
                except Exception as exc:
                    logger.error(f"Unhandled sample coroutine exception: {exc}")
                    sid, status = "unknown", "failed"

                if status == "success":
                    counts["ok"] += 1
                elif status == "skipped":
                    counts["skip"] += 1
                else:
                    counts["fail"] += 1

                pbar.set_postfix(counts, refresh=False)
                pbar.update(1)

    # --- Summary (sequential pass over completed metrics.json files) ---
    logger.debug("Computing evaluation summary")

    total_samples = len(sample_ids)
    summary: dict[str, Any] = {
        "eval_id": eval_id,
        "run_id": config["run_id"],
        "total_samples": total_samples,
        "evaluated": counts["ok"],
        "skipped": counts["skip"],
        "failed": counts["fail"],
        "success_rate": counts["ok"] / total_samples if total_samples > 0 else 0.0,
        "metrics": metric_names,
    }

    metric_stats: dict[str, Any] = {}
    for metric_name in metric_names:
        passed = 0
        total = 0
        for sample_id in sample_ids:
            metrics_json = eval_samples_dir / sample_id / "metrics.json"
            if metrics_json.exists():
                with open(metrics_json, "r") as f:
                    data = json.load(f)
                if metric_name in data.get("metrics", {}):
                    total += 1
                    metric_result = data["metrics"][metric_name]
                    if "status" in metric_result:
                        if metric_result["status"] == "success":
                            passed += 1
                    elif "horizontal_scroll_needed" in metric_result:
                        if not metric_result.get("horizontal_scroll_needed", True):
                            passed += 1
        if total > 0:
            metric_stats[metric_name] = {
                "passed": passed,
                "total": total,
                "pass_rate": passed / total,
            }

    summary["metric_stats"] = metric_stats

    summary_path = eval_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Evaluation complete: {eval_dir}")
    logger.info(
        f"Summary: total={total_samples}, ok={counts['ok']}, "
        f"skip={counts['skip']}, fail={counts['fail']}"
    )


def run_evaluation_sync(config_path: Path) -> None:
    """
    Synchronous wrapper for run_evaluation.

    Args:
        config_path: Path to evaluation config JSON file
    """
    asyncio.run(run_evaluation(config_path))
