#!/usr/bin/env python3
"""
Evaluate metrics on a flat directory of HTML files.

Scans <input-dir>/*.html and runs the requested metrics on each file,
writing all results to a single JSON file.

Usage:
    conda run -n responsive python scripts/eval_dataset.py \\
        --input-dir dataset/references \\
        --eval-types accessibility \\
        --output results/references_accessibility.json \\
        --concurrency 4
"""

import argparse
import asyncio
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from responsive.eval.metrics.registry import get_metric

EVAL_TYPES: dict[str, list[str]] = {
    "accessibility": ["accessibility", "accessible_form_labels"],
    "responsiveness": [
        "viewport_meta_tag",
        "heading_order",
        "horizontal_scroll",
        "tap_targets",
        "line_spacing",
        "responsive_media",
        "relative_units",
        "computed_font_sizes",
        "rotation_friendly",
        "text_media_cutoff_overlap",
        "text_enlargement",
    ],
}

PLAYWRIGHT_CONFIG = {"headless": True, "timeout_ms": 30000}


async def _eval_one(
    sample_id: str,
    html_path: Path,
    metrics: list,
    tmp_dir: Path,
    semaphore: asyncio.Semaphore,
) -> tuple[str, dict]:
    async with semaphore:
        artifacts_dir = tmp_dir / sample_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        context = {
            "artifacts_dir": artifacts_dir,
            "playwright": PLAYWRIGHT_CONFIG,
            "viewport": {"width": 375, "height": 688},
            "judge": {},
            "sketches": {},
        }

        results = {}
        for metric in metrics:
            try:
                result, _ = await metric.run(None, html_path, context)
                results[metric.name] = result
            except Exception as e:
                results[metric.name] = {"status": "error", "error": str(e)}

        return sample_id, results


async def run(
    input_dir: Path,
    metrics: list,
    output_path: Path,
    concurrency: int,
    ids: list[str] | None,
) -> None:
    html_files = sorted(input_dir.glob("*.html"))
    if ids:
        id_set = set(ids)
        html_files = [f for f in html_files if f.stem in id_set]

    if not html_files:
        print(f"No HTML files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    metric_names = [m.name for m in metrics]
    print(f"Evaluating {len(html_files)} samples | metrics: {metric_names}")

    semaphore = asyncio.Semaphore(concurrency)
    all_results: dict[str, dict] = {}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        tasks = [
            _eval_one(f.stem, f, metrics, tmp_dir, semaphore)
            for f in html_files
        ]

        total = len(tasks)
        done = 0
        for coro in asyncio.as_completed(tasks):
            sample_id, results = await coro
            all_results[sample_id] = results
            done += 1
            errors = [k for k, v in results.items() if v.get("status") == "error"]
            status = f"FAIL({','.join(errors)})" if errors else "ok"
            print(f"  [{done}/{total}] {sample_id}  {status}", flush=True)

    failed_samples = [
        sid for sid, r in all_results.items()
        if any(v.get("status") == "error" for v in r.values())
    ]

    output = {
        "meta": {
            "input_dir": str(input_dir),
            "metrics": metric_names,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "summary": {
            "total": total,
            "succeeded": total - len(failed_samples),
            "failed": len(failed_samples),
            "failed_ids": sorted(failed_samples),
        },
        "samples": dict(sorted(all_results.items())),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults written to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate metrics on a flat directory of HTML files"
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory of *.html files")
    parser.add_argument("--output", type=Path, help="Output JSON path (default: results/<dir>_<types>.json)")
    parser.add_argument(
        "--eval-types",
        nargs="+",
        choices=list(EVAL_TYPES),
        default=["accessibility"],
        metavar="TYPE",
        help=f"Eval type groups: {list(EVAL_TYPES)}",
    )
    parser.add_argument("--metrics", nargs="+", default=[], metavar="METRIC", help="Additional individual metric names")
    parser.add_argument("--ids", nargs="+", metavar="ID", help="Limit to these sample IDs (e.g. 0001 0042)")
    parser.add_argument("--concurrency", type=int, default=4, help="Max parallel samples (default: 4)")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Expand eval types + any explicit metric names, deduplicated
    metric_names: list[str] = []
    for et in args.eval_types:
        metric_names.extend(EVAL_TYPES[et])
    metric_names.extend(args.metrics)
    metric_names = list(dict.fromkeys(metric_names))

    metrics = []
    for name in metric_names:
        try:
            metrics.append(get_metric(name))
        except ValueError as e:
            print(f"Warning: {e}", file=sys.stderr)

    if not metrics:
        print("No valid metrics resolved.", file=sys.stderr)
        sys.exit(1)

    types_str = "_".join(args.eval_types)
    default_filename = f"{input_dir.name}_{types_str}.json"

    if args.output is None:
        output_path = Path("results") / default_filename
    else:
        output_path = args.output.resolve()
        if output_path.is_dir():
            output_path = output_path / default_filename

    asyncio.run(run(input_dir, metrics, output_path, args.concurrency, args.ids))


if __name__ == "__main__":
    main()
