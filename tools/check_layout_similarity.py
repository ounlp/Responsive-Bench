"""
Sanity-check script: layout similarity between ground-truth HTML and an augmented HTML directory.

Compares every <id>.html found in --candidate-dir against the matching
dataset/source_html/<id>.html, prints a summary table, and writes a JSON results file.

Usage:
    conda run -n responsive python tools/check_layout_similarity.py
    conda run -n responsive python tools/check_layout_similarity.py \\
        --candidate-dir dataset/layout_similarity_study/html_and_screenshots \\
        --concurrency 4
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

# Make the responsive package importable without installing it.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from responsive.eval.metrics.layout_similarity import LayoutSimilarityMetric


async def _evaluate_one(
    sample_id: str,
    reference_html: Path,
    candidate_html: Path,
    artifacts_dir: Path,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        sample = SimpleNamespace(reference_html=reference_html)
        context = {
            "artifacts_dir": artifacts_dir,
            "playwright": {"headless": True, "timeout_ms": 60000},
        }
        try:
            result, _ = await LayoutSimilarityMetric().run(sample, candidate_html, context)
            return {"id": sample_id, **result}
        except Exception as exc:
            return {"id": sample_id, "status": "error", "error": str(exc)}


def _print_table(results: list[dict]) -> None:
    score_keys: list[str] = []
    for r in results:
        if r.get("status") == "success" and "multi_scores" in r:
            score_keys = sorted(r["multi_scores"].keys())
            break

    W_ID = 8
    W_IOU = 14
    W_CAT = 12

    header = f"{'id':<{W_ID}}  {'weighted_iou':<{W_IOU}}"
    for k in score_keys:
        header += f"  {k:<{W_CAT}}"
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    iou_values: list[float] = []
    for r in sorted(results, key=lambda x: x["id"]):
        if r.get("status") != "success":
            print(f"{r['id']:<{W_ID}}  {'ERROR':<{W_IOU}}  {r.get('error', '')}")
            continue
        iou = r["weighted_iou"]
        iou_values.append(iou)
        row = f"{r['id']:<{W_ID}}  {iou:<{W_IOU}.4f}"
        for k in score_keys:
            score = r.get("multi_scores", {}).get(k, {}).get("iou_score", float("nan"))
            row += f"  {score:<{W_CAT}.4f}"
        print(row)

    if iou_values:
        n = len(iou_values)
        mean_iou = sum(iou_values) / n
        print(sep)
        print(f"{'mean':<{W_ID}}  {mean_iou:<{W_IOU}.4f}")
        print(f"{'min':<{W_ID}}  {min(iou_values):<{W_IOU}.4f}")
        print(f"{'max':<{W_ID}}  {max(iou_values):<{W_IOU}.4f}")

    print(sep)


async def main(args: argparse.Namespace) -> None:
    candidate_dir = Path(args.candidate_dir)
    html_dir = Path(args.dataset_root) / "source_html"
    results_dir = Path(args.output_root) / f"sanity_layout_similarity_{candidate_dir.name}"
    results_dir.mkdir(parents=True, exist_ok=True)

    candidate_files = sorted(candidate_dir.glob("*.html"))
    if not candidate_files:
        print(f"No HTML files found in {candidate_dir}", file=sys.stderr)
        sys.exit(1)

    pairs: list[tuple[str, Path, Path]] = []
    for candidate_html in candidate_files:
        sample_id = candidate_html.stem
        reference_html = html_dir / f"{sample_id}.html"
        if not reference_html.exists():
            print(f"Warning: no reference HTML for {sample_id}, skipping", file=sys.stderr)
            continue
        pairs.append((sample_id, reference_html, candidate_html))

    print(f"Evaluating {len(pairs)} samples")
    print(f"  candidate : {candidate_dir}")
    print(f"  reference : {html_dir}")
    print(f"  results   : {results_dir}")
    print(f"  concurrency: {args.concurrency}\n")

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        _evaluate_one(
            sample_id=sid,
            reference_html=ref,
            candidate_html=cand,
            artifacts_dir=results_dir / sid,
            semaphore=semaphore,
        )
        for sid, ref, cand in pairs
    ]

    results: list[dict] = await asyncio.gather(*tasks)

    print()
    _print_table(results)

    iou_values = [r["weighted_iou"] for r in results if r.get("status") == "success"]
    summary: dict = {}
    if iou_values:
        summary = {
            "n": len(iou_values),
            "mean_weighted_iou": round(sum(iou_values) / len(iou_values), 4),
            "min_weighted_iou": round(min(iou_values), 4),
            "max_weighted_iou": round(max(iou_values), 4),
        }

    output = {
        "candidate_dir": str(candidate_dir),
        "dataset_root": str(args.dataset_root),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "samples": {
            r["id"]: {k: v for k, v in r.items() if k != "id"}
            for r in sorted(results, key=lambda x: x["id"])
        },
        "summary": summary,
    }

    results_json = results_dir / "results.json"
    results_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults written to: {results_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare layout similarity between ground-truth HTML and an augmented HTML directory."
    )
    parser.add_argument(
        "--candidate-dir",
        default="dataset/layout_similarity_study/html_and_screenshots",
        help="Directory of candidate HTML files (default: dataset/layout_similarity_study/html_and_screenshots)",
    )
    parser.add_argument(
        "--dataset-root",
        default="dataset",
        help="Dataset root containing source_html/ subdirectory (default: dataset)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max concurrent evaluations (default: 4)",
    )
    parser.add_argument(
        "--output-root",
        default="results",
        help="Root directory for results (default: results)",
    )
    asyncio.run(main(parser.parse_args()))
