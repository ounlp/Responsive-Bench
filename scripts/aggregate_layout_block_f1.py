#!/usr/bin/env python3
"""
Aggregate per-sample layout_block_f1 / layout_block_f1_rra metrics into
dataset-level statistics. Both metrics share the same tp/fp/fn/f1_weighted
schema, so one script covers either.

Reads results.jsonl from an eval run directory (falling back to walking
samples/*/metrics.json if results.jsonl is absent) and reports:
  - overall f1_weighted distribution across samples
  - per-category micro P/R/F1 (TP/FP/FN summed across all samples)
  - per-category macro F1 (mean of per-sample F1, over samples where the
    category actually appears in reference or generated output)

Usage:
    conda run -n responsive python scripts/aggregate_layout_block_f1.py \
        runs/run_accessible_js/eval/eval_accessible_js_block_f1
    conda run -n responsive python scripts/aggregate_layout_block_f1.py \
        runs/run_accessible_js/eval/eval_accessible_js_block_f1_rra \
        --metric-key layout_block_f1_rra --output summary.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def _iter_sample_metrics(eval_dir: Path, metric_key: str):
    """
    Yield (sample_id, metric_dict) for every sample in the eval run.

    Prefers samples/*/metrics.json: each file is the orchestrator's merged,
    deduplicated current state for that sample (one file per sample, written
    in place). results.jsonl is append-only across every eval invocation, so
    a sample touched by more than one run (e.g. an incremental metric
    backfill) has multiple lines there -- reading it directly would
    double-count that sample. It's kept only as a fallback for eval dirs that
    predate the samples/ layout.
    """
    samples_dir = eval_dir / "samples"
    if samples_dir.exists():
        for sample_dir in sorted(samples_dir.iterdir()):
            metrics_path = sample_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            record = json.loads(metrics_path.read_text())
            metric = record.get("metrics", {}).get(metric_key)
            if metric is not None:
                yield record.get("sample_id", sample_dir.name), metric
        return

    results_path = eval_dir / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"Neither samples/ nor results.jsonl found under {eval_dir}")
    with results_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            metric = record.get(metric_key)
            if metric is not None:
                yield record["id"], metric


def aggregate(eval_dir: Path, metric_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (overall_df, per_category_df) summary tables."""
    sample_rows = []
    category_rows = []

    for sample_id, metric in _iter_sample_metrics(eval_dir, metric_key):
        if metric.get("status") != "success":
            continue
        sample_rows.append({"sample_id": sample_id, "f1_weighted": metric["f1_weighted"]})
        for category, stats in metric.get("categories", {}).items():
            category_rows.append(
                {
                    "sample_id": sample_id,
                    "category": category,
                    "tp": stats["tp"],
                    "fp": stats["fp"],
                    "fn": stats["fn"],
                    "f1": stats["f1"],
                    "weight_fraction": stats["weight_fraction"],
                    "present": (stats["tp"] + stats["fp"] + stats["fn"]) > 0,
                }
            )

    if not sample_rows:
        raise ValueError(f"No successful layout_block_f1 samples found under {eval_dir}")

    samples_df = pd.DataFrame(sample_rows)
    cat_df = pd.DataFrame(category_rows)

    overall_df = pd.DataFrame(
        [
            {
                "num_samples": len(samples_df),
                "mean_f1_weighted": samples_df["f1_weighted"].mean(),
                "median_f1_weighted": samples_df["f1_weighted"].median(),
                "std_f1_weighted": samples_df["f1_weighted"].std(),
                "min_f1_weighted": samples_df["f1_weighted"].min(),
                "max_f1_weighted": samples_df["f1_weighted"].max(),
            }
        ]
    )

    def _summarize_category(group: pd.DataFrame) -> pd.Series:
        tp, fp, fn = group["tp"].sum(), group["fp"].sum(), group["fn"].sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        micro_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        present = group[group["present"]]
        return pd.Series(
            {
                "num_present": len(present),
                "num_total": len(group),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "micro_precision": precision,
                "micro_recall": recall,
                "micro_f1": micro_f1,
                "macro_f1_when_present": present["f1"].mean() if len(present) else 0.0,
                "mean_weight_fraction_when_present": present["weight_fraction"].mean() if len(present) else 0.0,
            }
        )

    per_category_df = (
        cat_df.groupby("category", sort=True).apply(_summarize_category, include_groups=False).reset_index()
    )

    return overall_df, per_category_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate layout_block_f1 metrics across an eval run's samples")
    parser.add_argument(
        "eval_dir",
        type=str,
        nargs="?",
        default="runs/run_accessible_js/eval/eval_accessible_js_block_f1",
        help="Path to the eval run directory (containing results.jsonl or samples/)",
    )
    parser.add_argument(
        "--metric-key",
        type=str,
        default="layout_block_f1",
        choices=["layout_block_f1", "layout_block_f1_rra"],
        help="Which metric's results to aggregate (default: layout_block_f1)",
    )
    parser.add_argument("--output", type=str, help="Optional path to write the per-category table (.csv or .json)")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        print(f"Error: eval dir not found: {eval_dir}", file=sys.stderr)
        return 1

    overall_df, per_category_df = aggregate(eval_dir, args.metric_key)

    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    print(f"\n=== Overall ({eval_dir}) ===")
    print(overall_df.to_string(index=False))

    print("\n=== Per category ===")
    print(per_category_df.sort_values("mean_weight_fraction_when_present", ascending=False).to_string(index=False))

    if args.output:
        out_path = Path(args.output)
        if out_path.suffix == ".json":
            out_path.write_text(
                json.dumps(
                    {
                        "overall": overall_df.to_dict(orient="records")[0],
                        "per_category": per_category_df.to_dict(orient="records"),
                    },
                    indent=2,
                )
            )
        else:
            per_category_df.to_csv(out_path, index=False)
        print(f"\nWrote per-category summary to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
