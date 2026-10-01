#!/usr/bin/env python3
"""
Per-category, per-viewport standard deviation of layout_block_f1_rra F1 scores
across samples.

layout_block_f1_rra computes an independent F1 per (category, viewport) for
every sample (see layout_block_f1_rra.py's `viewports` schema). This script
loads all samples in an eval run, groups F1 scores by (category, viewport),
and reports the standard deviation -- plus mean/count for context -- across
samples: one row per category, one std/mean/n column triple per viewport.

Requires the new 3-viewport schema (metric["viewports"][vp]["categories"]);
raises if it finds old-schema (pre-viewport) data.

Usage:
    conda run -n responsive python scripts/layout_block_f1_rra_category_std.py \
        runs/run_accessible_js/eval/eval_accessible_js_block_f1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

VIEWPORT_ORDER = ("mobile", "tablet", "desktop")


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


def collect_rows(eval_dir: Path) -> pd.DataFrame:
    """One row per (sample, viewport, category) with its F1 score."""
    rows = []
    for sample_id, metric in _iter_sample_metrics(eval_dir, "layout_block_f1_rra"):
        if metric.get("status") != "success":
            continue
        viewports = metric.get("viewports")
        if not viewports:
            raise ValueError(
                f"Sample {sample_id} has old-schema layout_block_f1_rra data (no 'viewports' key). "
                "Rerun the eval with the current metric code before using this script."
            )
        for vp_name, vp_data in viewports.items():
            for category, stats in vp_data.get("categories", {}).items():
                rows.append(
                    {
                        "sample_id": sample_id,
                        "viewport": vp_name,
                        "category": category,
                        "f1": stats["f1"],
                    }
                )

    if not rows:
        raise ValueError(f"No successful layout_block_f1_rra samples found under {eval_dir}")
    return pd.DataFrame(rows)


def build_std_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to one row per category, std/mean/n column triple per viewport."""
    grouped = df.groupby(["category", "viewport"])["f1"].agg(["std", "mean", "count"]).reset_index()

    present_viewports = [v for v in VIEWPORT_ORDER if v in grouped["viewport"].unique()]

    std_table = grouped.pivot(index="category", columns="viewport", values="std").reindex(columns=present_viewports)
    mean_table = grouped.pivot(index="category", columns="viewport", values="mean").reindex(columns=present_viewports)
    count_table = grouped.pivot(index="category", columns="viewport", values="count").reindex(columns=present_viewports)

    std_table.columns = [f"std_f1_{v}" for v in present_viewports]
    mean_table.columns = [f"mean_f1_{v}" for v in present_viewports]
    count_table.columns = [f"n_{v}" for v in present_viewports]

    ordered_cols = [c for v in present_viewports for c in (f"std_f1_{v}", f"mean_f1_{v}", f"n_{v}")]
    combined = pd.concat([std_table, mean_table, count_table], axis=1)[ordered_cols]
    return combined.reset_index().sort_values("category")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Per-category, per-viewport standard deviation of layout_block_f1_rra F1 scores"
    )
    parser.add_argument(
        "eval_dir",
        type=str,
        nargs="?",
        default="runs/run_accessible_js/eval/eval_accessible_js_block_f1",
        help="Path to the eval run directory (containing results.jsonl or samples/)",
    )
    parser.add_argument("--output", type=str, help="Optional path to write the table (.csv or .json)")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        print(f"Error: eval dir not found: {eval_dir}", file=sys.stderr)
        return 1

    df = collect_rows(eval_dir)
    std_table = build_std_table(df)

    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    print(f"\n=== Per-category F1 standard deviation by viewport ({eval_dir}) ===")
    print(std_table.to_string(index=False))

    if args.output:
        out_path = Path(args.output)
        if out_path.suffix == ".json":
            out_path.write_text(json.dumps(std_table.to_dict(orient="records"), indent=2))
        else:
            std_table.to_csv(out_path, index=False)
        print(f"\nWrote table to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
