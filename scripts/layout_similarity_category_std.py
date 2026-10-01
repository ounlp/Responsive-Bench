#!/usr/bin/env python3
"""
Per-viewport standard deviation of layout_similarity weighted IoU across
samples, at both the global (whole-page) level and per semantic category.

layout_similarity computes an independent category-weighted IoU at each
viewport for every sample (see layout_similarity/metric.py's `viewports`
schema: metric["viewports"][vp]["weighted_iou"] for the global score,
metric["viewports"][vp]["multi_scores"][category]["iou_score"] per category).
This script loads all samples in an eval run and reports the standard
deviation -- plus mean/count for context -- of both: one row per viewport for
the global table, one row per (category, viewport) for the category table.

Requires the new 3-viewport schema (metric["viewports"]); raises if it finds
old-schema (pre-viewport) data.

Usage:
    conda run -n responsive python scripts/layout_similarity_category_std.py \
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


def collect_rows(eval_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (global_df, category_df):
      - global_df: one row per (sample, viewport) with its overall weighted IoU.
      - category_df: one row per (sample, viewport, category) with its IoU score.
    """
    global_rows = []
    category_rows = []
    for sample_id, metric in _iter_sample_metrics(eval_dir, "layout_similarity"):
        if metric.get("status") != "success":
            continue
        viewports = metric.get("viewports")
        if not viewports:
            raise ValueError(
                f"Sample {sample_id} has old-schema layout_similarity data (no 'viewports' key). "
                "Rerun the eval with the current metric code before using this script."
            )
        for vp_name, vp_data in viewports.items():
            global_rows.append(
                {
                    "sample_id": sample_id,
                    "viewport": vp_name,
                    "weighted_iou": vp_data["weighted_iou"],
                }
            )
            for category, stats in vp_data.get("multi_scores", {}).items():
                category_rows.append(
                    {
                        "sample_id": sample_id,
                        "viewport": vp_name,
                        "category": category,
                        "iou_score": stats["iou_score"],
                    }
                )

    if not global_rows:
        raise ValueError(f"No successful layout_similarity samples found under {eval_dir}")
    return pd.DataFrame(global_rows), pd.DataFrame(category_rows)


def build_global_std_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per viewport with std/mean/n of the overall weighted IoU across samples."""
    grouped = df.groupby("viewport")["weighted_iou"].agg(["std", "mean", "count"]).reset_index()
    grouped = grouped.rename(columns={"std": "std_weighted_iou", "mean": "mean_weighted_iou", "count": "n"})

    present_viewports = [v for v in VIEWPORT_ORDER if v in grouped["viewport"].unique()]
    grouped["viewport"] = pd.Categorical(grouped["viewport"], categories=present_viewports, ordered=True)
    return grouped.sort_values("viewport").reset_index(drop=True)


def build_category_std_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to one row per category, std/mean/n column triple per viewport."""
    grouped = df.groupby(["category", "viewport"])["iou_score"].agg(["std", "mean", "count"]).reset_index()

    present_viewports = [v for v in VIEWPORT_ORDER if v in grouped["viewport"].unique()]

    std_table = grouped.pivot(index="category", columns="viewport", values="std").reindex(columns=present_viewports)
    mean_table = grouped.pivot(index="category", columns="viewport", values="mean").reindex(columns=present_viewports)
    count_table = grouped.pivot(index="category", columns="viewport", values="count").reindex(columns=present_viewports)

    std_table.columns = [f"std_iou_{v}" for v in present_viewports]
    mean_table.columns = [f"mean_iou_{v}" for v in present_viewports]
    count_table.columns = [f"n_{v}" for v in present_viewports]

    ordered_cols = [c for v in present_viewports for c in (f"std_iou_{v}", f"mean_iou_{v}", f"n_{v}")]
    combined = pd.concat([std_table, mean_table, count_table], axis=1)[ordered_cols]
    return combined.reset_index().sort_values("category")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Per-viewport standard deviation of layout_similarity weighted IoU, "
        "at both the global and per-category level"
    )
    parser.add_argument(
        "eval_dir",
        type=str,
        nargs="?",
        default="runs/run_accessible_js/eval/eval_accessible_js_block_f1",
        help="Path to the eval run directory (containing results.jsonl or samples/)",
    )
    parser.add_argument(
        "--output", type=str, help="Optional path to write the tables (.csv or .json; category table gets a _by_category suffix)"
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        print(f"Error: eval dir not found: {eval_dir}", file=sys.stderr)
        return 1

    global_df, category_df = collect_rows(eval_dir)
    global_table = build_global_std_table(global_df)
    category_table = build_category_std_table(category_df)

    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    print(f"\n=== Global weighted IoU standard deviation by viewport ({eval_dir}) ===")
    print(global_table.to_string(index=False))

    print(f"\n=== Per-category IoU standard deviation by viewport ({eval_dir}) ===")
    print(category_table.to_string(index=False))

    if args.output:
        out_path = Path(args.output)
        category_path = out_path.with_stem(out_path.stem + "_by_category")
        if out_path.suffix == ".json":
            out_path.write_text(json.dumps(global_table.to_dict(orient="records"), indent=2))
            category_path.write_text(json.dumps(category_table.to_dict(orient="records"), indent=2))
        else:
            global_table.to_csv(out_path, index=False)
            category_table.to_csv(category_path, index=False)
        print(f"\nWrote global table to {out_path}")
        print(f"Wrote per-category table to {category_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
