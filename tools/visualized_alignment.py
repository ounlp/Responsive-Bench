"""
Publication-quality human-vs-LLM alignment figure for the responsive LLM judge.

Reuses the annotation loading, sample matching, and statistics helpers from
scripts/export_human_alignment.py rather than reimplementing a second matching
pipeline. This script adds the one thing that module does not expose: raw
per-annotator ratings and a Responsive-Bench / stress-test subset label per
sample-dimension pair, which are needed for the scatter figure and its CSV.

Output (default: results/):
  human_llm_alignment_scatter.pdf   vector figure for LaTeX \\includegraphics
  human_llm_alignment_scatter.svg   editable vector figure
  human_llm_alignment_scatter.png   300 DPI raster for quick inspection
  human_llm_alignment_plot_data.csv exact plotted data (one row per sample-dimension pair)

Usage:
  conda run -n responsive python tools/visualized_alignment.py \\
      [--annotations-file human_study/annotations.jsonl] [--runs-root runs] [--output-dir results] \\
      [--generate-alignment-figure | --no-generate-alignment-figure]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import export_human_alignment as eha  # noqa: E402 — reused matching/stats helpers

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Published statistics this script validates its recomputation against.
# These are targets, not the values that get drawn on the figure — the
# figure always shows numbers freshly computed from the plotted data.
# ---------------------------------------------------------------------------
EXPECTED_STATS: dict[str, float] = {
    "pearson": 0.844,
    "spearman": 0.752,
    "kendall": 0.666,
    "mae": 0.480,
    "acc1": 0.919,
}
EXPECTED_N = 310
EXPECTED_SUBSET_COUNTS = {"responsive_bench": 30, "stress_test": 26}
STATS_TOLERANCE = 0.01

SUBSET_RESPONSIVE_BENCH = "responsive_bench"
SUBSET_STRESS_TEST = "stress_test"

JITTER_SEED = 20260406  # fixed seed: jitter is reproducible, stats never see it
JITTER_STD = 0.06


def collect_sample_dimension_rows(
    annotations: list[dict[str, Any]],
    runs_root: Path,
    data_dir: Path | None,
) -> tuple[list[dict[str, Any]], list[str], int, dict[tuple[str, str, str], dict[str, dict[str, float]]]]:
    """Build one row per matched (sample, dimension) pair.

    Mirrors the two-pass approach in export_human_alignment.run_alignment (group
    human scores per unique sample, then pair with that sample's LLM judge score
    via the shared _find_metrics_file / _extract_llm_scores helpers), but keeps
    per-annotator scores and a subset label instead of collapsing straight to an
    aggregate — export_human_alignment's aggregate-only output can't support the
    per-annotator CSV columns or the subset-colored scatter this figure needs.

    Returns (rows, annotator_ids, n_skipped_samples, sample_dim_annotator) — the
    last two are returned for the row-count diagnostic in main().
    """
    # Pass 1: group raw per-annotator human scores by unique sample.
    sample_dim_annotator: dict[tuple[str, str, str], dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for ann in annotations:
        run_id = ann.get("run_id")
        eval_id = ann.get("eval_id")
        sample_id = ann.get("sample_id")
        annotator_id = ann.get("annotator_id")
        human_scores = ann.get("scores")
        if not (run_id and eval_id and sample_id and annotator_id and isinstance(human_scores, dict)):
            continue
        key = (run_id, eval_id, sample_id)
        for dim in eha.ALL_DIMENSIONS:
            h = human_scores.get(dim)
            if isinstance(h, (int, float)) and not isinstance(h, bool):
                sample_dim_annotator[key][dim][annotator_id] = float(h)

    # Fixed ordering so "annotator_1/2/3" CSV columns are stable across runs.
    annotator_ids = sorted(
        {aid for dims in sample_dim_annotator.values() for scores in dims.values() for aid in scores}
    )

    rows: list[dict[str, Any]] = []
    n_skipped = 0

    # Pass 2: for each unique sample, locate its LLM judge scores and pair per dimension.
    for (run_id, eval_id, sample_id), dim_annotators in sorted(sample_dim_annotator.items()):
        metrics_file = eha._find_metrics_file(run_id, eval_id, sample_id, runs_root, data_dir)
        if metrics_file is None:
            n_skipped += 1
            continue
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        llm_scores = eha._extract_llm_scores(metrics_data)
        if not llm_scores:
            n_skipped += 1
            continue

        # A Responsive-Bench sample is judged with sketches attached
        # (unified_responsive_v1, 6 dimensions incl. sketch_fidelity); a
        # stress-test sample has no sketch and is judged with sketch_free_v1
        # (5 dimensions), so sketch_fidelity is simply absent from its scores.
        subset = SUBSET_RESPONSIVE_BENCH if "sketch_fidelity" in llm_scores else SUBSET_STRESS_TEST

        for dim in eha.ALL_DIMENSIONS:
            annotator_scores = dim_annotators.get(dim, {})
            llm_score = llm_scores.get(dim)
            # Exclude missing/not-applicable dimensions rather than zero-filling.
            if not annotator_scores or llm_score is None:
                continue
            mean_human = sum(annotator_scores.values()) / len(annotator_scores)
            row: dict[str, Any] = {
                "sample_id": sample_id,
                "run_id": run_id,
                "eval_id": eval_id,
                "subset": subset,
                "dimension": dim,
                "mean_human_rating": mean_human,
                "llm_rating": float(llm_score),
            }
            for i, aid in enumerate(annotator_ids, start=1):
                row[f"annotator_{i}_rating"] = annotator_scores.get(aid)
            rows.append(row)

    return rows, annotator_ids, n_skipped, sample_dim_annotator


def diagnose_row_count(
    rows: list[dict[str, Any]],
    sample_dim_annotator: dict[tuple[str, str, str], dict[str, dict[str, float]]],
    n_skipped_samples: int,
) -> None:
    """Explain why the plotted row count differs from the expected 310, if it does."""
    print(f"  Row-count diagnostic (expected {EXPECTED_N} sample-dimension pairs, got {len(rows)}):")
    print(f"    by subset: {dict(Counter(r['subset'] for r in rows))}")
    print(f"    by dimension: {dict(Counter(r['dimension'] for r in rows))}")
    print(f"    annotated samples with no matched metrics.json: {n_skipped_samples}")

    sample_id_to_keys: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for key in sample_dim_annotator:
        sample_id_to_keys[key[2]].append(key)
    dupes = {sid: keys for sid, keys in sample_id_to_keys.items() if len(keys) > 1}
    if dupes:
        print(f"    sample_id values reused across multiple (run_id, eval_id) pairs: {dupes}")


def compute_and_validate_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute alignment statistics from the (unjittered) plotted rows.

    EXPECTED_STATS are validation targets from the published report, not the
    source of the numbers shown on the figure — those always come from `metrics`.
    """
    human = [r["mean_human_rating"] for r in rows]
    llm = [r["llm_rating"] for r in rows]
    metrics = eha._compute_metrics(human, llm)  # n, mae, acc1, pearson, spearman, kendall (see export_human_alignment)

    if metrics["n"] != EXPECTED_N:
        warnings.warn(
            f"Plotted sample-dimension count is {metrics['n']}, expected {EXPECTED_N}. "
            "See the row-count diagnostic above for likely causes."
        )

    for key, expected in EXPECTED_STATS.items():
        computed = metrics.get(key)
        if computed is None:
            warnings.warn(f"Could not recompute '{key}' from the plotted data.")
            continue
        if abs(computed - expected) > STATS_TOLERANCE:
            warnings.warn(
                f"Recomputed {key}={computed:.4f} differs from the published value "
                f"{expected:.4f} by more than the {STATS_TOLERANCE} tolerance."
            )
        else:
            print(f"    matches published value: {key}={computed:.4f} (expected {expected:.4f})")

    return metrics


def export_csv(rows: list[dict[str, Any]], annotator_ids: list[str], output_path: Path) -> None:
    """Write the exact (unjittered) plotted data — one row per sample-dimension pair."""
    fieldnames = ["sample_id", "run_id", "eval_id", "subset", "dimension"]
    fieldnames += [f"annotator_{i}_rating" for i in range(1, len(annotator_ids) + 1)]
    fieldnames += ["mean_human_rating", "llm_rating"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})


def render_alignment_figure(
    rows: list[dict[str, Any]],
    stats: dict[str, Any],
    output_dir: Path,
    seed: int = JITTER_SEED,
    jitter_std: float = JITTER_STD,
) -> dict[str, Path]:
    """Render the ACL-column-width scatter figure and save it as PDF/SVG/PNG."""
    rng = np.random.default_rng(seed)  # visualization-only jitter; stats above never see this

    human = np.array([r["mean_human_rating"] for r in rows], dtype=float)
    llm = np.array([r["llm_rating"] for r in rows], dtype=float)
    subset = np.array([r["subset"] for r in rows])

    human_jittered = human + rng.normal(0, jitter_std, size=len(rows))
    llm_jittered = llm + rng.normal(0, jitter_std, size=len(rows))

    slope, intercept = np.polyfit(human, llm, 1)  # fit on unjittered data
    fit_x = np.array([0.75, 5.25])

    # Legend counts are unique samples (30/26), not sample-dimension rows (180/130).
    unique_sample_counts: dict[str, int] = {}
    for key in (SUBSET_RESPONSIVE_BENCH, SUBSET_STRESS_TEST):
        n_unique = len({(r["run_id"], r["eval_id"], r["sample_id"]) for r in rows if r["subset"] == key})
        unique_sample_counts[key] = n_unique
        if n_unique != EXPECTED_SUBSET_COUNTS[key]:
            warnings.warn(
                f"Subset '{key}' has {n_unique} unique samples, expected {EXPECTED_SUBSET_COUNTS[key]}."
            )
    n_bench = unique_sample_counts[SUBSET_RESPONSIVE_BENCH]
    n_stress = unique_sample_counts[SUBSET_STRESS_TEST]

    plt.rcParams.update(
        {
            "font.size": 7,
            "font.family": "serif",
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 5.8,
            "pdf.fonttype": 42,  # embed as real (non-Type3) fonts
            "svg.fonttype": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(3.4, 3.5))

    # Marker shape (not just color) carries the subset distinction so the
    # figure stays interpretable when printed in grayscale.
    style = {
        SUBSET_RESPONSIVE_BENCH: dict(
            marker="o", facecolor="#1f77b4", label=f"Responsive-Bench (n={n_bench})"
        ),
        SUBSET_STRESS_TEST: dict(
            marker="^", facecolor="none", edgecolor="#d62728", label=f"Stress-test (n={n_stress})"
        ),
    }
    for key, s in style.items():
        mask = subset == key
        ax.scatter(
            human_jittered[mask],
            llm_jittered[mask],
            s=15,
            marker=s["marker"],
            facecolors=s["facecolor"],
            edgecolors=s.get("edgecolor", "black"),
            linewidths=0.45,
            alpha=0.55,
            label=s["label"],
            zorder=3,
        )

    ax.plot([1, 5], [1, 5], linestyle="--", color="gray", linewidth=1.0, zorder=1, label="Perfect agreement")
    ax.plot(fit_x, slope * fit_x + intercept, color="black", linewidth=1.1, zorder=2, label="Linear fit")

    ax.set_xlim(0.75, 5.25)
    ax.set_ylim(0.75, 5.25)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Mean human rating")
    ax.set_ylabel("LLM judge rating")
    ax.grid(True, linewidth=0.4, alpha=0.3, zorder=0)
    ax.legend(loc="lower right", frameon=True, framealpha=0.85, borderpad=0.4, handletextpad=0.4)

    stats_text = (
        f"Pearson r = {stats['pearson']:.3f}\n"
        f"Spearman ρ = {stats['spearman']:.3f}\n"
        f"Kendall τ = {stats['kendall']:.3f}\n"
        f"MAE = {stats['mae']:.3f}\n"
        f"±1 agreement = {stats['acc1'] * 100:.1f}%\n"
        f"n = {stats['n']}"
    )
    ax.text(
        0.03,
        0.97,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=5.6,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.6", alpha=0.9),
        zorder=4,
    )

    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for ext in ("pdf", "svg", "png"):
        p = output_dir / f"human_llm_alignment_scatter.{ext}"
        fig.savefig(p, format=ext, dpi=300, bbox_inches="tight")
        paths[ext] = p
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the publication-quality human-vs-LLM alignment scatter figure and CSV.",
    )
    parser.add_argument("--annotations-file", type=Path, default=Path("human_study/annotations.jsonl"), help="JSONL file of human annotations (default: human_study/annotations.jsonl)")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"), help="Root directory of runs (default: runs)")
    parser.add_argument("--frontend-public", type=Path, default=Path("frontend/public"), help="Frontend public dir for fallback metrics lookup (default: frontend/public)")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Directory for figure + CSV outputs (default: results)")
    parser.add_argument("--jitter-seed", type=int, default=JITTER_SEED, help="Seed for reproducible visualization-only jitter")
    parser.add_argument("--jitter-std", type=float, default=JITTER_STD, help="Std-dev of the Gaussian jitter applied for plotting only")
    parser.add_argument(
        "--generate-alignment-figure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate the scatter figure + CSV (default: enabled; pass --no-generate-alignment-figure to only print stats)",
    )
    args = parser.parse_args()

    runs_root = (REPO_ROOT / args.runs_root).resolve()
    data_dir = (REPO_ROOT / args.frontend_public / "data").resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()

    annotations = eha.load_annotations((REPO_ROOT / args.annotations_file).resolve())
    print(f"Loaded {len(annotations)} annotations from {args.annotations_file}.")

    rows, annotator_ids, n_skipped, sample_dim_annotator = collect_sample_dimension_rows(
        annotations, runs_root, data_dir
    )
    print(f"Matched {len(rows)} sample-dimension pairs across {len(annotator_ids)} annotators: {annotator_ids}")

    if len(rows) != EXPECTED_N:
        diagnose_row_count(rows, sample_dim_annotator, n_skipped)

    print("Validating recomputed statistics against published values:")
    stats = compute_and_validate_stats(rows)

    if not args.generate_alignment_figure:
        print("Skipping figure/CSV generation (--no-generate-alignment-figure).")
        return

    csv_path = output_dir / "human_llm_alignment_plot_data.csv"
    export_csv(rows, annotator_ids, csv_path)

    figure_paths = render_alignment_figure(
        rows, stats, output_dir, seed=args.jitter_seed, jitter_std=args.jitter_std
    )

    print("Generated artifacts:")
    print(f"  {csv_path}")
    for ext in ("pdf", "svg", "png"):
        print(f"  {figure_paths[ext]}")


if __name__ == "__main__":
    main()
