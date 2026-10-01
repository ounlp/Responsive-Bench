#!/usr/bin/env python3
"""
Export human-annotation data to static JSON files for the frontend.

Human ratings are read from human_study/annotations.jsonl (three pseudonymized
annotators, 56 webpages); LLM judge scores are read from the eval outputs
under runs/.

Tasks:
  alignment        — per-dimension human vs LLM judge alignment metrics.
                     The default (non-override) run also writes two fixed
                     subset exports alongside the combined file:
                     alignment_benchmark.json (the 30 Responsive-Bench
                     samples) and alignment_stress.json (the 26 stress-test
                     samples) -- see BENCHMARK_SUBSET_KEY / STRESS_SUBSET_KEY.
                     It also (always, regardless of --eval-override) writes
                     inter_annotator_{benchmark,stress,combined}.json: human
                     vs human agreement, split the same three ways.
  annotator-scores — per-annotator/per-sample raw scores for the drill-down view

Usage:
  python scripts/export_human_alignment.py \\
      [--task {alignment,annotator-scores,all}] \\
      [--annotations-file human_study/annotations.jsonl] \\
      [--runs-root runs] \\
      [--frontend-public frontend/public] \\
      [--alignment-output frontend/public/data/human_alignment.json] \\
      [--annotator-scores-output frontend/public/data/annotator_scores.json]
"""

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

REQUIRED_DIMENSIONS = [
    "fluid_layout_and_container_adaptation",
    "flexible_grid_reflow",
    "responsive_media",
    "intentional_content_visibility",
    "proper_alignment_and_spacing",
]
OPTIONAL_DIMENSIONS = ["sketch_fidelity"]
ALL_DIMENSIONS = REQUIRED_DIMENSIONS + OPTIONAL_DIMENSIONS

DIMENSION_LABELS = {
    "fluid_layout_and_container_adaptation": "Fluid Layout & Container Adaptation",
    "flexible_grid_reflow": "Flexible Grid Reflow",
    "responsive_media": "Responsive Media",
    "intentional_content_visibility": "Intentional Content Visibility",
    "proper_alignment_and_spacing": "Proper Alignment & Spacing",
    "sketch_fidelity": "Sketch Fidelity",
}

# The two human-annotated subsets that make up the full 56-webpage validation set
# (see Section 3.3.2 of the paper): 30 pages sampled from the released
# 200-page Responsive-Bench dataset (full 6-dim rubric incl. Sketch Fidelity),
# and 26 curated stress-test pages (sketch-independent 5-dim rubric, no aligned
# sketches). These (run_id, eval_id) pairs match
# exactly 30 and 26 unique sample_ids respectively.
BENCHMARK_SUBSET_KEY = ("run_accessible_js", "eval_accessible_js")
STRESS_SUBSET_KEY = ("test_adversarial", "eval_test_adversarial")

# ---------------------------------------------------------------------------
# Statistics helpers (pure Python — no scipy required)
# ---------------------------------------------------------------------------

def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = _mean(xs), _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sdx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sdy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sdx == 0 or sdy == 0:
        return None
    return num / (sdx * sdy)


def _rank(xs: list[float]) -> list[float]:
    """Average-tie ranks (1-indexed)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) - 1 and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    return _pearson(_rank(xs), _rank(ys))


def _kendall_tau_b(xs: list[float], ys: list[float]) -> float | None:
    """Kendall's τ-b (handles ties)."""
    n = len(xs)
    if n < 2:
        return None
    nc = nd = tx = ty = 0
    pairs = n * (n - 1) // 2
    for i in range(n):
        for j in range(i + 1, n):
            a = xs[i] - xs[j]
            b = ys[i] - ys[j]
            if a * b > 0:
                nc += 1
            elif a * b < 0:
                nd += 1
            if a == 0:
                tx += 1
            if b == 0:
                ty += 1
    denom_sq = (pairs - tx) * (pairs - ty)
    if denom_sq <= 0:
        return None
    return (nc - nd) / math.sqrt(denom_sq)


def _compute_metrics(human: list[float], llm: list[float]) -> dict:
    n = len(human)
    if n == 0:
        return {"n": 0, "mse": None, "mae": None, "acc": None, "acc1": None,
                "pearson": None, "spearman": None, "kendall": None}
    diffs = [h - l for h, l in zip(human, llm)]
    mse = _mean([d ** 2 for d in diffs])
    mae = _mean([abs(d) for d in diffs])
    acc = sum(1 for d in diffs if d == 0) / n
    acc1 = sum(1 for d in diffs if abs(d) <= 1) / n
    return {
        "n": n,
        "mse": round(mse, 4),
        "mae": round(mae, 4),
        "acc": round(acc, 4),
        "acc1": round(acc1, 4),
        "pearson": _round_or_none(_pearson(human, llm)),
        "spearman": _round_or_none(_spearman(human, llm)),
        "kendall": _round_or_none(_kendall_tau_b(human, llm)),
    }


def _round_or_none(v: float | None, digits: int = 4) -> float | None:
    return round(v, digits) if v is not None else None


def _avg_defined(values: list[float | None], digits: int = 4) -> float | None:
    """Average of the non-None values, over the count of non-None values --
    not over len(values). A metric (most often Pearson/Spearman/Kendall)
    can be undefined for one annotator pair (e.g. zero variance on a
    near-unanimous dimension) without being undefined for the others;
    dividing by the total pair count instead of the defined count would
    silently bias the average toward zero.
    """
    defined = [v for v in values if v is not None]
    if not defined:
        return None
    return round(sum(defined) / len(defined), digits)


def _std(xs: list[float]) -> float | None:
    """Sample standard deviation (n-1 denominator)."""
    n = len(xs)
    if n < 2:
        return None
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def _compute_score_stats(human: list[float], llm: list[float]) -> dict:
    n = len(human)
    if n == 0:
        return {"human_mean": None, "human_std": None, "llm_mean": None, "llm_std": None}
    return {
        "human_mean": round(_mean(human), 4),
        "human_std": _round_or_none(_std(human)),
        "llm_mean": round(_mean(llm), 4),
        "llm_std": _round_or_none(_std(llm)),
    }


def _percentile_ci(estimates: list[float], ci: float = 0.95) -> tuple[float | None, float | None]:
    """Central `ci` interval from a list of bootstrap replicate estimates."""
    if not estimates:
        return None, None
    estimates = sorted(estimates)
    lo_idx = int((1 - ci) / 2 * len(estimates))
    hi_idx = min(int((1 + ci) / 2 * len(estimates)), len(estimates) - 1)
    return round(estimates[lo_idx], 4), round(estimates[hi_idx], 4)


def _pooled_corr_ci(
    clusters: list[list[tuple[float, float]]],
    stat_fn,
    n_boot: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[float | None, float | None]:
    """
    95% CI for a pooled correlation statistic (pass _pearson or _spearman as
    stat_fn) via cluster (webpage-level) bootstrap.

    Ratings from the same webpage are not independent -- resampling
    individual (x, y) pairs would understate uncertainty. Instead each
    element of `clusters` is all the pairs belonging to one webpage (a
    singleton list when computing a single dimension's CI, since a page
    contributes at most one rating per dimension; a multi-item list when
    pooling across dimensions, as for an Overall row). We resample webpages
    with replacement, pool the resampled webpages' pairs, and recompute the
    statistic each time.
    """
    n = len(clusters)
    if n < 2:
        return None, None
    rng = random.Random(seed)
    estimates = []
    for _ in range(n_boot):
        pooled_x: list[float] = []
        pooled_y: list[float] = []
        for _ in range(n):
            c = clusters[rng.randrange(n)]
            pooled_x.extend(p[0] for p in c)
            pooled_y.extend(p[1] for p in c)
        v = stat_fn(pooled_x, pooled_y)
        if v is not None:
            estimates.append(v)
    if len(estimates) < n_boot // 2:
        return None, None
    return _percentile_ci(estimates, ci)


def _inter_annotator_ci(
    sample_keys: list,
    by_sample: dict,
    pairs: list[tuple[str, str]],
    dims: list[str],
    stat_fn,
    n_boot: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[float | None, float | None]:
    """
    95% CI for the pair-averaged correlation statistic reported by
    run_inter_annotator (pass _pearson or _spearman as stat_fn), via
    webpage-level cluster bootstrap.

    Each replicate resamples webpages with replacement, then -- for every
    annotator pair -- pools that pair's raw scores across `dims` (a single
    dimension, or all of them for an Overall row) on the resampled
    webpages and computes the statistic; the replicate's value is the
    average across pairs with at least one pooled observation, mirroring
    how the point estimate itself is computed.
    """
    n = len(sample_keys)
    if n < 2:
        return None, None
    rng = random.Random(seed)
    estimates = []
    for _ in range(n_boot):
        resampled_keys = [sample_keys[rng.randrange(n)] for _ in range(n)]
        pair_values = []
        for a1, a2 in pairs:
            xs: list[float] = []
            ys: list[float] = []
            for key in resampled_keys:
                by_ann = by_sample[key]
                s1 = by_ann.get(a1, {})
                s2 = by_ann.get(a2, {})
                for dim in dims:
                    v1 = s1.get(dim)
                    v2 = s2.get(dim)
                    if (
                        isinstance(v1, (int, float)) and not isinstance(v1, bool)
                        and isinstance(v2, (int, float)) and not isinstance(v2, bool)
                    ):
                        xs.append(float(v1))
                        ys.append(float(v2))
            v = stat_fn(xs, ys)
            if v is not None:
                pair_values.append(v)
        if pair_values:
            estimates.append(sum(pair_values) / len(pair_values))
    if len(estimates) < n_boot // 2:
        return None, None
    return _percentile_ci(estimates, ci)

# ---------------------------------------------------------------------------
# LLM judge score extraction (mirrors build_runs_data.py logic)
# ---------------------------------------------------------------------------

def _extract_llm_scores(metrics_data: dict) -> dict[str, int] | None:
    """Extract per-dimension raw 1-5 scores from metrics.json."""
    metrics = metrics_data.get("metrics") or {}
    lj = metrics.get("llm_judge_screenshots")
    if not isinstance(lj, dict) or lj.get("status") != "success":
        return None
    scores = lj.get("scores")
    if not isinstance(scores, dict):
        return None
    result = {}
    for dim in ALL_DIMENSIONS:
        v = scores.get(dim)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            result[dim] = int(round(float(v)))
        elif isinstance(v, dict):
            raw = v.get("score")
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                result[dim] = int(round(float(raw)))
    return result if result else None


def _find_metrics_file(
    run_id: str,
    eval_id: str,
    sample_id: str,
    runs_root: Path,
    data_dir: Path | None,
) -> Path | None:
    """Locate metrics data for a sample.

    Tries two locations:
    1. runs/{run_id}/eval/{eval_id}/samples/{sample_id}/metrics.json  (source of truth)
    2. frontend/public/data/runs/{run_id}/evals/{eval_id}/samples/{sample_id}.json
       (pre-built by build_runs_data.py; used when the run has no local eval/ dir)
    """
    primary = runs_root / run_id / "eval" / eval_id / "samples" / sample_id / "metrics.json"
    if primary.exists():
        return primary
    if data_dir is not None:
        fallback = data_dir / "runs" / run_id / "evals" / eval_id / "samples" / f"{sample_id}.json"
        if fallback.exists():
            return fallback
    return None


def _read_judge_model(run_id: str, eval_id: str, runs_root: Path) -> str | None:
    """Read the judge.model used by a specific eval, from its saved config.json."""
    config_path = runs_root / run_id / "eval" / eval_id / "config.json"
    if not config_path.exists():
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        return None
    judge = config.get("judge") or {}
    return judge.get("model")


def _slugify_model(text: str) -> str:
    """A short filesystem-safe slug for a judge model string (used in output filenames)."""
    name = text.rsplit("/", 1)[-1]  # drop any org/repo prefix, e.g. "Qwen/Qwen3-VL-235B..."
    name = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return name or "model"


def build_alignment_index(data_dir: Path) -> None:
    """Scan data_dir for human_alignment*.json variants and write a manifest the
    frontend dropdown reads to discover what's available (see AlignmentComparison.tsx).
    """
    index_path = data_dir / "human_alignment_index.json"
    variants = []
    for f in sorted(data_dir.glob("human_alignment*.json")):
        if f.name in ("human_alignment_index.json",):
            continue
        try:
            with open(f, "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        is_default = f.name == "human_alignment.json"
        label = d.get("label") or ("Default" if is_default else f.stem.replace("human_alignment_", ""))
        variants.append({
            "file": f.name,
            "label": label,
            "judge_models": d.get("judge_models"),
            "generated_at": d.get("generated_at"),
            "n_matched_samples": d.get("n_matched_samples"),
        })

    variants.sort(key=lambda v: (v["file"] != "human_alignment.json", v["label"]))

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"generated_at": datetime.now(timezone.utc).isoformat(), "variants": variants}, f, indent=2)
        f.write("\n")
    print(f"  Wrote {index_path} ({len(variants)} variant(s))")

# ---------------------------------------------------------------------------
# Task: alignment
# ---------------------------------------------------------------------------

def run_alignment(
    annotations: list[dict],
    runs_root: Path,
    data_dir: Path | None,
    output_path: Path,
    eval_overrides: dict[str, str] | None = None,
    label: str = "Default",
) -> dict:
    """Compute human-vs-LLM-judge alignment and write it to output_path.

    eval_overrides maps run_id -> eval_id: for any annotation whose run_id is a key
    here, its LLM judge score is looked up under the given eval_id instead of the
    eval_id stored on the annotation itself. This is how a second judge model can be
    compared against the same human ratings without re-annotating anything — e.g.
    {"test_adversarial": "eval_test_adversarial_qwen3_235b"} compares the existing
    human scores against qwen3-235b's judging of the exact same generated pages,
    instead of the default eval's judge.

    Returns the output dict (also written to output_path) so callers (main()) don't
    need to re-read the file back just to build the variant index.
    """
    eval_overrides = eval_overrides or {}
    dim_pairs: dict[str, list[tuple[float, float]]] = {d: [] for d in ALL_DIMENSIONS}
    # All (human, llm) pairs for one webpage, pooled across its dimensions --
    # the resampling unit for the clustered bootstrap CI below.
    sample_pairs: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
    n_matched = 0
    n_skipped = 0
    judge_models: dict[str, str | None] = {}

    # Pass 1: group human scores by unique sample across all annotations.
    sample_human: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for ann in annotations:
        run_id = ann.get("run_id")
        eval_id = ann.get("eval_id")
        sample_id = ann.get("sample_id")
        human_scores = ann.get("scores")
        if not (run_id and eval_id and sample_id and isinstance(human_scores, dict)):
            continue
        key = (run_id, eval_id, sample_id)
        for dim in ALL_DIMENSIONS:
            h = human_scores.get(dim)
            if isinstance(h, (int, float)) and not isinstance(h, bool):
                sample_human[key][dim].append(float(h))

    # Pass 2: for each unique sample, average human scores then pair with LLM score.
    for (run_id, eval_id, sample_id), dim_scores in sorted(sample_human.items()):
        resolved_eval_id = eval_overrides.get(run_id, eval_id)
        metrics_file = _find_metrics_file(run_id, resolved_eval_id, sample_id, runs_root, data_dir)
        if metrics_file is None:
            n_skipped += 1
            continue

        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        llm_scores = _extract_llm_scores(metrics_data)
        if not llm_scores:
            n_skipped += 1
            continue

        if run_id not in judge_models:
            judge_models[run_id] = _read_judge_model(run_id, resolved_eval_id, runs_root)

        for dim in ALL_DIMENSIONS:
            human_list = dim_scores.get(dim, [])
            l = llm_scores.get(dim)
            if human_list and isinstance(l, (int, float)) and not isinstance(l, bool):
                mean_h = sum(human_list) / len(human_list)
                pair = (mean_h, float(l))
                dim_pairs[dim].append(pair)
                sample_pairs.setdefault((run_id, eval_id, sample_id), []).append(pair)

        n_matched += 1

    print(f"  Alignment — matched: {n_matched}, skipped: {n_skipped}")
    if eval_overrides:
        print(f"    eval_overrides: {eval_overrides}")
    print(f"    judge_models: {judge_models}")

    dim_results = []
    all_human: list[float] = []
    all_llm: list[float] = []

    for dim in ALL_DIMENSIONS:
        pairs = dim_pairs[dim]
        if not pairs:
            continue
        human = [p[0] for p in pairs]
        llm = [p[1] for p in pairs]
        all_human.extend(human)
        all_llm.extend(llm)
        # Each page contributes at most one pair per dimension, so a
        # singleton-list-per-page "cluster" here is just an ordinary
        # per-page bootstrap; the clustering matters once pairs are pooled
        # across dimensions below (Overall).
        dim_clusters = [[p] for p in pairs]
        p_lo, p_hi = _pooled_corr_ci(dim_clusters, _pearson)
        s_lo, s_hi = _pooled_corr_ci(dim_clusters, _spearman)
        dim_results.append({
            "key": dim,
            "label": DIMENSION_LABELS.get(dim, dim),
            **_compute_metrics(human, llm),
            **_compute_score_stats(human, llm),
            "pearson_ci_low": p_lo,
            "pearson_ci_high": p_hi,
            "spearman_ci_low": s_lo,
            "spearman_ci_high": s_hi,
        })

    pooled_clusters = list(sample_pairs.values())
    pearson_ci_low, pearson_ci_high = _pooled_corr_ci(pooled_clusters, _pearson)
    spearman_ci_low, spearman_ci_high = _pooled_corr_ci(pooled_clusters, _spearman)

    overall = {
        "label": "Overall",
        **_compute_metrics(all_human, all_llm),
        **_compute_score_stats(all_human, all_llm),
        "pearson_ci_low": pearson_ci_low,
        "pearson_ci_high": pearson_ci_high,
        "spearman_ci_low": spearman_ci_low,
        "spearman_ci_high": spearman_ci_high,
    }

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "eval_overrides": eval_overrides,
        "judge_models": judge_models,
        "n_annotations": len(annotations),
        "n_matched_samples": n_matched,
        "dimensions": dim_results,
        "overall": overall,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    print(f"  Wrote {output_path}")
    if dim_results:
        print(
            f"    Overall n={overall['n']}, Pearson={overall['pearson']} "
            f"[{pearson_ci_low}, {pearson_ci_high}], "
            f"Spearman={overall['spearman']} [{spearman_ci_low}, {spearman_ci_high}]"
        )

    return output

# ---------------------------------------------------------------------------
# Task: inter-annotator agreement
# ---------------------------------------------------------------------------

def run_inter_annotator(
    annotations: list[dict],
    output_path: Path,
    subset_key: tuple[str, str] | None = None,
    label: str = "Combined",
) -> dict:
    """
    Compute human--human (inter-annotator) agreement and write it to output_path.

    Unlike run_alignment (one side is the single LLM judge score), here every
    pair of annotators is compared directly on their *raw*, unaveraged scores
    over the samples both of them rated. Each dimension's reported value is
    the average of the per-pair metrics across every annotator pair; an
    Overall row does the same after pooling each pair's raw score pairs
    across all dimensions first (mirroring run_alignment's Overall row).

    subset_key, if given, restricts to annotations with that (run_id, eval_id)
    -- e.g. BENCHMARK_SUBSET_KEY or STRESS_SUBSET_KEY -- so the same
    pooling-across-subsets caution that applies to human--LLM agreement can
    be checked for human--human agreement too.
    """
    if subset_key is not None:
        annotations = [a for a in annotations if (a.get("run_id"), a.get("eval_id")) == subset_key]

    # sample_key -> annotator_id -> {dim: score}
    by_sample: dict[tuple[str, str, str], dict[str, dict[str, float]]] = defaultdict(dict)
    annotator_ids: set[str] = set()
    for ann in annotations:
        run_id = ann.get("run_id")
        eval_id = ann.get("eval_id")
        sample_id = ann.get("sample_id")
        annotator_id = ann.get("annotator_id")
        scores = ann.get("scores")
        if not (run_id and eval_id and sample_id and annotator_id and isinstance(scores, dict)):
            continue
        key = (run_id, eval_id, sample_id)
        by_sample[key][annotator_id] = scores
        annotator_ids.add(annotator_id)

    annotator_ids_sorted = sorted(annotator_ids)
    pairs = list(combinations(annotator_ids_sorted, 2))

    def pair_scores(a1: str, a2: str, dim: str) -> tuple[list[float], list[float]]:
        xs: list[float] = []
        ys: list[float] = []
        for by_ann in by_sample.values():
            v1 = by_ann.get(a1, {}).get(dim)
            v2 = by_ann.get(a2, {}).get(dim)
            if isinstance(v1, (int, float)) and not isinstance(v1, bool) and isinstance(v2, (int, float)) and not isinstance(v2, bool):
                xs.append(float(v1))
                ys.append(float(v2))
        return xs, ys

    sample_keys = list(by_sample.keys())

    dim_results = []
    pooled_pair_scores: dict[tuple[str, str], tuple[list[float], list[float]]] = {p: ([], []) for p in pairs}
    present_dims: list[str] = []

    for dim in ALL_DIMENSIONS:
        per_pair_metrics = []
        ns = []
        for a1, a2 in pairs:
            xs, ys = pair_scores(a1, a2, dim)
            if not xs:
                continue
            per_pair_metrics.append(_compute_metrics(xs, ys))
            ns.append(len(xs))
            pooled_pair_scores[(a1, a2)][0].extend(xs)
            pooled_pair_scores[(a1, a2)][1].extend(ys)

        if not per_pair_metrics:
            continue
        present_dims.append(dim)

        avg = {
            key: _avg_defined([m[key] for m in per_pair_metrics])
            for key in ("mse", "mae", "acc", "acc1", "pearson", "spearman", "kendall")
        }
        n = ns[0] if len(set(ns)) == 1 else round(sum(ns) / len(ns))
        p_lo, p_hi = _inter_annotator_ci(sample_keys, by_sample, pairs, [dim], _pearson)
        s_lo, s_hi = _inter_annotator_ci(sample_keys, by_sample, pairs, [dim], _spearman)
        dim_results.append({
            "key": dim,
            "label": DIMENSION_LABELS.get(dim, dim),
            "n": n,
            **avg,
            "pearson_ci_low": p_lo,
            "pearson_ci_high": p_hi,
            "spearman_ci_low": s_lo,
            "spearman_ci_high": s_hi,
        })

    pooled_metrics = [
        _compute_metrics(xs, ys) for xs, ys in pooled_pair_scores.values() if xs
    ]
    if pooled_metrics:
        overall_p_lo, overall_p_hi = _inter_annotator_ci(sample_keys, by_sample, pairs, present_dims, _pearson)
        overall_s_lo, overall_s_hi = _inter_annotator_ci(sample_keys, by_sample, pairs, present_dims, _spearman)
        overall = {
            "label": "Overall",
            "n": pooled_metrics[0]["n"] if len(set(m["n"] for m in pooled_metrics)) == 1
            else round(sum(m["n"] for m in pooled_metrics) / len(pooled_metrics)),
            **{
                key: _avg_defined([m[key] for m in pooled_metrics])
                for key in ("mse", "mae", "acc", "acc1", "pearson", "spearman", "kendall")
            },
            "pearson_ci_low": overall_p_lo,
            "pearson_ci_high": overall_p_hi,
            "spearman_ci_low": overall_s_lo,
            "spearman_ci_high": overall_s_hi,
        }
    else:
        overall = {"label": "Overall", "n": 0, "mse": None, "mae": None, "acc": None, "acc1": None,
                   "pearson": None, "spearman": None, "kendall": None,
                   "pearson_ci_low": None, "pearson_ci_high": None,
                   "spearman_ci_low": None, "spearman_ci_high": None}

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "annotator_ids": annotator_ids_sorted,
        "annotator_pairs": [list(p) for p in pairs],
        "n_webpages": len(by_sample),
        "dimensions": dim_results,
        "overall": overall,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    print(f"  Wrote {output_path}")
    if dim_results:
        print(
            f"    Overall n={overall['n']}, Pearson={overall['pearson']} "
            f"[{overall['pearson_ci_low']}, {overall['pearson_ci_high']}], "
            f"Spearman={overall['spearman']} [{overall['spearman_ci_low']}, {overall['spearman_ci_high']}]"
        )

    return output

# ---------------------------------------------------------------------------
# Task: annotator-scores
# ---------------------------------------------------------------------------

def run_annotator_scores(annotations: list[dict], output_path: Path) -> None:
    # dim -> sample_id -> annotator -> score
    dim_sample_annotator: dict[str, dict[str, dict[str, float]]] = {
        d: defaultdict(dict) for d in ALL_DIMENSIONS
    }
    all_annotators: set[str] = set()

    for ann in annotations:
        annotator_id = ann.get("annotator_id")
        sample_id = ann.get("sample_id")
        scores = ann.get("scores")
        if not (annotator_id and sample_id and isinstance(scores, dict)):
            continue
        all_annotators.add(annotator_id)
        for dim in ALL_DIMENSIONS:
            v = scores.get(dim)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                dim_sample_annotator[dim][sample_id][annotator_id] = float(v)

    annotators_sorted = sorted(all_annotators)

    dimensions_out = []
    for dim in ALL_DIMENSIONS:
        sample_map = dim_sample_annotator[dim]
        if not sample_map:
            continue

        annotator_scores_all: dict[str, list[float]] = defaultdict(list)
        for scores_by_ann in sample_map.values():
            for ann_id, score in scores_by_ann.items():
                annotator_scores_all[ann_id].append(score)

        annotator_averages = {
            ann_id: round(sum(vs) / len(vs), 4)
            for ann_id, vs in annotator_scores_all.items()
        }

        samples_out = [
            {"sample_id": sid, "scores": dict(scores_by_ann)}
            for sid, scores_by_ann in sorted(sample_map.items())
        ]

        dimensions_out.append({
            "key": dim,
            "label": DIMENSION_LABELS.get(dim, dim),
            "annotator_averages": annotator_averages,
            "samples": samples_out,
        })

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "annotators": annotators_sorted,
        "dimensions": dimensions_out,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    print(f"  Wrote {output_path}")
    print(f"    {len(annotators_sorted)} annotators: {annotators_sorted}")
    print(f"    {len(dimensions_out)} dimensions with data")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_annotations(path: Path) -> list[dict]:
    """Read one annotation record per line: run_id, eval_id, sample_id, annotator_id, scores."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export annotation data to static JSON files for the frontend.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Tasks:\n"
            "  alignment        human vs LLM judge alignment metrics\n"
            "  annotator-scores per-annotator/per-sample raw scores\n"
            "  all              both tasks (default)\n"
        ),
    )
    parser.add_argument(
        "--task",
        choices=["alignment", "annotator-scores", "all"],
        default="all",
        help="Which data to export (default: all)",
    )
    parser.add_argument(
        "--annotations-file",
        type=Path,
        default=Path("human_study/annotations.jsonl"),
        help="JSONL file of human annotations (default: human_study/annotations.jsonl)",
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"), help="Root directory of runs (default: runs)")
    parser.add_argument("--frontend-public", type=Path, default=Path("frontend/public"), help="Frontend public directory for fallback metrics lookup (default: frontend/public)")
    parser.add_argument(
        "--alignment-output",
        type=Path,
        default=None,
        help=(
            "Output path for alignment data. Defaults to frontend/public/data/human_alignment.json "
            "when no --eval-override is given, or an auto-named human_alignment_<judge-model>.json "
            "alongside it otherwise, so overridden-judge comparisons never overwrite the default."
        ),
    )
    parser.add_argument(
        "--eval-override",
        action="append",
        default=None,
        metavar="RUN_ID=EVAL_ID",
        help=(
            "Compare existing human annotations against a DIFFERENT eval's LLM judge scores for the "
            "same underlying generated pages, instead of the eval_id stored on the annotation. "
            "RUN_ID must match an annotation's run_id; EVAL_ID must be an eval under that same run_id "
            "(same generated HTML, different judge config) — e.g. "
            "test_adversarial=eval_test_adversarial_qwen3_235b. Repeatable."
        ),
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Human-readable label embedded in the output and shown in the frontend dropdown. "
        "Auto-derived from the overridden judge model(s) if not given.",
    )
    parser.add_argument(
        "--annotator-scores-output",
        type=Path,
        default=Path("frontend/public/data/annotator_scores.json"),
        help="Output path for annotator scores data (default: frontend/public/data/annotator_scores.json)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    runs_root = (repo_root / args.runs_root).resolve()
    data_dir = (repo_root / args.frontend_public / "data").resolve()

    annotations = load_annotations((repo_root / args.annotations_file).resolve())
    print(f"Loaded {len(annotations)} annotations from {args.annotations_file}.")

    if args.task in ("alignment", "all"):
        print("Running task: alignment")

        eval_overrides: dict[str, str] = {}
        for item in (args.eval_override or []):
            if "=" not in item:
                print(f"ERROR: --eval-override must be RUN_ID=EVAL_ID, got: {item!r}", file=sys.stderr)
                sys.exit(1)
            run_id, eval_id = item.split("=", 1)
            eval_overrides[run_id.strip()] = eval_id.strip()

        alignment_output = args.alignment_output
        label = args.label
        if eval_overrides:
            override_models = [
                (run_id, eval_id, _read_judge_model(run_id, eval_id, runs_root))
                for run_id, eval_id in eval_overrides.items()
            ]
            if alignment_output is None:
                slug = "_".join(_slugify_model(model or eval_id) for _, eval_id, model in override_models)
                alignment_output = Path("frontend/public/data") / f"human_alignment_{slug}.json"
            if label is None:
                label = ", ".join(f"{run_id} → {model or eval_id}" for run_id, eval_id, model in override_models)
        else:
            if alignment_output is None:
                alignment_output = Path("frontend/public/data/human_alignment.json")
            if label is None:
                label = "Default"

        run_alignment(
            annotations,
            runs_root,
            data_dir,
            (repo_root / alignment_output).resolve(),
            eval_overrides=eval_overrides,
            label=label,
        )
        build_alignment_index(data_dir)

        # Also regenerate the fixed benchmark-only / stress-only subset exports
        # alongside the default combined run. These are separate files (not
        # prefixed "human_alignment_") so build_alignment_index()'s glob --
        # which feeds the unrelated judge-model comparison dropdown -- doesn't
        # pick them up.
        if not eval_overrides:
            for subset_key, subset_label, subset_filename in (
                (BENCHMARK_SUBSET_KEY, "Responsive-Bench", "alignment_benchmark.json"),
                (STRESS_SUBSET_KEY, "Stress test", "alignment_stress.json"),
            ):
                subset_annotations = [
                    a for a in annotations
                    if a.get("run_id") == subset_key[0] and a.get("eval_id") == subset_key[1]
                ]
                print(f"Running task: alignment (subset: {subset_label}, {len(subset_annotations)} annotations)")
                run_alignment(
                    subset_annotations,
                    runs_root,
                    data_dir,
                    (repo_root / "frontend/public/data" / subset_filename).resolve(),
                    eval_overrides={},
                    label=subset_label,
                )

        # Inter-annotator (human--human) agreement, split the same three ways
        # as the human--LLM subset exports above. Judge-independent, so this
        # regenerates every run regardless of --eval-override.
        for subset_key, subset_label, subset_filename in (
            (BENCHMARK_SUBSET_KEY, "Responsive-Bench", "inter_annotator_benchmark.json"),
            (STRESS_SUBSET_KEY, "Stress test", "inter_annotator_stress.json"),
            (None, "Combined", "inter_annotator_combined.json"),
        ):
            print(f"Running task: inter-annotator (subset: {subset_label})")
            run_inter_annotator(
                annotations,
                (repo_root / "frontend/public/data" / subset_filename).resolve(),
                subset_key=subset_key,
                label=subset_label,
            )

    if args.task in ("annotator-scores", "all"):
        print("Running task: annotator-scores")
        run_annotator_scores(
            annotations,
            (repo_root / args.annotator_scores_output).resolve(),
        )


if __name__ == "__main__":
    main()
