"""
Reading-order rank alignment (RRA): a structural complement to layout_block_f1's
IoU-based matching, scoring per-category count/order structure with zero
geometric overlap or coordinate comparison involved.

For each semantic category, boxes are ordered into rows (grouped by vertical
overlap, top-to-bottom) and left-to-right within each row, then each box's
position in that order is normalized to a rank in [0, 1] via index/(count-1).
The reference and generated rank sequences are aligned with a
Needleman-Wunsch-style DP: a substitution (match) is allowed only when the
rank gap is within `rank_tolerance`, and skipping a box (insertion/deletion)
costs `gap_cost`. Backtracking yields a monotonic one-to-one match, exactly
like layout_block_f1's greedy IoU matcher but ordered by rank instead of
geometry.

Because rank is index-normalized, ref/gen sequences of equal length always
align diagonally at zero cost (rank_ref[i] == rank_gen[i] for every i) — so a
category with matching box counts scores f1=1.0 regardless of how the boxes
are actually arranged. The metric only discriminates when counts differ,
where the DP finds the least rank-distorting place to insert/delete a box.
It therefore measures "do per-category counts structurally line up, and if
not, how cleanly can one side's sequence be produced by inserting/deleting
from the other" — not whether reading order itself is preserved when counts
already match. This is a deliberate scope choice, not an oversight.

Same category weighting (area-based union) and output schema as
layout_block_f1, so results are directly comparable side by side.

Computed independently at mobile, tablet, and desktop viewports (see
a11y_helpers.VIEWPORTS) — reading order is derived from box positions, which
genuinely differ by viewport width, so this is three independent alignments
per sample, not one alignment reused three times. Reported per-viewport with
no combined score across viewports.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from .a11y_helpers import VIEWPORTS
from .base import Metric
from .layout_similarity.iou_computation import compute_list_iou_shapely
from .layout_similarity.metric import _merge_text_blocks
from .layout_similarity.visual_extractor import extract_visual_components

logger = logging.getLogger("responsive.eval.metrics.layout_block_f1_rra")

DEFAULT_RANK_TOLERANCE = 0.15
DEFAULT_ROW_OVERLAP_THRESHOLD = 0.5


def _reading_order(items: list[dict[str, Any]], row_overlap_threshold: float) -> list[int]:
    """
    Row-banded reading order: cluster boxes into rows by vertical overlap,
    order rows top-to-bottom, order boxes left-to-right within each row.

    Returns a list of original indices into `items`, in reading order.
    """
    if not items:
        return []

    order = sorted(range(len(items)), key=lambda i: items[i]["box"]["y"])

    rows: list[list[int]] = []
    row_bounds: list[tuple[float, float]] = []  # (min_y, max_y) of the row band
    for idx in order:
        box = items[idx]["box"]
        y0, y1 = box["y"], box["y"] + box["height"]
        placed = False
        if rows:
            band_y0, band_y1 = row_bounds[-1]
            overlap = max(0.0, min(band_y1, y1) - max(band_y0, y0))
            min_h = min(band_y1 - band_y0, y1 - y0)
            if min_h > 0 and overlap / min_h >= row_overlap_threshold:
                rows[-1].append(idx)
                row_bounds[-1] = (min(band_y0, y0), max(band_y1, y1))
                placed = True
        if not placed:
            rows.append([idx])
            row_bounds.append((y0, y1))

    reading_order: list[int] = []
    for row in rows:
        row.sort(key=lambda i: items[i]["box"]["x"])
        reading_order.extend(row)
    return reading_order


def _rank_align(
    ref_order: list[int], gen_order: list[int], rank_tolerance: float, gap_cost: float
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """
    Needleman-Wunsch alignment of two rank sequences (given as reading-order
    index lists). Substitution cost is the normalized rank gap, allowed only
    within `rank_tolerance`; skipping an element costs `gap_cost`.

    Returns (matches, unmatched_ref_indices, unmatched_gen_indices), where
    matches is a list of (ref_index, gen_index, rank_gap) using the *original*
    item indices (not reading-order positions).
    """
    n, m = len(ref_order), len(gen_order)
    if n == 0 and m == 0:
        return [], [], []

    def rank(i: int, count: int) -> float:
        return i / (count - 1) if count > 1 else 0.0

    ref_ranks = [rank(pos, n) for pos in range(n)]
    gen_ranks = [rank(pos, m) for pos in range(m)]

    INF = float("inf")
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    # Backpointers recorded during the forward pass: "D" = diagonal match,
    # "U" = skip ref (up), "L" = skip gen (left). Backtracking follows these
    # directly instead of re-deriving the path via float equality checks,
    # which are unreliable here (e.g. i * gap_cost != (i-1) * gap_cost + gap_cost
    # for several i under float64 rounding).
    back = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap_cost
        back[i][0] = "U"
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + gap_cost
        back[0][j] = "L"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            gap = abs(ref_ranks[i - 1] - gen_ranks[j - 1])
            sub_cost = gap if gap <= rank_tolerance else INF
            diag = dp[i - 1][j - 1] + sub_cost
            up = dp[i - 1][j] + gap_cost
            left = dp[i][j - 1] + gap_cost
            best = min(diag, up, left)
            dp[i][j] = best
            back[i][j] = "D" if best == diag else ("U" if best == up else "L")

    matches: list[tuple[int, int, float]] = []
    unmatched_ref: list[int] = []
    unmatched_gen: list[int] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = back[i][j]
        if move == "D":
            gap = abs(ref_ranks[i - 1] - gen_ranks[j - 1])
            matches.append((ref_order[i - 1], gen_order[j - 1], round(gap, 6)))
            i -= 1
            j -= 1
        elif move == "U":
            unmatched_ref.append(ref_order[i - 1])
            i -= 1
        else:
            unmatched_gen.append(gen_order[j - 1])
            j -= 1

    matches.reverse()
    unmatched_ref.reverse()
    unmatched_gen.reverse()
    return matches, unmatched_ref, unmatched_gen


def _prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


class LayoutBlockF1RRAMetric(Metric):
    """Precision/recall/F1 between generated and reference layout blocks, matched by reading-order rank."""

    @property
    def name(self) -> str:
        return "layout_block_f1_rra"

    @property
    def requires_browser(self) -> bool:
        return True

    async def run(
        self, sample: Any, generated_html_path: Path, context: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Path]]:
        if sample is None:
            raise ValueError("Sample object is required for layout_block_f1_rra metric")
        if not hasattr(sample, "reference_html") or sample.reference_html is None:
            raise ValueError("Sample must have reference_html attribute for layout_block_f1_rra")

        reference_html_path = Path(sample.reference_html)
        if not reference_html_path.exists():
            raise FileNotFoundError(f"Reference HTML not found: {reference_html_path}")
        if not generated_html_path.exists():
            raise FileNotFoundError(f"Generated HTML not found: {generated_html_path}")

        artifacts_dir: Path = context["artifacts_dir"]
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        rra_config = context.get("layout_block_f1_rra") or {}
        rank_tolerance = float(rra_config.get("rank_tolerance", DEFAULT_RANK_TOLERANCE))
        gap_cost = float(rra_config.get("gap_cost", rank_tolerance))
        row_overlap_threshold = float(rra_config.get("row_overlap_threshold", DEFAULT_ROW_OVERLAP_THRESHOLD))

        viewport_results: dict[str, Any] = {}
        viewport_detail: dict[str, Any] = {}

        for vp_name, vp_size in VIEWPORTS.items():
            logger.info(f"[layout_block_f1_rra] [{vp_name}] extracting reference layout")
            ref_layout = await asyncio.to_thread(
                extract_visual_components, reference_html_path, None, None, vp_size
            )
            logger.info(f"[layout_block_f1_rra] [{vp_name}] extracting generated layout")
            gen_layout = await asyncio.to_thread(
                extract_visual_components, generated_html_path, None, None, vp_size
            )

            ref_layout = _merge_text_blocks(ref_layout)
            gen_layout = _merge_text_blocks(gen_layout)

            all_categories = sorted(set(ref_layout.keys()) | set(gen_layout.keys()))

            categories_summary: dict[str, Any] = {}
            categories_detail: dict[str, Any] = {}
            total_weight = 0.0
            weighted_f1_sum = 0.0

            for cat in all_categories:
                gen_items = gen_layout.get(cat, [])
                ref_items = ref_layout.get(cat, [])

                ref_order = _reading_order(ref_items, row_overlap_threshold)
                gen_order = _reading_order(gen_items, row_overlap_threshold)

                matches, unmatched_ref, unmatched_gen = _rank_align(ref_order, gen_order, rank_tolerance, gap_cost)
                tp, fp, fn = len(matches), len(unmatched_gen), len(unmatched_ref)
                precision, recall, f1 = _prf1(tp, fp, fn)

                _, weight = compute_list_iou_shapely(gen_items, ref_items)

                categories_summary[cat] = {
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    "weight": round(weight, 6),
                }
                categories_detail[cat] = {
                    "reference_reading_order": ref_order,
                    "generated_reading_order": gen_order,
                    "matches": [
                        {"ref_index": ri, "gen_index": gi, "rank_gap": gap} for ri, gi, gap in matches
                    ],
                    "unmatched_reference": [{"index": ri, "box": ref_items[ri]["box"]} for ri in unmatched_ref],
                    "unmatched_generated": [{"index": gi, "box": gen_items[gi]["box"]} for gi in unmatched_gen],
                }

                total_weight += weight
                weighted_f1_sum += weight * f1

            for entry in categories_summary.values():
                entry["weight_fraction"] = round(entry["weight"] / total_weight, 6) if total_weight > 0 else 0.0

            f1_weighted = weighted_f1_sum / total_weight if total_weight > 0 else 0.0

            logger.info(
                f"[layout_block_f1_rra] [{vp_name}] f1_weighted={f1_weighted:.4f}  "
                + "  ".join(f"{k}={v['f1']:.3f}(w={v['weight_fraction']:.3f})" for k, v in categories_summary.items())
            )

            viewport_results[vp_name] = {
                "f1_weighted": float(f1_weighted),
                "categories": categories_summary,
            }
            viewport_detail[vp_name] = {"categories": categories_detail}

        result = {
            "status": "success",
            "rank_tolerance": rank_tolerance,
            "gap_cost": gap_cost,
            "row_overlap_threshold": row_overlap_threshold,
            "viewports": viewport_results,
        }

        matches_path = artifacts_dir / "layout_block_f1_rra_matches.json"
        matches_path.write_text(json.dumps({"viewports": viewport_detail}, indent=2), encoding="utf-8")
        artifacts = {"layout_block_f1_rra_matches": matches_path}

        return result, artifacts
