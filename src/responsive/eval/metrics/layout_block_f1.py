"""
Block-level precision/recall/F1 metric built on the same object set layout_similarity
scores (extract_visual_components + the metric's own text_block merge pass).

For each semantic category, generated blocks are greedily matched to reference
blocks by descending IoU (accepted only above `iou_threshold`), giving a
per-category TP/FP/FN count that is turned into precision/recall/F1. Categories
are aggregated into one score using the same area-based weight layout_similarity
uses (union area of reference+generated boxes in that category).

Computed independently at mobile, tablet, and desktop viewports (see
a11y_helpers.VIEWPORTS) — box positions and category weights genuinely differ
by viewport width, so this is three independent matching passes per sample, not
one pass reused three times. Reported per-viewport with no combined score across
viewports; each width's structural agreement is its own diagnostic.

Self-contained like every other metric here: it runs its own extraction pass
rather than sharing state with layout_similarity, so enabling both in the same
eval config means additional browser passes per sample.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from .a11y_helpers import VIEWPORTS
from .base import Metric
from .layout_similarity.box_utils import bounding_box_to_polygon
from .layout_similarity.iou_computation import compute_list_iou_shapely
from .layout_similarity.metric import _merge_text_blocks
from .layout_similarity.visual_extractor import extract_visual_components

logger = logging.getLogger("responsive.eval.metrics.layout_block_f1")

DEFAULT_IOU_THRESHOLD = 0.5


def _pairwise_iou(box_a: dict[str, float], box_b: dict[str, float]) -> float:
    poly_a = bounding_box_to_polygon(box_a)
    poly_b = bounding_box_to_polygon(box_b)
    union = poly_a.union(poly_b).area
    if union <= 0:
        return 0.0
    return poly_a.intersection(poly_b).area / union


def _match_category(
    gen_items: list[dict[str, Any]], ref_items: list[dict[str, Any]], iou_threshold: float
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    """Greedy one-to-one IoU matching: highest-IoU pairs first, thresholded."""
    candidates: list[tuple[float, int, int]] = []
    for gi, g in enumerate(gen_items):
        for ri, r in enumerate(ref_items):
            iou = _pairwise_iou(g["box"], r["box"])
            if iou > 0:
                candidates.append((iou, gi, ri))
    candidates.sort(key=lambda c: c[0], reverse=True)

    matched_gen: set[int] = set()
    matched_ref: set[int] = set()
    matches: list[dict[str, Any]] = []
    for iou, gi, ri in candidates:
        if iou < iou_threshold:
            break  # sorted descending — nothing further can clear the threshold
        if gi in matched_gen or ri in matched_ref:
            continue
        matched_gen.add(gi)
        matched_ref.add(ri)
        matches.append({"gen_index": gi, "ref_index": ri, "iou": round(iou, 6)})

    unmatched_gen = [gi for gi in range(len(gen_items)) if gi not in matched_gen]
    unmatched_ref = [ri for ri in range(len(ref_items)) if ri not in matched_ref]
    return matches, unmatched_gen, unmatched_ref


def _prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


class LayoutBlockF1Metric(Metric):
    """Precision/recall/F1 between generated and reference layout blocks, per category, per viewport."""

    @property
    def name(self) -> str:
        return "layout_block_f1"

    @property
    def requires_browser(self) -> bool:
        return True

    async def run(
        self, sample: Any, generated_html_path: Path, context: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Path]]:
        if sample is None:
            raise ValueError("Sample object is required for layout_block_f1 metric")
        if not hasattr(sample, "reference_html") or sample.reference_html is None:
            raise ValueError("Sample must have reference_html attribute for layout_block_f1")

        reference_html_path = Path(sample.reference_html)
        if not reference_html_path.exists():
            raise FileNotFoundError(f"Reference HTML not found: {reference_html_path}")
        if not generated_html_path.exists():
            raise FileNotFoundError(f"Generated HTML not found: {generated_html_path}")

        artifacts_dir: Path = context["artifacts_dir"]
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        iou_threshold = float((context.get("layout_block_f1") or {}).get("iou_threshold", DEFAULT_IOU_THRESHOLD))

        viewport_results: dict[str, Any] = {}
        viewport_detail: dict[str, Any] = {}

        for vp_name, vp_size in VIEWPORTS.items():
            logger.info(f"[layout_block_f1] [{vp_name}] extracting reference layout")
            ref_layout = await asyncio.to_thread(
                extract_visual_components, reference_html_path, None, None, vp_size
            )
            logger.info(f"[layout_block_f1] [{vp_name}] extracting generated layout")
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

                matches, unmatched_gen, unmatched_ref = _match_category(gen_items, ref_items, iou_threshold)
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
                    "matches": matches,
                    "unmatched_generated": [{"index": gi, "box": gen_items[gi]["box"]} for gi in unmatched_gen],
                    "unmatched_reference": [{"index": ri, "box": ref_items[ri]["box"]} for ri in unmatched_ref],
                }

                total_weight += weight
                weighted_f1_sum += weight * f1

            for entry in categories_summary.values():
                entry["weight_fraction"] = round(entry["weight"] / total_weight, 6) if total_weight > 0 else 0.0

            f1_weighted = weighted_f1_sum / total_weight if total_weight > 0 else 0.0

            logger.info(
                f"[layout_block_f1] [{vp_name}] f1_weighted={f1_weighted:.4f}  "
                + "  ".join(f"{k}={v['f1']:.3f}(w={v['weight_fraction']:.3f})" for k, v in categories_summary.items())
            )

            viewport_results[vp_name] = {
                "f1_weighted": float(f1_weighted),
                "categories": categories_summary,
            }
            viewport_detail[vp_name] = {"categories": categories_detail}

        result = {
            "status": "success",
            "iou_threshold": iou_threshold,
            "viewports": viewport_results,
        }

        matches_path = artifacts_dir / "layout_block_f1_matches.json"
        matches_path.write_text(json.dumps({"viewports": viewport_detail}, indent=2), encoding="utf-8")
        artifacts = {"layout_block_f1_matches": matches_path}

        return result, artifacts
