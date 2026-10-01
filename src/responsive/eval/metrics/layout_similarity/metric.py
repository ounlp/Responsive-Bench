"""
Layout similarity metric implementation.

Compares the layout of generated HTML with reference HTML by extracting visual
components and computing weighted IoU (Intersection over Union) scores.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from ..a11y_helpers import VIEWPORTS
from ..base import Metric
from .visual_extractor import extract_visual_components
from .iou_computation import compute_weighted_iou_shapely

logger = logging.getLogger("responsive.eval.metrics.layout_similarity")

# Maximum normalized vertical gap between two text_blocks to be merged.
# At 0.02, a 2000px-tall page merges blocks separated by up to 40px —
# enough to span paragraph margins but not section breaks.
_TEXT_MERGE_GAP = 0.02


def _merge_text_blocks(elements: dict) -> dict:
    """
    Merge vertically adjacent text_blocks that share the same horizontal column.

    Generated HTML frequently splits what the reference renders as one large
    paragraph block into many thin per-<p> boxes. Merging before IoU removes
    this fragmentation artifact so the comparison reflects regional coverage
    rather than DOM structure.
    """
    blocks = elements.get("text_block", [])
    if len(blocks) < 2:
        return elements

    blocks = sorted(blocks, key=lambda e: e["box"]["y"])
    merged = [{"box": dict(blocks[0]["box"])}]

    for current in blocks[1:]:
        prev_box = merged[-1]["box"]
        cur_box = current["box"]

        # Skip merge if vertical gap is too large (different sections)
        gap_y = cur_box["y"] - (prev_box["y"] + prev_box["height"])
        if gap_y > _TEXT_MERGE_GAP:
            merged.append({"box": dict(cur_box)})
            continue

        # Skip merge if blocks are in different horizontal columns
        px0, px1 = prev_box["x"], prev_box["x"] + prev_box["width"]
        cx0, cx1 = cur_box["x"], cur_box["x"] + cur_box["width"]
        overlap_x = max(0.0, min(px1, cx1) - max(px0, cx0))
        min_w = min(prev_box["width"], cur_box["width"])
        if min_w <= 0 or overlap_x / min_w < 0.5:
            merged.append({"box": dict(cur_box)})
            continue

        # Expand the last merged block to encompass current
        new_x = min(px0, cx0)
        new_right = max(px1, cx1)
        new_bottom = max(prev_box["y"] + prev_box["height"], cur_box["y"] + cur_box["height"])
        merged[-1]["box"] = {
            "x": new_x,
            "y": prev_box["y"],
            "width": new_right - new_x,
            "height": new_bottom - prev_box["y"],
        }

    result = dict(elements)
    result["text_block"] = merged
    return result


def _serialize_multi_scores(multi_scores: dict) -> dict:
    return {
        key: {"iou_score": float(score), "weight": float(weight)}
        for key, (score, weight) in multi_scores.items()
    }


class LayoutSimilarityMetric(Metric):
    """Metric that computes layout similarity between generated and reference HTML."""

    @property
    def name(self) -> str:
        return "layout_similarity"

    @property
    def requires_browser(self) -> bool:
        return True

    async def run(
        self, sample: Any, generated_html_path: Path, context: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Path]]:
        if sample is None:
            raise ValueError("Sample object is required for layout similarity metric")

        if not hasattr(sample, "reference_html") or sample.reference_html is None:
            raise ValueError("Sample must have reference_html attribute for layout similarity")

        reference_html_path = Path(sample.reference_html)
        if not reference_html_path.exists():
            raise FileNotFoundError(f"Reference HTML not found: {reference_html_path}")

        if not generated_html_path.exists():
            raise FileNotFoundError(f"Generated HTML not found: {generated_html_path}")

        artifacts_dir: Path = context["artifacts_dir"]
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        viewport_results: dict[str, Any] = {}
        artifacts: dict[str, Path] = {}

        for vp_name, vp_size in VIEWPORTS.items():
            ref_screenshot = artifacts_dir / f"layout_reference_annotated_{vp_name}.png"
            gen_screenshot = artifacts_dir / f"layout_generated_annotated_{vp_name}.png"

            logger.info(f"[layout_similarity] [{vp_name}] extracting reference layout")
            ref_layout = await asyncio.to_thread(
                extract_visual_components, reference_html_path, ref_screenshot, None, vp_size
            )
            n_ref = sum(len(v) for v in ref_layout.values())
            logger.debug(f"[layout_similarity] [{vp_name}] {n_ref} reference components")

            logger.info(f"[layout_similarity] [{vp_name}] extracting generated layout")
            gen_layout = await asyncio.to_thread(
                extract_visual_components, generated_html_path, gen_screenshot, None, vp_size
            )
            n_gen = sum(len(v) for v in gen_layout.values())
            logger.debug(f"[layout_similarity] [{vp_name}] {n_gen} generated components")

            ref_tb_before = len(ref_layout.get("text_block", []))
            gen_tb_before = len(gen_layout.get("text_block", []))
            ref_layout = _merge_text_blocks(ref_layout)
            gen_layout = _merge_text_blocks(gen_layout)
            logger.debug(
                f"[layout_similarity] [{vp_name}] text_block merge: "
                f"ref {ref_tb_before}→{len(ref_layout.get('text_block', []))}  "
                f"gen {gen_tb_before}→{len(gen_layout.get('text_block', []))}"
            )

            weighted_iou, multi_scores = compute_weighted_iou_shapely(gen_layout, ref_layout)

            logger.info(
                f"[layout_similarity] [{vp_name}] weighted_iou={weighted_iou:.4f}  "
                + "  ".join(
                    f"{k}={s:.3f}(w={w:.3f})"
                    for k, (s, w) in multi_scores.items()
                )
            )

            viewport_results[vp_name] = {
                "weighted_iou": float(weighted_iou),
                "multi_scores": _serialize_multi_scores(multi_scores),
            }
            artifacts[f"reference_annotated_{vp_name}"] = ref_screenshot
            artifacts[f"generated_annotated_{vp_name}"] = gen_screenshot

        combined_weighted_iou = (
            sum(v["weighted_iou"] for v in viewport_results.values()) / len(viewport_results)
            if viewport_results
            else 0.0
        )

        result = {
            "status": "success",
            "viewports": viewport_results,
            "weighted_iou": float(combined_weighted_iou),
        }

        return result, artifacts
