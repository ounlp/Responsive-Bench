"""Layout similarity metric implementation."""

from .metric import LayoutSimilarityMetric
from .visual_extractor import extract_visual_components
from .box_utils import (
    boxes_adjacent,
    merge_boxes,
    is_within,
    intersection,
    bounding_box_to_polygon,
)
from .iou_computation import (
    compute_list_iou,
    compute_weighted_iou,
    compute_list_iou_shapely,
    compute_weighted_iou_shapely,
)

__all__ = [
    "LayoutSimilarityMetric",
    "extract_visual_components",
    "boxes_adjacent",
    "merge_boxes",
    "is_within",
    "intersection",
    "bounding_box_to_polygon",
    "compute_list_iou",
    "compute_weighted_iou",
    "compute_list_iou_shapely",
    "compute_weighted_iou_shapely",
]
