"""
Metric registry for evaluation metrics.

Maps metric names to metric classes for dynamic instantiation.
"""

from typing import Type

from .base import Metric
from .accessibility import AccessibilityMetric
from .accessible_active_nav_item import AccessibleActiveNavItemMetric
from .accessible_form_labels import AccessibleFormLabelsMetric
from .accessible_hamburger_menu import AccessibleHamburgerMenuMetric
from .accessible_landmarks import AccessibleLandmarksMetric
from .accessible_no_keyboard_trap import AccessibleNoKeyboardTrapMetric
from .computed_font_sizes import ComputedFontSizesMetric
from .heading_order import HeadingOrderMetric
from .horizontal_scroll import HorizontalScrollMetric
from .layout_block_f1 import LayoutBlockF1Metric
from .layout_block_f1_rra import LayoutBlockF1RRAMetric
from .layout_similarity import LayoutSimilarityMetric
from .line_spacing import LineSpacingMetric
from .llm_judge_screenshots import LLMJudgeScreenshotsMetric
from .relative_units import RelativeUnitsMetric
from .responsive_media import ResponsiveMediaMetric
from .rotation_friendly import RotationFriendlyMetric
from .tap_targets import TapTargetsMetric
from .text_enlargement import TextEnlargementMetric
from .text_media_cutoff_overlap import TextMediaCutoffOverlapMetric
from .viewport_meta_tag import ViewportMetaTagMetric

# Registry mapping metric names to classes
METRICS: dict[str, Type[Metric]] = {
    "accessibility": AccessibilityMetric,
    "accessible_landmarks": AccessibleLandmarksMetric,
    "accessible_hamburger_menu": AccessibleHamburgerMenuMetric,
    "accessible_active_nav_item": AccessibleActiveNavItemMetric,
    "accessible_no_keyboard_trap": AccessibleNoKeyboardTrapMetric,
    "accessible_form_labels": AccessibleFormLabelsMetric,
    "layout_similarity": LayoutSimilarityMetric,
    "layout_block_f1": LayoutBlockF1Metric,
    "layout_block_f1_rra": LayoutBlockF1RRAMetric,
    "horizontal_scroll": HorizontalScrollMetric,
    "llm_judge_screenshots": LLMJudgeScreenshotsMetric,
    "tap_targets": TapTargetsMetric,
    "line_spacing": LineSpacingMetric,
    "viewport_meta_tag": ViewportMetaTagMetric,
    "responsive_media": ResponsiveMediaMetric,
    "relative_units": RelativeUnitsMetric,
    "computed_font_sizes": ComputedFontSizesMetric,
    "rotation_friendly": RotationFriendlyMetric,
    "text_media_cutoff_overlap": TextMediaCutoffOverlapMetric,
    "heading_order": HeadingOrderMetric,
    "text_enlargement": TextEnlargementMetric,
}


def get_metric(name: str) -> Metric:
    """
    Get a metric instance by name.

    Args:
        name: Metric name (e.g., "horizontal_scroll", "layout_similarity", "llm_judge_screenshots")

    Returns:
        Metric instance

    Raises:
        ValueError: If metric name is not recognized
    """
    if name not in METRICS:
        available = ", ".join(sorted(METRICS.keys()))
        raise ValueError(f"Unknown metric '{name}'. Available: {available}")

    metric_class = METRICS[name]
    return metric_class()
