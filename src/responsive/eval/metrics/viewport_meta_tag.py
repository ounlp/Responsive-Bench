"""
Viewport meta tag evaluation metric.

Checks if HTML file has proper viewport meta tag in head.
"""

import logging
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger("responsive.eval.metrics.viewport_meta_tag")

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

from .base import Metric


class ViewportMetaTagMetric(Metric):
    """Metric that checks for proper viewport meta tag."""

    @property
    def name(self) -> str:
        """Return metric name."""
        return "viewport_meta_tag"

    @property
    def requires_browser(self) -> bool:
        """Viewport meta tag check does not require a browser."""
        return False

    async def run(
        self, sample: Any, generated_html_path: Path, context: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Path]]:
        """
        Run viewport meta tag evaluation.

        Args:
            sample: Sample object (unused)
            generated_html_path: Path to generated HTML file
            context: Context dict (unused for this metric)

        Returns:
            Tuple of (result_dict, artifacts_dict)

        Raises:
            ImportError: If beautifulsoup4 is not installed
            FileNotFoundError: If generated_html_path does not exist
        """
        if BeautifulSoup is None:
            raise ImportError(
                "beautifulsoup4 not installed. Install with: pip install beautifulsoup4"
            )

        if not generated_html_path.exists():
            raise FileNotFoundError(f"Generated HTML not found: {generated_html_path}")

        try:
            logger.debug(f"Starting viewport meta tag evaluation for: {generated_html_path}")

            # Read HTML file content
            html_content = generated_html_path.read_text(encoding="utf-8")
            soup = BeautifulSoup(html_content, "html.parser")

            # Find viewport meta tag in head
            viewport_meta = soup.find("meta", attrs={"name": "viewport"})

            has_viewport_meta = viewport_meta is not None
            viewport_content = None
            has_correct_attributes = False

            if has_viewport_meta:
                viewport_content = viewport_meta.get("content", "")

                # Check for required attributes in content
                content_lower = viewport_content.lower().replace(" ", "")

                # Check if it contains the essential responsive attributes
                has_width_device = "width=device-width" in content_lower
                has_initial_scale = "initial-scale=1" in content_lower

                has_correct_attributes = has_width_device and has_initial_scale

                logger.debug(
                    f"Found viewport meta tag: content='{viewport_content}', "
                    f"correct={has_correct_attributes}"
                )
            else:
                logger.debug("No viewport meta tag found")

        except Exception as e:
            raise RuntimeError(
                f"Viewport meta tag metric failed for {generated_html_path}: {e}\n"
                f"Traceback: {traceback.format_exc()}"
            ) from e

        # Build result dict
        result = {
            "status": "success",
            "html_file": generated_html_path.name,
            "has_viewport_meta": has_viewport_meta,
            "viewport_content": viewport_content,
            "has_correct_attributes": has_correct_attributes,
            "expected_content": "width=device-width, initial-scale=1",
            "passed": has_viewport_meta and has_correct_attributes,
        }

        logger.info(
            f"Viewport meta tag: {'passed' if result['passed'] else 'failed'} "
            f"(has_tag={has_viewport_meta}, correct={has_correct_attributes})"
        )

        # Build artifacts dict (no artifacts for this metric)
        artifacts: dict[str, Path] = {}

        return result, artifacts
