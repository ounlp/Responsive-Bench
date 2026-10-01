"""
Heading order evaluation metric (nesting-aware).

Rule:
- Headings can start at any level globally (e.g. a page may start with h2/h3).
- But when a heading appears *nested inside* the container subtree of another heading,
  it must be a *higher* level number than its nearest ancestor heading (e.g. h1 -> h2/h3/...).

This matches the common "section within section" semantics where nested sections should
use deeper heading levels, without enforcing that the entire document must start at h1.
"""

import logging
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger("responsive.eval.metrics.heading_order")

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

from .base import Metric


class HeadingOrderMetric(Metric):
    """Metric that checks if headings follow proper hierarchical order."""

    @property
    def name(self) -> str:
        """Return metric name."""
        return "heading_order"

    @property
    def requires_browser(self) -> bool:
        """Heading order check does not require a browser."""
        return False

    async def run(
        self, sample: Any, generated_html_path: Path, context: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Path]]:
        """
        Run heading order evaluation.

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
            logger.debug(f"Starting heading order evaluation for: {generated_html_path}")

            # Read HTML file content
            html_content = generated_html_path.read_text(encoding="utf-8")
            soup = BeautifulSoup(html_content, "html.parser")

            # Find all headings (h1-h6) in document order
            headings = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])

            violations = []
            heading_details = []

            def _level(tag_name: str) -> int:
                return int(tag_name[1])

            def _is_descendant_of(ancestor, node) -> bool:
                # BeautifulSoup Tag supports .parents; compare by identity
                try:
                    return any(p is ancestor for p in node.parents)
                except Exception:
                    return False

            # Stack of active "section" containers based on nesting.
            # Each entry: {heading_tag, heading_level, container_tag}
            # container_tag is the heading's parent element (a pragmatic "section container").
            stack: list[dict[str, Any]] = []

            for heading in headings:
                tag_name = heading.name.lower()
                level = _level(tag_name)
                text_content = heading.get_text(strip=True)

                container = heading.parent if getattr(heading, "parent", None) is not None else soup

                heading_details.append(
                    {
                        "level": level,
                        "tag": tag_name,
                        "text": text_content[:100],  # Truncate long text
                    }
                )

                # Pop stack until we find a container that actually contains this heading.
                # This approximates "nested sections": if a heading is within the subtree
                # of a previous heading's container, it's considered nested.
                while stack and not _is_descendant_of(stack[-1]["container"], heading):
                    stack.pop()

                # If we are nested within a parent heading's container, enforce increasing level.
                if stack:
                    parent_level = stack[-1]["level"]
                    if level <= parent_level:
                        violations.append(
                            {
                                "level": level,
                                "tag": tag_name,
                                "text": text_content[:100],
                                "parent_level": parent_level,
                                "parent_tag": f"h{parent_level}",
                                "message": (
                                    f"{tag_name.upper()} is nested under H{parent_level} container "
                                    f"but is not a deeper heading level (expected > h{parent_level})."
                                ),
                            }
                        )

                # Push this heading as a new potential parent scope
                stack.append({"level": level, "container": container, "tag": tag_name})

            # Combine violations (nesting-based only)
            all_violations = violations

            passed = len(all_violations) == 0

            logger.debug(
                f"Found {len(headings)} headings, {len(all_violations)} violations"
            )

        except Exception as e:
            raise RuntimeError(
                f"Heading order metric failed for {generated_html_path}: {e}\n"
                f"Traceback: {traceback.format_exc()}"
            ) from e

        # Build result dict
        result = {
            "status": "success",
            "html_file": generated_html_path.name,
            "total_headings": len(headings),
            "heading_levels_found": sorted({_level(h.name.lower()) for h in headings}),
            "heading_details": heading_details,
            "violations": all_violations,
            "violation_count": len(all_violations),
            "passed": passed,
        }

        logger.info(
            f"Heading order: {'passed' if passed else 'failed'} "
            f"(headings={len(headings)}, violations={len(all_violations)})"
        )

        # Build artifacts dict (no artifacts for this metric)
        artifacts: dict[str, Path] = {}

        return result, artifacts
