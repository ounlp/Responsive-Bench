"""
Base class for evaluation metrics.

All metrics must inherit from Metric and implement the run() method.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Metric(ABC):
    """
    Abstract base class for evaluation metrics.

    Each metric evaluates a generated HTML output and returns results and artifacts.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return metric name identifier (e.g., 'horizontal_scroll', 'layout_similarity')."""
        pass

    @property
    @abstractmethod
    def requires_browser(self) -> bool:
        """
        Return whether this metric requires a browser (Playwright).

        If True, the orchestrator will provide a browser context in the context dict.
        """
        pass

    @abstractmethod
    async def run(
        self, sample: Any, generated_html_path: Path, context: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Path]]:
        """
        Run the metric evaluation.

        Args:
            sample: Sample object from dataset loader (may be None if not needed)
            generated_html_path: Path to the generated HTML file to evaluate
            context: Context dictionary containing:
                - viewport: dict with width/height
                - playwright: dict with headless, timeout_ms
                - artifacts_dir: Path to directory for saving artifacts

        Returns:
            Tuple of (result_dict, artifacts_dict):
                - result_dict: JSON-serializable dict with metric results
                - artifacts_dict: Dict mapping artifact names to file paths

        Raises:
            Exception: On metric execution errors (will be caught and logged by orchestrator)
        """
        pass
