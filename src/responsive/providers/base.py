"""
Base provider interface for multi-modal generation.

All providers must implement the Provider abstract base class.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ProviderResult:
    """Result from a provider generation call."""

    text: str
    raw: dict[str, Any]
    usage: Optional[dict[str, Any]] = None
    latency_ms: float = 0.0
    model: str = ""
    provider: str = ""


class Provider(ABC):
    """
    Abstract base class for generation providers.

    All providers must implement generate() which accepts prompt text and images.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return provider name identifier (e.g., 'openai', 'gemini')."""
        pass

    @abstractmethod
    def generate(
        self, prompt_text: str, images: list[Path], generation: dict[str, Any]
    ) -> ProviderResult:
        """
        Generate output from prompt text and images.

        Args:
            prompt_text: Text prompt string
            images: List of image file paths (order matters)
            generation: Provider-specific generation parameters (temperature, max_tokens, etc.)

        Returns:
            ProviderResult with text output, raw response, usage info, and latency

        Raises:
            Exception: On API errors or invalid inputs
        """
        pass

    def _measure_latency(self, func, *args, **kwargs) -> tuple[Any, float]:
        """
        Measure execution time of a function call.

        Args:
            func: Function to call
            *args, **kwargs: Arguments to pass to function

        Returns:
            Tuple of (function result, latency_ms)
        """
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        latency_ms = (end - start) * 1000.0
        return result, latency_ms
