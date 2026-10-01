"""
Google Gemini provider implementation using the google-genai SDK.
"""

import base64
import os
from pathlib import Path
from typing import Any

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

from .base import Provider, ProviderResult


class GeminiProvider(Provider):
    """Google Gemini provider using Gemini 1.5 (vision-capable) models."""

    def __init__(self, model: str, **kwargs: Any):
        """
        Initialize Gemini provider.

        Args:
            model: Model identifier (e.g., "gemini-1.5-pro", "gemini-1.5-flash")
            **kwargs: Ignored

        Raises:
            ValueError: If GEMINI_API_KEY / GOOGLE_API_KEY is not set
            ImportError: If google-genai package is not installed
        """
        if genai is None:
            raise ImportError(
                "google-genai package not installed. Install with: pip install google-genai"
            )

        # Prefer GEMINI_API_KEY, fall back to legacy GOOGLE_API_KEY for convenience
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY (or legacy GOOGLE_API_KEY) environment variable not set"
            )

        self.client = genai.Client(api_key=api_key)
        self.model_name = model

    @property
    def name(self) -> str:
        """Return provider name."""
        return "gemini"

    def generate(
        self, prompt_text: str, images: list[Path], generation: dict[str, Any]
    ) -> ProviderResult:
        """
        Generate output using Gemini API.

        Args:
            prompt_text: Text prompt
            images: List of image file paths
            generation: Generation parameters (temperature, max_tokens, etc.)

        Returns:
            ProviderResult with generated text and metadata
        """
        # Build user parts: text first, then images
        parts: list[Any] = [types.Part.from_text(text=prompt_text)]

        # Add images as bytes parts
        for img_path in images:
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")

            image_bytes = img_path.read_bytes()
            parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/png"))

        # Prepare generation config
        generation_config = types.GenerateContentConfig(
            temperature=generation.get("temperature", 0.7),
            max_output_tokens=generation.get("max_tokens", 2000),
        )

        def call_api():
            return self.client.models.generate_content(
                model=self.model_name,
                contents=parts,  # SDK will convert parts to Content automatically
                config=generation_config,
            )

        response, latency_ms = self._measure_latency(call_api)

        # Extract text from response
        text = getattr(response, "text", "") or ""

        # Extract usage info if available
        usage_info = None
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            usage_info = {
                "prompt_tokens": getattr(usage, "prompt_token_count", 0),
                "completion_tokens": getattr(usage, "candidates_token_count", 0),
                "total_tokens": getattr(usage, "total_token_count", 0),
            }

        # Basic raw summary (avoid stuffing full response)
        raw_summary = {
            "model": self.model_name,
        }

        return ProviderResult(
            text=text,
            raw=raw_summary,
            usage=usage_info,
            latency_ms=latency_ms,
            model=self.model_name,
            provider="gemini",
        )
