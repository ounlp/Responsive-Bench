"""
Anthropic Claude provider implementation using anthropic SDK.
"""

import base64
import os
from pathlib import Path
from typing import Any

try:
    from anthropic import Anthropic, BadRequestError
except ImportError:
    Anthropic = None
    BadRequestError = None

from .base import Provider, ProviderResult


class ClaudeProvider(Provider):
    """Anthropic Claude provider using Claude 3 Vision or similar models."""

    def __init__(self, model: str, **kwargs: Any):
        """
        Initialize Claude provider.

        Args:
            model: Model identifier (e.g., "claude-3-opus-20240229", "claude-3-sonnet-20240229")
            **kwargs: Ignored

        Raises:
            ValueError: If ANTHROPIC_API_KEY is not set
            ImportError: If anthropic package is not installed
        """
        if Anthropic is None:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            )

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        self.client = Anthropic(api_key=api_key)
        self.model = model

    @property
    def name(self) -> str:
        """Return provider name."""
        return "claude"

    def generate(
        self, prompt_text: str, images: list[Path], generation: dict[str, Any]
    ) -> ProviderResult:
        """
        Generate output using Claude API.

        Args:
            prompt_text: Text prompt
            images: List of image file paths
            generation: Generation parameters (temperature, max_tokens, etc.)

        Returns:
            ProviderResult with generated text and metadata
        """
        # Build content list with text and images
        content = [{"type": "text", "text": prompt_text}]

        # Add images as base64-encoded data
        for img_path in images:
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")

            image_data = img_path.read_bytes()
            base64_image = base64.b64encode(image_data).decode("utf-8")
            mime_type = "image/png"  # Assuming PNG from dataset

            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": base64_image,
                    },
                }
            )

        # Prepare API call
        messages = [{"role": "user", "content": content}]

        def call_api():
            # Long HTML-generation requests can exceed the SDK's 10-minute non-streaming
            # limit (https://github.com/anthropics/anthropic-sdk-python#long-requests), so
            # this always goes over the streaming transport, then collapses back to the
            # same Message shape .create() would have returned.
            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": generation.get("max_tokens", 2000),
                "temperature": generation.get("temperature", 0.7),
                "messages": messages,
            }
            try:
                with self.client.messages.stream(**kwargs) as stream:
                    return stream.get_final_message()
            except BadRequestError as e:
                # Some newer models (e.g. extended-thinking Claude models) reject a
                # temperature override outright rather than just ignoring it — drop it
                # and retry once instead of failing the whole sample.
                if "temperature" in kwargs and "temperature" in str(e).lower() and "deprecated" in str(e).lower():
                    kwargs.pop("temperature")
                    with self.client.messages.stream(**kwargs) as stream:
                        return stream.get_final_message()
                raise

        response, latency_ms = self._measure_latency(call_api)

        # Extract text from response (Claude returns a list of content blocks)
        text = ""
        if response.content:
            text_parts = [
                block.text for block in response.content if hasattr(block, "text") and block.type == "text"
            ]
            text = "".join(text_parts)

        # Extract usage info
        usage_info = None
        if hasattr(response, "usage") and response.usage:
            usage_info = {
                "input_tokens": getattr(response.usage, "input_tokens", 0),
                "output_tokens": getattr(response.usage, "output_tokens", 0),
            }
            usage_info["total_tokens"] = (
                usage_info["input_tokens"] + usage_info["output_tokens"]
            )

        return ProviderResult(
            text=text,
            raw={
                "id": response.id,
                "model": response.model,
                "stop_reason": getattr(response, "stop_reason", None),
            },
            usage=usage_info,
            latency_ms=latency_ms,
            model=self.model,
            provider="claude",
        )
