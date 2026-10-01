"""
OpenAI provider implementation using official OpenAI SDK.
"""

import base64
import os
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from .base import Provider, ProviderResult


class OpenAIProvider(Provider):
    """OpenAI provider using GPT-4 Vision or similar models."""

    def __init__(self, model: str, **kwargs: Any):
        """
        Initialize OpenAI provider.

        Args:
            model: Model identifier (e.g., "gpt-4o-mini", "gpt-4o")
            **kwargs: Ignored

        Raises:
            ValueError: If OPENAI_API_KEY is not set
            ImportError: If openai package is not installed
        """
        if OpenAI is None:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    @property
    def name(self) -> str:
        """Return provider name."""
        return "openai"

    def generate(
        self, prompt_text: str, images: list[Path], generation: dict[str, Any]
    ) -> ProviderResult:
        """
        Generate output using OpenAI API.

        Args:
            prompt_text: Text prompt
            images: List of image file paths
            generation: Generation parameters (temperature, max_tokens, etc.)

        Returns:
            ProviderResult with generated text and metadata
        """
        # Build content list with text and images
        content = [{"type": "text", "text": prompt_text}]

        # Add images as base64-encoded data URLs
        for img_path in images:
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")

            # Read and encode image
            image_data = img_path.read_bytes()
            base64_image = base64.b64encode(image_data).decode("utf-8")
            mime_type = "image/png"  # Assuming PNG from dataset

            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}",
                    },
                }
            )

        # Prepare API call
        messages: list[dict[str, Any]] = []
        system_prompt = generation.get("system_prompt")
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "user", "content": content})

        # GPT-5.2+ models require max_completion_tokens instead of max_tokens
        # Detect GPT-5.2+ models (gpt-5.x, o1, o3 series)
        model_lower = self.model.lower()
        is_gpt52_plus = (
            "gpt-5" in model_lower
            or model_lower.startswith("o1")
            or model_lower.startswith("o3")
        )

        # Build API parameters
        api_params = {
            "model": self.model,
            "messages": messages,
            "temperature": generation.get("temperature", 0.7),
        }

        # Use appropriate token limit parameter
        max_tokens_value = generation.get("max_tokens", 2000)
        if is_gpt52_plus:
            api_params["max_completion_tokens"] = max_tokens_value
        else:
            api_params["max_tokens"] = max_tokens_value

        if "response_format" in generation:
            fmt = generation["response_format"]
            if isinstance(fmt, str):
                fmt = {"type": fmt}
            api_params["response_format"] = fmt

        def call_api():
            return self.client.chat.completions.create(**api_params)

        response, latency_ms = self._measure_latency(call_api)

        # Extract text from response
        text = response.choices[0].message.content or ""

        # Extract usage info
        usage_info = None
        if response.usage:
            usage_info = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return ProviderResult(
            text=text,
            raw={
                "id": response.id,
                "model": response.model,
                "created": response.created,
            },
            usage=usage_info,
            latency_ms=latency_ms,
            model=self.model,
            provider="openai",
        )
