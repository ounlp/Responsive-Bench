"""
vLLM provider implementation using an OpenAI-compatible Chat Completions API.

This provider is intended for self-hosted vLLM servers that expose /v1/chat/completions.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from .base import Provider, ProviderResult


class VLLMProvider(Provider):
    """Provider for OpenAI-compatible vLLM endpoints."""

    def __init__(self, model: str, **kwargs: Any):
        if OpenAI is None:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

        base_url = os.getenv("VLLM_BASE_URL")
        if not base_url:
            raise ValueError("VLLM_BASE_URL environment variable not set (e.g. http://host:8000/v1)")

        # vLLM OpenAI-compatible servers typically ignore api_key, but the SDK requires one.
        self.client = OpenAI(api_key="vllm", base_url=base_url)
        self.model = model

    @property
    def name(self) -> str:
        return "vllm"

    def generate(
        self, prompt_text: str, images: list[Path], generation: dict[str, Any]
    ) -> ProviderResult:
        # Build content list with text + image_url parts (data: URLs)
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]

        for img_path in images:
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")

            image_data = img_path.read_bytes()
            base64_image = base64.b64encode(image_data).decode("utf-8")
            mime_type = "image/png"  # dataset images are PNG

            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                }
            )

        messages: list[dict[str, Any]] = []
        system_prompt = generation.get("system_prompt")
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "user", "content": content})

        api_params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": generation.get("temperature", 0.7),
            "max_tokens": generation.get("max_tokens", 2000),
        }

        if "response_format" in generation:
            fmt = generation["response_format"]
            if isinstance(fmt, str):
                fmt = {"type": fmt}
            api_params["response_format"] = fmt

        if "extra_body" in generation:
            api_params["extra_body"] = generation["extra_body"]

        def call_api():
            return self.client.chat.completions.create(**api_params)

        response, latency_ms = self._measure_latency(call_api)

        text = response.choices[0].message.content or ""

        usage_info = None
        if getattr(response, "usage", None):
            usage_info = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        raw = {"model": getattr(response, "model", self.model)}
        if getattr(response, "id", None) is not None:
            raw["id"] = response.id
        if getattr(response, "created", None) is not None:
            raw["created"] = response.created

        return ProviderResult(
            text=text,
            raw=raw,
            usage=usage_info,
            latency_ms=latency_ms,
            model=self.model,
            provider="vllm",
        )

