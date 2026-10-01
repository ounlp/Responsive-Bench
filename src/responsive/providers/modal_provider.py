"""
Modal (modal.com) provider implementation using an OpenAI-compatible Chat
Completions API.

Modal web endpoints protected with Modal's proxy-auth feature require two
custom headers on every request -- Modal-Key and Modal-Secret -- generated as
a token pair (id prefixed "wk-", secret prefixed "ws-"). This is separate from
whatever bearer token (if any) the underlying OpenAI-compatible server itself
expects, so both are sent.
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


class ModalProvider(Provider):
    """Provider for OpenAI-compatible endpoints deployed on Modal."""

    def __init__(self, model: str, **kwargs: Any):
        if OpenAI is None:
            raise ImportError("openai package not installed. Install with: pip install openai")

        base_url = os.getenv("MODAL_ENDPOINT")
        if not base_url:
            raise ValueError(
                "MODAL_ENDPOINT environment variable not set "
                "(e.g. https://<workspace>--<app>-<fn>.modal.run/v1)"
            )

        token_id = os.getenv("MODAL_TOKEN_ID")
        token_secret = os.getenv("MODAL_TOKEN_SECRET")
        if not token_id or not token_secret:
            raise ValueError("MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables must both be set")

        default_headers = {"Modal-Key": token_id, "Modal-Secret": token_secret}

        # The underlying OpenAI-compatible server (e.g. vLLM) also requires an
        # api_key argument even when it doesn't check it; reuse the Modal
        # token secret unless the server has its own bearer token configured.
        api_key = os.getenv("MODAL_API_KEY", token_secret)

        self.client = OpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers)
        self.model = model

    @property
    def name(self) -> str:
        return "modal"

    def generate(
        self, prompt_text: str, images: list[Path], generation: dict[str, Any]
    ) -> ProviderResult:
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

        message = response.choices[0].message
        text = message.content or ""

        usage_info = None
        if getattr(response, "usage", None):
            usage_info = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            reasoning_tokens = getattr(response.usage, "reasoning_tokens", None)
            if reasoning_tokens is not None:
                usage_info["reasoning_tokens"] = reasoning_tokens

        raw = {"model": getattr(response, "model", self.model)}
        if getattr(response, "id", None) is not None:
            raw["id"] = response.id
        if getattr(response, "created", None) is not None:
            raw["created"] = response.created
        # Reasoning models (e.g. Kimi K3) spend part of max_tokens on a
        # separate reasoning_content field before the final answer -- if the
        # budget runs out mid-reasoning, `text` comes back empty even though
        # completion_tokens is nonzero. Keep it around for debugging that.
        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content:
            raw["reasoning_content"] = reasoning_content
        finish_reason = getattr(response.choices[0], "finish_reason", None)
        if finish_reason is not None:
            raw["finish_reason"] = finish_reason

        return ProviderResult(
            text=text,
            raw=raw,
            usage=usage_info,
            latency_ms=latency_ms,
            model=self.model,
            provider="modal",
        )
