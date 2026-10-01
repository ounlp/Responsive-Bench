"""
Provider registry for instantiating providers by name.
"""

import threading
from typing import Any

from .base import Provider
from .fake_provider import FakeProvider
from .openai_provider import OpenAIProvider
from .gemini_provider import GeminiProvider
from .claude_provider import ClaudeProvider
from .vllm_provider import VLLMProvider
from .hf_provider import HFProvider
from .modal_provider import ModalProvider


_PROVIDER_CLASSES: dict[str, type[Provider]] = {
    "fake": FakeProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "claude": ClaudeProvider,
    "vllm": VLLMProvider,
    "hf": HFProvider,
    "modal": ModalProvider,
}

# Providers are cached by (name, model, kwargs) for the lifetime of the process.
# Cheap for API-backed providers (avoids reconnecting); essential for "hf", which
# loads model weights onto GPU(s) and must not be reconstructed on every call —
# callers like the judge client otherwise call get_provider() fresh per request.
_provider_cache: dict[tuple[str, str, str], Provider] = {}
_provider_cache_lock = threading.Lock()


def get_provider(name: str, model: str, **kwargs: Any) -> Provider:
    """
    Get a provider instance by name, reusing a cached instance when available.

    Args:
        name: Provider name (e.g., "openai", "gemini", "claude", "vllm", "hf", "modal", "fake")
        model: Model identifier for the provider
        **kwargs: Additional provider-specific initialization arguments

    Returns:
        Provider instance

    Raises:
        ValueError: If provider name is not recognized
    """
    if name not in _PROVIDER_CLASSES:
        available = ", ".join(sorted(_PROVIDER_CLASSES.keys()))
        raise ValueError(f"Unknown provider '{name}'. Available: {available}")

    cache_key = (name, model, repr(sorted(kwargs.items())))
    with _provider_cache_lock:
        cached = _provider_cache.get(cache_key)
        if cached is not None:
            return cached

        provider_class = _PROVIDER_CLASSES[name]
        instance = provider_class(model=model, **kwargs)
        _provider_cache[cache_key] = instance
        return instance
