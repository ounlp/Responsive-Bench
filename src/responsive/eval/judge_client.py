"""
Provider-agnostic LLM judge caller for evaluation metrics.

This wraps the existing provider registry used by the generation pipeline so
evaluation can use OpenAI, Claude, etc. consistently.
"""

from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from ..providers.registry import get_provider


def extract_json_object(text: str) -> str:
    """Strip common LLM response wrappers and return the first JSON object found.

    Handles:
      - <think>...</think> blocks (Qwen3 chain-of-thought; stripped before JSON search)
      - ```json\\n{...}\\n```  (fenced code block with language tag)
      - ```\\n{...}\\n```      (fenced code block without tag)
      - Leading "json" text after the fence
      - Surrounding prose before/after the first {...} block
    """
    t = (text or "").strip()
    if not t:
        return t
    # Strip <think>...</think> blocks produced by Qwen3 and similar reasoning models.
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
            t = "\n".join(lines[1:-1]).strip()
    if t.lower().startswith("json"):
        t = t[4:].lstrip()
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start : end + 1]
    return t


def _to_jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    try:
        import json

        json.dumps(obj)
        return obj
    except Exception:
        return repr(obj)


def call_judge(
    *,
    prompt_text: str,
    image_paths: list[Path],
    judge_cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Call an LLM judge provider with a text prompt + images.

    Args:
        prompt_text: Prompt text (should instruct strict JSON output).
        image_paths: Screenshot paths, in the order the provider should see them.
        judge_cfg: Dict with at least provider/model, plus generation params like
            temperature/max_tokens.

    Returns:
        Normalized dict containing provider output and metadata.
    """
    provider_name = judge_cfg.get("provider")
    model = judge_cfg.get("model")
    if not provider_name or not model:
        raise ValueError("judge config must include non-empty 'provider' and 'model'")

    provider = get_provider(str(provider_name), str(model))
    generation: dict[str, Any] = {
        "temperature": judge_cfg.get("temperature", 0.0),
        "max_tokens": judge_cfg.get("max_tokens", 1000),
    }
    if "response_format" in judge_cfg:
        generation["response_format"] = judge_cfg["response_format"]
    if "extra_body" in judge_cfg:
        generation["extra_body"] = judge_cfg["extra_body"]
    if judge_cfg.get("system_prompt"):
        generation["system_prompt"] = judge_cfg["system_prompt"]

    result = provider.generate(prompt_text, image_paths, generation)

    return {
        "provider": getattr(result, "provider", provider_name),
        "model": getattr(result, "model", model),
        "text": getattr(result, "text", ""),
        "usage": _to_jsonable(getattr(result, "usage", None)),
        "latency_ms": _to_jsonable(getattr(result, "latency_ms", None)),
        "raw": _to_jsonable(getattr(result, "raw", None)),
    }

