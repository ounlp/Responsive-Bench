"""
Hugging Face provider implementation: loads a model directly via `transformers`
and runs inference in-process, instead of calling a hosted API or a vLLM server.

Intended for local/merged checkpoints (e.g. SFT/RL training outputs) that would
otherwise require standing up a vLLM server to evaluate.
"""

import os

# Avoid importing a broken/unrelated TensorFlow install via transformers.
os.environ.setdefault("USE_TF", "0")
# On macOS, conda-forge's libomp and PyTorch's bundled OpenMP runtime both get linked,
# which aborts the process at import time ("OMP: Error #15"). Must be set before torch
# is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import re
import threading
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
except ImportError:
    AutoModelForImageTextToText = None
    AutoProcessor = None

try:
    from PIL import Image
except ImportError:
    Image = None

from .base import Provider, ProviderResult

# Full-page screenshots in this dataset are tall and high-resolution. Left uncapped,
# the processor's default pixel budget produces thousands of vision tokens per image
# and the vision encoder runs out of memory. Sketch
# images are exempt from this cap (see _is_sketch_image) — full detail matters for the
# judge/generation prompts that compare against them.
DEFAULT_MAX_PIXELS = 4096 * 28 * 28

# Raw dataset sketch filenames: "{id}-{d|m}-v2.png" (src/responsive/dataset/loader.py).
_DATASET_SKETCH_PATTERN = re.compile(r"^\d{4}-[dm]-v2\.png$", re.IGNORECASE)


def _is_sketch_image(path: Path) -> bool:
    """True for sketch images from either the raw dataset or the eval pipeline's
    copies (which are named sketch_*.png; see orchestrator.py / llm_judge_screenshots.py)."""
    name = path.name
    return "sketch" in name.lower() or bool(_DATASET_SKETCH_PATTERN.match(name))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "")


class HFProvider(Provider):
    """Provider that loads a model directly from Hugging Face (hub id or local path)."""

    def __init__(self, model: str, **kwargs: Any):
        if torch is None:
            raise ImportError("torch not installed. Install with: pip install torch")
        if AutoModelForImageTextToText is None or AutoProcessor is None:
            raise ImportError(
                "transformers not installed. Install with: pip install transformers"
            )
        if Image is None:
            raise ImportError("Pillow is required for the hf provider")

        device_map = os.getenv("HF_DEVICE_MAP", "auto")
        dtype_name = os.getenv("HF_DTYPE", "bfloat16")
        dtype = getattr(torch, dtype_name) if dtype_name != "auto" else "auto"
        trust_remote_code = _env_bool("HF_TRUST_REMOTE_CODE", True)
        attn_implementation = os.getenv("HF_ATTN_IMPLEMENTATION")

        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = AutoModelForImageTextToText.from_pretrained(model, **model_kwargs)
        self.model.eval()

        # Loaded at the processor's own (generous) default pixel budget — we downscale
        # non-sketch images ourselves in generate() instead of capping at the processor
        # level, so sketch images can be excluded from the cap.
        self.processor = AutoProcessor.from_pretrained(
            model, trust_remote_code=trust_remote_code
        )

        max_pixels_env = os.getenv("HF_MAX_PIXELS")
        self._max_pixels = int(max_pixels_env) if max_pixels_env else DEFAULT_MAX_PIXELS

        self.model_id = model
        # device_map="auto" shards the model across GPUs and dispatches ops via
        # accelerate hooks; concurrent generate() calls from multiple threads are
        # not safe against that dispatch, so inference is fully serialized.
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "hf"

    @staticmethod
    def _downscale_to_pixel_budget(img: "Image.Image", max_pixels: int) -> "Image.Image":
        width, height = img.size
        if width * height <= max_pixels:
            return img
        scale = (max_pixels / (width * height)) ** 0.5
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return img.resize(new_size, Image.LANCZOS)

    def generate(
        self, prompt_text: str, images: list[Path], generation: dict[str, Any]
    ) -> ProviderResult:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
        for img_path in images:
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
            img = Image.open(img_path).convert("RGB")
            if not _is_sketch_image(img_path):
                img = self._downscale_to_pixel_budget(img, self._max_pixels)
            content.append({"type": "image", "image": img})

        messages: list[dict[str, Any]] = []
        system_prompt = generation.get("system_prompt")
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "user", "content": content})

        temperature = generation.get("temperature", 0.7)
        max_new_tokens = generation.get("max_tokens", 2000)
        do_sample = bool(temperature and temperature > 0)
        gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        if do_sample:
            gen_kwargs["temperature"] = temperature
            if "top_p" in generation:
                gen_kwargs["top_p"] = generation["top_p"]

        def call_model() -> tuple[str, int, int]:
            with self._lock:
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.model.device)

                with torch.inference_mode():
                    generated_ids = self.model.generate(**inputs, **gen_kwargs)

                input_len = inputs["input_ids"].shape[1]
                trimmed = generated_ids[:, input_len:]
                text = self.processor.batch_decode(
                    trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                return text, input_len, trimmed.shape[1]

        (text, prompt_tokens, completion_tokens), latency_ms = self._measure_latency(call_model)

        usage_info = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        return ProviderResult(
            text=text,
            raw={"model": self.model_id},
            usage=usage_info,
            latency_ms=latency_ms,
            model=self.model_id,
            provider="hf",
        )
