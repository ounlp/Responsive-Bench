"""
Fake provider for testing and smoke tests.

Returns deterministic HTML output without making API calls.
Sketch pipeline step 1 returns minimal JSON when generation["_responsive_step"] == "sketch_analyze".
"""

import json
import os
from pathlib import Path
from typing import Any

from .base import Provider, ProviderResult


class FakeProvider(Provider):
    """Fake provider that returns static HTML for testing."""

    def __init__(self, model: str, **kwargs: Any):
        """
        Initialize fake provider.

        Args:
            model: Model identifier (ignored, but kept for interface consistency)
            **kwargs: Ignored
        """
        self.model = model

    @property
    def name(self) -> str:
        """Return provider name."""
        return "fake"

    def generate(
        self, prompt_text: str, images: list[Path], generation: dict[str, Any]
    ) -> ProviderResult:
        """
        Generate fake HTML output.

        Args:
            prompt_text: Prompt text (ignored)
            images: List of image paths (validated to exist)
            generation: Generation parameters (ignored)

        Returns:
            ProviderResult with deterministic HTML output
        """
        # Validate images exist
        for img_path in images:
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")

        image_count = len(images)
        step = generation.get("_responsive_step")
        if step == "sketch_analyze":
            fake_obj = {
                "patterns_found": [],
                "applicable_criteria": ["K1", "K3"],
                "criteria_not_applicable": ["V1"],
                "implementation_priorities": {"mobile": [], "desktop": []},
                "sketch_notes": "Fake provider smoke output; no patterns inferred.",
            }
            body = json.dumps(fake_obj, ensure_ascii=False)
            result, latency_ms = self._measure_latency(lambda: None)
            return ProviderResult(
                text=body,
                raw={"provider": "fake", "model": self.model, "step": step},
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                latency_ms=latency_ms,
                model=self.model,
                provider="fake",
            )

        # Return deterministic HTML (normal generation or sketch augment step)
        fake_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fake Generated Page</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
        }}
        @media (max-width: 375px) {{
            body {{ background-color: #f0f0f0; }}
        }}
        @media (min-width: 376px) and (max-width: 768px) {{
            body {{ background-color: #e0e0e0; }}
        }}
        @media (min-width: 769px) {{
            body {{ background-color: #d0d0d0; }}
        }}
    </style>
</head>
<body>
    <h1>Fake Generated Responsive Page</h1>
    <p>This is a deterministic fake output for testing.</p>
    <p>Images processed: {image_count}</p>
</body>
</html>"""

        result, latency_ms = self._measure_latency(lambda: None)

        return ProviderResult(
            text=fake_html,
            raw={"provider": "fake", "model": self.model, "images_count": len(images)},
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            latency_ms=latency_ms,
            model=self.model,
            provider="fake",
        )
