"""Jinja2-based loader for eval judge prompt templates.

Templates live under src/responsive/eval/prompts/ in subdirectories by domain:
  llm_judge/   — responsive screenshot judge prompts
  accessible/  — accessibility metric judge prompts

Usage:
    from responsive.eval.prompts.loader import render_judge_prompt
    text = render_judge_prompt("llm_judge/unified_responsive_v1", {"rubric_guidance": ...})
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
    _JINJA2_AVAILABLE = True
except ImportError:
    _JINJA2_AVAILABLE = False

_PROMPTS_DIR = Path(__file__).parent


def _get_env() -> "Environment":
    if not _JINJA2_AVAILABLE:
        raise ImportError(
            "Jinja2 is required for prompt templates. Install with: pip install Jinja2>=3.1.0"
        )
    return Environment(
        loader=FileSystemLoader(str(_PROMPTS_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )


def render_judge_prompt(template_id: str, vars: dict[str, Any] | None = None) -> str:
    """Render a judge prompt template by ID.

    Args:
        template_id: Path relative to the prompts directory, without .j2 extension.
                     E.g. "llm_judge/unified_responsive_v1" or "accessible/landmarks_v1".
        vars: Optional dict of Jinja2 template variables.

    Returns:
        Rendered prompt string.

    Raises:
        ImportError: If Jinja2 is not installed.
        jinja2.TemplateNotFound: If the template file does not exist.
        jinja2.UndefinedError: If a required template variable is missing.
    """
    env = _get_env()
    template = env.get_template(f"{template_id}.j2")
    return template.render(**(vars or {}))
