from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

try:
    from playwright.async_api import Page
except ImportError:  # pragma: no cover
    Page = Any  # type: ignore[misc,assignment]

ViewportName = Literal["mobile", "tablet", "desktop"]

VIEWPORTS: dict[ViewportName, dict[str, int]] = {
    "mobile": {"width": 375, "height": 688},
    "tablet": {"width": 768, "height": 1024},
    "desktop": {"width": 1280, "height": 800},
}


def prefer_default_playwright_browsers_path() -> None:
    """
    Cursor's sandbox sets PLAYWRIGHT_BROWSERS_PATH to a temp cache that can be missing
    the correct arch binaries. For evaluation runs, prefer Playwright's default browser
    discovery by clearing this override.
    """
    if "PLAYWRIGHT_BROWSERS_PATH" in os.environ:
        os.environ.pop("PLAYWRIGHT_BROWSERS_PATH", None)


def to_file_url(path: Path) -> str:
    abs_path = path.resolve()
    path_str = str(abs_path).replace("\\", "/")
    if len(path_str) > 1 and path_str[1] == ":":
        path_str = "/" + path_str
    return f"file://{path_str}"


ARIA_LINE_RE = re.compile(
    r"^(?P<role>[a-zA-Z_][a-zA-Z0-9_-]*)"
    r'(?:\s+"(?P<name>[^"]*)")?'
    r"(?:\s+\[(?P<attrs>[^\]]+)\])?"
    r"\s*$"
)


def _parse_attrs(attr_text: str | None) -> dict[str, Any]:
    if not attr_text:
        return {}
    out: dict[str, Any] = {}
    parts = [p.strip() for p in attr_text.split(",") if p.strip()]
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"')
            if v.lower() in ("true", "false"):
                out[k] = v.lower() == "true"
            else:
                out[k] = v
        else:
            out[p] = True
    return out


def parse_aria_snapshot_yaml(yaml_text: str) -> dict[str, Any]:
    """
    Parse Playwright locator.aria_snapshot() YAML into a simple tree:
      {"type": role, "label": name, "properties": {...}, "children": [...]}
    """
    root: dict[str, Any] = {"type": "root", "label": None, "properties": {}, "children": []}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw in (yaml_text or "").splitlines():
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if not line.startswith("- "):
            continue
        payload = line[2:].strip()
        m = ARIA_LINE_RE.match(payload)
        if not m:
            continue
        role = m.group("role")
        name = m.group("name")
        props = _parse_attrs(m.group("attrs"))

        node: dict[str, Any] = {"type": role, "label": name, "properties": props, "children": []}

        while stack and indent <= stack[-1][0]:
            stack.pop()
        stack[-1][1]["children"].append(node)
        stack.append((indent, node))

    return root


def iter_nodes(tree: dict[str, Any]):
    if not isinstance(tree, dict):
        return
    yield tree
    for c in tree.get("children") or []:
        if isinstance(c, dict):
            yield from iter_nodes(c)


def find_node(tree: dict[str, Any], node_type: str) -> dict[str, Any] | None:
    for n in iter_nodes(tree):
        if n.get("type") == node_type:
            return n
    return None


def get_nodes_by_type(tree: dict[str, Any], node_type: str) -> list[dict[str, Any]]:
    return [n for n in iter_nodes(tree) if n.get("type") == node_type]


def is_nested_inside(tree: dict[str, Any], *, child: str, parent: str) -> bool:
    def walk(n: dict[str, Any], ancestors: list[str]) -> bool:
        t = str(n.get("type") or "")
        if t == child and parent in ancestors:
            return True
        new_anc = ancestors + ([t] if t else [])
        for c in n.get("children") or []:
            if isinstance(c, dict) and walk(c, new_anc):
                return True
        return False

    return walk(tree, [])


async def aria_snapshot_tree(page: Page) -> dict[str, Any]:
    body = page.locator("body")
    if not hasattr(body, "aria_snapshot"):
        raise RuntimeError(
            "Playwright does not support aria_snapshot() in this environment. Upgrade Playwright to >= 1.59."
        )
    yaml_text = await body.aria_snapshot()
    return parse_aria_snapshot_yaml(yaml_text)


async def aria_snapshot_yaml(page: Page) -> str:
    body = page.locator("body")
    if not hasattr(body, "aria_snapshot"):
        raise RuntimeError(
            "Playwright does not support aria_snapshot() in this environment. Upgrade Playwright to >= 1.59."
        )
    return str(await body.aria_snapshot() or "")


def score_from_failures(*, failures: list[str], critical: list[str] | None = None) -> int:
    critical = critical or []
    if not failures:
        return 5
    if any(f in critical for f in failures):
        return 1
    if len(failures) == 1:
        return 3
    return 2

