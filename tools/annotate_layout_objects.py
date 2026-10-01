"""
Annotate every HTML file in a directory with the exact object set layout_similarity scores.

For each *.html file directly inside the input directory, this:
  1. Extracts visual components the same way the layout_similarity metric does
     (extract_visual_components — same 7 categories, same first-pass text_block merge).
  2. Applies the metric's second text_block merge pass (metric._merge_text_blocks),
     producing the *final* object set actually fed into the weighted-IoU computation.
  3. Draws that final object set (not the raw/pre-merge extraction) onto a clean
     screenshot of the page, one color per category.
  4. Writes a per-file + aggregate object-count summary to results.json.

Both merge passes are imported from the metric package rather than reimplemented,
so the counts reported here always match what layout_similarity actually scores.

Usage:
    conda run -n responsive python tools/annotate_layout_objects.py <input_dir>
    conda run -n responsive python tools/annotate_layout_objects.py dataset/augmented_screenshot \\
        --output-dir results/layout_objects_augmented_screenshot --concurrency 4
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from PIL import ImageDraw, ImageFont  # noqa: E402

from responsive.eval.metrics.layout_similarity.visual_extractor import extract_visual_components  # noqa: E402
from responsive.eval.metrics.layout_similarity.metric import _merge_text_blocks  # noqa: E402

CATEGORY_COLORS: dict[str, str] = {
    "image": "#1f77b4",
    "text_block": "#2ca02c",
    "button": "#d62728",
    "nav_bar": "#9467bd",
    "form_table": "#ff7f0e",
    "divider": "#8c564b",
    "video": "#17becf",
}
DEFAULT_COLOR = "#e377c2"


def _draw_final_layout(clean_image, final_layout: dict[str, list[dict]], total_width: float, total_height: float):
    """Draw the metric's final (post-merge) boxes onto a clean screenshot, one color per category."""
    image = clean_image.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for category, items in final_layout.items():
        color = CATEGORY_COLORS.get(category, DEFAULT_COLOR)
        for item in items:
            box = item["box"]
            x = box["x"] * total_width
            y = box["y"] * total_height
            w = box["width"] * total_width
            h = box["height"] * total_height
            draw.rectangle(((x, y), (x + w, y + h)), outline=color, width=2)
            draw.text((x, y), category, fill=color, font=font)

    return image


def _annotate_one(html_path: Path, annotated_out: Path) -> dict[str, Any]:
    """Synchronous worker: extract, apply both merge passes, draw, and count."""
    extra: dict[str, Any] = {}
    try:
        # save_path=None: the image extract_visual_components would save reflects only its
        # own first-pass merge, not the metric's second pass, so we draw our own image below.
        layout = extract_visual_components(html_path, save_path=None, extra_out=extra)
    except Exception as exc:
        return {"status": "error", "error": f"extraction failed: {exc}"}

    if "clean_image" not in extra:
        return {"status": "error", "error": "extraction failed: no screenshot captured"}

    final_layout = _merge_text_blocks(layout)
    counts = {category: len(items) for category, items in final_layout.items()}
    total_objects = sum(counts.values())

    annotated_out.parent.mkdir(parents=True, exist_ok=True)
    image = _draw_final_layout(extra["clean_image"], final_layout, extra["total_width"], extra["total_height"])
    image.save(annotated_out)

    return {
        "status": "success",
        "counts": counts,
        "total_objects": total_objects,
        "annotated_image": str(annotated_out),
    }


async def _annotate_one_async(
    html_path: Path, annotated_out: Path, semaphore: asyncio.Semaphore
) -> tuple[str, dict[str, Any]]:
    async with semaphore:
        result = await asyncio.to_thread(_annotate_one, html_path, annotated_out)
    return html_path.stem, result


def _print_table(files_result: dict[str, dict[str, Any]], categories: list[str]) -> None:
    W_ID = max((len(k) for k in files_result), default=8) + 2
    W_ID = max(W_ID, 10)
    W_COL = 12

    header = f"{'file':<{W_ID}}  {'total':<{W_COL}}"
    for cat in categories:
        header += f"  {cat:<{W_COL}}"
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    totals: list[int] = []
    per_category: dict[str, list[int]] = {cat: [] for cat in categories}
    for stem, r in sorted(files_result.items()):
        if r.get("status") != "success":
            print(f"{stem:<{W_ID}}  {'ERROR':<{W_COL}}  {r.get('error', '')}")
            continue
        totals.append(r["total_objects"])
        row = f"{stem:<{W_ID}}  {r['total_objects']:<{W_COL}}"
        for cat in categories:
            count = r["counts"].get(cat, 0)
            per_category[cat].append(count)
            row += f"  {count:<{W_COL}}"
        print(row)

    if totals:
        print(sep)
        stats: list[tuple[str, Any]] = [
            ("mean", lambda vs: f"{sum(vs) / len(vs):<{W_COL}.2f}"),
            ("min", lambda vs: f"{min(vs):<{W_COL}}"),
            ("max", lambda vs: f"{max(vs):<{W_COL}}"),
        ]
        for stat_name, fmt in stats:
            row = f"{stat_name:<{W_ID}}  {fmt(totals)}"
            for cat in categories:
                row += f"  {fmt(per_category[cat])}"
            print(row)
    print(sep)


async def main(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        print(f"Error: not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    html_files = sorted(input_dir.glob("*.html"))
    if not html_files:
        print(f"No HTML files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path("results") / f"layout_objects_{input_dir.name}"
    annotated_dir = output_dir / "annotated"

    print(f"Input directory : {input_dir}")
    print(f"HTML files      : {len(html_files)}")
    print(f"Output directory: {output_dir}")
    print(f"Concurrency     : {args.concurrency}\n")
    print("Annotating…", flush=True)

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        _annotate_one_async(html_path, annotated_dir / f"{html_path.stem}_annotated.png", semaphore)
        for html_path in html_files
    ]
    results = await asyncio.gather(*tasks)
    files_result: dict[str, dict[str, Any]] = dict(results)

    categories = sorted(
        {cat for r in files_result.values() if r.get("status") == "success" for cat in r["counts"]}
    )

    print()
    _print_table(files_result, categories)

    successes = [r for r in files_result.values() if r.get("status") == "success"]
    summary: dict[str, Any] = {}
    if successes:
        totals = [r["total_objects"] for r in successes]
        summary = {
            "n_files": len(successes),
            "n_errors": len(files_result) - len(successes),
            "mean_total_objects": round(sum(totals) / len(totals), 4),
            "min_total_objects": min(totals),
            "max_total_objects": max(totals),
            "mean_counts_by_category": {
                cat: round(sum(r["counts"].get(cat, 0) for r in successes) / len(successes), 4)
                for cat in categories
            },
        }

    output = {
        "input_dir": str(input_dir),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "files": files_result,
        "summary": summary,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    results_json = output_dir / "results.json"
    results_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults written to: {results_json}")
    print(f"Annotated images in: {annotated_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Annotate every HTML file in a directory with the object set layout_similarity "
            "actually scores (post text_block merge), and report per-category counts."
        )
    )
    parser.add_argument("input_dir", help="Directory of HTML files to annotate (top-level *.html only)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for annotated images + results.json (default: results/layout_objects_<input_dir name>)",
    )
    parser.add_argument("--concurrency", type=int, default=4, help="Max concurrent browser sessions (default: 4)")
    asyncio.run(main(parser.parse_args()))
