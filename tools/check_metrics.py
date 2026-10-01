"""
Original vs. modified Layout Similarity (paper Table 12 and Figure 2).

Scores reconstructed HTML against the original Sketch2Code reference HTML
(dataset/source_html/<id>.html) with both formulations:

  layout_similarity           Modified metric: geometric-union IoU + text-block merging
                              (the benchmark metric, src/responsive/eval/metrics/layout_similarity).
  layout_similarity_original  Sketch2Code's original implementation (pairwise box IoU,
                              no text-block merging).

Modes:
  directory  (default) - evaluate every *.html in one or more candidate directories
                         against the matching dataset/source_html/<id>.html reference
  file                 - compare specific HTML files against one reference

Output:
  directory mode  {output_root}/{metric}/{candidate_dir_name}/results.json
  file mode       stdout  (+ {output_root}/file/{candidate_stem}/ for artifacts)

Usage examples:
  # Table 12: both settings, both metric variants
  python tools/check_metrics.py

  # One candidate dir
  python tools/check_metrics.py --metric layout_similarity \\
      --candidate-dir dataset/layout_similarity_study/html_and_screenshots

  # File mode
  python tools/check_metrics.py \\
      --reference dataset/source_html/0041.html \\
      --candidate dataset/layout_similarity_study/html_and_screenshots/0041.html
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── layout_similarity ─────────────────────────────────────────────────────────

async def _run_layout_similarity(
    ref_path: Path,
    gen_path: Path,
    artifacts_dir: Path,
    semaphore: asyncio.Semaphore,
) -> dict:
    from responsive.eval.metrics.layout_similarity import LayoutSimilarityMetric
    async with semaphore:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        sample = SimpleNamespace(reference_html=ref_path)
        context = {"artifacts_dir": artifacts_dir, "playwright": {"headless": True, "timeout_ms": 60000}}
        try:
            result, _ = await LayoutSimilarityMetric().run(sample, gen_path, context)
            return result
        except Exception as exc:
            return {"status": "error", "error": str(exc)}


# ── layout_similarity_original (Sketch2Code paper exact implementation) ───────
#
# Differences from our updated layout_similarity:
#   1. Browser launched without --allow-file-access-from-files
#   2. page.goto without wait_until="networkidle"
#   3. No scroll loop to trigger lazy-loaded images
#   4. No _merge_text_blocks post-processing step in metric.py
# Everything else — selectors, boxes_adjacent text merging, normalization,
# compute_weighted_iou_shapely — is identical to the paper.

def _extract_visual_components_original(url, save_path):
    """
    extract_visual_components from Sketch2Code html_utils.py, verbatim.
    Only adaptation: accepts pathlib.Path in addition to str.
    """
    import io as _io
    import os as _os
    from PIL import Image, ImageDraw, ImageFont
    from playwright.sync_api import sync_playwright
    from responsive.eval.metrics.layout_similarity.box_utils import (
        boxes_adjacent, merge_boxes, is_within,
    )

    if isinstance(url, Path):
        url = str(url)
    if _os.path.exists(url):
        url = "file://" + _os.path.abspath(url)

    screenshot_image = None
    element_data = {}

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()   # no --allow-file-access-from-files
            page = browser.new_page()
            page.goto(url, timeout=60000)   # no wait_until="networkidle", no scroll

            total_width  = page.evaluate("() => document.documentElement.scrollWidth")
            total_height = page.evaluate("() => document.documentElement.scrollHeight")

            selectors = {
                'video':      'video',
                'image':      'img',
                'text_block': 'p, span, a, strong, h1, h2, h3, h4, h5, h6, li, th, td, label, code, pre, div',
                'form_table': 'form, table, div.form',
                'button':     'button, input[type="button"], input[type="submit"], [role="button"]',
                'nav_bar':    'nav, [role="navigation"], .navbar, [class~="nav"], [class~="navigation"], [class~="menu"], [class~="navbar"], [id="menu"], [id="nav"], [id="navigation"], [id="navbar"]',
                'divider':    'hr, [class*="separator"], [class*="divider"], [id="separator"], [id="divider"], [role="separator"]',
            }

            for key, value in selectors.items():
                element_data[key] = []
                elements = page.query_selector_all(value)
                for element in elements:
                    if not element.is_visible():
                        continue
                    box = element.bounding_box()
                    if box and box['width'] > 0 and box['height'] > 0:
                        text_content = None
                        if key == 'text_block':
                            is_direct_text = element.evaluate("""
                                (el) => Array.from(el.childNodes).some(node =>
                                    node.nodeType === Node.TEXT_NODE && node.textContent.trim() !== '')
                            """)
                            tag_name = element.evaluate("el => el.tagName.toLowerCase()")
                            if tag_name == 'div' and not is_direct_text:
                                continue
                            text_content = element.text_content().strip()
                            if not text_content:
                                continue
                        element_data[key].append({
                            'type': key,
                            'box': {'x': box['x'], 'y': box['y'], 'width': box['width'], 'height': box['height']},
                            'text_content': text_content,
                        })

            # Process adjacent text blocks (identical to paper)
            text_blocks = element_data['text_block']
            text_blocks.sort(key=lambda block: (block['box']['y'], block['box']['x']))
            merged_text_blocks = []
            while text_blocks:
                current = text_blocks.pop(0)
                index = 0
                add = True
                while index < len(text_blocks):
                    if is_within(text_blocks[index]['box'], current['box']):
                        del text_blocks[index]
                        continue
                    elif is_within(current['box'], text_blocks[index]['box']):
                        add = False
                        break
                    if boxes_adjacent(current['box'], text_blocks[index]['box']):
                        if current['box']['x'] < text_blocks[index]['box']['x'] or current['box']['y'] < text_blocks[index]['box']['y']:
                            current['text_content'] += " " + text_blocks[index]['text_content']
                        else:
                            current['text_content'] = text_blocks[index]['text_content'] + " " + current['text_content']
                        current['box'] = merge_boxes(current['box'], text_blocks[index]['box'])
                        del text_blocks[index]
                    else:
                        index += 1
                if add:
                    merged_text_blocks.append(current)
            element_data['text_block'] = merged_text_blocks

            image_bytes = page.screenshot(full_page=True, animations="disabled", timeout=60000)
            image_buffer = _io.BytesIO(image_bytes)
            screenshot_image = Image.open(image_buffer)
            draw = ImageDraw.Draw(screenshot_image)
            font = ImageFont.load_default()
            for key in element_data:
                for item in element_data[key]:
                    x = item['box']['x']; y = item['box']['y']
                    w = item['box']['width']; h = item['box']['height']
                    draw.rectangle(((x, y), (x + w, y + h)), outline="red", width=2)
                    draw.text((x, y), item['type'], fill="red", font=font)
            if save_path:
                screenshot_image.save(save_path)
                print(f"Saved annotated image to {save_path}")

            # Normalize to relative coordinates
            for key in element_data:
                for item in element_data[key]:
                    b = item['box']
                    item['box'] = {
                        'x': b['x'] / total_width,
                        'y': b['y'] / total_height,
                        'width': b['width'] / total_width,
                        'height': b['height'] / total_height,
                    }
            browser.close()

    except Exception as e:
        print(f"Failed to take screenshot due to: {e}. Generating a blank image.")
        import traceback; traceback.print_exc()
        screenshot_image = Image.new('RGB', (1280, 960), color='white')
        if save_path:
            screenshot_image.save(save_path)

    return element_data


async def _run_layout_similarity_original(
    ref_path: Path,
    gen_path: Path,
    artifacts_dir: Path,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        ref_screenshot = artifacts_dir / "layout_reference_annotated.png"
        gen_screenshot = artifacts_dir / "layout_generated_annotated.png"
        try:
            from responsive.eval.metrics.layout_similarity.iou_computation import (
                compute_weighted_iou_shapely,
            )
            ref_layout = await asyncio.to_thread(
                _extract_visual_components_original, ref_path, ref_screenshot
            )
            gen_layout = await asyncio.to_thread(
                _extract_visual_components_original, gen_path, gen_screenshot
            )
            weighted_iou, multi_scores = compute_weighted_iou_shapely(gen_layout, ref_layout)
            return {
                "status": "success",
                "weighted_iou": float(weighted_iou),
                "multi_scores": {
                    k: {"iou_score": float(s), "weight": float(w)}
                    for k, (s, w) in multi_scores.items()
                },
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}


# ── score key per metric ──────────────────────────────────────────────────────

_SCORE_KEYS = {
    "layout_similarity": "weighted_iou",
    "layout_similarity_original": "weighted_iou",
}

_METRIC_DESCS = {
    "layout_similarity": "Weighted IoU of visual element bounding boxes (updated: Shapely unions + text-block merge)",
    "layout_similarity_original": "Weighted IoU — Sketch2Code paper exact implementation (no scroll, no networkidle, no post-merge)",
}


# ── directory runners ─────────────────────────────────────────────────────────

async def _run_dir_layout_similarity(
    candidate_dir: Path,
    html_dir: Path,
    artifacts_root: Path,
    concurrency: int,
) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    pairs = []
    for gen_path in sorted(candidate_dir.glob("*.html")):
        sid = gen_path.stem
        ref_path = html_dir / f"{sid}.html"
        if not ref_path.exists():
            print(f"  Warning: no reference for {sid}, skipping", file=sys.stderr)
            continue
        pairs.append((sid, ref_path, gen_path))

    async def _one(sid, ref_path, gen_path):
        r = await _run_layout_similarity(ref_path, gen_path, artifacts_root / sid, semaphore)
        return {"id": sid, **r}

    return list(await asyncio.gather(*[_one(sid, r, g) for sid, r, g in pairs]))



async def _run_dir_layout_similarity_original(
    candidate_dir: Path,
    html_dir: Path,
    artifacts_root: Path,
    concurrency: int,
) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    pairs = []
    for gen_path in sorted(candidate_dir.glob("*.html")):
        sid = gen_path.stem
        ref_path = html_dir / f"{sid}.html"
        if not ref_path.exists():
            print(f"  Warning: no reference for {sid}, skipping", file=sys.stderr)
            continue
        pairs.append((sid, ref_path, gen_path))

    async def _one(sid, ref_path, gen_path):
        r = await _run_layout_similarity_original(ref_path, gen_path, artifacts_root / sid, semaphore)
        return {"id": sid, **r}

    return list(await asyncio.gather(*[_one(sid, r, g) for sid, r, g in pairs]))


async def _run_dir_metric(
    metric: str,
    candidate_dir: Path,
    html_dir: Path,
    output_dir: Path,
    concurrency: int,
) -> list[dict]:
    if metric == "layout_similarity":
        artifacts_root = output_dir / "artifacts"
        return await _run_dir_layout_similarity(candidate_dir, html_dir, artifacts_root, concurrency)
    elif metric == "layout_similarity_original":
        artifacts_root = output_dir / "artifacts"
        return await _run_dir_layout_similarity_original(candidate_dir, html_dir, artifacts_root, concurrency)
    else:
        raise ValueError(f"Unknown metric: {metric}")


# ── printing ──────────────────────────────────────────────────────────────────

def _print_dir_table(metric: str, results: list[dict], label: str) -> None:
    key = _SCORE_KEYS[metric]
    W_ID, W_SCORE = 8, 16

    # Extra columns per metric
    extra_cols: list[tuple[str, str]] = []   # (header, result_key)
    if metric in ("layout_similarity", "layout_similarity_original"):
        cat_keys: list[str] = []
        for r in results:
            if r.get("status") == "success" and "multi_scores" in r:
                cat_keys = sorted(r["multi_scores"])
                break
        extra_cols = [(k, k) for k in cat_keys]

    header = f"{'id':<{W_ID}}  {key:<{W_SCORE}}"
    for col_hdr, _ in extra_cols:
        header += f"  {col_hdr:<12}"

    sep = "-" * len(header)
    print(f"\n[{metric}]  {label}")
    print(sep)
    print(header)
    print(sep)

    values: list[float] = []
    for r in sorted(results, key=lambda x: x["id"]):
        if r.get("status") != "success":
            print(f"{r['id']:<{W_ID}}  ERROR  {r.get('error', '')}")
            continue
        v = r.get(key, float("nan"))
        values.append(v)
        row = f"{r['id']:<{W_ID}}  {v:<{W_SCORE}.4f}"
        for _, rkey in extra_cols:
            extra_v = r.get("multi_scores", {}).get(rkey, {}).get("iou_score", float("nan"))
            row += f"  {extra_v:<12.4f}"
        print(row)

    if values:
        print(sep)
        print(f"{'mean':<{W_ID}}  {sum(values)/len(values):<{W_SCORE}.4f}")
        print(f"{'min':<{W_ID}}  {min(values):<{W_SCORE}.4f}")
        print(f"{'max':<{W_ID}}  {max(values):<{W_SCORE}.4f}")
    print(sep)


def _print_unified_dir_table(all_results: dict[str, list[dict]], label: str) -> None:
    """One row per sample, one column per metric — shown when >1 metric selected."""
    W_ID, W_COL = 8, 14
    metrics = list(all_results.keys())

    by_metric: dict[str, dict[str, float | None]] = {}
    all_ids: set[str] = set()
    for metric, results in all_results.items():
        key = _SCORE_KEYS[metric]
        by_metric[metric] = {}
        for r in results:
            sid = r["id"]
            all_ids.add(sid)
            by_metric[metric][sid] = (
                r[key] if r.get("status") == "success" and key in r else None
            )

    header = f"{'id':<{W_ID}}"
    for metric in metrics:
        header += f"  {_SCORE_KEYS[metric]:<{W_COL}}"
    sep = "-" * len(header)

    print(f"\n[summary]  {label}")
    print(sep)
    print(header)
    print(sep)

    per_metric_values: dict[str, list[float]] = {m: [] for m in metrics}
    for sid in sorted(all_ids):
        row = f"{sid:<{W_ID}}"
        for metric in metrics:
            v = by_metric[metric].get(sid)
            if v is None:
                row += f"  {'ERR':<{W_COL}}"
            else:
                per_metric_values[metric].append(v)
                row += f"  {v:<{W_COL}.4f}"
        print(row)

    print(sep)
    for stat_name, fn in [("mean", lambda vs: sum(vs) / len(vs)), ("min", min), ("max", max)]:
        row = f"{stat_name:<{W_ID}}"
        for metric in metrics:
            vs = per_metric_values[metric]
            row += f"  {fn(vs):<{W_COL}.4f}" if vs else f"  {'---':<{W_COL}}"
        print(row)
    print(sep)


def _print_file_table(reference: Path, candidate: Path, metric_scores: dict) -> None:
    W = 22
    sep = "  " + "-" * 46

    print(f"\n  reference : {reference}")
    print(f"  candidate : {candidate}")

    print(f"\n  [summary]")
    print(sep)
    for metric, result in metric_scores.items():
        key = _SCORE_KEYS[metric]
        if result.get("status") != "success":
            print(f"  {metric:<{W}}  ERROR  {result.get('error', '')}")
        else:
            score = result.get(key, float("nan"))
            print(f"  {metric:<{W}}  {score:.4f}  ({key})")
    print(sep)

    for ls_metric in ("layout_similarity", "layout_similarity_original"):
        if ls_metric in metric_scores:
            result = metric_scores[ls_metric]
            if result.get("status") == "success" and "multi_scores" in result:
                print(f"\n  [{ls_metric} detail]")
                print(sep)
                for cat, scores in sorted(result["multi_scores"].items()):
                    if scores["weight"] > 0:
                        print(
                            f"  {cat:<20}  iou={scores['iou_score']:.4f}"
                            f"  weight={scores['weight']:.4f}"
                        )
                print(sep)

    print()


# ── JSON writing ──────────────────────────────────────────────────────────────

def _write_dir_json(
    metric: str,
    results: list[dict],
    candidate_dir: Path,
    dataset_root: str,
    output_dir: Path,
) -> None:
    key = _SCORE_KEYS[metric]
    values = [r[key] for r in results if r.get("status") == "success" and key in r]
    summary: dict = {}
    if values:
        n = len(values)
        summary = {
            "n": n,
            f"mean_{key}": round(sum(values) / n, 4),
            f"min_{key}": round(min(values), 4),
            f"max_{key}": round(max(values), 4),
        }

    output = {
        "metric": metric,
        "definition": _METRIC_DESCS[metric],
        "candidate_dir": str(candidate_dir),
        "dataset_root": dataset_root,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "samples": {
            r["id"]: {k: v for k, v in r.items() if k != "id"}
            for r in sorted(results, key=lambda x: x["id"])
        },
        "summary": summary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "results.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"  → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    file_mode = args.reference is not None or args.candidate is not None

    if file_mode:
        if not args.reference or not args.candidate:
            print("Error: --reference and --candidate must both be provided in file mode.", file=sys.stderr)
            sys.exit(1)

        ref_path = Path(args.reference)
        if not ref_path.exists():
            print(f"Error: --reference path not found: {ref_path}", file=sys.stderr)
            sys.exit(1)

        semaphore = asyncio.Semaphore(1)

        for candidate_str in args.candidate:
            gen_path = Path(candidate_str)
            if not gen_path.exists():
                print(f"Warning: --candidate path not found: {gen_path}, skipping", file=sys.stderr)
                continue

            metric_scores: dict = {}

            for metric in args.metric:
                if metric == "layout_similarity":
                    artifacts_dir = Path(args.output) / "file" / gen_path.stem
                    result = await _run_layout_similarity(ref_path, gen_path, artifacts_dir, semaphore)
                    metric_scores[metric] = result
                    if result.get("status") == "success":
                        print(f"  Annotated PNGs saved to: {artifacts_dir}")

                elif metric == "layout_similarity_original":
                    artifacts_dir = Path(args.output) / "file_original" / gen_path.stem
                    result = await _run_layout_similarity_original(ref_path, gen_path, artifacts_dir, semaphore)
                    metric_scores[metric] = result
                    if result.get("status") == "success":
                        print(f"  Annotated PNGs saved to: {artifacts_dir}")

            _print_file_table(ref_path, gen_path, metric_scores)

    else:
        html_dir = Path(args.dataset_root) / "source_html"
        if not html_dir.exists():
            print(f"Reference HTML dir not found: {html_dir}", file=sys.stderr)
            sys.exit(1)

        candidate_dirs = args.candidate_dir or [
            "dataset/layout_similarity_study/html_and_screenshots",
            "dataset/layout_similarity_study/screenshots_only",
        ]

        for candidate_dir_str in candidate_dirs:
            candidate_dir = Path(candidate_dir_str)
            if not candidate_dir.exists():
                print(f"Skipping {candidate_dir} (not found)", file=sys.stderr)
                continue

            all_results: dict[str, list[dict]] = {}
            for metric in args.metric:
                output_dir = Path(args.output) / metric / candidate_dir.name
                print(f"\nRunning [{metric}] on {candidate_dir.name} …")
                all_results[metric] = await _run_dir_metric(
                    metric, candidate_dir, html_dir, output_dir, args.concurrency
                )

            if len(args.metric) > 1:
                _print_unified_dir_table(all_results, candidate_dir.name)

            for metric, results in all_results.items():
                _print_dir_table(metric, results, candidate_dir.name)

            for metric, results in all_results.items():
                output_dir = Path(args.output) / metric / candidate_dir.name
                _write_dir_json(metric, results, candidate_dir, args.dataset_root, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Original vs. modified Layout Similarity against the Sketch2Code reference HTML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--metric",
        action="append",
        choices=["layout_similarity", "layout_similarity_original"],
        metavar="NAME",
        default=None,
        help=(
            "Metric to compute (repeatable). "
            "Choices: layout_similarity, layout_similarity_original. "
            "Default: both. "
            "layout_similarity_original uses the exact Sketch2Code paper implementation "
            "(no scroll, no networkidle wait, no _merge_text_blocks post-processing)."
        ),
    )
    parser.add_argument(
        "--candidate-dir",
        action="append",
        nargs="+",
        metavar="DIR",
        default=None,
        help=(
            "Candidate HTML directories (repeatable, accepts multiple values per flag). "
            "Default: dataset/layout_similarity_study/html_and_screenshots and dataset/layout_similarity_study/screenshots_only."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        default="dataset",
        help="Dataset root containing source_html/ subdirectory (default: dataset).",
    )
    parser.add_argument(
        "--reference",
        metavar="FILE",
        default=None,
        help="Reference HTML file (triggers file mode).",
    )
    parser.add_argument(
        "--candidate",
        nargs="+",
        metavar="FILE",
        default=None,
        help="Candidate HTML file(s) (triggers file mode; accepts multiple values).",
    )
    parser.add_argument(
        "-o", "--output",
        default="results/metrics",
        help=(
            "Root output directory. "
            "Directory mode: {root}/{metric}/{candidate_dir}/results.json. "
            "File mode: {root}/file/{candidate_stem}/ for artifacts. "
            "Default: results/metrics."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max concurrent browser evaluations for layout_similarity (default: 4).",
    )

    parsed = parser.parse_args()
    if parsed.metric is None:
        parsed.metric = ["layout_similarity", "layout_similarity_original"]
    # Flatten nested lists produced by action="append" + nargs="+"
    if parsed.candidate_dir:
        parsed.candidate_dir = [d for group in parsed.candidate_dir for d in group]

    asyncio.run(main(parsed))
