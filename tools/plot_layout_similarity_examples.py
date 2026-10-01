"""
Publication figure for an ACL paper illustrating why the modified Layout
Similarity metric (geometric-union IoU + text-block merge, see
Appendix D of the paper) tracks visual layout similarity better than the original
Sketch2Code pairwise-IoU formulation — using four real sample pairs.

Reference/generated annotated screenshots come straight from the metric's own
artifacts (results/metrics/layout_similarity/<condition>/artifacts/<id>/), and
the Original/Modified scores are read from the corresponding results.json
files (layout_similarity_original vs layout_similarity), not hand-typed.
Run tools/check_metrics.py first to produce them; <condition> is the name of
the candidate dataset directory.

Samples were picked as the largest score-improvement examples in each
condition, skipping degenerate pages (e.g. HTTP error pages) — see
CANDIDATES below for the full ranked list this was chosen from.

Output: results/layout_similarity_examples.pdf (+ .png preview)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_ROOT = REPO_ROOT / "results" / "metrics"

COLOR_ORIGINAL = "#d03b3b"  # dataviz-skill status "critical"
COLOR_MODIFIED = "#0ca30c"  # dataviz-skill status "good"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
PANEL_EDGE = "#9a9a95"

# The four examples shown: (condition, sample_id, short description).
# Chosen as the two largest original->modified score deltas in each condition
# (0044/screenshots_only, the next-largest, was skipped — its
# reference page is a degenerate HTTP 503 error page, not a useful illustration).
EXAMPLES: list[dict[str, Any]] = [
    dict(condition="html_and_screenshots", sample_id="0049", desc="HTML + screenshots"),
    dict(condition="html_and_screenshots", sample_id="0043", desc="HTML + screenshots"),
    dict(condition="screenshots_only", sample_id="0039", desc="Screenshots only"),
    dict(condition="screenshots_only", sample_id="0048", desc="Screenshots only"),
]


def load_score(condition: str, sample_id: str) -> tuple[float, float]:
    """Read (original, modified) weighted_iou for one sample from the real results.json files."""
    orig = json.loads((METRICS_ROOT / "layout_similarity_original" / condition / "results.json").read_text())
    mod = json.loads((METRICS_ROOT / "layout_similarity" / condition / "results.json").read_text())
    return orig["samples"][sample_id]["weighted_iou"], mod["samples"][sample_id]["weighted_iou"]


def load_images(condition: str, sample_id: str) -> tuple[Image.Image, Image.Image]:
    art_dir = METRICS_ROOT / "layout_similarity" / condition / "artifacts" / sample_id
    ref = Image.open(art_dir / "layout_reference_annotated.png")
    gen = Image.open(art_dir / "layout_generated_annotated.png")
    return ref, gen


def score_box(ax, original: float, modified: float) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.04, 0.05), 0.92, 0.90, boxstyle="round,pad=0.0,rounding_size=0.10",
            facecolor="white", edgecolor=PANEL_EDGE, linewidth=1.0, zorder=1,
        )
    )
    ax.plot([0.5, 0.5], [0.15, 0.85], color=PANEL_EDGE, linewidth=0.8, zorder=2)
    ax.text(0.27, 0.70, "Original", ha="center", va="center", fontsize=6.2, color=TEXT_SECONDARY, zorder=3)
    ax.text(0.27, 0.30, f"{original:.2f}", ha="center", va="center", fontsize=10.5, fontweight="bold", color=COLOR_ORIGINAL, zorder=3)
    ax.text(0.73, 0.70, "Modified", ha="center", va="center", fontsize=6.2, color=TEXT_SECONDARY, zorder=3)
    ax.text(0.73, 0.30, f"{modified:.2f}", ha="center", va="center", fontsize=10.5, fontweight="bold", color=COLOR_MODIFIED, zorder=3)


def build_figure():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "text.color": TEXT_PRIMARY,
            "pdf.fonttype": 42,
        }
    )

    # ── Load everything first so the layout can be computed from real image
    # aspect ratios instead of guessed at (these pages vary a lot in length).
    for ex in EXAMPLES:
        ex["ref_img"], ex["gen_img"] = load_images(ex["condition"], ex["sample_id"])
        ex["orig_score"], ex["mod_score"] = load_score(ex["condition"], ex["sample_id"])

    # ── Absolute layout, in inches. GridSpec's fractional hspace/wspace
    # doesn't compose cleanly with per-example content of differing aspect
    # ratio, so every element gets an explicit inch-valued position instead.
    # All four examples sit in a single row (4 examples x 2 images each).
    FIG_W = 6.75
    SIDE_MARGIN = 0.08
    COL_GAP = 0.16
    IMG_GAP = 0.06
    N_EXAMPLES = len(EXAMPLES)
    W_IMG = (FIG_W - 2 * SIDE_MARGIN - (N_EXAMPLES - 1) * COL_GAP - N_EXAMPLES * IMG_GAP) / (2 * N_EXAMPLES)

    TAG_H = 0.18
    GAP1 = 0.04
    TITLE_H = 0.13
    GAP2 = 0.04
    GAP3 = 0.08
    SCORE_H = 0.50
    TOP_MARGIN = 0.10
    GAP_LEGEND = 0.10
    LEGEND_H = 0.32
    ANNOT_H = 0.38
    BOTTOM_MARGIN = 0.08

    for ex in EXAMPLES:
        ex["ref_h"] = W_IMG * ex["ref_img"].height / ex["ref_img"].width
        ex["gen_h"] = W_IMG * ex["gen_img"].height / ex["gen_img"].width
        ex["img_h"] = max(ex["ref_h"], ex["gen_h"])

    row_img_h = max(ex["img_h"] for ex in EXAMPLES)
    row_block_h = TAG_H + GAP1 + TITLE_H + GAP2 + row_img_h + GAP3 + SCORE_H

    FIG_H = TOP_MARGIN + row_block_h + GAP_LEGEND + LEGEND_H + ANNOT_H + BOTTOM_MARGIN

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")

    def axes_at(x_in: float, y_from_top_in: float, w_in: float, h_in: float):
        return fig.add_axes(
            [x_in / FIG_W, (FIG_H - y_from_top_in - h_in) / FIG_H, w_in / FIG_W, h_in / FIG_H]
        )

    for idx, ex in enumerate(EXAMPLES):
        x0 = SIDE_MARGIN + idx * (2 * W_IMG + IMG_GAP + COL_GAP)
        y0 = TOP_MARGIN
        tag = f"({'abcd'[idx]})"

        ax_tag = axes_at(x0, y0, 2 * W_IMG + IMG_GAP, TAG_H)
        ax_tag.axis("off")
        ax_tag.set_xlim(0, 1)
        ax_tag.set_ylim(0, 1)
        ax_tag.text(
            0.0, 0.5, f"{tag} {ex['sample_id']}",
            ha="left", va="center", fontsize=7.2, fontweight="bold", color=TEXT_PRIMARY,
            clip_on=True, transform=ax_tag.transAxes,
        )

        y_title = y0 + TAG_H + GAP1
        y_img = y_title + TITLE_H + GAP2

        for col_idx, (label, img, img_h) in enumerate(
            [("Reference", ex["ref_img"], ex["ref_h"]), ("Generated", ex["gen_img"], ex["gen_h"])]
        ):
            xi = x0 + col_idx * (W_IMG + IMG_GAP)

            ax_title = axes_at(xi, y_title, W_IMG, TITLE_H)
            ax_title.axis("off")
            ax_title.text(0.5, 0.5, label, ha="center", va="center", fontsize=7.8, color=TEXT_SECONDARY, transform=ax_title.transAxes)

            ax_img = axes_at(xi, y_img, W_IMG, img_h)
            ax_img.imshow(img)
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            for spine in ax_img.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(PANEL_EDGE)
                spine.set_linewidth(0.9)

        # Positioned under this example's own image height (not the row max)
        # so examples shorter than the tallest one don't carry a dead gap.
        y_score = y_img + ex["img_h"] + GAP3
        ax_score = axes_at(x0, y_score, 2 * W_IMG + IMG_GAP, SCORE_H)
        score_box(ax_score, ex["orig_score"], ex["mod_score"])

    y_legend = TOP_MARGIN + row_block_h + GAP_LEGEND
    ax_legend = axes_at(SIDE_MARGIN, y_legend, FIG_W - 2 * SIDE_MARGIN, LEGEND_H)
    ax_legend.axis("off")
    ax_legend.text(
        0.5, 0.5,
        "(a)-(b): HTML + screenshots condition. (c)-(d): Screenshots-only condition.\n"
        "Red boxes are the layout elements extracted by the metric (Playwright), labeled by category "
        "(text_block, image, button, nav_bar, form_table, divider).",
        ha="center", va="center", fontsize=6.4, color=TEXT_SECONDARY, style="italic",
        linespacing=1.5, transform=ax_legend.transAxes,
    )

    y_annot = y_legend + LEGEND_H
    ax_annot = axes_at(SIDE_MARGIN, y_annot, FIG_W - 2 * SIDE_MARGIN, ANNOT_H)
    ax_annot.axis("off")
    ax_annot.text(
        0.5, 1.0,
        "Modified Layout Similarity is more robust to DOM fragmentation and overlapping elements,\n"
        "producing scores that better reflect visual layout similarity.",
        ha="center", va="top", fontsize=7.6, style="italic", color=TEXT_SECONDARY, linespacing=1.4, transform=ax_annot.transAxes,
    )

    return fig


def main() -> None:
    fig = build_figure()
    out_pdf = REPO_ROOT / "results" / "layout_similarity_examples.pdf"
    out_png = out_pdf.with_suffix(".png")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, facecolor="white", bbox_inches="tight")
    fig.savefig(out_png, dpi=300, facecolor="white", bbox_inches="tight")
    print(f"Saved -> {out_pdf}")
    print(f"Saved -> {out_png}")


if __name__ == "__main__":
    main()
