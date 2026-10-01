"""
Generates a clear before/after figure explaining the text-block merge
preprocessing step used in the Geometric Union layout similarity metric.

Shows how a page that renders as one continuous text region gets fragmented
into many individual DOM boxes, and how our merge step corrects for this.

Output: results/text_merge_explanation.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
from pathlib import Path

# ── Colours ──────────────────────────────────────────────────────────────────
BOX_FILL    = "#E8F0FE"    # light blue — individual element boxes
BOX_EDGE    = "#5C85D6"    # blue edge
MERGED_FILL = "#D4EDDA"    # light green — merged region
MERGED_EDGE = "#27AE60"    # green edge
GAP_COLOR   = "#E74C3C"    # red — gap annotation
TAG_COLORS  = {
    "h1": ("#1A3A6B", "#FFFFFF"),   # dark navy bg, white text
    "h2": ("#2471A3", "#FFFFFF"),   # medium blue
    "p":  ("#5D6D7E", "#FFFFFF"),   # gray
    "a":  ("#117A65", "#FFFFFF"),   # teal/link
}
PANEL_BG    = "#F7F9FC"
PAGE_BG     = "#FFFFFF"
NAV_BG      = "#4A90D9"

# ── Element data (sample 0045 — Capron VA Sales Tax page) ─────────────────────
# (tag, short label, box height in axes units)
ELEMENTS = [
    ("h1", "Capron, VA Sales Tax Rate",             0.115),
    ("p",  "The current total local sales tax...",  0.095),
    ("h2", "Sales Tax Breakdown",                   0.075),
    ("a",  "Virginia State  ↗",                     0.055),
    ("a",  "Southampton County  ↗",                 0.055),
    ("a",  "Capron",                                0.055),
    ("h2", "Capron Details",                        0.075),
    ("p",  "Capron, VA is in Southampton County.",  0.080),
]
GAP = 0.016            # gap between consecutive boxes (axes units)
BOX_X, BOX_W = 0.10, 0.80   # left edge and width of all boxes


def compute_boxes(elements, gap, y_start=0.14):
    """Return list of (tag, label, x, y_bottom, w, h) from bottom up."""
    boxes = []
    y = y_start
    for tag, label, h in elements[::-1]:   # bottom → top
        boxes.append((tag, label, BOX_X, y, BOX_W, h))
        y += h + gap
    return boxes[::-1]   # restore top-down order


BOXES = compute_boxes(ELEMENTS, GAP)
MERGED_Y     = BOXES[-1][3]                      # bottom of bottom box
MERGED_TOP   = BOXES[0][3] + BOXES[0][5]         # top of top box
MERGED_H     = MERGED_TOP - MERGED_Y


# ── Helpers ───────────────────────────────────────────────────────────────────
def page_bg(ax, nav=True):
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.04, 0.04), 0.92, 0.92,
        boxstyle="round,pad=0.005",
        facecolor=PAGE_BG, edgecolor="#CCCCCC", linewidth=1.0, zorder=0
    ))
    if nav:
        ax.add_patch(mpatches.Rectangle(
            (0.04, 0.90), 0.92, 0.06,
            facecolor=NAV_BG, edgecolor="none", zorder=1
        ))


def tag_badge(ax, tag, x, y, h):
    """Draw the element-type badge on the left side of a box."""
    bg, fg = TAG_COLORS.get(tag, ("#888", "#FFF"))
    badge_w = 0.055
    ax.add_patch(mpatches.Rectangle(
        (x, y), badge_w, h,
        facecolor=bg, edgecolor="none", zorder=4, clip_on=False
    ))
    ax.text(x + badge_w / 2, y + h / 2, f"<{tag}>",
            fontsize=7.5, ha="center", va="center",
            color=fg, fontweight="bold", zorder=5,
            fontfamily="monospace")


def draw_element_box(ax, tag, label, x, y, w, h, alpha=1.0):
    # main box fill
    ax.add_patch(mpatches.Rectangle(
        (x, y), w, h,
        facecolor=BOX_FILL, edgecolor=BOX_EDGE,
        linewidth=1.3, linestyle="--", zorder=3, alpha=alpha
    ))
    # left tag badge
    tag_badge(ax, tag, x, y, h)
    # content label
    ax.text(x + 0.065, y + h / 2, label,
            fontsize=8.5, va="center", color="#2C3E50",
            zorder=5, style="normal",
            clip_on=True)


def setup_ax(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_facecolor(PANEL_BG)
    ax.axis("off")
    page_bg(ax)


# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor("#FFFFFF")

# Three columns: before | arrow | after
gs = fig.add_gridspec(1, 3,
                      width_ratios=[5, 1, 5],
                      left=0.03, right=0.97,
                      top=0.87, bottom=0.10,
                      wspace=0.0)

ax_before = fig.add_subplot(gs[0])
ax_arrow  = fig.add_subplot(gs[1])
ax_after  = fig.add_subplot(gs[2])


# ═══════════════════════════════════════════════════════════════════════════════
# LEFT — before merge: individual DOM element boxes
# ═══════════════════════════════════════════════════════════════════════════════
setup_ax(ax_before)
ax_before.set_title("Before merge\n8 individual text_block boxes",
                    fontsize=12, fontweight="bold", color=BOX_EDGE,
                    pad=8, loc="center")

for tag, label, x, y, w, h in BOXES:
    draw_element_box(ax_before, tag, label, x, y, w, h)

# Annotate the gap between h2 and first <a> — keep entirely inside left panel
b1 = BOXES[2]   # h2 "Sales Tax Breakdown"
b2 = BOXES[3]   # a  "Virginia State"
gap_y_bottom = b2[3] + b2[5]   # top of Virginia State box
gap_y_top    = b1[3]            # bottom of Sales Tax Breakdown box
gap_mid      = (gap_y_bottom + gap_y_top) / 2
gap_x        = BOX_X + BOX_W * 0.60   # 60% across — well inside the panel

# Vertical double-headed arrow in the gap
ax_before.annotate("",
    xy=(gap_x, gap_y_top), xytext=(gap_x, gap_y_bottom),
    arrowprops=dict(arrowstyle="<->", color=GAP_COLOR, lw=1.5)
)
ax_before.text(gap_x + 0.025, gap_mid,
               f"gap ≈ {int(GAP * 2000)} px  (≤ 40 px ✓)",
               fontsize=8, color=GAP_COLOR, va="center", fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.25", fc="white",
                         ec=GAP_COLOR, alpha=0.9, lw=0.8))

# Callout: count
ax_before.text(0.50, 0.06, "8 separate boxes → metric sees 8 disconnected fragments",
               fontsize=9, ha="center", va="center", color=BOX_EDGE,
               style="italic", transform=ax_before.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", fc="#EBF5FB",
                         ec=BOX_EDGE, alpha=0.9, lw=0.8))


# ═══════════════════════════════════════════════════════════════════════════════
# MIDDLE — merge arrow
# ═══════════════════════════════════════════════════════════════════════════════
ax_arrow.set_xlim(0, 1)
ax_arrow.set_ylim(0, 1)
ax_arrow.axis("off")
ax_arrow.set_facecolor("#FFFFFF")

# Fat arrow — at true vertical centre
ax_arrow.annotate("",
    xy=(0.90, 0.50), xytext=(0.10, 0.50),
    arrowprops=dict(arrowstyle="-|>",
                    color=MERGED_EDGE,
                    lw=6.0,
                    mutation_scale=45)
)
# Label clearly above the arrow
ax_arrow.text(0.50, 0.72, "text-block\nmerge",
              fontsize=11, ha="center", va="bottom",
              color=MERGED_EDGE, fontweight="bold", linespacing=1.5)
# Criterion clearly below the arrow
ax_arrow.text(0.50, 0.28,
              "same column\n+  gap ≤ 40 px",
              fontsize=9, ha="center", va="top",
              color="#555555", style="italic", linespacing=1.6)


# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT — after merge: single unified region
# ═══════════════════════════════════════════════════════════════════════════════
setup_ax(ax_after)
ax_after.set_title("After merge\n1 unified text_block region",
                   fontsize=12, fontweight="bold", color=MERGED_EDGE,
                   pad=8, loc="center")

# Draw ghost outlines of the original boxes (faint, to show what was merged)
for tag, label, x, y, w, h in BOXES:
    ax_after.add_patch(mpatches.Rectangle(
        (x, y), w, h,
        facecolor="none", edgecolor="#AAAAAA",
        linewidth=0.7, linestyle=":", zorder=2, alpha=0.5
    ))

# Draw the merged region on top
ax_after.add_patch(mpatches.Rectangle(
    (BOX_X, MERGED_Y), BOX_W, MERGED_H,
    facecolor=MERGED_FILL, edgecolor=MERGED_EDGE,
    linewidth=2.5, zorder=3, alpha=0.85
))

# Brace-style label on the RIGHT side (avoids collision with middle arrow)
brace_x = BOX_X + BOX_W + 0.06
ax_after.annotate("",
    xy=(brace_x, MERGED_Y), xytext=(brace_x, MERGED_TOP),
    arrowprops=dict(arrowstyle="<->", color=MERGED_EDGE, lw=1.8)
)
ax_after.text(brace_x + 0.015, (MERGED_Y + MERGED_TOP) / 2,
              "1 merged\nregion",
              fontsize=9, ha="left", va="center",
              color=MERGED_EDGE, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.25", fc="white",
                        ec=MERGED_EDGE, alpha=0.9, lw=0.8))

# Centre label inside the merged box
ax_after.text(BOX_X + BOX_W / 2, MERGED_Y + MERGED_H / 2,
              "✓  All 8 fragments collapsed\n    into one geometric region",
              fontsize=10.5, ha="center", va="center",
              color="#1A5E34", fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.4", fc="white",
                        ec=MERGED_EDGE, alpha=0.95, lw=1.2))

# Callout: count
ax_after.text(0.50, 0.06, "1 region → metric measures true visual coverage",
              fontsize=9, ha="center", va="center", color=MERGED_EDGE,
              style="italic", transform=ax_after.transAxes,
              bbox=dict(boxstyle="round,pad=0.3", fc="#EAFAF1",
                        ec=MERGED_EDGE, alpha=0.9, lw=0.8))


# ── Super-title ───────────────────────────────────────────────────────────────
fig.suptitle(
    "Text-Block Merge  —  Sample 0045  (text_block category)\n"
    "The LLM generates one box per HTML element. Visually, it's one continuous text region. "
    "Our preprocessing merges adjacent boxes so the metric reflects that.",
    fontsize=11.5, y=0.97, color="#222222", linespacing=1.5
)

# ── Save ──────────────────────────────────────────────────────────────────────
out = Path(__file__).resolve().parents[1] / "results" / "text_merge_explanation.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out}")
plt.show()
