"""
Generates a two-panel figure illustrating the difference between the original
Sketch2Code layout similarity metric and our Geometric Union + Text-Merge approach,
using sample 0045 (Capron VA Sales Tax page) as the reference example.

Output: results/layout_similarity_comparison.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Colours ──────────────────────────────────────────────────────────────────
REF_FILL   = "#2B7FBF"   # blue   – reference text_blocks
GEN_FILL   = "#E07B2A"   # orange – generated text_blocks
INTER_OLD  = "#C0392B"   # red    – pairwise-counted intersection
INTER_NEW  = "#27AE60"   # green  – correct geometric intersection
NAV_COLOR  = "#4A90D9"
PAGE_BG    = "#FFFFFF"
PANEL_BG   = "#F4F5F7"


# ── Helpers ───────────────────────────────────────────────────────────────────
def rect_inter(r1, r2):
    """Axis-aligned rectangle intersection. Boxes: (x, y_bottom, w, h)."""
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    ix = max(x1, x2);  iw = min(x1+w1, x2+w2) - ix
    iy = max(y1, y2);  ih = min(y1+h1, y2+h2) - iy
    return (ix, iy, iw, ih) if iw > 0 and ih > 0 else None


def filled(ax, box, color, alpha, zorder=2, hatch=None, ec=None, lw=0):
    x, y, w, h = box
    ax.add_patch(mpatches.Rectangle(
        (x, y), w, h, facecolor=color, alpha=alpha, linewidth=lw,
        edgecolor=ec or color, hatch=hatch, zorder=zorder
    ))


def outlined(ax, box, ec, lw=1.8, zorder=3):
    x, y, w, h = box
    ax.add_patch(mpatches.Rectangle(
        (x, y), w, h, facecolor="none",
        edgecolor=ec, linewidth=lw, zorder=zorder
    ))


def setup_ax(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_facecolor(PANEL_BG)
    ax.axis("off")
    # page outline
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.04, 0.04), 0.92, 0.91,
        boxstyle="round,pad=0.005",
        facecolor=PAGE_BG, edgecolor="#CCCCCC", linewidth=1.2, zorder=0
    ))
    # nav bar
    ax.add_patch(mpatches.Rectangle(
        (0.04, 0.89), 0.92, 0.06,
        facecolor=NAV_COLOR, edgecolor="none", zorder=1
    ))


# ── Box coordinates (sample 0045, text_block category only) ──────────────────
#
# REFERENCE — the original page wraps most content in one large text_block.
REF_BOX = (0.10, 0.13, 0.80, 0.68)      # main content container

# GENERATED — the LLM page emits one tight box per element (~10 fragments).
GEN_BOXES = [
    (0.10, 0.76, 0.80, 0.08),   # h1 title
    (0.10, 0.64, 0.80, 0.08),   # intro paragraph
    (0.10, 0.57, 0.48, 0.05),   # h2 "Sales Tax Breakdown"
    (0.10, 0.51, 0.23, 0.04),   # table link – Virginia State
    (0.10, 0.46, 0.23, 0.04),   # table link – Southampton County
    (0.10, 0.41, 0.23, 0.04),   # table text – Capron
    (0.10, 0.34, 0.48, 0.05),   # h2 "Capron Details"
    (0.10, 0.27, 0.48, 0.05),   # details paragraph
]

# MERGED — after text-merge preprocessing, adjacent same-column fragments unify.
MERGED_GEN = (0.10, 0.27, 0.80, 0.57)   # single merged region


# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor("#FFFFFF")

gs = fig.add_gridspec(1, 2, left=0.03, right=0.97,
                      top=0.84, bottom=0.18, wspace=0.08)
ax_old = fig.add_subplot(gs[0])
ax_new = fig.add_subplot(gs[1])


# ═══════════════════════════════════════════════════════════════════════════════
# LEFT — original Sketch2Code pairwise method
# ═══════════════════════════════════════════════════════════════════════════════
setup_ax(ax_old)

# Reference box
filled(ax_old, REF_BOX, REF_FILL, alpha=0.18, zorder=2)
outlined(ax_old, REF_BOX, REF_FILL, lw=2.2)

# Generated boxes + their intersection with the reference (red hatching)
for gb in GEN_BOXES:
    filled(ax_old, gb, GEN_FILL, alpha=0.22, zorder=2)
    outlined(ax_old, gb, GEN_FILL, lw=1.3)
    inter = rect_inter(gb, REF_BOX)
    if inter:
        filled(ax_old, inter, INTER_OLD, alpha=0.55, zorder=4,
               hatch="////", ec=INTER_OLD)

# "reference" label — bottom-left corner of the reference box, outside hatching
rx, ry, rw, rh = REF_BOX
ax_old.text(rx + 0.015, ry + 0.018, "reference  text_block",
            fontsize=8.5, color=REF_FILL, va="bottom", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec=REF_FILL, alpha=0.92, lw=0.9))

# Annotation: show the many-to-one pairing
ax_old.annotate(
    f"← {len(GEN_BOXES)} generated boxes,\n   each paired with\n   this one ref box",
    xy=(rx + rw, ry + rh * 0.55),
    xytext=(rx + rw + 0.04, ry + rh * 0.42),
    fontsize=8.5, color=INTER_OLD,
    arrowprops=dict(arrowstyle="->", color=INTER_OLD, lw=1.3),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=INTER_OLD, alpha=0.92),
    va="center"
)

# Panel title — above the axes, via set_title
ax_old.set_title("Original  (Sketch2Code — pairwise sum)",
                 fontsize=12, fontweight="bold", color=INTER_OLD,
                 pad=8, loc="center")

# Score badge below the panel
ax_old.text(0.50, -0.10,
            "IoU (text_block)  =  0.467",
            fontsize=13, ha="center", va="center", color=INTER_OLD,
            fontweight="bold", transform=ax_old.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", fc="#FDECEA", ec=INTER_OLD, lw=1.8))


# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT — our method: text-merge → geometric union IoU
# ═══════════════════════════════════════════════════════════════════════════════
setup_ax(ax_new)

# Reference box
filled(ax_new, REF_BOX, REF_FILL, alpha=0.18, zorder=2)
outlined(ax_new, REF_BOX, REF_FILL, lw=2.2)

# Merged generated region
filled(ax_new, MERGED_GEN, GEN_FILL, alpha=0.22, zorder=2)
outlined(ax_new, MERGED_GEN, GEN_FILL, lw=2.5)

# Single green intersection
inter = rect_inter(MERGED_GEN, REF_BOX)
if inter:
    filled(ax_new, inter, INTER_NEW, alpha=0.42, zorder=4)
    outlined(ax_new, inter, INTER_NEW, lw=1.0, zorder=5)

# "reference" label
rx, ry, rw, rh = REF_BOX
ax_new.text(rx + 0.015, ry + 0.018, "reference  text_block",
            fontsize=8.5, color=REF_FILL, va="bottom", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec=REF_FILL, alpha=0.92, lw=0.9))

# "merged" annotation — outside the right edge
mx, my, mw, mh = MERGED_GEN
ax_new.annotate(
    "8 fragmented boxes\nmerged → 1 union region",
    xy=(mx + mw, my + mh * 0.75),
    xytext=(mx + mw + 0.04, my + mh * 0.60),
    fontsize=8.5, color=GEN_FILL,
    arrowprops=dict(arrowstyle="->", color=GEN_FILL, lw=1.3),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GEN_FILL, alpha=0.92),
    va="center"
)

# "counted once" badge inside the green intersection
ix, iy, iw, ih = inter
ax_new.text(ix + iw * 0.5, iy + ih * 0.5,
            "✓  intersection\n    counted once",
            fontsize=10, ha="center", va="center", color=INTER_NEW,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec=INTER_NEW, alpha=0.94, lw=1.2))

# Panel title — above the axes, via set_title
ax_new.set_title("Ours  (Text-Merge  →  Geometric Union IoU)",
                 fontsize=12, fontweight="bold", color=INTER_NEW,
                 pad=8, loc="center")

# Score badge below the panel
ax_new.text(0.50, -0.10,
            "IoU (text_block)  =  0.556",
            fontsize=13, ha="center", va="center", color=INTER_NEW,
            fontweight="bold", transform=ax_new.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", fc="#E9F7EF", ec=INTER_NEW, lw=1.8))


# ── Shared legend ─────────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor=REF_FILL,  alpha=0.5,
                   label="Reference  text_block"),
    mpatches.Patch(facecolor=GEN_FILL,  alpha=0.5,
                   label="Generated  text_block(s)"),
    mpatches.Patch(facecolor=INTER_OLD, alpha=0.65, hatch="////",
                   label="Pairwise intersection (old — each pair counted)"),
    mpatches.Patch(facecolor=INTER_NEW, alpha=0.5,
                   label="Geometric intersection (ours — counted once)"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=4,
           fontsize=9.5, framealpha=0.95,
           bbox_to_anchor=(0.50, 0.02), edgecolor="#AAAAAA")

# ── Super-title ───────────────────────────────────────────────────────────────
fig.suptitle(
    "Layout Similarity — Sample 0045   (text_block category, desktop 1280 px)\n"
    "Reference: 1 large container box   ·   Generated: 8 fragmented element boxes",
    fontsize=12.5, y=0.96, color="#222222"
)

# ── Save ──────────────────────────────────────────────────────────────────────
out = Path(__file__).resolve().parents[1] / "results" / "layout_similarity_comparison.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out}")
plt.show()
