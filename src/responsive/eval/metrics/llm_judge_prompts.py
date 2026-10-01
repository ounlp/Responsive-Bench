"""Registry of LLM judge prompts for responsive web evaluation.

This module defines a single, unified judge prompt that scores a generated webpage
using three screenshots (mobile, tablet, desktop).

The judge returns per-dimension 1-5 Likert scores plus an overall score.
"""

from __future__ import annotations

import textwrap

from .llm_judge_unified_data import (
    UNIFIED_DIMENSION_ORDER,
    UNIFIED_RUBRIC_DISPLAY_TITLES,
    UNIFIED_RUBRIC_GUIDANCE,
)

# Schema identifier for response parsing
# - "score_1_5": {"score": 1-5, "rationale": "..."}
# - "multi_score_1_5": unified multi-dimension JSON
SCHEMA_SCORE_1_5 = "score_1_5"
SCHEMA_MULTI_SCORE_1_5 = "multi_score_1_5"


# -----------------------------------------------------------------------------
# Dimensions (6)
# -----------------------------------------------------------------------------

JUDGE_DIMENSION_KEYS = [
    "fluid_layout_and_container_adaptation",
    "flexible_grid_reflow",
    "responsive_media",
    "intentional_content_visibility",
    "proper_alignment_and_spacing",
    "sketch_fidelity",
]


# -----------------------------------------------------------------------------
# Scoring tables (separated from the prompt content for easier reuse)
# -----------------------------------------------------------------------------

LIKERT_1_5_LABELS = {
    1: "Strongly disagree",
    2: "Disagree",
    3: "Neutral",
    4: "Agree",
    5: "Strongly agree",
}


# Per-dimension rubric: evidence-only text (Likert labels are in LIKERT_1_5_TABLE_MD).
# Keep these descriptions grounded in what can be seen in screenshots.
DIMENSION_RUBRIC_1_5: dict[str, dict[int, str]] = {
    "fluid_layout_and_container_adaptation": {
        1: (
            "The layout is visibly fixed or rigid in multiple areas. At least one "
            "section produces a horizontal scrollbar or has content cut off at the viewport edge, or a "
            "large section occupies a fixed width with a conspicuous blank column beside it, across one "
            "or more screenshots."
        ),
        2: (
            "Some sections resize, but two or more areas show obvious rigid behavior, either "
            "dead space flanking them, content clipped at the edge, or sections that do not grow to fill "
            "the available viewport width on tablet or mobile."
        ),
        3: (
            "The majority of the page fills each viewport correctly, but one clearly visible "
            "section either overflows its container, is clipped at the viewport edge, or leaves a "
            "noticeable blank column beside it at one or more screen sizes."
        ),
        4: (
            "Nearly all sections fill their viewport and resize cleanly across all three "
            "screenshots. At most one minor area shows slightly rigid sizing, but it does not produce "
            "overflow, clipping, or a distracting blank column."
        ),
        5: (
            "Every major section fills its viewport width fluidly in all three "
            "screenshots. No horizontal scrollbar appears, no content is clipped at the viewport edge, "
            "and no section leaves a conspicuous blank column beside it at any screen size."
        ),
    },
    "flexible_grid_reflow": {
        1: (
            "One or more multi-column areas retain their desktop column count on "
            "tablet or mobile, making columns too narrow to read, causing items to overlap horizontally, "
            "or squeezing cells to the point of illegibility."
        ),
        2: (
            "Some multi-column areas reflow correctly, but at least one prominent grid or card "
            "row does not reduce its column count on smaller viewports, resulting in cells that are "
            "clearly cramped or hard to read."
        ),
        3: (
            "Most multi-column areas reflow to fewer columns, but one noticeable case does not, "
            "for example, two equal-width columns that remain side by side on mobile when each has become "
            "too narrow for its content."
        ),
        4: (
            "Multi-column layouts reduce their column count logically across all three viewports "
            "and all content stays comfortably readable. At most one minor inconsistency exists that does "
            "not materially affect readability."
        ),
        5: (
            "Every multi-column area transitions cleanly to fewer columns and then to a "
            "single column as the viewport narrows. Content order is preserved, no cell is too narrow to "
            "read, and no horizontal overlapping occurs in any screenshot."
        ),
    },
    "responsive_media": {
        1: (
            "One or more images or videos overflow their container boundary, display a "
            "visibly incorrect aspect ratio (stretched or squashed), or are cropped so severely that the "
            "primary subject is obscured, across at least one screenshot."
        ),
        2: (
            "Some media scales correctly, but at least one prominent image or video has a "
            "clearly wrong aspect ratio, is wider than its container, or is cropped in a way that hides "
            "meaningful content on tablet or mobile."
        ),
        3: (
            "The majority of media scales correctly, but one noticeable image or video shows an "
            "obviously incorrect aspect ratio, overflows its container, or is cropped in a way that cuts "
            "off a meaningful part of the subject."
        ),
        4: (
            "Nearly all images and videos scale proportionally and fit inside their containers "
            "across all three viewports. At most one minor cropping or sizing issue exists that does not "
            "distort the subject or break the visual composition."
        ),
        5: (
            "Every visible image and video fits cleanly inside its container in all three "
            "screenshots. Aspect ratios are visually correct, no media overflows its container boundary, "
            "and no subject is cropped in a way that obscures meaningful content."
        ),
    },
    "intentional_content_visibility": {
        1: (
            "Clearly secondary elements (sidebars, auxiliary navigation panels, "
            "decorative sections) remain fully displayed on mobile and occupy screen space comparable to "
            "or greater than the primary content, making the primary content hard to locate or access."
        ),
        2: (
            "Some secondary content is reduced on smaller viewports, but at least one clearly "
            "non-primary element remains prominently visible on mobile or tablet and noticeably competes "
            "with the primary content for screen space."
        ),
        3: (
            "Primary content is accessible on all viewports, but at least one identifiable "
            "secondary element remains fully visible on mobile or tablet when it could reasonably be "
            "collapsed or hidden without losing important functionality."
        ),
        4: (
            "Most secondary content is collapsed, hidden, or clearly visually subordinated on mobile "
            "and tablet. The primary content is the dominant element. At most one minor secondary element "
            "remains that does not materially compete with the primary content."
        ),
        5: (
            "On all three viewports the primary content is the clear dominant element. "
            "Secondary and auxiliary content is collapsed, hidden, or visually reduced on mobile and "
            "tablet, and no secondary element occupies space that meaningfully competes with the primary "
            "content."
        ),
    },
    "proper_alignment_and_spacing": {
        1: (
            "Multiple sections show elements that touch or overlap unintentionally, or "
            "have dramatically inconsistent margins, or large unexpected gaps appear between adjacent "
            "sections — making the layout look visually broken in more than one area across the "
            "screenshots."
        ),
        2: (
            "Noticeable spacing or alignment problems appear in two or more sections: elements "
            "crowd each other, margin sizes are visibly uneven between similar components, or an "
            "unexpected gap between adjacent sections disrupts the visual rhythm."
        ),
        3: (
            "Most sections appear well-spaced and consistently aligned, but one area shows a "
            "clear problem, such as elements touching at one viewport, an inconsistent left-edge "
            "alignment within a column, or padding that disappears asymmetrically at one screen size."
        ),
        4: (
            "Consistent alignment and balanced spacing appear throughout all three screenshots. At "
            "most one minor inconsistency exists, such as a slightly uneven margin between two items, that "
            "does not disrupt the overall visual rhythm of the layout."
        ),
        5: (
            "Alignment and spacing are consistent and balanced across all three viewports. "
            "No elements touch or overlap unintentionally, no large unexpected gaps appear between "
            "adjacent sections, and padding and margins appear visually even within similar components."
        ),
    },
}


# Long-form dimension text (what to evaluate).
_DIM_FLUID_LAYOUT = (
    "The page layout and its major sections (headers, content areas, sidebars, footers) should resize to "
    "fill the available viewport width at each screen size. Containers must not clip their internal "
    "content, produce horizontal scrollbars, or leave conspicuous blank columns beside fixed-width "
    "blocks. Judge this by looking at the overall page structure across all three screenshots and "
    "checking whether sections fill the viewport cleanly, whether any content is cut off at the viewport "
    "edge, and whether any section remains rigidly fixed in a way that leaves dead space beside it or "
    "forces horizontal scrolling."
)

_DIM_FLEXIBLE_GRID = (
    "Multi-column layouts on the desktop screenshot (card grids, feature rows, article lists, "
    "sidebar-plus-content layouts) should reduce their column count on smaller viewports so that content "
    "stacks into fewer columns or a single column rather than remaining side-by-side and becoming too "
    "narrow to read. Judge this by identifying every area that displays multiple items in a row on "
    "desktop, then checking in the tablet and mobile screenshots whether those columns have reduced and "
    "stacked logically, whether content order is preserved, and whether every cell remains wide enough "
    "to be comfortably readable."
)

_DIM_RESPONSIVE_MEDIA = (
    "Images and videos must scale proportionally within their containers across all three viewports, "
    "without overflowing the container boundary, showing a visibly incorrect aspect ratio (stretched or "
    "squashed), or being cropped so severely that the primary subject of the image is obscured. Judge "
    "this by inspecting every visible image and video in all three screenshots and checking whether the "
    "media fits inside its container, whether the aspect ratio looks visually correct, and whether any "
    "meaningful portion of the subject is cut off by the container edge."
)

_DIM_INTENTIONAL_VISIBILITY = (
    "Secondary and auxiliary content (sidebars, supplementary navigation panels, decorative banners, "
    "non-essential widgets, and similar supporting elements) should be hidden, collapsed, or visually "
    "subordinated on smaller viewports so that primary content remains the dominant visible element. "
    "Primary content is whatever the page exists to communicate: the main article body, hero message, "
    "primary call to action, or product detail. Judge this by identifying what the primary content is, "
    "then checking whether secondary elements on mobile and tablet are taking up screen space that "
    "competes with or buries that primary content."
)

_DIM_ALIGNMENT_SPACING = (
    "Elements should maintain consistent horizontal alignment and balanced whitespace between them "
    "across all three viewports, without elements touching or overlapping unintentionally, and without "
    "large unexpected gaps appearing between adjacent sections. Judge this by scanning each screenshot "
    "for elements that share a visual column or row and checking whether their edges align, whether "
    "padding and margins appear visually even within similar components, and whether any two adjacent "
    "sections are either crowded together or separated by a gap that is clearly inconsistent with the "
    "surrounding rhythm of the layout."
)


# A concise markdown table used inside the prompt (kept here as a separate artifact).
LIKERT_1_5_TABLE_MD = """| Score | Label |
|---:|---|
| 1 | Strongly disagree |
| 2 | Disagree |
| 3 | Neutral |
| 4 | Agree |
| 5 | Strongly agree |"""



# -----------------------------------------------------------------------------
# Unified prompt
# -----------------------------------------------------------------------------


def _format_unified_rubric_guidance_markdown() -> str:
    """Build the 'Rubric guidance by dimension' block (markdown-style bullets)."""
    chunks: list[str] = []
    for key in UNIFIED_DIMENSION_ORDER:
        title = UNIFIED_RUBRIC_DISPLAY_TITLES[key]
        chunks.append(f"- {title}")
        for n in range(1, 6):
            body = UNIFIED_RUBRIC_GUIDANCE[key][n]
            wrapped = textwrap.fill(
                body,
                width=85,
                initial_indent=f"  - {n}: ",
                subsequent_indent="    ",
            )
            chunks.append(wrapped)
        chunks.append("")
    return "\n".join(chunks).rstrip() + "\n"


def build_unified_responsive_judge_prompt() -> str:
    """Assemble the full unified judge prompt from static framing + formatted rubric."""
    return (
        UNIFIED_PROMPT_LEADING
        + _format_unified_rubric_guidance_markdown()
        + UNIFIED_PROMPT_TRAILING
    )


# Everything before the rubric bullet list (through the rubric section header).
UNIFIED_PROMPT_LEADING = """You are an expert responsive web design evaluator.

Task:
You will be given the following inputs:
- A desktop sketch (wireframe or hand-drawn layout) showing the intended design at desktop width
- A mobile sketch (wireframe or hand-drawn layout) showing the intended design at mobile width
- Three screenshots of the same generated webpage at three viewport sizes:
  - Mobile (375px width)
  - Tablet (768px width)
  - Desktop (1280px width)

Each screenshot shows the full rendered height and width of the page, but the region that is
visible to a user without scrolling is marked by a semi-transparent red border. Treat everything
inside the red border as the visible viewport. Content that falls outside the red border is
below the fold or to the right of the viewport edge and requires scrolling to reach. When
evaluating clipping, overflow, or truncation, assess whether content is cut off at the red
border boundary — not at the edge of the screenshot image itself. Content that extends beyond
the red border horizontally indicates a layout failure; content that extends below the red
border vertically is normal scrollable content and should not be penalized.

How to use the sketches:
The sketches show the designer's intended layout at desktop and mobile widths. Use them as
reference for evaluating intent in three dimensions: Flexible Grid Reflow, Intentional Content
Visibility, and Responsive Media. When the screenshot behavior is ambiguous — for example, it
is unclear whether a sidebar should be visible on mobile — consult the mobile sketch to resolve
the ambiguity. If the sketch clearly shows an element hidden on mobile but the screenshot shows
it visible, that counts as a failure. If the sketch is unclear or does not address a specific
element, rely on visual evidence in the screenshots alone. The sketches clarify intent; they
do not replace visual evidence. No sketch is provided for tablet — tablet behavior should be
inferred as an intermediate state between the desktop and mobile sketches.

Evaluate responsiveness across the 6 dimensions below. Base your judgments on visible evidence
in the screenshots, using the sketches as intent references where specified.

---

Dimensions:

1) Fluid Layout & Container Adaptation: The page layout and its major sections (headers, content
areas, sidebars, footers) should resize to fill the available viewport width at each screen size.
Containers must not clip their internal content, produce horizontal scrollbars, or leave
conspicuous blank columns beside fixed-width blocks. Judge this by looking at the overall page
structure across all three screenshots and checking whether sections fill the red-bordered
viewport cleanly, whether any content is cut off at the right edge of the red border, and whether
any section remains rigidly fixed in a way that leaves dead space beside it or forces horizontal
scrolling. When text, a heading, or a list item ends abruptly mid-word or mid-sentence at the
right edge of the red border without punctuation or a natural line break, treat this as container
overflow or viewport clipping — not as screenshot cropping.

2) Flexible Grid Reflow: Multi-column layouts on the desktop screenshot (card grids, feature rows,
article lists, sidebar-plus-content layouts) should reduce their column count on smaller viewports
so that content stacks into fewer columns or a single column rather than remaining side-by-side
and becoming too narrow to read. Judge this by identifying every area that displays multiple items
in a row on desktop, then checking in the tablet and mobile screenshots whether those columns have
reduced and stacked logically, whether content order is preserved, and whether every cell remains
wide enough to be comfortably readable. Use the desktop sketch to confirm which areas are intended
as multi-column, and use the mobile sketch to confirm how those areas are intended to stack on
mobile. If the mobile screenshot does not match the stacking behavior shown in the mobile sketch,
treat this as a reflow failure even if the stacking is otherwise plausible. If no multi-column
layout is visible in the desktop screenshot, first confirm that the desktop screenshot does not
contain any side-by-side content areas, card rows, or sidebar-plus-content arrangements within
the red border. Only if genuinely no such areas exist, assign a score of 5 and note in the
rationale that no multi-column layout was present to evaluate.

3) Responsive Media: Images and videos must scale proportionally within their containers across
all three viewports, without overflowing the container boundary, showing a visibly incorrect
aspect ratio (stretched or squashed), or being cropped so severely that the primary subject of
the image is obscured. Judge this by inspecting every visible image and video in all three
screenshots — including media below the red border — and checking whether the media fits inside
its container and within the red border horizontally, whether the aspect ratio looks visually
correct, and whether any meaningful portion of the subject is cut off by the container edge.
Use the desktop and mobile sketches to confirm where media elements are intended to appear and
at what relative size. If the sketch shows a media element prominently placed but the screenshot
omits it entirely or places it in a significantly different position, note this as a deviation.
An image or video that maintains a correct aspect ratio but still extends horizontally beyond
the right edge of the red border is a responsive media failure. Correct proportions do not
offset overflow; both conditions must be satisfied independently. If no images or videos are
visible anywhere in any screenshot, assign a score of 5 and note that no media was present
to evaluate.

4) Intentional Content Visibility: Secondary and auxiliary content (sidebars, supplementary
navigation panels, decorative banners, non-essential widgets, and similar supporting elements)
should be hidden, collapsed, or visually subordinated on smaller viewports so that primary content
remains the dominant visible element. Primary content is whatever the page exists to communicate:
the main article body, hero message, primary call to action, or product detail. Judge this by
identifying what the primary content is on desktop, then checking whether secondary elements on
mobile and tablet are taking up space within the red-bordered viewport that competes with or
buries that primary content. Use the mobile sketch as the authoritative reference for which
elements should be visible, hidden, or collapsed on mobile. If the mobile sketch shows an element
absent or collapsed but the mobile screenshot shows it fully visible and competing with primary
content, this is a score 1 or 2 failure regardless of whether it might seem reasonable in
isolation. If the mobile sketch is ambiguous about a specific element, fall back to visual
judgment. Pay particular attention to what occupies the viewport on first load — the area inside
the red border — since that is what the user sees before any scrolling.

5) Proper Alignment and Spacing: Elements should maintain consistent horizontal alignment and
balanced whitespace between them across all three viewports, without elements touching or
overlapping unintentionally, and without large unexpected gaps appearing between adjacent
sections. Judge this by scanning each screenshot — including content below the red border —
for elements that share a visual column or row and checking whether their edges align, whether
padding and margins appear visually even within similar components, and whether any two adjacent
sections are either crowded together or separated by a gap that is clearly inconsistent with
the surrounding rhythm of the layout. The sketches are not used for this dimension; evaluate
purely from visual evidence in the screenshots.

6) Sketch Fidelity: The generated screenshots should structurally correspond to the desktop and
mobile sketches in terms of layout composition, section order, column arrangement, navigation
placement, and the relative prominence of major content regions. This dimension scores how
closely the implementation reproduces the structural intent of the sketches — not pixel
precision, but whether the overall compositional skeleton is preserved. Judge this by comparing
the desktop screenshot against the desktop sketch and the mobile screenshot against the mobile
sketch. For each pair, check: (a) whether the same major sections are present and in the same
vertical order; (b) whether the column count and arrangement of major regions matches; (c)
whether navigation elements appear in the intended position; (d) whether the relative size and
prominence of major content regions corresponds to the sketch. The tablet screenshot is not
directly compared to a sketch. Do not penalize reasonable visual elaborations (color, typography,
imagery) that are not specified in the sketch — only structural and compositional deviations
count as failures.

---

Scoring (5-point Likert scale):
| Score | Label |
|---:|---|
| 1 | Strongly disagree |
| 2 | Disagree |
| 3 | Neutral |
| 4 | Agree |
| 5 | Strongly agree |

---

Rubric guidance by dimension (use these to pick 1-5):

"""


UNIFIED_PROMPT_TRAILING = """
---

Output requirements:
- Respond with exactly one JSON object.
- All scores must be integers from 1 to 5.
- Each rationale must be one paragraph grounded in specific screenshot evidence (what you see on
  mobile vs tablet vs desktop), referencing the red border where relevant to justify clipping or
  visibility judgments, and referencing the sketches where relevant to justify sketch-based
  comparisons.
- Do not mention policies, hidden reasoning, or speculation about the underlying HTML/CSS.

Output format (strict JSON):
{
  "scores": {
    "fluid_layout_and_container_adaptation": {"score": <1-5>, "rationale": "<one paragraph>"},
    "flexible_grid_reflow": {"score": <1-5>, "rationale": "<one paragraph>"},
    "responsive_media": {"score": <1-5>, "rationale": "<one paragraph>"},
    "intentional_content_visibility": {"score": <1-5>, "rationale": "<one paragraph>"},
    "proper_alignment_and_spacing": {"score": <1-5>, "rationale": "<one paragraph>"},
    "sketch_fidelity": {"score": <1-5>, "rationale": "<one paragraph>"}
  },
  "overall_score": <arithmetic mean of the 6 final scores, rounded to nearest integer>,
  "overall_rationale": "<one paragraph summarizing the main strengths and weaknesses across\n  dimensions>"
}

Begin evaluation now.
"""


UNIFIED_RESPONSIVE_JUDGE_PROMPT_TEMPLATE = build_unified_responsive_judge_prompt()



# Backwards-compatible registry structure: one unified dimension/prompt.
JUDGE_DIMENSIONS = [
    {
        "key": "responsive_overall",
        "name": "Responsive Overall (6 dimensions + sketches)",
        "output_schema": SCHEMA_MULTI_SCORE_1_5,
        "prompt_template": UNIFIED_RESPONSIVE_JUDGE_PROMPT_TEMPLATE,
    }
]
