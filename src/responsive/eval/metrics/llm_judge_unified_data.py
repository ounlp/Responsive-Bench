"""Canonical text for the unified responsive LLM judge (6 dimensions).

Single source for rubric lines embedded in UNIFIED_RESPONSIVE_JUDGE_PROMPT_TEMPLATE.
Wording matches the unified judge prompt (red-border rules).
"""

from __future__ import annotations

# Short titles used in the prompt rubric section and UI labels.
UNIFIED_RUBRIC_DISPLAY_TITLES: dict[str, str] = {
    "fluid_layout_and_container_adaptation": "Fluid Layout & Container Adaptation",
    "flexible_grid_reflow": "Flexible Grid Reflow",
    "responsive_media": "Responsive Media",
    "intentional_content_visibility": "Intentional Content Visibility",
    "proper_alignment_and_spacing": "Proper Alignment and Spacing",
    "sketch_fidelity": "Sketch Fidelity",
}

# Order matches JUDGE_DIMENSION_KEYS in llm_judge_prompts.py
UNIFIED_DIMENSION_ORDER: tuple[str, ...] = (
    "fluid_layout_and_container_adaptation",
    "flexible_grid_reflow",
    "responsive_media",
    "intentional_content_visibility",
    "proper_alignment_and_spacing",
    "sketch_fidelity",
)

# Verbal rubric lines (unified prompt; red-border / sketch-aware). One paragraph per score.
UNIFIED_RUBRIC_GUIDANCE: dict[str, dict[int, str]] = {
    "fluid_layout_and_container_adaptation": {
        1: (
            "The layout is visibly fixed or rigid in multiple areas. At least one section produces a "
            "horizontal scrollbar or has content cut off at the right edge of the red border, or a large "
            "section occupies a fixed width with a conspicuous blank column beside it, across one or more "
            "screenshots."
        ),
        2: (
            "Some sections resize, but two or more areas show obvious rigid behavior — either dead space "
            "flanking them, content clipped at the right edge of the red border, or sections that do not "
            "grow to fill the available viewport width on tablet or mobile."
        ),
        3: (
            "The majority of the page fills each viewport correctly, but one clearly visible section "
            "either overflows past the right edge of the red border, is clipped there, or leaves a "
            "noticeable blank column beside it at one or more screen sizes."
        ),
        4: (
            "Nearly all sections fill their viewport and resize cleanly across all three screenshots. At "
            "most one minor area shows slightly rigid sizing, but it does not produce overflow, clipping, or "
            "a distracting blank column."
        ),
        5: (
            "Every major section fills its viewport width fluidly in all three screenshots. No content "
            "is clipped at the right edge of the red border, and no section leaves a conspicuous blank "
            "column beside it at any screen size."
        ),
    },
    "flexible_grid_reflow": {
        1: (
            "One or more multi-column areas retain their desktop column count on tablet or mobile, "
            "making columns too narrow to read, causing items to overlap horizontally, or squeezing cells "
            "to the point of illegibility. Or the mobile screenshot shows a column arrangement that "
            "contradicts the stacking behavior clearly indicated in the mobile sketch."
        ),
        2: (
            "Some multi-column areas reflow correctly, but at least one prominent grid or card row does "
            "not reduce its column count on smaller viewports, resulting in cells that are clearly cramped "
            "or hard to read. Or the mobile screenshot partially matches the mobile sketch but one "
            "noticeable area stacks differently than intended."
        ),
        3: (
            "Most multi-column areas reflow to fewer columns, but one noticeable case does not — for "
            "example, two equal-width columns that remain side by side on mobile when each has become too "
            "narrow for its content, or when the mobile sketch indicates they should stack."
        ),
        4: (
            "Multi-column layouts reduce their column count logically across all three viewports and all "
            "content stays comfortably readable, with at most one minor inconsistency that does not "
            "materially affect readability or deviate meaningfully from the mobile sketch."
        ),
        5: (
            "Every multi-column area transitions cleanly to fewer columns and then to a single column "
            "as the viewport narrows, in a manner consistent with the mobile sketch. Content order is "
            "preserved, no cell is too narrow to read, and no horizontal overlapping occurs."
        ),
    },
    "responsive_media": {
        1: (
            "Any of the following visible in at least one screenshot is sufficient for a score of 1, "
            "independently: (a) one or more images or videos extend horizontally beyond the right edge of "
            "the red border, regardless of whether the aspect ratio appears correct; (b) one or more images "
            "or videos display a visibly incorrect aspect ratio (stretched or squashed); (c) one or more "
            "images or videos are cropped so severely that the primary subject is obscured; (d) a media "
            "element shown prominently in the sketch is entirely absent from the corresponding screenshot. "
            "A correct aspect ratio does not offset overflow, and the absence of cropping does not offset "
            "distortion."
        ),
        2: (
            "No image or video overflows the red border and no severe distortion or subject-obscuring "
            "crop is present. However, at least one prominent image has a noticeably wrong aspect ratio "
            "falling short of severe distortion, is wider than its immediate container without crossing the "
            "red border, is cropped in a way that hides a meaningful but non-central portion of the subject, "
            "or appears in a significantly different position than indicated in the sketch."
        ),
        3: (
            "The majority of media scales correctly, but one noticeable image or video shows an "
            "obviously incorrect aspect ratio, overflows its container, or is cropped in a way that cuts "
            "off a meaningful part of the subject."
        ),
        4: (
            "Nearly all images and videos scale proportionally and fit inside their containers across "
            "all three viewports. At most one minor cropping or sizing issue exists that does not distort "
            "the subject or break the visual composition, and media placement broadly matches the sketches."
        ),
        5: (
            "Every visible image and video fits cleanly inside its container in all three screenshots. "
            "Aspect ratios are visually correct, no media overflows its container boundary or the red "
            "border, no subject is cropped in a way that obscures meaningful content, and media elements "
            "appear in the positions indicated by the sketches."
        ),
    },
    "intentional_content_visibility": {
        1: (
            "Clearly secondary elements remain fully displayed within the red-bordered viewport on "
            "mobile and occupy screen space comparable to or greater than the primary content, making the "
            "primary content hard to locate or access. Or the mobile screenshot shows elements that the "
            "mobile sketch clearly indicates should be hidden or collapsed, and those elements dominate "
            "the visible viewport."
        ),
        2: (
            "Some secondary content is reduced on smaller viewports, but at least one clearly "
            "non-primary element remains prominently visible within the red-bordered viewport on mobile "
            "or tablet and noticeably competes with the primary content for screen space. Or the mobile "
            "sketch indicates certain elements should be collapsed but they remain partially visible and "
            "take noticeable space."
        ),
        3: (
            "Primary content is accessible on all viewports, but at least one identifiable secondary "
            "element remains fully visible within the red-bordered viewport on mobile or tablet when the "
            "mobile sketch indicates it should be hidden, or when it could reasonably be collapsed without "
            "losing important functionality."
        ),
        4: (
            "Most secondary content is collapsed, hidden, or clearly visually subordinated on mobile "
            "and tablet in a manner consistent with the mobile sketch. The primary content is the dominant "
            "element within the red-bordered viewport. At most one minor secondary element remains that "
            "does not materially compete with the primary content."
        ),
        5: (
            "On all three viewports the primary content is the clear dominant element within the "
            "red-bordered viewport. The mobile screenshot matches the mobile sketch in terms of which "
            "elements are visible and which are hidden or collapsed, and no secondary element occupies "
            "space that meaningfully competes with the primary content."
        ),
    },
    "proper_alignment_and_spacing": {
        1: (
            "Multiple sections show elements that touch or overlap unintentionally, or have dramatically "
            "inconsistent margins, or large unexpected gaps appear between adjacent sections — making the "
            "layout look visually broken in more than one area across the screenshots."
        ),
        2: (
            "Noticeable spacing or alignment problems appear in two or more sections: elements crowd "
            "each other, margin sizes are visibly uneven between similar components, or an unexpected gap "
            "between adjacent sections disrupts the visual rhythm."
        ),
        3: (
            "Most sections appear well-spaced and consistently aligned, but one area shows a clear "
            "problem, such as elements touching at one viewport, an inconsistent left-edge alignment within "
            "a column, or padding that disappears asymmetrically at one screen size."
        ),
        4: (
            "Consistent alignment and balanced spacing appear throughout all three screenshots. At most "
            "one minor inconsistency exists that does not disrupt the overall visual rhythm of the layout."
        ),
        5: (
            "Alignment and spacing are consistent and balanced across all three viewports and across the "
            "full rendered page. No elements touch or overlap unintentionally, no large unexpected gaps "
            "appear between adjacent sections, and padding and margins appear visually even within similar "
            "components."
        ),
    },
    "sketch_fidelity": {
        1: (
            "The desktop or mobile screenshot deviates fundamentally from the corresponding sketch in "
            "structural terms — major sections are missing, the overall column arrangement is entirely "
            "different, the section order is reversed or scrambled, or the primary content region is placed "
            "in a structurally different position. More than one major structural deviation is present "
            "across the desktop and mobile comparisons."
        ),
        2: (
            "The screenshots broadly attempt the intended layout but one major structural element "
            "deviates significantly from the sketch — for example, a sidebar that should appear on the "
            "right appears on the left, a section that should appear above the fold is pushed far below, "
            "or a two-column arrangement on desktop is rendered as a single column when the sketch shows "
            "two."
        ),
        3: (
            "The overall compositional skeleton is recognizable from the sketch, but one noticeable "
            "structural deviation exists — such as a navigation element in a different position, a content "
            "section in a different vertical order, or a major region that is present but sized very "
            "differently than indicated in the sketch."
        ),
        4: (
            "The screenshots closely follow the structural intent of the sketches. Section order, "
            "column arrangement, and the relative prominence of major content regions correspond well to "
            "both the desktop and mobile sketches. At most one minor compositional difference exists that "
            "does not change the overall reading of the layout."
        ),
        5: (
            "The desktop screenshot faithfully reproduces the structural skeleton of the desktop sketch "
            "and the mobile screenshot faithfully reproduces the structural skeleton of the mobile sketch. "
            "Section order, column arrangement, navigation placement, and the relative prominence of all "
            "major content regions correspond to the sketches with no meaningful structural deviations."
        ),
    },
}
