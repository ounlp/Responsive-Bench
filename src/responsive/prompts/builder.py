"""
Prompt builder for responsive webpage generation.

Builds prompts from templates and sample data, ensuring consistent image ordering.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..dataset.loader import Sample

# Unified image ordering: mobile first (Image 1), desktop second (Image 2).
# All prompt templates and builder methods use this order.
IMAGE_ORDER = ["mobile", "desktop"]


@dataclass
class PromptPackage:
    """A complete prompt package with text, images, and metadata."""

    text: str
    images: list[Path]
    meta: dict


class PromptBuilder:
    """
    Builds prompts from templates and sample data.

    Image ordering is always mobile first (Image 1), desktop second (Image 2).
    All .j2 templates reflect this order.
    """

    def __init__(self, templates_dir: Optional[Path] = None):
        """
        Initialize prompt builder.

        Args:
            templates_dir: Directory containing prompt templates. Defaults to
                src/responsive/prompts/templates/ relative to this file.
        """
        if templates_dir is None:
            # Default to templates/ directory relative to this file
            templates_dir = Path(__file__).parent / "templates"
        self.templates_dir = Path(templates_dir).resolve()

        if not self.templates_dir.exists():
            raise FileNotFoundError(f"Templates directory not found: {self.templates_dir}")

    def build(self, sample: Sample, template_id: str = "base_v1") -> PromptPackage:
        """
        Build a prompt package from a sample and template.

        Args:
            sample: Sample object with sketches and metadata
            template_id: Template identifier (e.g., "base_v1")

        Returns:
            PromptPackage with text, ordered images, and metadata

        Raises:
            FileNotFoundError: If template file does not exist
        """
        # Load template
        template_path = self.templates_dir / f"{template_id}.j2"
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        template_text = template_path.read_text(encoding="utf-8")

        # Build ordered image list: mobile (Image 1), desktop (Image 2)
        images = [sample.sketches[device] for device in IMAGE_ORDER]

        # Build metadata
        meta = {
            "sample_id": sample.id,
            "template_id": template_id,
            "image_ordering": IMAGE_ORDER,
            "page_type": sample.page_type,
        }

        return PromptPackage(
            text=template_text,
            images=images,
            meta=meta,
        )

    def build_accessible_js(self, sample: Sample) -> PromptPackage:
        """
        Build a prompt package for accessible_js mode.

        Image order: mobile (Image 1), desktop (Image 2).
        """
        template_id = "generate_a11y"
        template_path = self.templates_dir / f"{template_id}.j2"
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        template_text = template_path.read_text(encoding="utf-8")
        images = [sample.sketches[device] for device in IMAGE_ORDER]

        meta = {
            "sample_id": sample.id,
            "template_id": template_id,
            "image_ordering": IMAGE_ORDER,
            "page_type": sample.page_type,
        }
        return PromptPackage(text=template_text, images=images, meta=meta)

    def build_sketch_analyzer(self, sample: Sample, reference_html: str) -> PromptPackage:
        """
        Build prompt package for sketch analyzer step (mobile + desktop sketches only).

        Image order: mobile (Image 1), desktop (Image 2).
        """
        template_path = self.templates_dir / "augment_from_sketch_analyze.j2"
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        text = template_path.read_text(encoding="utf-8")
        text = text.replace("{{page_type}}", sample.page_type)
        text = text.replace("{{reference_html}}", reference_html)

        images = [sample.sketches[device] for device in IMAGE_ORDER]
        meta = {
            "sample_id": sample.id,
            "template_id": "augment_from_sketch_analyze",
            "image_ordering": IMAGE_ORDER,
            "page_type": sample.page_type,
        }
        return PromptPackage(text=text, images=images, meta=meta)

    def build_augment_reference_screenshot(
        self,
        sample: Sample,
        reference_html: str,
        desktop_img: Path,
        tablet_img: Path,
        mobile_img: Path,
    ) -> PromptPackage:
        """
        Build prompt package for screenshot-based reference HTML augmentation.

        No sketches used. Attaches desktop (Image 1), tablet (Image 2),
        mobile (Image 3) screenshots of the reference page.
        """
        template_path = self.templates_dir / "augment_from_screenshot.j2"
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        text = template_path.read_text(encoding="utf-8")
        text = text.replace("{{reference_html}}", reference_html)

        image_ordering = ["desktop", "tablet", "mobile"]
        images = [desktop_img, tablet_img, mobile_img]
        meta = {
            "sample_id": sample.id,
            "template_id": "augment_from_screenshot",
            "image_ordering": image_ordering,
            "page_type": sample.page_type,
        }
        return PromptPackage(text=text, images=images, meta=meta)

    def build_screenshot_only_augment(
        self,
        sample: Sample,
        desktop_img: Path,
        tablet_img: Path,
        mobile_img: Path,
    ) -> PromptPackage:
        """
        Build prompt package for screenshot-only reconstruction (no HTML source).

        Attaches desktop (Image 1), tablet (Image 2), mobile (Image 3) screenshots.
        The model reconstructs the page entirely from vision.
        """
        template_path = self.templates_dir / "reconstruct_from_screenshot.j2"
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        text = template_path.read_text(encoding="utf-8")

        image_ordering = ["desktop", "tablet", "mobile"]
        images = [desktop_img, tablet_img, mobile_img]
        meta = {
            "sample_id": sample.id,
            "template_id": "reconstruct_from_screenshot",
            "image_ordering": image_ordering,
            "page_type": sample.page_type,
        }
        return PromptPackage(text=text, images=images, meta=meta)

    def build_sketch_html_augment(
        self, sample: Sample, reference_html: str, sketch_analysis_json: str
    ) -> PromptPackage:
        """
        Build prompt package for reference HTML augmentation from sketch analysis.

        Embeds reference HTML and analysis JSON; attaches mobile then desktop sketches.
        """
        template_path = self.templates_dir / "augment_from_sketch_html.j2"
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        text = template_path.read_text(encoding="utf-8")
        text = text.replace("{{reference_html}}", reference_html)
        text = text.replace("{{sketch_analysis_json}}", sketch_analysis_json)

        images = [sample.sketches[device] for device in IMAGE_ORDER]
        meta = {
            "sample_id": sample.id,
            "template_id": "augment_from_sketch_html",
            "image_ordering": IMAGE_ORDER,
            "page_type": sample.page_type,
        }
        return PromptPackage(text=text, images=images, meta=meta)
