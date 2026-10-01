"""
Dataset loader for responsive webpage generation dataset.

Provides a single source of truth for accessing sketches, HTML references,
and metadata for each sample in the dataset.
"""

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Literal

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# Allowed device breakpoints (enforced order for deterministic output)
ALLOWED_DEVICES = ("desktop", "mobile")

# Expected file naming patterns
SKETCH_PATTERN = re.compile(r"^(\d{4})-([dm])-v2\.png$")
HTML_PATTERN = re.compile(r"^(\d{4})\.html$")


@dataclass(frozen=True)
class Sample:
    """A single dataset sample with resolved paths."""

    # Sample identifier (e.g., "0001")
    id: str

    # Dictionary mapping device type to absolute sketch path
    # Keys: "desktop", "mobile" (in that order)
    sketches: dict[str, Path]

    # Absolute path to reference HTML file
    reference_html: Path

    # ISO 8601 timestamp of metadata creation
    created_at: str

    # Page category (e.g., "unknown", "blog", "ecommerce")
    page_type: str

    # Absolute path to the metadata JSON file
    meta_path: Path

    # Absolute path to the original (pre-augmentation) Sketch2Code HTML, if listed
    source_html: Path | None = None


class ResponsiveDataset:
    """
    Dataset loader for responsive webpage wireframe sketches and reference HTML.

    Provides validated access to the dataset organized as:
    - dataset/sketches/{id}-{d|m}-v2.png
    - dataset/references/{id}.html        (benchmark reference HTML)
    - dataset/source_html/{id}.html       (original Sketch2Code HTML)
    - dataset/metadata/{id}.json          (reference_html / source_html point at the files above)
    """

    def __init__(self, root: str | Path, *, strict: bool = True):
        """
        Initialize dataset loader.

        Args:
            root: Path to dataset root directory (containing sketches/, metadata/, and reference HTML)
            strict: Whether to enforce strict validation (raises on missing optional fields)
        """
        self.root = Path(root).resolve()
        self.metadata_dir = self.root / "metadata"
        self.sketches_dir = self.root / "sketches"
        self.strict = strict

        # Validate directory structure exists
        if not self.metadata_dir.exists():
            raise FileNotFoundError(f"Metadata directory not found: {self.metadata_dir}")
        if not self.sketches_dir.exists():
            raise FileNotFoundError(f"Sketches directory not found: {self.sketches_dir}")

    def ids(self) -> list[str]:
        """
        Return sorted list of sample IDs discovered from metadata/*.json files.

        Returns:
            Sorted list of sample IDs (e.g., ["0001", "0002", ...])
        """
        ids = []
        for path in self.metadata_dir.glob("*.json"):
            # Extract ID from filename (e.g., "0001.json" -> "0001")
            sample_id = path.stem
            ids.append(sample_id)
        return sorted(ids)

    def get(self, sample_id: str) -> Sample:
        """
        Load a single sample with resolved absolute paths.

        Args:
            sample_id: Four-digit sample identifier (e.g., "0001")

        Returns:
            Sample object with resolved paths

        Raises:
            FileNotFoundError: If metadata file does not exist
            ValueError: If metadata is invalid or ID mismatch
        """
        meta_path = self.metadata_dir / f"{sample_id}.json"

        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        # Load and parse metadata
        with open(meta_path, "r") as f:
            meta = json.load(f)

        # Validate required fields exist
        self._validate_metadata_fields(meta, meta_path)

        # Validate ID matches filename
        if meta["id"] != sample_id:
            raise ValueError(
                f"ID mismatch in {meta_path}: "
                f"filename={sample_id}, metadata={meta['id']}"
            )

        # Validate and resolve sketch paths in deterministic order
        sketches = self._resolve_sketches(meta, sample_id, meta_path)

        # Validate and resolve reference HTML path
        reference_html = self._resolve_html(meta, sample_id, meta_path)
        source_html = (
            self._resolve_html(meta, sample_id, meta_path, field="source_html")
            if meta.get("source_html")
            else None
        )

        return Sample(
            id=meta["id"],
            sketches=sketches,
            reference_html=reference_html,
            created_at=meta["created_at"],
            page_type=meta.get("page_type", "unknown") if not self.strict else meta["page_type"],
            meta_path=meta_path,
            source_html=source_html,
        )

    def _validate_metadata_fields(self, meta: dict, meta_path: Path) -> None:
        """
        Validate that required metadata fields exist and have valid types.

        Args:
            meta: Metadata dictionary
            meta_path: Path to metadata file (for error messages)

        Raises:
            ValueError: If required fields are missing or invalid
        """
        required_fields = ["id", "sketches", "reference_html", "created_at"]

        # Check required fields
        for field in required_fields:
            if field not in meta:
                raise ValueError(f"Missing required field '{field}' in {meta_path}")

        # Validate page_type (required in strict mode, defaults in relaxed)
        if self.strict and "page_type" not in meta:
            raise ValueError(f"Missing required field 'page_type' in {meta_path}")

        # Validate page_type is a string if present
        if "page_type" in meta and not isinstance(meta["page_type"], str):
            raise ValueError(f"Field 'page_type' must be a string in {meta_path}")

        # Validate sketches is a dict
        if not isinstance(meta["sketches"], dict):
            raise ValueError(f"Field 'sketches' must be a dictionary in {meta_path}")

        # Validate created_at is parseable ISO timestamp
        try:
            datetime.fromisoformat(meta["created_at"])
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Field 'created_at' must be a valid ISO 8601 timestamp in {meta_path}: {e}"
            )

    def _resolve_sketches(self, meta: dict, sample_id: str, meta_path: Path) -> dict[str, Path]:
        """
        Resolve and validate sketch paths with device key enforcement.

        Args:
            meta: Metadata dictionary
            sample_id: Sample identifier
            meta_path: Path to metadata file (for error messages)

        Returns:
            Dictionary mapping device to absolute path (in deterministic order)

        Raises:
            ValueError: If device keys are invalid or paths are unsafe
        """
        sketches_meta = meta["sketches"]

        # Enforce exactly the allowed device keys
        provided_devices = set(sketches_meta.keys())
        expected_devices = set(ALLOWED_DEVICES)

        if provided_devices != expected_devices:
            missing = expected_devices - provided_devices
            extra = provided_devices - expected_devices
            error_parts = []
            if missing:
                error_parts.append(f"missing: {sorted(missing)}")
            if extra:
                error_parts.append(f"unexpected: {sorted(extra)}")
            raise ValueError(
                f"Invalid device keys in {meta_path}: {', '.join(error_parts)}. "
                f"Expected exactly: {sorted(ALLOWED_DEVICES)}"
            )

        # Resolve paths in deterministic order and validate
        sketches = {}
        for device in ALLOWED_DEVICES:  # Use ordered tuple
            rel_path = sketches_meta[device]

            # Validate file extension
            if not rel_path.endswith(".png"):
                raise ValueError(
                    f"Sketch for '{device}' must be a .png file in {meta_path}: {rel_path}"
                )

            # Validate naming convention
            filename = Path(rel_path).name
            match = SKETCH_PATTERN.match(filename)
            if not match:
                raise ValueError(
                    f"Sketch filename must match pattern '{{id}}-{{d|m}}-v2.png' "
                    f"in {meta_path}: {filename}"
                )

            # Validate ID in filename matches sample ID
            file_id = match.group(1)
            if file_id != sample_id:
                raise ValueError(
                    f"Sketch filename ID '{file_id}' does not match sample ID '{sample_id}' "
                    f"in {meta_path}: {filename}"
                )

            # Resolve path and validate it's within dataset root
            abs_path = (self.root / rel_path).resolve()
            self._validate_path_safety(abs_path, meta_path)

            sketches[device] = abs_path

        return sketches

    def _resolve_html(
        self, meta: dict, sample_id: str, meta_path: Path, field: str = "reference_html"
    ) -> Path:
        """
        Resolve and validate an HTML path from metadata.

        Args:
            meta: Metadata dictionary
            sample_id: Sample identifier
            meta_path: Path to metadata file (for error messages)
            field: Metadata field holding the path ("reference_html" or "source_html")

        Returns:
            Absolute path to reference HTML

        Raises:
            ValueError: If path is invalid or unsafe
        """
        rel_path = meta[field]

        # Validate file extension
        if not rel_path.endswith(".html"):
            raise ValueError(
                f"Reference HTML must be a .html file in {meta_path}: {rel_path}"
            )

        # Validate naming convention
        filename = Path(rel_path).name
        match = HTML_PATTERN.match(filename)
        if not match:
            raise ValueError(
                f"HTML filename must match pattern '{{id}}.html' in {meta_path}: {filename}"
            )

        # Validate ID in filename matches sample ID
        file_id = match.group(1)
        if file_id != sample_id:
            raise ValueError(
                f"HTML filename ID '{file_id}' does not match sample ID '{sample_id}' "
                f"in {meta_path}: {filename}"
            )

        # Resolve path and validate it's within dataset root
        abs_path = (self.root / rel_path).resolve()
        self._validate_path_safety(abs_path, meta_path)

        return abs_path

    def _validate_path_safety(self, path: Path, meta_path: Path) -> None:
        """
        Validate that a resolved path does not escape the dataset root.

        Args:
            path: Resolved absolute path to validate
            meta_path: Path to metadata file (for error messages)

        Raises:
            ValueError: If path escapes dataset root
        """
        try:
            path.relative_to(self.root)
        except ValueError:
            raise ValueError(
                f"Path escapes dataset root in {meta_path}: {path} "
                f"is not relative to {self.root}"
            )

    def iter_samples(
        self,
        ids: list[str] | None = None,
        *,
        validate: bool = True,
        on_error: Literal["raise", "skip", "warn"] = "raise"
    ) -> Iterator[Sample]:
        """
        Iterate over samples deterministically.

        Args:
            ids: Optional list of sample IDs to iterate over (uses all if None)
            validate: Whether to validate each sample before yielding
            on_error: How to handle validation/loading errors:
                - "raise": Raise exception on first error
                - "skip": Skip invalid samples silently
                - "warn": Print warning and skip invalid samples

        Yields:
            Sample objects in ID order
        """
        # Use provided IDs or discover all IDs
        sample_ids = ids if ids is not None else self.ids()

        for sample_id in sample_ids:
            try:
                sample = self.get(sample_id)

                # Optionally validate before yielding
                if validate:
                    errors = self.validate_sample(sample)
                    if errors:
                        error_msg = f"Validation failed for {sample_id}: {errors}"
                        if on_error == "raise":
                            raise ValueError(error_msg)
                        elif on_error == "warn":
                            print(f"Warning: {error_msg}")
                            continue
                        else:  # skip
                            continue

                yield sample

            except Exception as e:
                error_msg = f"Error loading {sample_id}: {e}"
                if on_error == "raise":
                    raise
                elif on_error == "warn":
                    print(f"Warning: {error_msg}")
                # skip continues automatically

    def validate_sample(self, sample: Sample, *, check_images: bool = False) -> list[str]:
        """
        Validate a single sample for completeness and file existence.

        Args:
            sample: Sample object to validate
            check_images: Whether to validate image integrity with PIL (slower)

        Returns:
            List of error messages (empty list means valid)
        """
        errors = []

        # Check metadata file exists
        if not sample.meta_path.exists():
            errors.append(f"Metadata file missing: {sample.meta_path}")

        # Check each sketch exists (in deterministic order)
        for device in ALLOWED_DEVICES:
            if device not in sample.sketches:
                errors.append(f"Missing {device} sketch in sample")
                continue

            path = sample.sketches[device]
            if not path.exists():
                errors.append(f"{device} sketch missing: {path}")
            elif path.stat().st_size == 0:
                errors.append(f"{device} sketch is empty: {path}")
            elif check_images:
                # Optional: validate image integrity with PIL
                img_errors = self._validate_image(path, device)
                errors.extend(img_errors)

        # Check reference HTML exists
        if not sample.reference_html.exists():
            errors.append(f"Reference HTML missing: {sample.reference_html}")
        elif sample.reference_html.stat().st_size == 0:
            errors.append(f"Reference HTML is empty: {sample.reference_html}")

        if sample.source_html is not None and not sample.source_html.exists():
            errors.append(f"Source HTML missing: {sample.source_html}")

        return errors

    def _validate_image(self, path: Path, device: str) -> list[str]:
        """
        Validate image file integrity using PIL.

        Args:
            path: Path to image file
            device: Device type (for error messages)

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        if not PIL_AVAILABLE:
            errors.append(f"PIL not available for {device} image validation")
            return errors

        try:
            with Image.open(path) as img:
                # Attempt to load the image data
                img.verify()
            # Need to reopen after verify()
            with Image.open(path) as img:
                img.load()
        except Exception as e:
            errors.append(f"{device} sketch is corrupted or invalid PNG: {path} ({e})")

        return errors

    def validate_all(
        self,
        *,
        ids: list[str] | None = None,
        check_images: bool = False
    ) -> dict[str, list[str]]:
        """
        Validate all samples and return mapping of invalid samples to their errors.

        Args:
            ids: Optional list of sample IDs to validate (validates all if None)
            check_images: Whether to validate image integrity with PIL (slower)

        Returns:
            Dictionary mapping sample ID to list of errors (only includes invalid samples)
        """
        invalid = {}

        # Use provided IDs or discover all IDs
        sample_ids = ids if ids is not None else self.ids()

        for sample_id in sample_ids:
            try:
                sample = self.get(sample_id)
                errors = self.validate_sample(sample, check_images=check_images)
                if errors:
                    invalid[sample_id] = errors
            except Exception as e:
                invalid[sample_id] = [str(e)]

        return invalid

    def write_index(
        self,
        out_path: str | Path,
        *,
        ids: list[str] | None = None
    ) -> None:
        """
        Write a dataset index snapshot to a JSONL file.

        Creates a reproducible snapshot of dataset samples for consistent
        evaluation runs and dataset versioning.

        Args:
            out_path: Path to output index file (will be overwritten)
            ids: Optional list of sample IDs to include (includes all if None)
        """
        out_path = Path(out_path)

        # Use provided IDs or discover all IDs
        sample_ids = ids if ids is not None else self.ids()

        # Write one line per sample ID
        with open(out_path, "w") as f:
            for sample_id in sample_ids:
                f.write(json.dumps({"id": sample_id}) + "\n")

        # Write metadata file for reproducibility
        meta_path = out_path.with_suffix(".meta.json")
        metadata_hash = self._compute_metadata_hash(sample_ids)

        meta_info = {
            "dataset_root": self.root.name,
            "timestamp": datetime.now().isoformat(),
            "sample_count": len(sample_ids),
            "metadata_hash": metadata_hash
        }

        with open(meta_path, "w") as f:
            json.dump(meta_info, f, indent=2)

        print(f"Wrote index with {len(sample_ids)} samples to {out_path}")
        print(f"Wrote index metadata to {meta_path}")

    def _compute_metadata_hash(self, sample_ids: list[str]) -> str:
        """
        Compute hash of metadata file contents for reproducibility tracking.

        Args:
            sample_ids: List of sample IDs to include in hash

        Returns:
            Hex digest of combined metadata content
        """
        hasher = hashlib.sha256()

        for sample_id in sorted(sample_ids):  # Sort for determinism
            meta_path = self.metadata_dir / f"{sample_id}.json"
            if meta_path.exists():
                hasher.update(meta_path.read_bytes())

        return hasher.hexdigest()

    def load_index(self, index_path: str | Path) -> list[str]:
        """
        Load sample IDs from a JSONL index file.

        Args:
            index_path: Path to index file created by write_index()

        Returns:
            List of sample IDs in file order
        """
        index_path = Path(index_path)

        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        ids = []
        with open(index_path, "r") as f:
            for line in f:
                record = json.loads(line.strip())
                ids.append(record["id"])

        return ids


# CLI entry point for validation and index creation
def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Dataset loader for responsive webpage generation dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate all samples in the dataset"
    )
    validate_parser.add_argument(
        "--root",
        type=str,
        default="dataset",
        help="Path to dataset root directory (default: dataset)"
    )
    validate_parser.add_argument(
        "--check-images",
        action="store_true",
        help="Validate image integrity with PIL (slower)"
    )
    validate_parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Use relaxed mode (allow missing optional fields)"
    )

    # Index command
    index_parser = subparsers.add_parser(
        "index",
        help="Create a dataset index snapshot"
    )
    index_parser.add_argument(
        "--root",
        type=str,
        default="dataset",
        help="Path to dataset root directory (default: dataset)"
    )
    index_parser.add_argument(
        "--out",
        type=str,
        help="Output path for index file (default: <root>/index.jsonl)"
    )

    # Show command
    show_parser = subparsers.add_parser(
        "show",
        help="Show details for a specific sample"
    )
    show_parser.add_argument(
        "--root",
        type=str,
        default="dataset",
        help="Path to dataset root directory (default: dataset)"
    )
    show_parser.add_argument(
        "--id",
        type=str,
        required=True,
        help="Sample ID to show"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Initialize dataset
    strict = not getattr(args, "no_strict", False)
    dataset = ResponsiveDataset(args.root, strict=strict)

    if args.command == "validate":
        # Validate all samples
        print(f"Validating dataset in {args.root}...")
        if args.check_images and not PIL_AVAILABLE:
            print("Warning: PIL not available, skipping image integrity checks")
            check_images = False
        else:
            check_images = args.check_images

        invalid = dataset.validate_all(check_images=check_images)

        if not invalid:
            print(f"✓ All {len(dataset.ids())} samples are valid")
            return 0
        else:
            print(f"✗ Found {len(invalid)} invalid samples:")
            for sample_id, errors in invalid.items():
                print(f"\n{sample_id}:")
                for error in errors:
                    print(f"  - {error}")
            return 1

    elif args.command == "index":
        # Create index file
        out_path = args.out if args.out else Path(args.root) / "index.jsonl"

        print(f"Creating index for dataset in {args.root}...")
        dataset.write_index(out_path)
        return 0

    elif args.command == "show":
        # Show details for a single sample
        try:
            sample = dataset.get(args.id)
            print(f"\nSample {sample.id}:")
            print(f"  Created: {sample.created_at}")
            print(f"  Type: {sample.page_type}")
            print(f"  Sketches:")
            for device in ALLOWED_DEVICES:  # Show in deterministic order
                path = sample.sketches[device]
                exists = "✓" if path.exists() else "✗"
                print(f"    {device}: {exists} {path.relative_to(dataset.root)}")
            html_exists = "✓" if sample.reference_html.exists() else "✗"
            print(f"  Reference HTML: {html_exists} {sample.reference_html.relative_to(dataset.root)}")

            # Validate
            errors = dataset.validate_sample(sample)
            if errors:
                print(f"\n  Validation errors:")
                for error in errors:
                    print(f"    - {error}")
            else:
                print(f"\n  ✓ Valid")

            return 0

        except Exception as e:
            print(f"Error: {e}")
            return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
