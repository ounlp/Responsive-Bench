#!/usr/bin/env python3
"""
Convert ResponsiveBench dataset to Parquet and upload to HuggingFace.

Usage:
  python scripts/upload_to_huggingface.py --repo OWNER/REPO [--private] [--output dataset.parquet]

Requirements:
  pip install datasets pyarrow huggingface_hub Pillow

You must be logged in to HuggingFace:
  huggingface-cli login
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from datasets import Dataset, Features, Value, Image as HFImage
except ImportError:
    print("ERROR: 'datasets' not installed. Run: pip install datasets pyarrow huggingface_hub", file=sys.stderr)
    sys.exit(1)

try:
    from PIL import Image as PILImage
except ImportError:
    print("ERROR: 'Pillow' not installed. Run: pip install Pillow", file=sys.stderr)
    sys.exit(1)


DATASET_ROOT = Path(__file__).resolve().parent.parent / "dataset"


def load_sample(sample_id: str) -> dict | None:
    meta_path = DATASET_ROOT / "metadata" / f"{sample_id}.json"
    if not meta_path.exists():
        print(f"  SKIP {sample_id}: missing {meta_path.name}", file=sys.stderr)
        return None

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    html_path = DATASET_ROOT / meta["reference_html"]
    source_html_path = DATASET_ROOT / meta["source_html"]
    desktop_path = DATASET_ROOT / meta["sketches"]["desktop"]
    mobile_path = DATASET_ROOT / meta["sketches"]["mobile"]

    for p in (html_path, source_html_path, desktop_path, mobile_path):
        if not p.exists():
            print(f"  SKIP {sample_id}: missing {p.name}", file=sys.stderr)
            return None

    html = html_path.read_text(encoding="utf-8")
    source_html = source_html_path.read_text(encoding="utf-8")

    # Load images as PIL so HuggingFace serializes them correctly
    desktop_img = PILImage.open(desktop_path).convert("RGB")
    mobile_img = PILImage.open(mobile_path).convert("RGB")

    return {
        "id": meta.get("id", sample_id),
        "page_type": meta.get("page_type", "unknown"),
        "created_at": meta.get("created_at", ""),
        "sketch_desktop": desktop_img,
        "sketch_mobile": mobile_img,
        "reference_html": html,
        "source_html": source_html,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload ResponsiveBench to HuggingFace")
    parser.add_argument("--repo", required=True, help="HuggingFace repo ID, e.g. ounlp/ResponsiveBench")
    parser.add_argument("--private", action="store_true", help="Create a private repository")
    parser.add_argument("--output", type=Path, default=None, help="Also save Parquet locally at this path")
    args = parser.parse_args()

    metadata_dir = DATASET_ROOT / "metadata"
    sample_ids = sorted(p.stem for p in metadata_dir.glob("*.json"))
    print(f"Found {len(sample_ids)} samples in {metadata_dir}")

    records = []
    for sid in sample_ids:
        rec = load_sample(sid)
        if rec:
            records.append(rec)

    print(f"Loaded {len(records)} valid samples.")

    if not records:
        print("ERROR: No valid samples found.", file=sys.stderr)
        sys.exit(1)

    features = Features({
        "id": Value("string"),
        "page_type": Value("string"),
        "created_at": Value("string"),
        "sketch_desktop": HFImage(),
        "sketch_mobile": HFImage(),
        "reference_html": Value("string"),
        "source_html": Value("string"),
    })

    dataset = Dataset.from_list(records, features=features)
    print(f"Dataset schema:\n{dataset.features}\n")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(str(args.output))
        print(f"Saved Parquet to {args.output}")

    print(f"Pushing to HuggingFace: {args.repo} ({'private' if args.private else 'public'}) ...")
    dataset.push_to_hub(
        args.repo,
        private=args.private,
        commit_message="Add ResponsiveBench dataset (200 samples, desktop+mobile sketches, reference HTML)",
    )
    print(f"Done. Dataset live at: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
