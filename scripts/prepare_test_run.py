#!/usr/bin/env python3
"""
Prepare test HTML files for evaluation.

Takes HTML files from a directory and prepares them in a run structure
that can be evaluated using the standard evaluation pipeline.

Usage:
    python scripts/prepare_test_run.py --input stress_test/ --run-id test_adversarial
    python scripts/prepare_test_run.py --input stress_test/ --run-id test_adversarial --clean
"""

import argparse
import shutil
import sys
from pathlib import Path


def prepare_test_run(
    input_dir: Path,
    run_id: str = "test_adversarial",
    output_root: Path = Path("runs"),
    clean: bool = False,
) -> None:
    """
    Prepare test HTML files for evaluation.

    Args:
        input_dir: Directory containing HTML files to test
        run_id: Run ID for the test run (default: "test_adversarial")
        output_root: Root directory for runs (default: "runs")
        clean: If True, remove existing run directory before creating new one

    Raises:
        FileNotFoundError: If input directory does not exist
        ValueError: If no HTML files found
    """
    input_dir = Path(input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Find all HTML files
    html_files = sorted(input_dir.glob("*.html"))
    if not html_files:
        raise ValueError(f"No HTML files found in: {input_dir}")

    print(f"Found {len(html_files)} HTML file(s) in {input_dir}")

    # Create run directory structure
    run_dir = output_root / run_id
    samples_dir = run_dir / "samples"

    # Clean existing run if requested
    if clean and run_dir.exists():
        print(f"Cleaning existing run directory: {run_dir}")
        shutil.rmtree(run_dir)

    # Create samples directory
    samples_dir.mkdir(parents=True, exist_ok=True)
    print(f"Preparing run directory: {run_dir}")

    # Use project root assets/rick.jpg (not input_dir)
    repo_root = Path(__file__).resolve().parent.parent
    rick_src = repo_root / "assets" / "rick.jpg"
    copy_rick = rick_src.exists()
    if not copy_rick:
        print(f"Note: {rick_src} not found; skipping rick.jpg copy into samples.")

    # Copy HTML files to run structure
    sample_ids = []
    for i, html_file in enumerate(html_files, start=1):
        # Generate sample ID from filename (without extension) or use sequential ID
        sample_id = html_file.stem
        # Sanitize sample ID (remove spaces, special chars)
        sample_id = "".join(c if c.isalnum() or c in "_-" else "_" for c in sample_id)
        if not sample_id or sample_id.startswith("_"):
            sample_id = f"test_{i:03d}"

        sample_dir = samples_dir / sample_id
        sample_dir.mkdir(exist_ok=True)

        output_html = sample_dir / "output.html"
        shutil.copy2(html_file, output_html)
        if copy_rick:
            shutil.copy2(rick_src, sample_dir / "rick.jpg")
        sample_ids.append(sample_id)
        print(f"  [{i}/{len(html_files)}] {html_file.name} -> {sample_id}/output.html")

    print(f"\n✓ Prepared {len(sample_ids)} sample(s) in {run_dir}")
    print(f"  Sample IDs: {', '.join(sample_ids)}")
    print(f"\nTo evaluate, create a config file pointing to run_id: '{run_id}'")
    print(f"  Example: python scripts/run_eval.py --config configs/eval_test_example.json")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare test HTML files for evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare HTML files from stress_test/ directory
  python scripts/prepare_test_run.py --input stress_test/

  # Use custom run ID
  python scripts/prepare_test_run.py --input stress_test/ --run-id my_test_run

  # Clean existing run before creating new one
  python scripts/prepare_test_run.py --input stress_test/ --clean
        """,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Directory containing HTML files to test",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="test_adversarial",
        help="Run ID for the test run (default: test_adversarial)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="runs",
        help="Root directory for runs (default: runs)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing run directory before creating new one",
    )

    args = parser.parse_args()

    try:
        prepare_test_run(
            input_dir=Path(args.input),
            run_id=args.run_id,
            output_root=Path(args.output_root),
            clean=args.clean,
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
