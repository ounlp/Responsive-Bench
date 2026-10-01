#!/usr/bin/env python3
"""
CLI entry point for running generation pipelines.

Usage:
    python scripts/run_generate.py --config configs/example_run.json
"""

import argparse
import sys
from pathlib import Path

# Load .env file if it exists (before importing anything that uses env vars)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, skip .env loading
    pass

# Add src to path
src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from responsive.runner.generate import run_from_config


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run a generation pipeline from a config file"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to run config JSON file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["generate_a11y", "augment_from_sketch", "augment_from_screenshot", "reconstruct_from_screenshot"],
        default=None,
        help="Override the config's run mode",
    )

    args = parser.parse_args()

    try:
        config_path = Path(args.config).resolve()
        run_from_config(config_path, mode_override=args.mode)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
