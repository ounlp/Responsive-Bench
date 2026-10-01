#!/usr/bin/env python3
"""
CLI entry point for running evaluation pipelines.

Usage:
    python scripts/run_eval.py --config configs/eval_example.json
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

from responsive.eval.orchestrator import run_evaluation_sync


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run an evaluation pipeline from a config file"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to evaluation config JSON file",
    )

    args = parser.parse_args()

    try:
        config_path = Path(args.config).resolve()
        run_evaluation_sync(config_path)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
