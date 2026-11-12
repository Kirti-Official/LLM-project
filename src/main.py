
"""
VERDICT - Main Execution Pipeline

Loads precomputed evidence and abstention decisions,
then generates evidence-grounded responses using Llama 3.1 8B.
"""

import argparse
import json
from pathlib import Path

from src.llm_generator import process_file


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ramdocs_abstention_decisions.jsonl"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ramdocs_llm_responses.jsonl"
)


def inspect_input_file(input_path):
    """Verify input file and count available records."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    with input_path.open("r", encoding="utf-8") as file:
        total_records = sum(
            1 for line in file if line.strip()
        )

    if total_records == 0:
        raise ValueError("Input file contains no records.")

    return total_records


def main():
    parser = argparse.ArgumentParser(
        description="VERDICT LLM Reasoning Pipeline"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of records to process"
    )

    args = parser.parse_args()

    print("=" * 60)
    print(" VERDICT: EVIDENCE-AWARE REASONING PIPELINE")
    print("=" * 60)

    print("\n[1/3] Checking input data...")

    total_records = inspect_input_file(args.input)

    print(f"Input records available: {total_records}")

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than zero.")

    print("\n[2/3] Connecting to Llama 3.1 8B...")
    print("Checking Ollama availability and model installation.")

    print("\n[3/3] Generating evidence-grounded responses...")

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    process_file(
        input_file=args.input,
        output_file=args.output,
        limit=args.limit
    )

    print("\n" + "=" * 60)
    print(" VERDICT PIPELINE COMPLETED")
    print("=" * 60)

    print(f"Input records: {total_records}")
    print(f"Output file: {args.output}")


if __name__ == "__main__":
    main()
