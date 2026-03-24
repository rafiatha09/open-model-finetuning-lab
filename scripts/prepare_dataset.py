"""Prepare a simple instruction dataset with train/validation splits."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.instruction_dataset import (  # noqa: E402
    load_instruction_jsonl,
    train_validation_split,
    write_instruction_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare train/validation JSONL splits for instruction fine-tuning.")
    parser.add_argument("--input", required=True, help="Path to the source instruction JSONL file.")
    parser.add_argument("--output-dir", required=True, help="Directory where train/validation JSONL files will be written.")
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Validation split ratio as a float between 0 and 1. Default: 0.2",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for the split.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    records = load_instruction_jsonl(args.input)
    train_records, validation_records = train_validation_split(
        records,
        validation_size=args.validation_size,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    train_path = write_instruction_jsonl(train_records, output_dir / "train.jsonl")
    validation_path = write_instruction_jsonl(validation_records, output_dir / "validation.jsonl")

    print("Prepared dataset")
    print(f"Input records:      {len(records)}")
    print(f"Training records:   {len(train_records)}")
    print(f"Validation records: {len(validation_records)}")
    print(f"Train path:         {train_path}")
    print(f"Validation path:    {validation_path}")


if __name__ == "__main__":
    main()
