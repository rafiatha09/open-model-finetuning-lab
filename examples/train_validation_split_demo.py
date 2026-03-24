"""Show how the repository splits instruction data into train and validation sets."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.instruction_dataset import load_instruction_jsonl, train_validation_split  # noqa: E402


def main() -> None:
    sample_path = ROOT / "data/sample/domain_assistant_examples.jsonl"
    records = load_instruction_jsonl(sample_path)
    train_records, validation_records = train_validation_split(records, validation_size=0.33, seed=42)

    print("Train/validation split demo")
    print(f"Input records:      {len(records)}")
    print(f"Training records:   {len(train_records)}")
    print(f"Validation records: {len(validation_records)}")
    print()
    print("Validation example instruction:")
    print(f"  {validation_records[0].instruction}")
    print()
    print("Takeaway:")
    print("  A validation split gives you a simple way to measure whether training behavior generalizes.")


if __name__ == "__main__":
    main()
