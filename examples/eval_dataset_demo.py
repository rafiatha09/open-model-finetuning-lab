"""Show the structure of the starter evaluation dataset."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.dataset import load_eval_jsonl  # noqa: E402


def main() -> None:
    eval_set_path = ROOT / "data/eval/sample_eval_set.jsonl"
    examples = load_eval_jsonl(eval_set_path)

    print("Evaluation dataset demo")
    print(f"Loaded examples: {len(examples)}")
    print()

    first = examples[0]
    print(f"ID: {first.id}")
    print(f"Instruction: {first.instruction}")
    print(f"Input: {first.input}")
    print(f"Reference: {first.reference}")
    print(f"Tags: {first.tags}")
    print(f"Checks: {first.checks}")


if __name__ == "__main__":
    main()
