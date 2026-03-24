"""Show how an instruction example becomes SFT training text."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.instruction_dataset import format_instruction_prompt, format_sft_text, load_instruction_jsonl  # noqa: E402


def main() -> None:
    sample_path = ROOT / "data/sample/domain_assistant_examples.jsonl"
    example = load_instruction_jsonl(sample_path)[0]

    print("Dataset formatting demo")
    print("Prompt only:")
    print(format_instruction_prompt(example))
    print()
    print("Prompt plus response:")
    print(format_sft_text(example, eos_token=" <eos>"))
    print()
    print("Takeaway:")
    print("  Fine-tuning learns from consistent formatted text, not from raw JSON fields directly.")


if __name__ == "__main__":
    main()
