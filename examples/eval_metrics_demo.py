"""Demonstrate the small heuristic metrics used in Phase 4."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.dataset import EvalExample  # noqa: E402
from evaluation.metrics import score_prediction  # noqa: E402


def main() -> None:
    example = EvalExample(
        id="metrics-demo",
        instruction="Explain LoRA briefly.",
        input="Use at most 2 sentences.",
        reference="LoRA adds small trainable adapters instead of updating every model weight. It reduces memory cost for fine-tuning.",
        tags=["demo"],
        checks={"must_include": ["LoRA", "memory"], "max_sentences": 2},
    )

    strong_prediction = (
        "LoRA adds small trainable adapters instead of updating every model weight. "
        "It reduces memory cost for fine-tuning."
    )
    weak_prediction = (
        "LoRA is interesting and used in many systems. "
        "It can be about models, tasks, scaling, deployment, and many other ideas."
    )

    print("Evaluation metrics demo")
    print("Strong prediction metrics:")
    print(score_prediction(strong_prediction, example))
    print()
    print("Weak prediction metrics:")
    print(score_prediction(weak_prediction, example))


if __name__ == "__main__":
    main()
