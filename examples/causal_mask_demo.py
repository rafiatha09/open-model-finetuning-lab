"""Show how a causal mask blocks attention to future tokens."""

from __future__ import annotations

import math


def softmax(values: list[float]) -> list[float]:
    max_value = max(values)
    shifted = [math.exp(value - max_value) if value > -1e8 else 0.0 for value in values]
    total = sum(shifted)
    return [value / total if total else 0.0 for value in shifted]


def make_causal_mask(length: int) -> list[list[int]]:
    return [[1 if col <= row else 0 for col in range(length)] for row in range(length)]


def apply_mask(scores: list[list[float]], mask: list[list[int]]) -> list[list[float]]:
    blocked = -1e9
    return [
        [score if allowed else blocked for score, allowed in zip(row, mask_row)]
        for row, mask_row in zip(scores, mask)
    ]


def main() -> None:
    tokens = ["I", "study", "open", "models"]
    scores = [
        [3.0, 1.0, 0.0, 0.0],
        [2.0, 3.0, 1.0, 0.0],
        [1.0, 2.0, 3.0, 1.0],
        [0.5, 1.0, 2.0, 3.0],
    ]
    mask = make_causal_mask(len(tokens))
    masked_scores = apply_mask(scores, mask)

    print("Causal mask matrix (1 means allowed, 0 means blocked):")
    for row in mask:
        print(f"  {row}")
    print()

    print("Attention weights after applying the mask:")
    for token, row in zip(tokens, masked_scores):
        weights = softmax(row)
        print(f"  Query token {token!r}: {['{:.3f}'.format(weight) for weight in weights]}")
    print()

    print("Takeaway:")
    print("  Each position can look backward, but never forward into future tokens.")


if __name__ == "__main__":
    main()
