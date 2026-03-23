"""A tiny embedding lookup demo using hand-written vectors."""

from __future__ import annotations

import math


def dot(left: list[float], right: list[float]) -> float:
    return sum(x * y for x, y in zip(left, right))


def norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return dot(left, right) / (norm(left) * norm(right))


def main() -> None:
    # Pretend this is a tiny embedding table learned by a model.
    embeddings = {
        "doctor": [0.9, 0.8, 0.1],
        "physician": [0.85, 0.75, 0.15],
        "coffee": [0.1, 0.2, 0.95],
    }

    print("Embedding demo")
    print("A token ID would normally look up one vector from the embedding table.")
    print()

    for token, vector in embeddings.items():
        print(f"  {token:>9}: {vector}")

    print()
    print("Cosine similarity:")
    print(f"  doctor vs physician: {cosine_similarity(embeddings['doctor'], embeddings['physician']):.3f}")
    print(f"  doctor vs coffee:    {cosine_similarity(embeddings['doctor'], embeddings['coffee']):.3f}")
    print()
    print("Takeaway:")
    print("  Embeddings turn token IDs into learned vectors, and similar tokens often end up closer together.")


if __name__ == "__main__":
    main()
