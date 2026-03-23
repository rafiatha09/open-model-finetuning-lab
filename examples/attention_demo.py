"""A small attention demo with hand-written vectors."""

from __future__ import annotations

import math


def dot(left: list[float], right: list[float]) -> float:
    # The dot product is our simple similarity score.
    # If two vectors point in a similar direction, this value gets larger.
    return sum(x * y for x, y in zip(left, right))


def softmax(values: list[float]) -> list[float]:
    # Softmax converts raw attention scores into normalized weights.
    # After this step, all weights are between 0 and 1 and sum to 1.
    max_value = max(values)
    shifted = [math.exp(value - max_value) for value in values]
    total = sum(shifted)
    return [value / total for value in shifted]


def weighted_sum(weights: list[float], vectors: list[list[float]]) -> list[float]:
    # This is the final "mixing" step.
    # Tokens with larger weights contribute more of their value vectors.
    width = len(vectors[0])
    combined = [0.0] * width
    for weight, vector in zip(weights, vectors):
        for index in range(width):
            combined[index] += weight * vector[index]
    return combined


def main() -> None:
    # We will compute attention for the token "love".
    # In a real transformer, every token gets its own query, key, and value.
    tokens = ["I", "love", "coffee"]

    # The query represents the token currently asking:
    # "Which other tokens are most relevant to me right now?"
    query_for_love = [1.0, 1.0]

    # Keys are what the query compares against.
    # One key per token.
    keys = [
        [1.0, 0.0],  # I
        [1.0, 1.0],  # love
        [0.0, 1.0],  # coffee
    ]

    # Values are the vectors we actually mix together after we know the weights.
    # Keys decide "where to look"; values decide "what information to pull in."
    values = [
        [1.0, 0.0],
        [0.5, 0.5],
        [0.0, 1.0],
    ]

    # Attention math in its simplest form:
    # score_i = q · k_i
    # weight_i = exp(score_i) / sum_j exp(score_j)
    # output = sum_i weight_i * v_i

    # Step 1:
    # Compare the query against every key to get raw relevance scores.
    scores = [dot(query_for_love, key) for key in keys]

    # Step 2:
    # Normalize those scores so they become attention weights.
    # Higher score -> higher weight.
    weights = softmax(scores)

    # Step 3:
    # Use the weights to blend the value vectors into one output vector.
    # This output is the updated representation for the query token.
    combined = weighted_sum(weights, values)

    print("Attention demo")
    print(f"Query token: {tokens[1]!r}")
    print()
    print("Raw attention scores:")
    for token, score in zip(tokens, scores):
        print(f"  {token:>6}: {score:.3f}")
    print()
    print("Attention weights after softmax:")
    for token, weight in zip(tokens, weights):
        print(f"  {token:>6}: {weight:.3f}")
    print()
    print("Weighted combination of value vectors:")
    print(f"  {combined}")
    print()
    print("Takeaway:")
    print("  Attention lets one token pull in information from the tokens most relevant to it.")


if __name__ == "__main__":
    main()
