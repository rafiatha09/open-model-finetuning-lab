"""A tiny next-token prediction demo with toy probabilities."""

from __future__ import annotations


def main() -> None:
    prompt = "I love"

    # Pretend this came from the model's final logits after softmax.
    next_token_probabilities = {
        " you": 0.45,
        " coffee": 0.30,
        " open": 0.15,
        " machine": 0.10,
    }

    greedy_choice = max(next_token_probabilities, key=next_token_probabilities.get)

    print("Next-token prediction demo")
    print(f"Prompt: {prompt!r}")
    print()
    print("Possible next-token probabilities:")
    for token, probability in next_token_probabilities.items():
        print(f"  {token!r}: {probability:.2f}")
    print()
    print(f"Greedy decoding would pick: {greedy_choice!r}")
    print()
    print("Takeaway:")
    print("  The model predicts a probability distribution first, then decoding chooses the next token.")


if __name__ == "__main__":
    main()
