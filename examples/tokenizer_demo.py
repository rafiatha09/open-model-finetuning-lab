"""A tiny tokenizer demo using only the Python standard library."""

from __future__ import annotations

import re


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


def simple_tokenize(text: str) -> list[str]:
    """Split text into word-like pieces and punctuation."""
    return TOKEN_PATTERN.findall(text)


def build_vocab(tokens: list[str]) -> dict[str, int]:
    """Assign a stable integer ID to each token in first-seen order."""
    vocab: dict[str, int] = {}
    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def main() -> None:
    text = "Fine-tuning open models is practical, but context length still matters. Fine-tuning is the best way to get desire result."
    tokens = simple_tokenize(text)
    vocab = build_vocab(tokens)
    token_ids = [vocab[token] for token in tokens]

    print("Input text:")
    print(f"  {text}")
    print()
    print("Tokens:")
    print(f"  {tokens}")
    print()
    print("Vocabulary:")
    for token, token_id in vocab.items():
        print(f"  {token_id:>2} -> {token!r}")
    print()
    print("Token IDs:")
    print(f"  {token_ids}")
    print()
    print("Takeaway:")
    print("  Models consume token IDs, not raw strings.")


if __name__ == "__main__":
    main()
