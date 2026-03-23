"""Show how a decoder-only model generates text one token at a time."""

from __future__ import annotations


NEXT_TOKEN_TABLE = {
    ("I",): "love",
    ("I", "love"): "open",
    ("I", "love", "open"): "models",
    ("I", "love", "open", "models"): ".",
}


def generate(seed_tokens: list[str], max_steps: int = 4) -> list[str]:
    tokens = seed_tokens[:]
    for _ in range(max_steps):
        next_token = NEXT_TOKEN_TABLE.get(tuple(tokens))
        if next_token is None:
            break
        tokens.append(next_token)
        if next_token == ".":
            break
    return tokens


def main() -> None:
    prompt_tokens = ["I"]
    generated = generate(prompt_tokens)

    print("Decoder-only generation demo")
    print(f"Start tokens: {prompt_tokens}")
    print()
    print("Generation steps:")
    for index in range(1, len(generated)):
        prefix = generated[:index]
        next_token = generated[index]
        print(f"  Prefix {prefix} -> next token {next_token!r}")
    print()
    print("Final sequence:")
    print(f"  {generated}")
    print()
    print("Takeaway:")
    print("  Decoder-only models keep extending the existing sequence one next token at a time.")


if __name__ == "__main__":
    main()
