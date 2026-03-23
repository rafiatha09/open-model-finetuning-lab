"""A tiny context-window demo showing truncation of older tokens."""

from __future__ import annotations


def keep_last_tokens(tokens: list[str], window_size: int) -> list[str]:
    return tokens[-window_size:]


def main() -> None:
    tokens = ["system", "you", "are", "helpful", "user", "tell", "me", "about", "lora"]
    window_size = 6
    visible_tokens = keep_last_tokens(tokens, window_size)

    print("Context-window demo")
    print(f"All tokens:     {tokens}")
    print(f"Window size:    {window_size}")
    print(f"Visible tokens: {visible_tokens}")
    print()
    print("Takeaway:")
    print("  If the sequence gets too long, older tokens may fall out of the active context window.")


if __name__ == "__main__":
    main()
