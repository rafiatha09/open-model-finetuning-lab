"""Show how a tokenizer chat template formats role-based messages."""

from __future__ import annotations

from _hf_demo_utils import load_demo_assets


def main() -> None:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about LoRA."},
    ]

    with load_demo_assets() as (_, tokenizer, _):
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )

        print("Chat template demo")
        print("Messages:")
        for message in messages:
            print(f"  {message['role']}: {message['content']}")
        print()
        print("Formatted prompt:")
        print(f"  {formatted_text}")
        print()
        print("Token IDs:")
        print(f"  {token_ids['input_ids']}")
        print()
        print("Takeaway:")
        print("  Chat templates turn role-based messages into the exact text format a chat model expects.")


if __name__ == "__main__":
    main()
