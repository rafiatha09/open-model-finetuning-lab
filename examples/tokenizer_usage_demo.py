"""Show common tokenizer encode/decode patterns with Transformers."""

from __future__ import annotations

from _hf_demo_utils import load_demo_assets


def main() -> None:
    texts = [
        "Tell me about LoRA",
        "Hello world",
    ]

    with load_demo_assets() as (_, tokenizer, _):
        encoded = tokenizer(texts, padding=True, return_tensors="pt")
        decoded = tokenizer.batch_decode(encoded["input_ids"], skip_special_tokens=False)

        print("Tokenizer usage demo")
        print("Input texts:")
        for text in texts:
            print(f"  {text!r}")
        print()
        print("Encoded input IDs:")
        for row in encoded["input_ids"]:
            print(f"  {row.tolist()}")
        print()
        print("Attention masks:")
        for row in encoded["attention_mask"]:
            print(f"  {row.tolist()}")
        print()
        print("Decoded text:")
        for text in decoded:
            print(f"  {text!r}")
        print()
        print("Takeaway:")
        print("  Tokenizers handle encoding, padding, tensor output, and decoding as part of the normal inference workflow.")


if __name__ == "__main__":
    main()
