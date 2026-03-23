"""Offline Hugging Face loading demo with tiny local assets."""

from __future__ import annotations

from _hf_demo_utils import load_demo_assets


def main() -> None:
    with load_demo_assets() as (model_dir, tokenizer, model):
        sample_text = "Tell me about LoRA"
        encoded = tokenizer(sample_text, return_tensors="pt")

        print("Hugging Face loading demo")
        print(f"Loaded from local path: {model_dir}")
        print()
        print(f"Tokenizer class: {tokenizer.__class__.__name__}")
        print(f"Model class:     {model.__class__.__name__}")
        print(f"Vocab size:      {tokenizer.vocab_size}")
        print()
        print(f"Input text: {sample_text!r}")
        print(f"Input IDs:  {encoded['input_ids'][0].tolist()}")
        print()
        print("Takeaway:")
        print("  In practice you load a tokenizer and model together, then pass tokenized tensors into generation.")


if __name__ == "__main__":
    main()
