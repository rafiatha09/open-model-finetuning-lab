"""Show how generation settings change output length and randomness."""

from __future__ import annotations

import torch

from _hf_demo_utils import load_demo_assets


def main() -> None:
    prompt = "Tell me about LoRA"

    with load_demo_assets() as (_, tokenizer, model):
        inputs = tokenizer(prompt, return_tensors="pt")

        short = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=2,
            pad_token_id=tokenizer.pad_token_id,
        )
        long = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=6,
            pad_token_id=tokenizer.pad_token_id,
        )

        torch.manual_seed(5)
        cooler = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            max_new_tokens=5,
            pad_token_id=tokenizer.pad_token_id,
        )
        torch.manual_seed(5)
        hotter = model.generate(
            **inputs,
            do_sample=True,
            temperature=1.2,
            top_p=0.9,
            max_new_tokens=5,
            pad_token_id=tokenizer.pad_token_id,
        )

        print("Generation controls demo")
        print(f"Prompt: {prompt!r}")
        print()
        print("max_new_tokens comparison:")
        print(f"  2 -> {tokenizer.decode(short[0], skip_special_tokens=False)!r}")
        print(f"  6 -> {tokenizer.decode(long[0], skip_special_tokens=False)!r}")
        print()
        print("temperature comparison with the same random seed:")
        print(f"  0.3 -> {tokenizer.decode(cooler[0], skip_special_tokens=False)!r}")
        print(f"  1.2 -> {tokenizer.decode(hotter[0], skip_special_tokens=False)!r}")
        print()
        print("Takeaway:")
        print("  max_new_tokens controls continuation length, while temperature changes how conservative or random sampling feels.")


if __name__ == "__main__":
    main()
