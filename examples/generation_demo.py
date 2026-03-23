"""Offline generation demo showing greedy decoding vs sampling."""

from __future__ import annotations

import torch

from _hf_demo_utils import load_demo_assets


def decode_generation(tokenizer, sequence) -> str:
    return tokenizer.decode(sequence[0], skip_special_tokens=False)


def main() -> None:
    with load_demo_assets() as (_, tokenizer, model):
        prompt = "Tell me about LoRA"
        inputs = tokenizer(prompt, return_tensors="pt")

        greedy = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=5,
            pad_token_id=tokenizer.pad_token_id,
        )

        torch.manual_seed(11)
        sampled = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            max_new_tokens=5,
            pad_token_id=tokenizer.pad_token_id,
        )

        print("Generation demo")
        print(f"Prompt: {prompt!r}")
        print()
        print("Greedy decoding")
        print("  do_sample=False, max_new_tokens=5")
        print(f"  {decode_generation(tokenizer, greedy)!r}")
        print()
        print("Sampling")
        print("  do_sample=True, temperature=0.8, top_p=0.9, max_new_tokens=5")
        print(f"  {decode_generation(tokenizer, sampled)!r}")
        print()
        print("Parameter meaning")
        print("  temperature: controls how sharp or random sampling feels")
        print("  top_p: keeps only the most likely cumulative probability mass")
        print("  max_new_tokens: limits how many new tokens can be generated")
        print("  greedy vs sampling: deterministic pick vs probabilistic pick")


if __name__ == "__main__":
    main()
