"""Compare a vague prompt and a more structured prompt."""

from __future__ import annotations

from _hf_demo_utils import load_demo_assets


def prompt_length(tokenizer, text: str) -> int:
    return len(tokenizer(text)["input_ids"])


def main() -> None:
    vague_prompt = "Explain LoRA"
    structured_prompt = (
        "Explain LoRA for an ML engineer new to LLM fine-tuning. "
        "Use 3 short bullets and mention memory savings."
    )

    with load_demo_assets() as (_, tokenizer, _):
        vague_length = prompt_length(tokenizer, vague_prompt)
        structured_length = prompt_length(tokenizer, structured_prompt)

        print("Prompting demo")
        print("Vague prompt:")
        print(f"  {vague_prompt}")
        print(f"  token count: {vague_length}")
        print()
        print("Structured prompt:")
        print(f"  {structured_prompt}")
        print(f"  token count: {structured_length}")
        print()
        print("Why the structured prompt is usually better:")
        print("  It gives the model clearer task, audience, format, and scope information.")
        print()
        print("Takeaway:")
        print("  Better prompts often improve outputs before you need any fine-tuning.")


if __name__ == "__main__":
    main()
