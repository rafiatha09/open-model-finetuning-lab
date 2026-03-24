"""Run local instruction-style inference against a saved checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from serving.config import load_serving_config  # noqa: E402
from serving.inference import build_inference_prompt, generate_response, load_model_bundle  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local inference with a saved causal LM checkpoint.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/serving/local_assistant.yaml"),
        help="Path to a serving YAML config file.",
    )
    parser.add_argument("--model-path", help="Optional override for the model/checkpoint path.")
    parser.add_argument("--instruction", help="Instruction or question for the assistant.")
    parser.add_argument("--input", default="", help="Optional extra task context.")
    parser.add_argument("--system-prompt", help="Optional override for the serving system prompt.")
    parser.add_argument("--max-new-tokens", type=int, help="Optional override for the generation length.")
    parser.add_argument("--temperature", type=float, help="Optional override for sampling temperature.")
    parser.add_argument("--top-p", type=float, help="Optional override for nucleus sampling.")
    parser.add_argument("--show-prompt", action="store_true", help="Print the formatted prompt before generation.")
    return parser


def _get_instruction(provided_instruction: str | None) -> str:
    if provided_instruction and provided_instruction.strip():
        return provided_instruction.strip()
    instruction = input("Instruction: ").strip()
    if not instruction:
        raise ValueError("An instruction is required for inference.")
    return instruction


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_serving_config(args.config)
    if args.model_path:
        config = replace(config, model_path=args.model_path)
    if args.system_prompt is not None:
        config = replace(config, system_prompt=args.system_prompt)
    if args.temperature is not None:
        config = replace(config, temperature=args.temperature)
    if args.top_p is not None:
        config = replace(config, top_p=args.top_p)
    if args.max_new_tokens is not None:
        config = replace(config, max_new_tokens=args.max_new_tokens)

    instruction = _get_instruction(args.instruction)
    prompt = build_inference_prompt(
        instruction=instruction,
        input_text=args.input,
        system_prompt=config.system_prompt,
    )

    if args.show_prompt:
        print("Prompt")
        print("------")
        print(prompt)
        print()

    bundle = load_model_bundle(config, ROOT)
    response = generate_response(bundle, prompt, config)

    print(f"Assistant: {config.assistant_name}")
    print("Response:")
    print(response or "<no text generated>")


if __name__ == "__main__":
    main()
