"""Offline demonstration of the evaluation flow using tiny local model assets."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

EXAMPLES = ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from _hf_demo_utils import build_demo_model, build_demo_tokenizer  # noqa: E402
from evaluation.runner import run_evaluation  # noqa: E402


def _write_demo_model(model_dir: Path, *, bias_lora_token: bool) -> None:
    tokenizer = build_demo_tokenizer()
    model = build_demo_model(tokenizer.vocab_size)
    if bias_lora_token:
        with torch.no_grad():
            model.lm_head.weight[18].add_(0.75)
            model.transformer.wte.weight[18].add_(0.75)

    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir, max_shard_size="10GB", safe_serialization=False)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="eval-demo-") as tmpdir:
        root = Path(tmpdir)
        base_dir = root / "base_model"
        candidate_dir = root / "candidate_model"
        output_dir = root / "reports"
        eval_set_path = root / "eval_set.jsonl"

        _write_demo_model(base_dir, bias_lora_token=False)
        _write_demo_model(candidate_dir, bias_lora_token=True)

        eval_rows = [
            {
                "id": "demo-lora",
                "instruction": "Tell me about LoRA",
                "input": "",
                "reference": "LoRA adapters reduce memory cost for fine tuning.",
                "tags": ["demo", "lora"],
                "checks": {"must_include": ["LoRA", "memory"], "max_sentences": 2},
            },
            {
                "id": "demo-question",
                "instruction": "What is LoRA",
                "input": "Keep it short.",
                "reference": "LoRA adapters reduce memory cost for fine tuning.",
                "tags": ["demo", "question"],
                "checks": {"must_include": ["LoRA"], "max_sentences": 2},
            },
        ]
        with eval_set_path.open("w", encoding="utf-8") as handle:
            for row in eval_rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

        summary = run_evaluation(
            eval_set_path=eval_set_path,
            base_model=str(base_dir),
            candidate_model=str(candidate_dir),
            output_dir=output_dir,
            repo_root=ROOT,
            base_name="demo-base",
            candidate_name="demo-tuned",
            max_new_tokens=12,
        )

        print("Eval demo complete")
        print(f"Examples: {summary.example_count}")
        print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
