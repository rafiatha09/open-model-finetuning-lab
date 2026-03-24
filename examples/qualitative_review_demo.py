"""Show how to inspect saved evaluation outputs after a comparison run."""

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
    with tempfile.TemporaryDirectory(prefix="qual-review-demo-") as tmpdir:
        root = Path(tmpdir)
        base_dir = root / "base_model"
        candidate_dir = root / "candidate_model"
        output_dir = root / "review_outputs"
        eval_set = root / "eval_set.jsonl"

        _write_demo_model(base_dir, bias_lora_token=False)
        _write_demo_model(candidate_dir, bias_lora_token=True)

        eval_set.write_text(
            '{"id":"review-1","instruction":"Tell me about LoRA","input":"","reference":"LoRA adapters reduce memory cost for fine tuning.","tags":["demo"],"checks":{"must_include":["LoRA"],"max_sentences":2}}\n',
            encoding="utf-8",
        )

        run_evaluation(
            eval_set_path=eval_set,
            base_model=str(base_dir),
            candidate_model=str(candidate_dir),
            output_dir=output_dir,
            repo_root=ROOT,
            base_name="demo-base",
            candidate_name="demo-tuned",
            max_new_tokens=10,
        )

        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        review_preview = (output_dir / "qualitative_review.md").read_text(encoding="utf-8").splitlines()[:10]

        print("Qualitative review demo")
        print(f"Candidate better count: {summary['candidate_better_count']}")
        print("Review preview:")
        for line in review_preview:
            print(line)


if __name__ == "__main__":
    main()
