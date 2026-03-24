from pathlib import Path
import subprocess
import sys
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from _hf_demo_utils import build_demo_model, build_demo_tokenizer  # noqa: E402


def test_phase4_docs_and_eval_set_exist() -> None:
    expected = [
        ROOT / "docs/04_evaluation/01_eval_fundamentals.md",
        ROOT / "docs/04_evaluation/02_before_after_comparison.md",
        ROOT / "docs/04_evaluation/03_hallucination_checks.md",
        ROOT / "docs/04_evaluation/04_instruction_following_checks.md",
        ROOT / "docs/04_evaluation/05_error_analysis.md",
        ROOT / "docs/04_evaluation/06_self_check_qa.md",
        ROOT / "data/eval/sample_eval_set.jsonl",
        ROOT / "scripts/evaluate_model.py",
        ROOT / "examples/eval_demo.py",
        ROOT / "examples/eval_dataset_demo.py",
        ROOT / "examples/eval_metrics_demo.py",
        ROOT / "examples/qualitative_review_demo.py",
    ]
    for path in expected:
        assert path.exists(), f"Missing Phase 4 material: {path}"
        if path.suffix == ".md":
            assert "## Why this matters in real LLM engineering" in path.read_text(encoding="utf-8")


def _write_demo_model(model_dir: Path, *, bias_lora_token: bool) -> None:
    tokenizer = build_demo_tokenizer()
    model = build_demo_model(tokenizer.vocab_size)
    if bias_lora_token:
        with torch.no_grad():
            model.lm_head.weight[18].add_(0.75)
            model.transformer.wte.weight[18].add_(0.75)

    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir, max_shard_size="10GB", safe_serialization=False)


def test_phase4_evaluation_script_runs() -> None:
    with tempfile.TemporaryDirectory(prefix="phase4-eval-") as tmpdir:
        root = Path(tmpdir)
        base_dir = root / "base_model"
        candidate_dir = root / "candidate_model"
        output_dir = root / "eval_report"
        eval_set = root / "eval_set.jsonl"

        _write_demo_model(base_dir, bias_lora_token=False)
        _write_demo_model(candidate_dir, bias_lora_token=True)

        eval_set.write_text(
            (
                '{"id":"demo-1","instruction":"Tell me about LoRA","input":"","reference":"LoRA adapters reduce memory cost for fine tuning.","tags":["demo"],"checks":{"must_include":["LoRA"],"max_sentences":2}}\n'
                '{"id":"demo-2","instruction":"What is LoRA","input":"Keep it short.","reference":"LoRA adapters reduce memory cost for fine tuning.","tags":["demo"],"checks":{"must_include":["LoRA"],"max_sentences":2}}\n'
            ),
            encoding="utf-8",
        )

        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/evaluate_model.py"),
                "--eval-set",
                str(eval_set),
                "--base-model",
                str(base_dir),
                "--candidate-model",
                str(candidate_dir),
                "--output-dir",
                str(output_dir),
                "--max-new-tokens",
                "8",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "Evaluation summary" in result.stdout
        assert (output_dir / "comparison_rows.jsonl").exists()
        assert (output_dir / "summary.json").exists()
        assert (output_dir / "qualitative_review.md").exists()


def test_phase4_example_runs() -> None:
    examples = [
        (ROOT / "examples/eval_demo.py", "Eval demo complete"),
        (ROOT / "examples/eval_dataset_demo.py", "Evaluation dataset demo"),
        (ROOT / "examples/eval_metrics_demo.py", "Evaluation metrics demo"),
        (ROOT / "examples/qualitative_review_demo.py", "Qualitative review demo"),
    ]
    for path, marker in examples:
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        assert marker in result.stdout
