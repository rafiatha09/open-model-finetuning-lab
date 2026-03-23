from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_phase1_docs_exist() -> None:
    expected = [
        ROOT / "docs/01_foundation/01_tokenization.md",
        ROOT / "docs/01_foundation/02_embeddings.md",
        ROOT / "docs/01_foundation/03_transformer_basics.md",
        ROOT / "docs/01_foundation/04_causal_masking.md",
        ROOT / "docs/01_foundation/05_decoder_only_models.md",
        ROOT / "docs/01_foundation/06_next_token_prediction.md",
        ROOT / "docs/01_foundation/07_training_vs_inference.md",
        ROOT / "docs/01_foundation/08_context_window_and_kv_cache.md",
        ROOT / "docs/01_foundation/09_self_check_qa.md",
    ]
    for path in expected:
        assert path.exists(), f"Missing Phase 1 doc: {path}"
        assert "## Why this matters in real LLM engineering" in path.read_text(encoding="utf-8")


def test_phase1_examples_run() -> None:
    examples = [
        ROOT / "examples/tokenizer_demo.py",
        ROOT / "examples/embeddings_demo.py",
        ROOT / "examples/attention_demo.py",
        ROOT / "examples/causal_mask_demo.py",
        ROOT / "examples/next_token_demo.py",
        ROOT / "examples/decoder_only_demo.py",
        ROOT / "examples/context_window_demo.py",
    ]
    for path in examples:
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip(), f"Example produced no output: {path}"
