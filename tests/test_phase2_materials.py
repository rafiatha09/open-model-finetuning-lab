from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_phase2_docs_exist() -> None:
    expected = [
        ROOT / "docs/02_open_models/01_huggingface_basics.md",
        ROOT / "docs/02_open_models/02_model_loading.md",
        ROOT / "docs/02_open_models/03_tokenizer_usage.md",
        ROOT / "docs/02_open_models/04_generation_parameters.md",
        ROOT / "docs/02_open_models/05_chat_templates.md",
        ROOT / "docs/02_open_models/06_basic_prompting.md",
        ROOT / "docs/02_open_models/07_self_check_qa.md",
    ]
    for path in expected:
        assert path.exists(), f"Missing Phase 2 doc: {path}"
        assert "## Why this matters in real LLM engineering" in path.read_text(encoding="utf-8")


def test_phase2_examples_run() -> None:
    examples = [
        ROOT / "examples/hf_loading_demo.py",
        ROOT / "examples/tokenizer_usage_demo.py",
        ROOT / "examples/generation_demo.py",
        ROOT / "examples/generation_controls_demo.py",
        ROOT / "examples/chat_template_demo.py",
        ROOT / "examples/prompting_demo.py",
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
