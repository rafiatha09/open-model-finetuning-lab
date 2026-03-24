from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def test_phase3_docs_and_configs_exist() -> None:
    expected = [
        ROOT / "docs/03_finetuning/01_dataset_formatting.md",
        ROOT / "docs/03_finetuning/02_sft.md",
        ROOT / "docs/03_finetuning/03_peft.md",
        ROOT / "docs/03_finetuning/04_lora.md",
        ROOT / "docs/03_finetuning/05_qlora.md",
        ROOT / "docs/03_finetuning/06_checkpointing.md",
        ROOT / "docs/03_finetuning/07_self_check_qa.md",
        ROOT / "configs/training/sft.yaml",
        ROOT / "configs/training/lora.yaml",
        ROOT / "configs/training/qlora.yaml",
    ]
    for path in expected:
        assert path.exists(), f"Missing Phase 3 material: {path}"
        if path.suffix == ".md":
            assert "## Why this matters in real LLM engineering" in path.read_text(encoding="utf-8")


def test_prepare_dataset_script_runs() -> None:
    with tempfile.TemporaryDirectory(prefix="phase3-dataset-") as tmpdir:
        output_dir = Path(tmpdir)
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/prepare_dataset.py"),
                "--input",
                str(ROOT / "data/sample/domain_assistant_examples.jsonl"),
                "--output-dir",
                str(output_dir),
                "--validation-size",
                "0.33",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "Prepared dataset" in result.stdout
        assert (output_dir / "train.jsonl").exists()
        assert (output_dir / "validation.jsonl").exists()


def test_phase3_training_scripts_support_dry_run() -> None:
    commands = [
        [sys.executable, str(ROOT / "scripts/run_sft.py"), "--config", str(ROOT / "configs/training/sft.yaml"), "--dry-run"],
        [sys.executable, str(ROOT / "scripts/run_lora.py"), "--config", str(ROOT / "configs/training/lora.yaml"), "--dry-run"],
    ]
    for command in commands:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "dry run" in result.stdout.lower()


def test_phase3_examples_run() -> None:
    examples = [
        ROOT / "examples/dataset_formatting_demo.py",
        ROOT / "examples/train_validation_split_demo.py",
        ROOT / "examples/training_plan_demo.py",
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
