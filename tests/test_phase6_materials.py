from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_phase6_docs_exist() -> None:
    expected = [
        ROOT / "docs/06_advanced/01_dpo.md",
        ROOT / "docs/06_advanced/02_reward_models.md",
        ROOT / "docs/06_advanced/03_grpo_and_rl.md",
        ROOT / "docs/06_advanced/04_distillation.md",
        ROOT / "docs/06_advanced/05_self_check_qa.md",
        ROOT / "docs/phases/06_advanced_post_training.md",
    ]
    for path in expected:
        assert path.exists(), f"Missing Phase 6 material: {path}"
        assert "## Why this matters in real LLM engineering" in path.read_text(
            encoding="utf-8"
        )
