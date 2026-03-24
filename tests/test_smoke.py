from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_phase_roadmap_docs_exist() -> None:
    phase_docs = sorted((ROOT / "docs/phases").glob("*.md"))
    assert len(phase_docs) == 6


def test_sample_data_files_exist() -> None:
    sample_files = sorted((ROOT / "data/sample").glob("*.jsonl"))
    assert len(sample_files) >= 3
