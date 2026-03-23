from open_model_finetuning_lab.roadmap import PHASES
from open_model_finetuning_lab.sample_data import load_jsonl


def test_roadmap_has_expected_phases() -> None:
    assert len(PHASES) == 7


def test_sample_data_loads() -> None:
    rows = load_jsonl()
    assert len(rows) >= 3
    assert "instruction" in rows[0]
    assert "response" in rows[0]
