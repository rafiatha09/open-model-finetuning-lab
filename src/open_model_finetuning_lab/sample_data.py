import json
from pathlib import Path

from open_model_finetuning_lab.paths import DATA_DIR


SAMPLE_DATA_PATH = DATA_DIR / "sample" / "domain_assistant_examples.jsonl"


def load_jsonl(path: Path | None = None) -> list[dict]:
    """Load a JSONL file into a list of dictionaries."""
    target = path or SAMPLE_DATA_PATH
    rows: list[dict] = []

    with target.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {target}") from exc

    return rows
