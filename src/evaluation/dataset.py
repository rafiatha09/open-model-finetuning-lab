"""Helpers for loading small JSONL evaluation sets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalExample:
    id: str
    instruction: str
    input: str = ""
    reference: str = ""
    tags: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_eval_record(record: dict[str, Any], line_number: int) -> EvalExample:
    instruction = _clean_text(record.get("instruction"))
    if not instruction:
        raise ValueError(f"Evaluation record on line {line_number} must include a non-empty 'instruction'.")

    example_id = _clean_text(record.get("id")) or f"example-{line_number}"
    input_text = _clean_text(record.get("input"))
    reference = _clean_text(record.get("reference") or record.get("response") or record.get("expected_output"))

    raw_tags = record.get("tags") or []
    if not isinstance(raw_tags, list):
        raise ValueError(f"Evaluation record on line {line_number} must use a list for 'tags'.")
    tags = [_clean_text(tag) for tag in raw_tags if _clean_text(tag)]

    raw_checks = record.get("checks") or {}
    if not isinstance(raw_checks, dict):
        raise ValueError(f"Evaluation record on line {line_number} must use a mapping for 'checks'.")

    return EvalExample(
        id=example_id,
        instruction=instruction,
        input=input_text,
        reference=reference,
        tags=tags,
        checks=raw_checks,
    )


def load_eval_jsonl(path: str | Path) -> list[EvalExample]:
    target = Path(path)
    records: list[EvalExample] = []

    with target.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {target}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"Evaluation record on line {line_number} in {target} must be a JSON object.")
            records.append(normalize_eval_record(raw, line_number))

    if not records:
        raise ValueError(f"No valid evaluation records found in {target}")

    return records
