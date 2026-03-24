"""Helpers for loading, validating, splitting, and formatting instruction data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any


@dataclass(frozen=True)
class InstructionExample:
    instruction: str
    input: str
    response: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_record(record: dict[str, Any]) -> InstructionExample:
    instruction = _clean_text(record.get("instruction"))
    response = _clean_text(record.get("response") or record.get("output"))
    input_text = _clean_text(record.get("input"))

    if not instruction:
        raise ValueError("Each record must include a non-empty 'instruction' field.")
    if not response:
        raise ValueError("Each record must include a non-empty 'response' or 'output' field.")

    return InstructionExample(
        instruction=instruction,
        input=input_text,
        response=response,
    )


def load_instruction_jsonl(path: Path | str) -> list[InstructionExample]:
    target = Path(path)
    records: list[InstructionExample] = []

    with target.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {target}") from exc
            records.append(normalize_record(raw))

    if not records:
        raise ValueError(f"No valid records found in {target}")

    return records


def train_validation_split(
    records: list[InstructionExample],
    validation_size: float | int = 0.2,
    seed: int = 42,
) -> tuple[list[InstructionExample], list[InstructionExample]]:
    if len(records) < 2:
        raise ValueError("Need at least 2 records to create a train/validation split.")

    shuffled = records[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    if isinstance(validation_size, float):
        if not 0.0 < validation_size < 1.0:
            raise ValueError("Float validation_size must be between 0 and 1.")
        validation_count = max(1, int(round(len(shuffled) * validation_size)))
    else:
        validation_count = int(validation_size)

    validation_count = min(max(1, validation_count), len(shuffled) - 1)
    validation_records = shuffled[:validation_count]
    train_records = shuffled[validation_count:]
    return train_records, validation_records


def write_instruction_jsonl(records: list[InstructionExample], path: Path | str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=True) + "\n")

    return target


def format_instruction_prompt(example: InstructionExample) -> str:
    parts = [f"### Instruction:\n{example.instruction}"]
    if example.input:
        parts.append(f"### Input:\n{example.input}")
    parts.append("### Response:\n")
    return "\n\n".join(parts)


def format_sft_text(example: InstructionExample, eos_token: str = "") -> str:
    text = format_instruction_prompt(example) + example.response
    if eos_token:
        text += eos_token
    return text
