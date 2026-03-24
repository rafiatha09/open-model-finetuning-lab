"""Configuration loading for local inference and serving helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ServingConfig:
    assistant_name: str
    model_path: str
    backend: str = "transformers"
    interface: str = "cli"
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool | None = None
    system_prompt: str = ""
    trust_remote_code: bool = False

    def resolved_model_path(self, root: Path) -> Path:
        model_path = Path(self.model_path)
        if model_path.is_absolute():
            return model_path
        return root / model_path

    @property
    def should_sample(self) -> bool:
        if self.do_sample is not None:
            return self.do_sample
        return self.temperature > 0.0


def load_serving_config(path: str | Path) -> ServingConfig:
    target = Path(path)
    with target.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Config file {target} must contain a YAML mapping.")

    return ServingConfig(
        assistant_name=str(raw.get("assistant_name", "local-assistant")),
        model_path=str(raw.get("model_path", "models/domain-assistant-sft")),
        backend=str(raw.get("backend", "transformers")),
        interface=str(raw.get("interface", "cli")),
        max_new_tokens=int(raw.get("max_new_tokens", 128)),
        temperature=float(raw.get("temperature", 0.0)),
        top_p=float(raw.get("top_p", 1.0)),
        do_sample=raw.get("do_sample"),
        system_prompt=str(raw.get("system_prompt", "")).strip(),
        trust_remote_code=bool(raw.get("trust_remote_code", False)),
    )
