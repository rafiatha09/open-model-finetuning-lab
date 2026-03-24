"""Configuration loading for SFT, LoRA, and QLoRA workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    train_path: str
    validation_path: str
    max_length: int = 512
    mask_prompt_tokens: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    num_train_epochs: float = 1.0
    learning_rate: float = 2.0e-4
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    weight_decay: float = 0.0
    logging_steps: int = 10
    save_steps: int = 50
    eval_steps: int = 50
    seed: int = 42


@dataclass(frozen=True)
class LoraTuningConfig:
    enabled: bool = False
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class QuantizationConfig:
    enabled: bool = False
    load_in_4bit: bool = False
    compute_dtype: str = "float16"


@dataclass(frozen=True)
class FineTuningConfig:
    experiment_name: str
    method: str
    base_model: str
    output_dir: str
    dataset: DatasetConfig
    training: TrainingConfig
    lora: LoraTuningConfig = field(default_factory=LoraTuningConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    trust_remote_code: bool = False
    resume_from_checkpoint: str | None = None

    @property
    def uses_lora(self) -> bool:
        return self.method.lower() in {"lora", "qlora"} or self.lora.enabled

    @property
    def uses_qlora(self) -> bool:
        return self.method.lower() == "qlora" or self.quantization.enabled


def _require_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{key}' to be a mapping.")
    return value


def load_training_config(path: str | Path) -> FineTuningConfig:
    target = Path(path)
    with target.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Config file {target} must contain a YAML mapping.")

    dataset = DatasetConfig(**_require_mapping(raw, "dataset"))
    training = TrainingConfig(**_require_mapping(raw, "training"))
    lora = LoraTuningConfig(**_require_mapping(raw, "lora"))
    quantization = QuantizationConfig(**_require_mapping(raw, "quantization"))

    return FineTuningConfig(
        experiment_name=str(raw.get("experiment_name", "unnamed-experiment")),
        method=str(raw.get("method", "sft")).lower(),
        base_model=str(raw["base_model"]),
        output_dir=str(raw["output_dir"]),
        dataset=dataset,
        training=training,
        lora=lora,
        quantization=quantization,
        trust_remote_code=bool(raw.get("trust_remote_code", False)),
        resume_from_checkpoint=raw.get("resume_from_checkpoint"),
    )
