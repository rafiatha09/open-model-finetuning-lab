"""Training entry points shared by SFT and LoRA scripts."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from data.instruction_dataset import load_instruction_jsonl
from training.config import FineTuningConfig
from training.dataset import CausalLMCollator, InstructionSFTDataset


@dataclass(frozen=True)
class TrainingPlan:
    config_path: str
    method: str
    base_model: str
    train_examples: int
    validation_examples: int
    output_dir: str
    notes: list[str]


def _is_installed(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _require_dependency(module_name: str, install_hint: str) -> None:
    if not _is_installed(module_name):
        raise RuntimeError(
            f"Missing optional dependency '{module_name}'. "
            f"Install it with `{install_hint}` and rerun the command."
        )


def _load_pretrained_with_local_fallback(loader, model_name: str, **kwargs):
    """Prefer locally cached assets before attempting any network lookup."""
    model_path = Path(model_name)
    if model_path.exists():
        return loader.from_pretrained(model_path, **kwargs)

    try:
        cached_snapshot = snapshot_download(model_name, local_files_only=True)
        return loader.from_pretrained(cached_snapshot, **kwargs)
    except Exception:
        pass

    local_only_kwargs = dict(kwargs)
    local_only_kwargs["local_files_only"] = True
    try:
        return loader.from_pretrained(model_name, **local_only_kwargs)
    except Exception:
        return loader.from_pretrained(model_name, **kwargs)


def build_training_plan(config: FineTuningConfig, config_path: str) -> TrainingPlan:
    train_records = load_instruction_jsonl(config.dataset.train_path)
    validation_records = load_instruction_jsonl(config.dataset.validation_path)

    notes = [
        f"method={config.method}",
        f"mask_prompt_tokens={config.dataset.mask_prompt_tokens}",
        f"max_length={config.dataset.max_length}",
    ]
    if config.uses_lora:
        notes.append("LoRA adapters enabled")
    if config.uses_qlora:
        notes.append("QLoRA-style quantization requested")

    return TrainingPlan(
        config_path=config_path,
        method=config.method,
        base_model=config.base_model,
        train_examples=len(train_records),
        validation_examples=len(validation_records),
        output_dir=config.output_dir,
        notes=notes,
    )


def _prepare_tokenizer(model_name: str, trust_remote_code: bool = False):
    tokenizer = _load_pretrained_with_local_fallback(
        AutoTokenizer,
        model_name,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return tokenizer


def _prepare_model(config: FineTuningConfig):
    model_kwargs: dict[str, Any] = {"trust_remote_code": config.trust_remote_code}

    if config.uses_qlora:
        _require_dependency("bitsandbytes", "python -m pip install bitsandbytes")
        from transformers import BitsAndBytesConfig

        compute_dtype = getattr(torch, config.quantization.compute_dtype)
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=config.quantization.load_in_4bit,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model_kwargs["device_map"] = "auto"

    model = _load_pretrained_with_local_fallback(
        AutoModelForCausalLM,
        config.base_model,
        **model_kwargs,
    )

    if config.uses_lora:
        _require_dependency("peft", "python -m pip install peft")
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=config.lora.target_modules,
        )
        model = get_peft_model(model, lora_config)

    return model


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _build_dataloaders(config: FineTuningConfig, tokenizer):
    train_records = load_instruction_jsonl(config.dataset.train_path)
    validation_records = load_instruction_jsonl(config.dataset.validation_path)

    train_dataset = InstructionSFTDataset(
        train_records,
        tokenizer=tokenizer,
        max_length=config.dataset.max_length,
        mask_prompt_tokens=config.dataset.mask_prompt_tokens,
    )
    validation_dataset = InstructionSFTDataset(
        validation_records,
        tokenizer=tokenizer,
        max_length=config.dataset.max_length,
        mask_prompt_tokens=config.dataset.mask_prompt_tokens,
    )

    collator = CausalLMCollator(tokenizer=tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.training.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    return train_loader, validation_loader


def _evaluate(model, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch_to_device(batch, device)
            outputs = model(**batch)
            losses.append(float(outputs.loss.detach().cpu()))
    model.train()
    if not losses:
        return 0.0
    return sum(losses) / len(losses)


def _save_checkpoint(model, tokenizer, output_dir: str | Path, step: int | None = None) -> Path:
    checkpoint_dir = Path(output_dir)
    if step is not None:
        checkpoint_dir = checkpoint_dir / f"checkpoint-step-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    return checkpoint_dir


def run_training(config: FineTuningConfig, *, dry_run: bool = False, config_path: str = "<memory>") -> TrainingPlan:
    plan = build_training_plan(config, config_path=config_path)
    if dry_run:
        return plan

    tokenizer = _prepare_tokenizer(config.base_model, trust_remote_code=config.trust_remote_code)
    model = _prepare_model(config)
    device = _resolve_device()
    model.to(device)

    train_loader, validation_loader = _build_dataloaders(config, tokenizer)
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    total_update_steps = max(
        1,
        math.ceil(len(train_loader) * config.training.num_train_epochs / config.training.gradient_accumulation_steps),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=total_update_steps,
    )

    torch.manual_seed(config.training.seed)
    global_step = 0
    optimizer.zero_grad()
    model.train()

    for epoch_index in range(int(math.ceil(config.training.num_train_epochs))):
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = _move_batch_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss / config.training.gradient_accumulation_steps
            loss.backward()

            should_step = (
                batch_index % config.training.gradient_accumulation_steps == 0
                or batch_index == len(train_loader)
            )
            if not should_step:
                continue

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % config.training.logging_steps == 0:
                print(f"[train] epoch={epoch_index + 1} step={global_step} loss={float(loss.detach().cpu()):.4f}")

            if global_step % config.training.eval_steps == 0:
                validation_loss = _evaluate(model, validation_loader, device)
                print(f"[eval] step={global_step} validation_loss={validation_loss:.4f}")

            if global_step % config.training.save_steps == 0:
                checkpoint_path = _save_checkpoint(model, tokenizer, config.output_dir, step=global_step)
                print(f"[save] checkpoint={checkpoint_path}")

    final_validation_loss = _evaluate(model, validation_loader, device)
    print(f"[eval] final_validation_loss={final_validation_loss:.4f}")
    _save_checkpoint(model, tokenizer, config.output_dir)
    return plan
