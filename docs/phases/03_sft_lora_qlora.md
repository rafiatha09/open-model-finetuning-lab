# Phase 3: SFT / LoRA / QLoRA

## Goal

Fine-tune efficiently on a focused instruction dataset.

## What to build here

- a training-ready dataset schema
- a minimal SFT script
- LoRA support
- QLoRA support when hardware is limited

## Phase 3 materials

Docs:

- `docs/03_finetuning/01_dataset_formatting.md`
- `docs/03_finetuning/02_sft.md`
- `docs/03_finetuning/03_peft.md`
- `docs/03_finetuning/04_lora.md`
- `docs/03_finetuning/05_qlora.md`
- `docs/03_finetuning/06_checkpointing.md`
- `docs/03_finetuning/07_self_check_qa.md`

Configs:

- `configs/training/sft.yaml`
- `configs/training/lora.yaml`
- `configs/training/qlora.yaml`

Scripts:

- `scripts/prepare_dataset.py`
- `scripts/run_sft.py`
- `scripts/run_lora.py`

Examples:

- `examples/dataset_formatting_demo.py`
- `examples/train_validation_split_demo.py`
- `examples/training_plan_demo.py`

Source modules:

- `src/data/`
- `src/training/`

## Exit criteria

You can train a small experiment end to end and save artifacts reproducibly.
