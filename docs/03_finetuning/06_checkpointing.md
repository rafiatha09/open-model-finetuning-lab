# Checkpointing

## Core idea

Checkpointing means saving intermediate training state so you can:

- resume interrupted runs
- compare model states across training
- keep the best or most recent artifact

## What usually gets saved

- model weights or adapters
- optimizer state
- scheduler state
- trainer state and metadata

For LoRA workflows, you often save the adapter separately from the base model.

## Why this matters early

Even small training runs benefit from checkpointing because:

- runs can fail
- experiments need comparison
- you may want to resume instead of restarting

## Practical tradeoffs

More checkpoints give you:

- better recovery
- more comparison points

But they also cost:

- disk space
- save time
- directory clutter

So checkpoint frequency should be deliberate.

## Repository guidance

For early runs:

- save often enough to recover progress
- keep output directories named clearly
- separate SFT, LoRA, and QLoRA experiments
- record which config produced each checkpoint

## See also

- `docs/03_finetuning/02_sft.md`
- `docs/03_finetuning/04_lora.md`
- `configs/training/sft.yaml`

## Why this matters in real LLM engineering

Training is not useful if you cannot reproduce, resume, or compare results.
Checkpoint discipline is part of making fine-tuning runs trustworthy.
