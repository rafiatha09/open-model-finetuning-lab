# QLoRA

## Core idea

QLoRA combines LoRA with low-bit model loading so you can fine-tune large models
with much lower memory use than full-precision training.

## Mental model

LoRA reduces how many parameters you train.

QLoRA goes further by also reducing the memory footprint of the frozen base
model during training.

So the practical idea is:

- keep the base model quantized
- train LoRA adapters on top

## Why it is useful

QLoRA is attractive when:

- hardware is limited
- you still want to work with larger checkpoints
- full fine-tuning is unrealistic

## Practical constraints

QLoRA usually adds more dependency and hardware complexity than plain LoRA.

You often need:

- quantization support
- compatible libraries such as `bitsandbytes`
- hardware and platform support that matches the stack

So it is powerful, but not always the easiest first run.

## Repository stance

This repository prioritizes:

1. understanding SFT
2. running LoRA cleanly
3. extending into QLoRA once the earlier path is stable

That keeps the learning curve manageable.

## See also

- `docs/03_finetuning/04_lora.md`
- `docs/03_finetuning/03_peft.md`
- `configs/training/qlora.yaml`

## Why this matters in real LLM engineering

QLoRA is one of the most practical ways to fine-tune larger open models on
limited hardware, but it also introduces extra systems complexity that you need
to understand before depending on it.
