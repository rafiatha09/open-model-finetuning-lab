# Phase 3 Self-Check Q&A

This file is a concise review sheet for `03_finetuning`.

## Core Questions

### 1. What is supervised fine-tuning (SFT)?

- SFT trains a model on prompt-response examples so it predicts the desired responses better.

### 2. What is the simple instruction dataset format used in this repository?

- JSONL with `instruction`, optional `input`, and `response`.

### 3. Why do we keep a validation split?

- To check generalization and avoid judging progress only on training data.

### 4. What is PEFT?

- Parameter-Efficient Fine-Tuning: adapting a model by training a small subset of parameters.

### 5. What is LoRA?

- A PEFT method that trains low-rank adapter weights instead of the whole model.

### 6. What is QLoRA?

- LoRA plus low-bit model loading to reduce memory use further.

### 7. Why does this repository prefer LoRA or QLoRA over full fine-tuning?

- They are usually cheaper, lighter, and easier to iterate on.

### 8. What does checkpointing give you during training?

- Resume capability, experiment comparison, and saved artifacts.

### 9. Why should prompt formatting be consistent across the dataset?

- The model learns patterns from the formatting, so inconsistent structure teaches inconsistent behavior.

### 10. What does `--dry-run` help with in the training scripts?

- It validates config and data flow before any model loading or training starts.

## Applied Questions

### 11. Why is a tiny end-to-end run better than jumping straight into a large fine-tuning job?

- It catches data, config, and formatting mistakes early and cheaply.

### 12. Why can data quality matter more than training complexity?

- A clean dataset teaches useful behavior, while a noisy dataset teaches the wrong patterns faster.

### 13. Why might a full fine-tuning run be the wrong first choice?

- It costs more memory and compute when LoRA may already be enough.

### 14. Why should you inspect formatted samples before training?

- To verify the exact text the model will learn from.

### 15. Why is config-driven training useful?

- It makes experiments reproducible, comparable, and easier to rerun.

## Short Explain-Like-I’m-Teaching

### 16. Explain LoRA in plain English.

- LoRA teaches a model through small adapter weights instead of retraining the whole model.

### 17. Explain why QLoRA is attractive but more complex.

- It saves memory, but usually needs extra dependencies and hardware support.

### 18. Explain why checkpoint names and output directories matter.

- Clear names make it easier to compare runs, resume training, and avoid confusion later.

## Mini Challenge

### 19. What is the cleanest Phase 3 workflow in this repository from raw JSONL to training?

- Prepare the dataset split.
- Review the YAML config.
- Run `run_sft.py` or `run_lora.py` with `--dry-run`.
- Then run a real training job once the plan looks correct.

## Why this matters in real LLM engineering

Fine-tuning is not just about calling a trainer. It depends on clean data,
repeatable configs, careful checkpointing, and choosing the lightest training
method that solves the problem.
