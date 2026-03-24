# Supervised Fine-Tuning (SFT)

## Core idea

Supervised fine-tuning teaches a model by showing it prompt-response examples
and updating the model so it predicts those responses better.

For this repository, that usually means:

- instruction-style prompt
- desired assistant response
- next-token prediction over the response text

## Mental model

If the base model already knows language, SFT is how you push it toward:

- your task format
- your response style
- your domain vocabulary

SFT does not create intelligence from nothing. It reshapes the model toward the
behavior shown in your dataset.

## The simple workflow

1. prepare an instruction dataset
2. split into train and validation
3. format each record into prompt plus response text
4. tokenize the examples
5. train with a causal language modeling objective
6. save checkpoints and compare outputs

## What SFT is good for

- teaching a domain-specific assistant style
- making outputs more consistent
- improving performance on narrow recurring tasks
- aligning a base model to your prompt format

## What SFT is not

- a substitute for good prompting
- a guarantee of factual correctness
- the best default choice when LoRA or QLoRA can do the same job cheaper

That is why this phase treats full SFT as the baseline concept, but recommends
parameter-efficient fine-tuning first in many practical cases.

## Practical advice

- start with a tiny dataset and dry-run the pipeline
- inspect formatted examples before training
- keep a validation split from the beginning
- compare against the untuned model baseline

## See also

- `docs/03_finetuning/01_dataset_formatting.md`
- `docs/03_finetuning/04_lora.md`
- `scripts/run_sft.py`

## Why this matters in real LLM engineering

SFT is the most direct path from a labeled instruction dataset to a tuned model.
Even if you later prefer LoRA or QLoRA, understanding SFT gives you the basic
mental model for how post-training pipelines are structured.
