# LoRA

## Core idea

LoRA stands for Low-Rank Adaptation. Instead of updating every model weight, it
learns small adapter matrices that modify the model's behavior efficiently.

## Why people use it

LoRA is usually cheaper than full fine-tuning in:

- memory
- storage
- experimentation cost

That makes it a strong default for many practical open-model projects.

## Mental model

You keep the base model mostly frozen and train a much smaller set of adapter
parameters.

So the question becomes:

"How can we change behavior without retraining the whole model?"

LoRA is one common answer.

## Practical tradeoff

Compared with full SFT:

- it is usually lighter and easier to iterate on
- it may be easier to store and swap multiple adapters
- it depends on a PEFT layer instead of changing the full model weights

For this repository, LoRA is the preferred starting point once the SFT pipeline
is conceptually clear.

## Common terms

- `r`: low-rank size
- `alpha`: LoRA scaling factor
- `dropout`: regularization on LoRA updates
- `target_modules`: which model layers get adapters

## Practical advice

- start with a known target-module list for your model family
- keep the first run tiny
- compare LoRA output to the untuned baseline
- save the adapter separately from the full base model

## See also

- `docs/03_finetuning/03_peft.md`
- `docs/03_finetuning/05_qlora.md`
- `scripts/run_lora.py`

## Why this matters in real LLM engineering

LoRA is one of the most common fine-tuning techniques for open models because it
gives a strong balance between quality, cost, and iteration speed.
