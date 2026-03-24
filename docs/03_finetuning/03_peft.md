# PEFT

## Core idea

PEFT means Parameter-Efficient Fine-Tuning.

It is the broader family of methods that adapt a model by training a relatively
small number of parameters instead of the whole network.

LoRA is one example of PEFT.

## Why PEFT matters

PEFT is useful because it lowers the barrier to experimentation.

It often helps with:

- GPU memory limits
- storage limits
- faster iteration
- maintaining multiple task-specific adapters

## Practical intuition

If full fine-tuning asks:

"Can we update everything?"

PEFT asks:

"What is the smallest useful part we need to update?"

That change in mindset is important for real LLM engineering work.

## Where PEFT fits in this repository

For this learning path:

- SFT explains the basic training pipeline
- PEFT explains why we usually avoid updating the whole model
- LoRA and QLoRA are the practical methods you are likely to try first

## Common engineering benefit

PEFT makes it easier to:

- keep one base model
- train multiple adapters
- compare task-specific variants
- ship smaller artifacts

## See also

- `docs/03_finetuning/04_lora.md`
- `docs/03_finetuning/05_qlora.md`
- `scripts/run_lora.py`

## Why this matters in real LLM engineering

PEFT is central to modern open-model fine-tuning because it makes adaptation
more affordable and easier to operationalize than full-model retraining.
