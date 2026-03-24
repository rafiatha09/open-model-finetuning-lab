# Dataset Formatting

## Core idea

A fine-tuning dataset should make the desired behavior explicit in a consistent
instruction format.

For this repository, the simple baseline schema is JSONL with:

- `instruction`
- optional `input`
- `response`

## Example record

```json
{"instruction":"Explain LoRA in two bullets.","input":"","response":"LoRA trains small adapter weights instead of the full model. This reduces memory cost and makes experimentation easier."}
```

## Why format consistency matters

The model learns from repeated patterns.

If your dataset format is inconsistent, the model gets mixed signals about:

- where the task starts
- whether extra context exists
- what the response should look like

## Train and validation split

You should split data before training so you can check whether the model is:

- learning useful behavior
- overfitting to the training set

Even a small validation set is better than none.

## Prompt formatting for SFT

This repository uses a simple prompt style:

```text
### Instruction:
...

### Input:
...

### Response:
...
```

If `input` is empty, the input section can be omitted.

## Practical advice

- keep records short and readable at first
- make outputs high quality and consistent
- check a few formatted examples before training
- save the prepared split so experiments are reproducible

## See also

- `docs/03_finetuning/02_sft.md`
- `docs/03_finetuning/06_checkpointing.md`
- `scripts/prepare_dataset.py`

## Why this matters in real LLM engineering

Fine-tuning quality depends heavily on data formatting. A clean schema and a
reproducible split often matter more than adding more code to the training loop.
