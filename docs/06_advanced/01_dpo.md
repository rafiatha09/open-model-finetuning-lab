# DPO

## What it is

DPO stands for Direct Preference Optimization. Instead of training only on
"good" answers, it learns from preference pairs such as:

- chosen answer
- rejected answer

The model is trained to prefer the chosen one.

## When to use it after SFT / LoRA

Use DPO after you already have:

- a working SFT or LoRA baseline
- a reasonable prompt format
- examples of outputs where one answer is clearly better than another

DPO is usually a next step when "the model basically works, but I want better
helpfulness, style, or ranking between responses."

## Why people use it

DPO is attractive because it gives a more direct path from human preference data
to model behavior than plain supervised fine-tuning.


## Why this matters in real LLM engineering

DPO is one of the clearest bridges from basic fine-tuning to preference-based
post-training, which is why it is worth understanding even if it is not your
first production milestone.
