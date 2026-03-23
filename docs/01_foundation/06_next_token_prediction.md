# Next-Token Prediction

## Core idea

A decoder-only language model is trained to predict the next token in a
sequence.

That sounds simple, but it is the objective that drives everything else:

- language modeling
- instruction following after tuning
- chat-style generation
- domain adaptation through supervised examples

## Intuition

The model is not trained with an explicit symbolic rulebook for grammar,
reasoning, or facts. Instead, it is repeatedly asked:

"Given this prefix, what token is likely to come next?"

By doing this at very large scale, the model learns patterns that look like
knowledge, style, task behavior, and reasoning traces.

## Mental model

Take this sequence:

`"LoRA reduces training"`

The model tries to predict the next token, maybe:

`" cost"`

During training, the true next token is known, and the model is nudged to assign
more probability to it.

This happens across many positions in many sequences, not just once per example.

## Why this objective is surprisingly powerful

Predicting the next token forces the model to learn many useful regularities:

- syntax
- common phrases
- domain terminology
- discourse structure
- question-answer patterns

Instruction tuning does not replace next-token prediction. It reshapes the data
so the next-token objective teaches assistant-like behavior.

## Common engineering implications

This perspective helps when you ask:

- why prompt wording changes outputs
- why formatting training data matters
- why the model imitates patterns in examples
- why low-quality targets can teach bad behavior

It also explains why "teaching a skill" often means giving the model many good
prefix-to-target examples, not writing rules by hand.

## See also

- `docs/01_foundation/04_causal_masking.md`
- `docs/01_foundation/07_training_vs_inference.md`
- `docs/01_foundation/05_decoder_only_models.md`

## Why this matters in real LLM engineering

Next-token prediction is the training objective behind most open LLMs you will
use. If you keep that objective in mind, prompting, SFT, evaluation design, and
failure analysis become much easier to interpret.
