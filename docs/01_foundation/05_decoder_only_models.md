# Decoder-Only Models

## Core idea

Most modern open LLMs used for chat and instruction following are decoder-only
transformers.

They generate text one token at a time by repeatedly predicting the next token
from the tokens already seen.

## Intuition

If your goal is generation, decoder-only models are a natural fit:

- they read the prompt from left to right
- they predict one next token
- they append it
- they repeat

That loop is simple, scalable, and aligns well with how chat generation works.

## How they relate to the transformer

A decoder-only model is not a completely different architecture. It is a
transformer configured for autoregressive generation with causal masking.

The important ingredients are:

- token embeddings
- repeated transformer blocks
- causal attention
- a final projection to vocabulary logits

## Mental model

Prompt:

`"Write a short summary of LoRA:"`

The model does not generate the whole answer at once. It generates:

1. first next token
2. then the next token after that
3. then the next one
4. until it hits a stop condition

At each step, the previously generated tokens become part of the context.

## Why chat models still fit this pattern

Even if the interface looks conversational, the model is still doing next-token
prediction over a serialized text prompt that contains system, user, and
assistant content in some format.

So many "chat" behaviors are really prompt-formatting and fine-tuning choices
built on top of a decoder-only base.

## Common engineering implications

This helps explain:

- why prompt templates matter so much
- why output can drift if the model starts generating in the wrong format
- why stop tokens and max token limits matter
- why serving performance is sensitive to generation length

## See also

- `docs/01_foundation/03_transformer_basics.md`
- `docs/01_foundation/04_causal_masking.md`
- `docs/01_foundation/07_training_vs_inference.md`

## Why this matters in real LLM engineering

Decoder-only models are the default working surface for open LLM application
engineering. Understanding their generation loop makes prompting, serving, and
fine-tuning decisions much more concrete.
