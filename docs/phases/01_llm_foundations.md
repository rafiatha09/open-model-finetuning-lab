# Phase 1: LLM Foundations

## Goal

Understand the minimum theory needed to make good engineering decisions later.

## What to learn here

- tokens and tokenization
- next-token prediction
- context windows
- prompting vs. fine-tuning
- decoding basics
- instruction tuning at a high level

## What to build here

- a glossary of core terms
- a short note on when not to fine-tune
- tiny examples that connect theory to later repo phases

## Phase 1 materials

Foundation docs:

- `docs/01_foundation/01_tokenization.md`
- `docs/01_foundation/02_embeddings.md`
- `docs/01_foundation/03_transformer_basics.md`
- `docs/01_foundation/04_causal_masking.md`
- `docs/01_foundation/05_decoder_only_models.md`
- `docs/01_foundation/06_next_token_prediction.md`
- `docs/01_foundation/07_training_vs_inference.md`
- `docs/01_foundation/08_context_window_and_kv_cache.md`
- `docs/01_foundation/09_self_check_qa.md`

Runnable examples:

- `examples/tokenizer_demo.py`
- `examples/embeddings_demo.py`
- `examples/attention_demo.py`
- `examples/causal_mask_demo.py`
- `examples/next_token_demo.py`
- `examples/decoder_only_demo.py`
- `examples/context_window_demo.py`

## Exit criteria

You can explain why a base model, a prompt, and a tuned model may all behave
differently on the same task.
