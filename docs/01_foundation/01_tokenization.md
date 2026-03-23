# Tokenization

## Core idea

A language model does not read raw text directly. It reads tokens, which are
chunks of text mapped to integer IDs.

Those chunks are usually not whole words. A token might be:

- a full word like `model`
- part of a word like `token` and `ization`
- punctuation like `.`
- whitespace patterns like a leading space before a word

## Intuition

Tokenization is the bridge between human text and the model's internal world.
If text is the user-facing layer, tokens are the model-facing layer.

You can think of a tokenizer as a compression-friendly text parser. It tries to
split text into units that are useful enough to represent meaning, while still
keeping the vocabulary size manageable.

## Mental model

Imagine the sentence:

`Fine-tuning small open models is practical.`

A tokenizer might turn it into something conceptually like:

`["Fine", "-", "tuning", " small", " open", " models", " is", " practical", "."]`

Then each token is converted into an integer ID.

The model never sees the original sentence string. It sees a sequence of IDs.

## What to remember

- different tokenizers split the same text differently
- token count is not the same as word count
- pricing, latency, and context-window usage are all token-based
- prompt formatting choices can change token usage a lot

## Common engineering implications

If you are debugging a model, tokenization often explains surprising behavior:

- a prompt is longer than expected because punctuation and whitespace become tokens
- a structured format works better because its token boundaries are consistent
- a domain term performs poorly because it gets split into awkward rare pieces

For fine-tuning, tokenization affects:

- maximum sequence length
- memory usage
- batching efficiency
- whether examples are truncated

## Simple rule of thumb

When an LLM workflow behaves strangely, ask:

1. how is the text tokenized?
2. how many tokens are we actually sending?
3. are important spans being truncated or fragmented?

## See also

- `examples/tokenizer_demo.py`
- `docs/01_foundation/08_context_window_and_kv_cache.md`
- `docs/01_foundation/06_next_token_prediction.md`

## Why this matters in real LLM engineering

Tokenization shapes cost, latency, prompt length, truncation, and even model
quality on domain-specific text. If you do not understand tokenization, it is
easy to misdiagnose prompt problems as model problems.
