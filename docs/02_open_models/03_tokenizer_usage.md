# Tokenizer Usage

## Core idea

Before a model can generate, text must be converted into token IDs by the
tokenizer, and generated token IDs must be decoded back into text.

## The basic pattern

```python
inputs = tokenizer("Tell me about LoRA", return_tensors="pt")
outputs = model.generate(**inputs)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

That one pattern covers a lot of LLM application work.

## What tokenizers do in practice

They handle:

- splitting text into tokens
- mapping tokens to IDs
- adding special tokens when needed
- returning PyTorch-ready tensors
- decoding IDs back to readable text

## Useful tokenizer methods

- `tokenizer(...)`: encode text
- `tokenizer.encode(...)`: get token IDs
- `tokenizer.decode(...)`: decode one token sequence
- `tokenizer.batch_decode(...)`: decode many sequences
- `tokenizer.apply_chat_template(...)`: format chat messages for chat models

## Practical options to know

- `return_tensors="pt"`: return PyTorch tensors
- `truncation=True`: cut inputs to a maximum length
- `padding=True`: pad shorter sequences
- `skip_special_tokens=True`: hide special tokens when decoding

## Why this matters for debugging

If a model output looks strange, inspect:

- the tokenized input IDs
- the decoded result with and without special tokens
- whether prompt formatting introduced unwanted tokens

This often reveals problems faster than staring at final text alone.

## Practical takeaway

Treat tokenization as part of the inference pipeline, not a preprocessing detail.

Bad tokenization setup can make a good model look bad.

## See also

- `docs/02_open_models/05_chat_templates.md`
- `docs/02_open_models/04_generation_parameters.md`
- `examples/hf_loading_demo.py`

## Why this matters in real LLM engineering

Most inference bugs show up first at the tokenizer boundary. If you can inspect
encoding and decoding clearly, prompt debugging and model comparison become much
easier.
