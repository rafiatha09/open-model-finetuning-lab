# Hugging Face Basics

## Core idea

Hugging Face Transformers gives you a standard Python interface for working with
open models, tokenizers, configs, and generation code.

For Phase 2, the most important idea is:

- a model is not enough on its own
- you usually need the tokenizer and model together

## The objects you will see most often

- `AutoTokenizer`: loads the tokenizer for a model
- `AutoModelForCausalLM`: loads a decoder-only generation model
- `AutoConfig`: loads model configuration

The `Auto*` classes are useful because they choose the right concrete class for
you from a model name or local model directory.

## The common workflow

The usual Hugging Face flow looks like this:

1. load the tokenizer
2. load the model
3. tokenize the prompt
4. run generation
5. decode tokens back into text

That is the basic loop behind many LLM demos and applications.

## Model names and local paths

You can load from:

- a model ID on Hugging Face Hub, like `gpt2`
- a local folder that already contains model files

For learning, both matter:

- Hub loading is common in real projects
- local loading is useful for offline work, custom checkpoints, and fine-tuned models

## Why tokenizers matter here too

The tokenizer is part of the model package, not an optional extra.

If you load the wrong tokenizer for a model, the token IDs will not line up with
what the model expects.

That means:

- inputs get encoded incorrectly
- outputs decode incorrectly
- behavior becomes unreliable

## Common beginner mistake

It is easy to think "I only need the weights."

In practice, you usually need:

- tokenizer files
- model config
- model weights
- sometimes generation or chat-template settings

## Practical rule of thumb

When you start with an open model, first confirm:

1. which tokenizer goes with it
2. whether it is a causal language model
3. whether the prompt should use a chat template
4. whether you are loading from the Hub or a local checkpoint

## See also

- `docs/02_open_models/02_model_loading.md`
- `docs/02_open_models/03_tokenizer_usage.md`
- `examples/hf_loading_demo.py`

## Why this matters in real LLM engineering

Most open-model work begins with loading the right tokenizer and model pair.
Understanding the Hugging Face object model makes inference, fine-tuning, and
checkpoint management much more predictable.
