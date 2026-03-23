# Model Loading

## Core idea

Model loading means turning saved model files into usable Python objects.

With Transformers, the usual entry points are:

- `AutoTokenizer.from_pretrained(...)`
- `AutoModelForCausalLM.from_pretrained(...)`

## The basic pattern

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("some-model")
model = AutoModelForCausalLM.from_pretrained("some-model")
```

That `"some-model"` can be:

- a Hub model ID
- a local folder path

## What gets loaded

For the tokenizer:

- vocabulary
- special tokens
- tokenizer config
- sometimes a chat template

For the model:

- architecture config
- learned weights
- generation-relevant settings like EOS or PAD token IDs

## Practical loading decisions

Important questions when loading a model:

- are you online or offline?
- do you want CPU or GPU inference?
- are you loading a base model or a fine-tuned checkpoint?
- does the model need a specific prompt format?

For this repository, the demos use tiny local assets so they run without a Hub download.

## Common beginner mistake

Loading succeeds, but generation still looks broken because:

- the prompt format is wrong
- the tokenizer does not match the model
- special tokens are missing or misconfigured

So "model loaded successfully" does not mean "system is ready."

## Practical checklist

After loading, confirm:

1. tokenizer vocabulary exists
2. model config looks reasonable
3. `model.eval()` is set for inference demos
4. a small prompt can be tokenized and generated

## See also

- `docs/02_open_models/01_huggingface_basics.md`
- `docs/02_open_models/05_chat_templates.md`
- `examples/hf_loading_demo.py`

## Why this matters in real LLM engineering

Open-model projects often fail at the "last mile" between checkpoint files and
working inference. Solid model-loading habits save time when switching models,
running offline, or serving fine-tuned checkpoints.
