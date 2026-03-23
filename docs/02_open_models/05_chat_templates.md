# Chat Templates

## Core idea

Many chat models do not expect raw plain text alone. They expect the conversation
to be serialized in a model-specific format.

A chat template is the rule that turns message dictionaries like:

```python
[
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Tell me about LoRA."},
]
```

into the exact text string or token sequence the model expects.

## Why this exists

Under the hood, chat models still perform next-token prediction over a sequence
of tokens.

The chat template defines how roles such as:

- system
- user
- assistant

are represented in that token sequence.

## The basic API

```python
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

Key idea:

- `tokenize=False` returns the formatted string
- `tokenize=True` returns token IDs
- `add_generation_prompt=True` appends the assistant prefix so generation starts in the right place

## Practical intuition

If the template is wrong, the model may:

- ignore the system instruction
- continue the user turn instead of answering
- produce awkward role markers
- behave much worse than expected

So chat formatting is part of prompt engineering, not just a convenience helper.

## Common beginner mistake

People often test a chat model with a plain string and then conclude:

- "the model is bad"
- "the checkpoint is broken"

Sometimes the real issue is simply that the prompt was not formatted with the
expected template.

## Practical checklist

When working with a chat model, confirm:

1. whether the tokenizer has a chat template
2. whether you should use a system message
3. whether `add_generation_prompt=True` is needed
4. what the final serialized text actually looks like

## See also

- `docs/02_open_models/03_tokenizer_usage.md`
- `docs/02_open_models/06_basic_prompting.md`
- `examples/chat_template_demo.py`

## Why this matters in real LLM engineering

Chat-template handling is one of the highest-leverage open-model details. A good
template can unlock strong behavior from a checkpoint, while a bad one can make
a good model look unusable.
