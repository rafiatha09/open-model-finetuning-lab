# Training vs Inference

## Core idea

Training and inference use the same model, but they solve different problems.

- training updates parameters
- inference uses fixed parameters to generate or score outputs

Confusing those two modes makes many LLM design choices look mysterious when
they are actually straightforward.

## Intuition

During training, the model is learning from examples. During inference, the
model is applying what it has already learned.

That means the priorities are different:

- training cares about gradients, losses, optimizer state, and data quality
- inference cares about latency, memory, throughput, decoding, and caching

## Mental model

Training loop:

1. feed tokenized examples
2. compute predictions
3. compare predictions to targets
4. compute gradients
5. update weights

Inference loop:

1. feed a prompt
2. compute next-token probabilities
3. choose a token according to a decoding strategy
4. append it to the prompt
5. repeat until stopping

The model architecture is related, but the runtime concerns are very different.

## Why this matters for LLM workflows

It explains common surprises:

- a model that is cheap to serve may still be expensive to fine-tune
- a model that trains with teacher forcing still generates one token at a time
- a fine-tuned model can improve quality while increasing serving complexity
- quantization might be a serving choice, a training choice, or both

## Teacher forcing vs generation

In training, the model usually sees the true previous tokens from the dataset.
In inference, it sees its own generated history.

That gap matters because mistakes can compound during generation even if
training loss looked good.

## Common engineering implications

Understanding the split helps you make better decisions about:

- hardware allocation
- what to log during experiments
- when to optimize decoding
- when a poor result is a data problem versus a serving problem

## See also

- `docs/01_foundation/06_next_token_prediction.md`
- `docs/01_foundation/08_context_window_and_kv_cache.md`
- `docs/01_foundation/05_decoder_only_models.md`

## Why this matters in real LLM engineering

Many practical LLM tradeoffs come from whether you are optimizing training or
inference. Keeping those modes separate helps you choose the right tooling,
metrics, and performance optimizations.
