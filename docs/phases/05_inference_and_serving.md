# Phase 5: Inference and Serving

## Goal

Expose the tuned model as a usable domain assistant.

## What to build here

- local inference helper
- CLI or API surface
- prompt and system-message conventions
- checkpoint loading from `models/`
- a repeatable command for local assistant testing

## Exit criteria

You can send a request to the assistant and receive a controlled response.

Starter command:

```bash
python scripts/run_inference.py \
  --config configs/serving/local_assistant.yaml \
  --instruction "Explain tokenizer usage in simple terms."
```
