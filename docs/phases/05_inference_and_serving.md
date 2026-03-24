# Phase 5: Inference and Serving

## Goal

Expose the tuned model as a usable domain assistant.

## What to build here

- local inference helper
- CLI or API surface
- prompt and system-message conventions
- checkpoint loading from `models/`
- a repeatable command for local assistant testing
- a serving backend that can later swap to vLLM or TGI

## Phase 5 materials

Docs:

- `docs/05_serving/01_quantization.md`
- `docs/05_serving/02_vllm_and_tgi_basics.md`
- `docs/05_serving/03_latency_vs_cost.md`
- `docs/05_serving/04_batching.md`
- `docs/05_serving/05_serving_api.md`
- `docs/05_serving/06_self_check_qa.md`

Scripts:

- `scripts/run_inference.py`
- `scripts/serve_model.py`

Source modules:

- `src/omlab/inference/generate.py`
- `src/omlab/inference/api.py`

Examples:

- `examples/serving_backend_demo.py`
- `examples/serving_api_demo.py`
- `examples/batching_demo.py`

## Exit criteria

You can send a request to the assistant and receive a controlled response.

Starter command:

```bash
python scripts/run_inference.py \
  --config configs/serving/local_assistant.yaml \
  --instruction "Explain tokenizer usage in simple terms."
```

API command:

```bash
python scripts/serve_model.py \
  --config configs/serving/local_assistant.yaml \
  --host 127.0.0.1 \
  --port 8000
```
