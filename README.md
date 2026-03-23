# open-model-finetuning-lab

`open-model-finetuning-lab` is a Python-first learning repository for Goal A:
fine-tune an open model, evaluate it, and serve it as a domain assistant.

It is designed for an ML engineer moving into LLM engineering, with a structure
that stays simple today and extends cleanly into deeper post-training later.

## What this repo covers

This repository is organized as a staged lab:

1. LLM foundations
2. Open-model usage
3. Fine-tuning with SFT, LoRA, and QLoRA
4. Evaluation
5. Inference and serving
6. Deployment basics
7. Later extension into DPO and advanced post-training

Each phase has:

- a short roadmap doc in `docs/phases/`
- a place in `src/` for reusable code
- lightweight config stubs in `configs/`
- simple runnable entry points for early smoke checks

## Repository structure

```text
open-model-finetuning-lab/
├── AGENTS.md
├── README.md
├── examples/
├── pyproject.toml
├── configs/
│   ├── evaluation/
│   ├── serving/
│   └── training/
├── data/
│   ├── processed/
│   ├── raw/
│   └── sample/
├── docs/
│   ├── 01_foundation/
│   └── phases/
├── experiments/
├── models/
├── prompts/
├── reports/
├── scripts/
├── src/
│   └── open_model_finetuning_lab/
└── tests/
```

## Why this structure

- `src/` holds reusable Python code instead of hiding logic in notebooks.
- `scripts/` holds runnable task entry points for local learning and smoke tests.
- `configs/` keeps training, evaluation, and serving settings explicit.
- `data/`, `models/`, `experiments/`, and `reports/` make the ML workflow visible.
- `docs/phases/` explains what each stage is for before implementation gets deep.

## Quickstart

Create a virtual environment and install the package in editable mode:

```bash
cd open-model-finetuning-lab
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Or install directly from `pyproject.toml`:

```bash
python -m pip install -e ".[dev]"
```

Run the starter commands:

```bash
python -m open_model_finetuning_lab roadmap
python -m open_model_finetuning_lab check
python scripts/preview_sample_data.py
```

## Learning roadmap

### Phase 1: LLM foundations

Build intuition for tokenization, next-token prediction, instruction tuning,
context windows, decoding, and why fine-tuning is not always the first tool.

Deliverables:

- overview in [`docs/phases/01_llm_foundations.md`](docs/phases/01_llm_foundations.md)
- guided notes in `docs/01_foundation/`
- simple runnable examples in `examples/`

Phase 1 learning path:

1. [`docs/01_foundation/01_tokenization.md`](docs/01_foundation/01_tokenization.md)
2. [`docs/01_foundation/02_embeddings.md`](docs/01_foundation/02_embeddings.md)
3. [`docs/01_foundation/03_transformer_basics.md`](docs/01_foundation/03_transformer_basics.md)
4. [`docs/01_foundation/04_causal_masking.md`](docs/01_foundation/04_causal_masking.md)
5. [`docs/01_foundation/05_decoder_only_models.md`](docs/01_foundation/05_decoder_only_models.md)
6. [`docs/01_foundation/06_next_token_prediction.md`](docs/01_foundation/06_next_token_prediction.md)
7. [`docs/01_foundation/07_training_vs_inference.md`](docs/01_foundation/07_training_vs_inference.md)
8. [`docs/01_foundation/08_context_window_and_kv_cache.md`](docs/01_foundation/08_context_window_and_kv_cache.md)
9. [`docs/01_foundation/09_self_check_qa.md`](docs/01_foundation/09_self_check_qa.md)
10. [`examples/tokenizer_demo.py`](examples/tokenizer_demo.py)
11. [`examples/embeddings_demo.py`](examples/embeddings_demo.py)
12. [`examples/attention_demo.py`](examples/attention_demo.py)
13. [`examples/causal_mask_demo.py`](examples/causal_mask_demo.py)
14. [`examples/next_token_demo.py`](examples/next_token_demo.py)
15. [`examples/decoder_only_demo.py`](examples/decoder_only_demo.py)
16. [`examples/context_window_demo.py`](examples/context_window_demo.py)

Suggested order:

- read the first three docs to build vocabulary
- run the tokenizer, embedding, and attention demos
- read causal masking, decoder-only models, and next-token prediction together
- run the causal-mask, next-token, and decoder-only demos
- finish with training vs inference and context window / KV cache, then run the context-window demo
- rerun the examples and explain the outputs in your own words

### Phase 2: Open-model usage

Learn how to select a model, run local inference, format prompts, and compare
small domain-assistant tasks before training anything.

Deliverables:

- prompt and baseline workflow
- model-selection notes
- local inference scripts

### Phase 3: SFT / LoRA / QLoRA

Start with supervised fine-tuning on a small domain dataset, then introduce
parameter-efficient training so compute requirements stay realistic.

Deliverables:

- dataset schema
- training config templates
- reproducible training entry point

### Phase 4: Evaluation

Measure more than loss. Add task-specific metrics, qualitative review, and a
simple evaluation harness to compare base vs. tuned models.

Deliverables:

- baseline evaluation config
- prompt-based eval examples
- report template in `reports/`

### Phase 5: Inference and serving

Package the tuned model behind a small local interface, then expose it as a
domain assistant with clear prompt/system behavior.

Deliverables:

- inference helpers
- serving config
- local API or CLI assistant

### Phase 6: Deployment basics

Prepare the repo so later deployment is straightforward: artifact naming,
environment handling, config separation, and evaluation before release.

Deliverables:

- deployment checklist
- packaging notes
- simple serving lifecycle doc

### Phase 7: DPO and advanced post-training

After the supervised path is stable, extend the lab with preference data,
ranking, and post-training methods such as DPO.

Deliverables:

- preference dataset schema
- experiment tracking conventions
- comparison plan between SFT and post-training

## What is already included

- a clean starter package with a tiny CLI
- sample JSONL data for a future domain assistant
- a Phase 1 foundations learning path with linked docs and demos
- phase docs and config placeholders
- an `AGENTS.md` file for future issue-style work

## What should be built next

The next smallest meaningful task is:

1. define the first domain assistant use case
2. expand the sample dataset into a small training-ready instruction dataset
3. add a baseline inference script against one open model
4. add an evaluation script that compares baseline outputs on 10 to 20 examples

That sequence keeps the lab educational and grounded. It avoids jumping into
fine-tuning before the task, data format, and baseline quality are clear.
