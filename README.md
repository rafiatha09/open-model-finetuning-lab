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
6. Advanced post-training and future extensions

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
│   ├── 02_open_models/
│   ├── 03_finetuning/
│   ├── 04_evaluation/
│   ├── 05_serving/
│   ├── 06_advanced/
│   └── phases/
├── experiments/
├── models/
├── prompts/
├── reports/
├── scripts/
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── serving/
│   └── training/
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

For Phase 2 Hugging Face examples:

```bash
python -m pip install -e ".[dev,llm]"
```

For Phase 3 training scripts:

```bash
python -m pip install -e ".[dev,llm]"
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

- overview in [`docs/phases/02_open_model_usage.md`](docs/phases/02_open_model_usage.md)
- guided notes in `docs/02_open_models/`
- runnable Hugging Face examples in `examples/`

Phase 2 learning path:

1. [`docs/02_open_models/01_huggingface_basics.md`](docs/02_open_models/01_huggingface_basics.md)
2. [`docs/02_open_models/02_model_loading.md`](docs/02_open_models/02_model_loading.md)
3. [`docs/02_open_models/03_tokenizer_usage.md`](docs/02_open_models/03_tokenizer_usage.md)
4. [`docs/02_open_models/04_generation_parameters.md`](docs/02_open_models/04_generation_parameters.md)
5. [`docs/02_open_models/05_chat_templates.md`](docs/02_open_models/05_chat_templates.md)
6. [`docs/02_open_models/06_basic_prompting.md`](docs/02_open_models/06_basic_prompting.md)
7. [`docs/02_open_models/07_self_check_qa.md`](docs/02_open_models/07_self_check_qa.md)
8. [`examples/hf_loading_demo.py`](examples/hf_loading_demo.py)
9. [`examples/tokenizer_usage_demo.py`](examples/tokenizer_usage_demo.py)
10. [`examples/generation_demo.py`](examples/generation_demo.py)
11. [`examples/generation_controls_demo.py`](examples/generation_controls_demo.py)
12. [`examples/chat_template_demo.py`](examples/chat_template_demo.py)
13. [`examples/prompting_demo.py`](examples/prompting_demo.py)

Run the Phase 2 examples:

```bash
python examples/hf_loading_demo.py
python examples/tokenizer_usage_demo.py
python examples/generation_demo.py
python examples/generation_controls_demo.py
python examples/chat_template_demo.py
python examples/prompting_demo.py
```

Notes:

- the demos use `transformers` and tiny offline local assets created at runtime
- the goal is API understanding, not model quality

### Phase 3: SFT / LoRA / QLoRA

Start with supervised fine-tuning on a small domain dataset, then introduce
parameter-efficient training so compute requirements stay realistic.

Deliverables:

- overview in [`docs/phases/03_sft_lora_qlora.md`](docs/phases/03_sft_lora_qlora.md)
- guided notes in `docs/03_finetuning/`
- config-driven training scripts and modules

Phase 3 learning path:

1. [`docs/03_finetuning/01_dataset_formatting.md`](docs/03_finetuning/01_dataset_formatting.md)
2. [`docs/03_finetuning/02_sft.md`](docs/03_finetuning/02_sft.md)
3. [`docs/03_finetuning/03_peft.md`](docs/03_finetuning/03_peft.md)
4. [`docs/03_finetuning/04_lora.md`](docs/03_finetuning/04_lora.md)
5. [`docs/03_finetuning/05_qlora.md`](docs/03_finetuning/05_qlora.md)
6. [`docs/03_finetuning/06_checkpointing.md`](docs/03_finetuning/06_checkpointing.md)
7. [`docs/03_finetuning/07_self_check_qa.md`](docs/03_finetuning/07_self_check_qa.md)
8. [`configs/training/sft.yaml`](configs/training/sft.yaml)
9. [`configs/training/lora.yaml`](configs/training/lora.yaml)
10. [`configs/training/qlora.yaml`](configs/training/qlora.yaml)
11. [`examples/dataset_formatting_demo.py`](examples/dataset_formatting_demo.py)
12. [`examples/train_validation_split_demo.py`](examples/train_validation_split_demo.py)
13. [`examples/training_plan_demo.py`](examples/training_plan_demo.py)
14. [`scripts/prepare_dataset.py`](scripts/prepare_dataset.py)
15. [`scripts/run_sft.py`](scripts/run_sft.py)
16. [`scripts/run_lora.py`](scripts/run_lora.py)

Training flow:

```bash
python scripts/prepare_dataset.py \
  --input data/sample/domain_assistant_examples.jsonl \
  --output-dir data/processed/domain_assistant

python examples/dataset_formatting_demo.py
python examples/train_validation_split_demo.py
python examples/training_plan_demo.py

python scripts/run_sft.py --config configs/training/sft.yaml --dry-run
python scripts/run_lora.py --config configs/training/lora.yaml --dry-run
```

Notes:

- use the prepared dataset format before touching the training loop
- prefer LoRA first for practical experimentation
- treat QLoRA as the memory-efficient extension once LoRA is clear
- actual LoRA/QLoRA runs require optional dependencies such as `peft`

### Phase 4: Evaluation

Measure more than loss. Add task-specific metrics, qualitative review, and a
simple evaluation harness to compare base vs. tuned models.

Deliverables:

- overview in [`docs/phases/04_evaluation.md`](docs/phases/04_evaluation.md)
- guided notes in `docs/04_evaluation/`
- a sample eval set, evaluation script, and a runnable demo

Phase 4 learning path:

1. [`docs/04_evaluation/01_eval_fundamentals.md`](docs/04_evaluation/01_eval_fundamentals.md)
2. [`docs/04_evaluation/02_before_after_comparison.md`](docs/04_evaluation/02_before_after_comparison.md)
3. [`docs/04_evaluation/03_hallucination_checks.md`](docs/04_evaluation/03_hallucination_checks.md)
4. [`docs/04_evaluation/04_instruction_following_checks.md`](docs/04_evaluation/04_instruction_following_checks.md)
5. [`docs/04_evaluation/05_error_analysis.md`](docs/04_evaluation/05_error_analysis.md)
6. [`docs/04_evaluation/06_self_check_qa.md`](docs/04_evaluation/06_self_check_qa.md)
7. [`data/eval/sample_eval_set.jsonl`](data/eval/sample_eval_set.jsonl)
8. [`scripts/evaluate_model.py`](scripts/evaluate_model.py)
9. [`examples/eval_dataset_demo.py`](examples/eval_dataset_demo.py)
10. [`examples/eval_metrics_demo.py`](examples/eval_metrics_demo.py)
11. [`examples/eval_demo.py`](examples/eval_demo.py)
12. [`examples/qualitative_review_demo.py`](examples/qualitative_review_demo.py)

Run the offline evaluation demo:

```bash
python examples/eval_dataset_demo.py
python examples/eval_metrics_demo.py
python examples/eval_demo.py
python examples/qualitative_review_demo.py
```

Run a real model-vs-model comparison:

```bash
python scripts/evaluate_model.py \
  --eval-set data/eval/sample_eval_set.jsonl \
  --base-model models/domain-assistant-sft/checkpoint-step-20 \
  --candidate-model models/domain-assistant-sft \
  --output-dir reports/eval/sft_compare \
  --max-examples 3
```

What the script writes:

- `comparison_rows.jsonl` for row-level outputs and metrics
- `summary.json` for aggregate metrics
- `qualitative_review.md` for manual inspection

### Phase 5: Inference and serving

Package the tuned model behind a small local interface, then expose it as a
domain assistant with clear prompt/system behavior.

Deliverables:

- overview in [`docs/phases/05_inference_and_serving.md`](docs/phases/05_inference_and_serving.md)
- guided notes in `docs/05_serving/`
- a minimal local API server and backend abstraction

Phase 5 learning path:

1. [`docs/05_serving/01_quantization.md`](docs/05_serving/01_quantization.md)
2. [`docs/05_serving/02_vllm_and_tgi_basics.md`](docs/05_serving/02_vllm_and_tgi_basics.md)
3. [`docs/05_serving/03_latency_vs_cost.md`](docs/05_serving/03_latency_vs_cost.md)
4. [`docs/05_serving/04_batching.md`](docs/05_serving/04_batching.md)
5. [`docs/05_serving/05_serving_api.md`](docs/05_serving/05_serving_api.md)
6. [`docs/05_serving/06_self_check_qa.md`](docs/05_serving/06_self_check_qa.md)
7. [`scripts/run_inference.py`](scripts/run_inference.py)
8. [`scripts/serve_model.py`](scripts/serve_model.py)
9. [`src/omlab/inference/generate.py`](src/omlab/inference/generate.py)
10. [`src/omlab/inference/api.py`](src/omlab/inference/api.py)
11. [`examples/serving_backend_demo.py`](examples/serving_backend_demo.py)
12. [`examples/serving_api_demo.py`](examples/serving_api_demo.py)
13. [`examples/batching_demo.py`](examples/batching_demo.py)

Run local inference against the saved SFT checkpoint:

```bash
python examples/serving_backend_demo.py
python examples/serving_api_demo.py
python examples/batching_demo.py

python scripts/run_inference.py \
  --config configs/serving/local_assistant.yaml \
  --instruction "Explain LoRA for an ML engineer new to LLM fine-tuning."
```

Note:

- the current default checkpoint is a tiny learning model, so expect the inference path to work before the answer quality becomes strong

Add optional task context:

```bash
python scripts/run_inference.py \
  --config configs/serving/local_assistant.yaml \
  --instruction "Summarize KV cache." \
  --input "Keep it to 2 short sentences."
```

Inspect the exact prompt sent to the model:

```bash
python scripts/run_inference.py \
  --config configs/serving/local_assistant.yaml \
  --instruction "What is tokenization?" \
  --show-prompt
```

Run the local serving API:

```bash
python scripts/serve_model.py \
  --config configs/serving/local_assistant.yaml \
  --host 127.0.0.1 \
  --port 8000
```

Then send a request:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"instruction":"Explain LoRA for a beginner.","max_new_tokens":40}'
```

### Phase 6: Advanced post-training and future extensions

After the supervised path is stable, extend the lab with post-training ideas
such as DPO, reward models, GRPO/RL, and distillation. This phase is for future
extension, not for the first production version.

Deliverables:

- advanced bridge docs in `docs/06_advanced/`
- clear notes on when each advanced method becomes useful after SFT / LoRA
- a practical boundary between core Goal A work and later upgrades

Phase 6 learning path:

1. [`docs/06_advanced/01_dpo.md`](docs/06_advanced/01_dpo.md)
2. [`docs/06_advanced/02_reward_models.md`](docs/06_advanced/02_reward_models.md)
3. [`docs/06_advanced/03_grpo_and_rl.md`](docs/06_advanced/03_grpo_and_rl.md)
4. [`docs/06_advanced/04_distillation.md`](docs/06_advanced/04_distillation.md)
5. [`docs/06_advanced/05_self_check_qa.md`](docs/06_advanced/05_self_check_qa.md)

## What is already included

- a clean starter package with a tiny CLI
- sample JSONL data for a future domain assistant
- Phase 1 and Phase 2 learning paths with linked docs and demos
- a Phase 3 fine-tuning scaffold with configs, scripts, and reusable modules
- phase docs and config placeholders
- an `AGENTS.md` file for future issue-style work

## What should be built next

The next smallest meaningful task is:

1. choose one real open instruct model for the first baseline
2. expand the instruction dataset from sample size to a meaningful train/eval split
3. run a real LoRA dry run, then a small training job
4. add a simple evaluation script that compares base vs tuned outputs

That sequence keeps the lab grounded and makes training measurable instead of
jumping straight from scaffolding to unchecked fine-tuning.
