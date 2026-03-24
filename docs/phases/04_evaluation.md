# Phase 4: Evaluation

## Goal

Measure whether tuning improved the assistant in ways that matter.

## What to build here

- a baseline-vs-tuned comparison script
- a small curated evaluation set
- a report template for wins, failures, and regressions
- lightweight heuristic metrics
- manual review artifacts you can inspect after every run

## Phase 4 materials

Docs:

- `docs/04_evaluation/01_eval_fundamentals.md`
- `docs/04_evaluation/02_before_after_comparison.md`
- `docs/04_evaluation/03_hallucination_checks.md`
- `docs/04_evaluation/04_instruction_following_checks.md`
- `docs/04_evaluation/05_error_analysis.md`
- `docs/04_evaluation/06_self_check_qa.md`

Dataset:

- `data/eval/sample_eval_set.jsonl`

Script:

- `scripts/evaluate_model.py`

Examples:

- `examples/eval_dataset_demo.py`
- `examples/eval_metrics_demo.py`
- `examples/eval_demo.py`
- `examples/qualitative_review_demo.py`

## Exit criteria

You can compare at least two model variants with readable evidence.

Starter command:

```bash
python scripts/evaluate_model.py \
  --eval-set data/eval/sample_eval_set.jsonl \
  --base-model models/domain-assistant-sft/checkpoint-step-20 \
  --candidate-model models/domain-assistant-sft \
  --output-dir reports/eval/sft_compare \
  --max-examples 3
```
