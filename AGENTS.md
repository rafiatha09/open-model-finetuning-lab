# AGENTS.md

## Project mission

This repository teaches and implements Goal A:
fine-tune an open model, evaluate it, and serve it as a domain assistant.

The repository should stay:

- Python-first
- beginner-to-intermediate friendly
- runnable from the command line
- educational rather than clever
- easy to extend into DPO and later post-training work

## How to approach work in this repo

Treat tasks like small GitHub issues.

Before editing:

1. restate the goal in one or two sentences
2. list the files you expect to change
3. state assumptions
4. define acceptance criteria

During implementation:

- prefer small, testable modules in `src/`
- prefer Python scripts in `scripts/` for runnable workflows
- keep configs explicit in `configs/`
- add comments only where they improve understanding
- update `README.md` or phase docs when behavior or workflow changes
- do not hide important logic only in notebooks

After editing:

1. summarize changes
2. list files changed
3. list commands to run
4. add verification notes
5. mention risks or limitations
6. recommend the next smallest task

## Repository conventions

### Code style

- favor explicitness over cleverness
- use descriptive function and file names
- keep data prep, training, evaluation, and serving separate
- default to standard library unless an external dependency adds clear value

### Architecture rules

- reusable code belongs in `src/open_model_finetuning_lab/`
- user-facing runnable commands belong in `scripts/`
- experiment settings belong in `configs/`
- reports, comparisons, and findings belong in `reports/`
- educational explanations belong in `docs/phases/`

### Dependency rules

- avoid heavy frameworks unless they directly support the learning goal
- add new packages only with a short reason in the PR/task summary
- prefer widely used OSS tools in the LLM ecosystem:
  `transformers`, `datasets`, `peft`, `trl`, `accelerate`, `evaluate`

## Preferred build order

Implement work in this order unless a task says otherwise:

1. define the assistant task and dataset shape
2. add baseline inference with an open model
3. add SFT training
4. add LoRA and QLoRA variants
5. add evaluation harness and comparison reports
6. add serving interface
7. extend into DPO or later post-training

## Definition of done

A task is complete when:

- code is readable and placed in the right layer
- the main command path is runnable
- docs reflect the new workflow
- obvious next work is written down

## Guardrails for future agents

- do not restructure the repo without updating the roadmap
- do not introduce hidden coupling between training and serving code
- do not replace simple scripts with frameworks unless there is a clear payoff
- do not delete educational docs when adding implementation details
