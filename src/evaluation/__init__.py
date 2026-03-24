"""Simple evaluation helpers for comparing model variants."""

from evaluation.dataset import EvalExample, load_eval_jsonl
from evaluation.runner import run_evaluation

__all__ = ["EvalExample", "load_eval_jsonl", "run_evaluation"]
