"""Compare two model variants on a small JSONL evaluation set."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.runner import run_evaluation  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare a base model and a tuned model on a JSONL eval set.")
    parser.add_argument("--eval-set", required=True, help="Path to the evaluation JSONL file.")
    parser.add_argument("--base-model", required=True, help="Base model repo id, local path, or checkpoint path.")
    parser.add_argument("--candidate-model", required=True, help="Candidate or tuned model repo id or local path.")
    parser.add_argument("--output-dir", required=True, help="Directory where eval outputs will be written.")
    parser.add_argument("--base-name", default="base", help="Display name for the base model in reports.")
    parser.add_argument("--candidate-name", default="candidate", help="Display name for the tuned model in reports.")
    parser.add_argument("--system-prompt", default="", help="Optional shared system guidance added to every eval prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=80, help="Maximum number of generated tokens per example.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. Use 0.0 for greedy decoding.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p nucleus sampling value when temperature > 0.")
    parser.add_argument("--max-examples", type=int, help="Optional limit for a quick smoke test.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    summary = run_evaluation(
        eval_set_path=args.eval_set,
        base_model=args.base_model,
        candidate_model=args.candidate_model,
        output_dir=args.output_dir,
        repo_root=ROOT,
        base_name=args.base_name,
        candidate_name=args.candidate_name,
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_examples=args.max_examples,
    )

    print("Evaluation summary")
    print(f"Examples:            {summary.example_count}")
    print(f"Base model:          {summary.base_model}")
    print(f"Candidate model:     {summary.candidate_model}")
    print(f"Base avg overlap F1: {summary.base_metrics['token_overlap_f1']:.4f}")
    print(f"Candidate avg F1:    {summary.candidate_metrics['token_overlap_f1']:.4f}")
    print(f"Candidate better:    {summary.candidate_better_count}")
    print(f"Base better:         {summary.base_better_count}")
    print(f"Ties:                {summary.ties}")


if __name__ == "__main__":
    main()
