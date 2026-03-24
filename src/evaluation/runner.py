"""Evaluation runner for comparing two model variants on the same eval set."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from evaluation.dataset import EvalExample, load_eval_jsonl
from evaluation.metrics import score_prediction
from serving.config import ServingConfig
from serving.inference import build_inference_prompt, generate_response, load_model_bundle


@dataclass(frozen=True)
class EvaluationSummary:
    base_model: str
    candidate_model: str
    example_count: int
    base_metrics: dict[str, float]
    candidate_metrics: dict[str, float]
    candidate_better_count: int
    base_better_count: int
    ties: int
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return path


def _write_json(record: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def _score_for_comparison(metrics: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(metrics["exact_match"]),
        float(metrics["instruction_following_pass"]),
        float(metrics["token_overlap_f1"]),
        float(metrics["keyword_recall"]),
        -float(metrics["possible_hallucination"]),
        -float(metrics["repetition_ratio"]),
    )


def _compare_rows(base_metrics: dict[str, Any], candidate_metrics: dict[str, Any]) -> str:
    base_score = _score_for_comparison(base_metrics)
    candidate_score = _score_for_comparison(candidate_metrics)
    if candidate_score > base_score:
        return "candidate"
    if base_score > candidate_score:
        return "base"
    return "tie"


def _average_metric(records: list[dict[str, Any]], metric_name: str) -> float:
    if not records:
        return 0.0
    values = [float(record[metric_name]) for record in records]
    return sum(values) / len(values)


def _summarize_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    metric_names = [
        "exact_match",
        "token_overlap_f1",
        "keyword_recall",
        "instruction_following_pass",
        "repetition_ratio",
        "possible_hallucination",
        "sentence_count",
    ]
    return {metric_name: _average_metric(records, metric_name) for metric_name in metric_names}


def _build_review_markdown(rows: list[dict[str, Any]], summary: EvaluationSummary, base_name: str, candidate_name: str) -> str:
    lines = [
        "# Evaluation Review",
        "",
        f"- Base: `{base_name}`",
        f"- Candidate: `{candidate_name}`",
        f"- Examples: {summary.example_count}",
        f"- Candidate better: {summary.candidate_better_count}",
        f"- Base better: {summary.base_better_count}",
        f"- Ties: {summary.ties}",
        "",
    ]

    for row in rows:
        lines.extend(
            [
                f"## {row['id']}",
                "",
                f"Instruction: {row['instruction']}",
                f"Input: {row['input'] or '<empty>'}",
                f"Reference: {row['reference'] or '<empty>'}",
                f"Winner: {row['winner']}",
                "",
                f"### {base_name}",
                row["base_output"] or "<no text generated>",
                "",
                f"Metrics: {json.dumps(row['base_metrics'], ensure_ascii=True)}",
                "",
                f"### {candidate_name}",
                row["candidate_output"] or "<no text generated>",
                "",
                f"Metrics: {json.dumps(row['candidate_metrics'], ensure_ascii=True)}",
                "",
                "Manual review: [ ] candidate better  [ ] tie  [ ] base better",
                "Notes:",
                "",
            ]
        )

    return "\n".join(lines).strip() + "\n"


def run_evaluation(
    *,
    eval_set_path: str | Path,
    base_model: str,
    candidate_model: str,
    output_dir: str | Path,
    repo_root: Path,
    base_name: str = "base",
    candidate_name: str = "candidate",
    system_prompt: str = "",
    max_new_tokens: int = 80,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_examples: int | None = None,
) -> EvaluationSummary:
    examples = load_eval_jsonl(eval_set_path)
    if max_examples is not None:
        examples = examples[: max(1, max_examples)]

    base_config = ServingConfig(
        assistant_name=base_name,
        model_path=base_model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0.0,
        system_prompt=system_prompt,
    )
    candidate_config = ServingConfig(
        assistant_name=candidate_name,
        model_path=candidate_model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0.0,
        system_prompt=system_prompt,
    )

    base_bundle = load_model_bundle(base_config, repo_root)
    candidate_bundle = load_model_bundle(candidate_config, repo_root)

    rows: list[dict[str, Any]] = []
    base_scores: list[dict[str, Any]] = []
    candidate_scores: list[dict[str, Any]] = []

    candidate_better_count = 0
    base_better_count = 0
    ties = 0

    for example in examples:
        prompt = build_inference_prompt(
            instruction=example.instruction,
            input_text=example.input,
            system_prompt=system_prompt,
        )
        base_output = generate_response(base_bundle, prompt, base_config)
        candidate_output = generate_response(candidate_bundle, prompt, candidate_config)

        base_metrics = score_prediction(base_output, example)
        candidate_metrics = score_prediction(candidate_output, example)
        winner = _compare_rows(base_metrics, candidate_metrics)

        if winner == "candidate":
            candidate_better_count += 1
        elif winner == "base":
            base_better_count += 1
        else:
            ties += 1

        base_scores.append(base_metrics)
        candidate_scores.append(candidate_metrics)

        rows.append(
            {
                "id": example.id,
                "instruction": example.instruction,
                "input": example.input,
                "reference": example.reference,
                "tags": example.tags,
                "checks": example.checks,
                "prompt": prompt,
                "base_output": base_output,
                "candidate_output": candidate_output,
                "base_metrics": base_metrics,
                "candidate_metrics": candidate_metrics,
                "winner": winner,
            }
        )

    output_path = Path(output_dir)
    rows_path = _write_jsonl(rows, output_path / "comparison_rows.jsonl")

    summary = EvaluationSummary(
        base_model=base_model,
        candidate_model=candidate_model,
        example_count=len(rows),
        base_metrics=_summarize_metrics(base_scores),
        candidate_metrics=_summarize_metrics(candidate_scores),
        candidate_better_count=candidate_better_count,
        base_better_count=base_better_count,
        ties=ties,
        output_dir=str(output_path),
    )

    _write_json(summary.to_dict(), output_path / "summary.json")
    (output_path / "qualitative_review.md").write_text(
        _build_review_markdown(rows, summary, base_name, candidate_name),
        encoding="utf-8",
    )

    print(f"Wrote row-level outputs to {rows_path}")
    print(f"Wrote summary to {output_path / 'summary.json'}")
    print(f"Wrote qualitative review to {output_path / 'qualitative_review.md'}")
    print(
        "Comparison summary: "
        f"{candidate_name} better on {summary.candidate_better_count}, "
        f"{base_name} better on {summary.base_better_count}, "
        f"ties on {summary.ties}."
    )
    return summary
