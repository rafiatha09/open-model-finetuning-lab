"""Small heuristic metrics for comparing generations on an eval set."""

from __future__ import annotations

import re
from typing import Any

from evaluation.dataset import EvalExample


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+")


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def exact_match(prediction: str, reference: str) -> float:
    if not reference.strip():
        return 0.0
    return float(prediction.strip().lower() == reference.strip().lower())


def token_overlap_f1(prediction: str, reference: str) -> float:
    prediction_tokens = _tokenize(prediction)
    reference_tokens = _tokenize(reference)
    if not prediction_tokens or not reference_tokens:
        return 0.0

    prediction_counts: dict[str, int] = {}
    for token in prediction_tokens:
        prediction_counts[token] = prediction_counts.get(token, 0) + 1

    reference_counts: dict[str, int] = {}
    for token in reference_tokens:
        reference_counts[token] = reference_counts.get(token, 0) + 1

    overlap = 0
    for token, count in prediction_counts.items():
        overlap += min(count, reference_counts.get(token, 0))

    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def keyword_recall(prediction: str, keywords: list[str]) -> float:
    cleaned_keywords = [keyword.strip().lower() for keyword in keywords if keyword and keyword.strip()]
    if not cleaned_keywords:
        return 1.0

    lowered_prediction = prediction.lower()
    hits = sum(1 for keyword in cleaned_keywords if keyword in lowered_prediction)
    return hits / len(cleaned_keywords)


def repetition_ratio(prediction: str) -> float:
    tokens = _tokenize(prediction)
    if len(tokens) < 2:
        return 0.0
    return 1.0 - (len(set(tokens)) / len(tokens))


def sentence_count(text: str) -> int:
    parts = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(text) if part.strip()]
    return max(1, len(parts)) if text.strip() else 0


def instruction_following_pass(prediction: str, example: EvalExample) -> bool:
    checks = example.checks
    must_include = checks.get("must_include") or []
    if must_include and keyword_recall(prediction, list(must_include)) < 1.0:
        return False

    max_sentences = checks.get("max_sentences")
    if max_sentences is not None and sentence_count(prediction) > int(max_sentences):
        return False

    return True


def possible_hallucination(prediction: str, reference: str) -> bool:
    """Very rough warning signal, not a factuality metric."""
    if not reference.strip():
        return False

    overlap = token_overlap_f1(prediction, reference)
    prediction_tokens = _tokenize(prediction)
    reference_tokens = _tokenize(reference)
    return len(prediction_tokens) >= max(8, len(reference_tokens) + 3) and overlap < 0.15


def score_prediction(prediction: str, example: EvalExample) -> dict[str, Any]:
    checks = example.checks
    must_include = list(checks.get("must_include") or [])

    return {
        "exact_match": exact_match(prediction, example.reference),
        "token_overlap_f1": token_overlap_f1(prediction, example.reference),
        "keyword_recall": keyword_recall(prediction, must_include),
        "instruction_following_pass": float(instruction_following_pass(prediction, example)),
        "repetition_ratio": repetition_ratio(prediction),
        "possible_hallucination": float(possible_hallucination(prediction, example.reference)),
        "sentence_count": sentence_count(prediction),
    }
