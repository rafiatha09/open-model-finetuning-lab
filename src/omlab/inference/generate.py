"""Minimal generation backend abstraction for local serving."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from serving.config import ServingConfig, load_serving_config
from serving.inference import build_inference_prompt, generate_response, load_model_bundle


@dataclass(frozen=True)
class GenerationRequest:
    instruction: str
    input: str = ""
    system_prompt: str | None = None
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None


@dataclass(frozen=True)
class GenerationResult:
    text: str
    prompt: str
    backend: str
    model_path: str


class GenerationBackend(Protocol):
    backend_name: str
    supports_batching: bool

    def generate(self, request: GenerationRequest) -> GenerationResult:
        ...

    def generate_many(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        ...


@dataclass
class TransformersGenerationBackend:
    """Simple local backend backed by Hugging Face Transformers."""

    config: ServingConfig
    repo_root: Path

    def __post_init__(self) -> None:
        self.backend_name = "transformers"
        self.supports_batching = False
        self.bundle = load_model_bundle(self.config, self.repo_root)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        system_prompt = self.config.system_prompt if request.system_prompt is None else request.system_prompt
        prompt = build_inference_prompt(
            instruction=request.instruction,
            input_text=request.input,
            system_prompt=system_prompt,
        )
        text = generate_response(
            self.bundle,
            prompt,
            self.config,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        return GenerationResult(
            text=text,
            prompt=prompt,
            backend=self.backend_name,
            model_path=self.config.model_path,
        )

    def generate_many(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        # This backend serves requests sequentially today. vLLM or TGI can later
        # replace this with true continuous batching behind the same interface.
        return [self.generate(request) for request in requests]


def build_backend_from_config(
    *,
    config_path: str | Path,
    repo_root: Path,
    model_path: str | None = None,
    backend_name: str | None = None,
) -> tuple[ServingConfig, TransformersGenerationBackend]:
    config = load_serving_config(config_path)
    if model_path:
        config = replace(config, model_path=model_path)

    selected_backend = config.backend if backend_name is None else backend_name
    if selected_backend != "transformers":
        raise ValueError(
            f"Unsupported backend '{selected_backend}'. "
            "This starter layer currently supports only 'transformers'."
        )

    backend = TransformersGenerationBackend(config=config, repo_root=repo_root)
    return config, backend
