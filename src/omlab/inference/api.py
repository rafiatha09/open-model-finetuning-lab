"""Minimal FastAPI application for local model serving."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI
from pydantic import BaseModel, Field

from omlab.inference.generate import GenerationBackend, GenerationRequest


class GeneratePayload(BaseModel):
    instruction: str = Field(..., description="Instruction or question for the assistant.")
    input: str = Field("", description="Optional extra context for the instruction.")
    system_prompt: str | None = Field(None, description="Optional system override for this request.")
    max_new_tokens: int | None = Field(None, ge=1, description="Optional generation length override.")
    temperature: float | None = Field(None, ge=0.0, description="Optional sampling temperature override.")
    top_p: float | None = Field(None, ge=0.0, le=1.0, description="Optional top-p override.")


class BatchGeneratePayload(BaseModel):
    requests: list[GeneratePayload]


def create_app(*, backend: GenerationBackend, assistant_name: str) -> FastAPI:
    app = FastAPI(
        title="Open Model Finetuning Lab API",
        version="0.1.0",
        description="Minimal local serving API for a tuned domain assistant.",
    )

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "assistant_name": assistant_name,
            "backend": backend.backend_name,
            "supports_batching": backend.supports_batching,
        }

    @app.post("/generate")
    def generate(payload: GeneratePayload) -> dict[str, object]:
        result = backend.generate(GenerationRequest(**payload.model_dump()))
        return asdict(result)

    @app.post("/generate_batch")
    def generate_batch(payload: BatchGeneratePayload) -> dict[str, object]:
        requests = [GenerationRequest(**request.model_dump()) for request in payload.requests]
        results = backend.generate_many(requests)
        return {
            "backend": backend.backend_name,
            "count": len(results),
            "results": [asdict(result) for result in results],
        }

    return app


def run_api(*, backend: GenerationBackend, assistant_name: str, host: str = "127.0.0.1", port: int = 8000) -> None:
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "uvicorn is required to run the local API server. "
            'Install it with `python -m pip install -e ".[serve]"` '
            'or `python -m pip install uvicorn`.'
        ) from exc

    app = create_app(backend=backend, assistant_name=assistant_name)
    uvicorn.run(app, host=host, port=port, log_level="info")
