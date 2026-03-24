from pathlib import Path
import sys
import tempfile

from fastapi.testclient import TestClient
import torch


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from _hf_demo_utils import build_demo_model, build_demo_tokenizer  # noqa: E402
from omlab.inference.api import create_app  # noqa: E402
from omlab.inference.generate import GenerationRequest, TransformersGenerationBackend  # noqa: E402
from serving.config import ServingConfig  # noqa: E402


def test_phase5_docs_and_files_exist() -> None:
    expected = [
        ROOT / "docs/05_serving/01_quantization.md",
        ROOT / "docs/05_serving/02_vllm_and_tgi_basics.md",
        ROOT / "docs/05_serving/03_latency_vs_cost.md",
        ROOT / "docs/05_serving/04_batching.md",
        ROOT / "docs/05_serving/05_serving_api.md",
        ROOT / "docs/05_serving/06_self_check_qa.md",
        ROOT / "scripts/serve_model.py",
        ROOT / "src/omlab/inference/generate.py",
        ROOT / "src/omlab/inference/api.py",
        ROOT / "examples/serving_backend_demo.py",
        ROOT / "examples/serving_api_demo.py",
        ROOT / "examples/batching_demo.py",
    ]
    for path in expected:
        assert path.exists(), f"Missing Phase 5 material: {path}"
        if path.suffix == ".md":
            assert "## Why this matters in real LLM engineering" in path.read_text(encoding="utf-8")


def _write_demo_model(model_dir: Path) -> None:
    tokenizer = build_demo_tokenizer()
    model = build_demo_model(tokenizer.vocab_size)
    with torch.no_grad():
        model.lm_head.weight[18].add_(0.75)
        model.transformer.wte.weight[18].add_(0.75)
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir, max_shard_size="10GB", safe_serialization=False)


def test_phase5_backend_and_api_run_locally() -> None:
    with tempfile.TemporaryDirectory(prefix="phase5-serve-") as tmpdir:
        model_dir = Path(tmpdir) / "model"
        _write_demo_model(model_dir)

        config = ServingConfig(
            assistant_name="demo-assistant",
            model_path=str(model_dir),
            backend="transformers",
            max_new_tokens=8,
            do_sample=False,
        )
        backend = TransformersGenerationBackend(config=config, repo_root=ROOT)

        result = backend.generate(GenerationRequest(instruction="Tell me about LoRA"))
        assert result.prompt
        assert isinstance(result.text, str)

        app = create_app(backend=backend, assistant_name=config.assistant_name)
        client = TestClient(app)

        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        response = client.post(
            "/generate",
            json={"instruction": "Tell me about LoRA", "max_new_tokens": 8},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["backend"] == "transformers"
        assert "prompt" in payload

        batch = client.post(
            "/generate_batch",
            json={
                "requests": [
                    {"instruction": "Tell me about LoRA", "max_new_tokens": 8},
                    {"instruction": "What is LoRA", "max_new_tokens": 8},
                ]
            },
        )
        assert batch.status_code == 200
        assert batch.json()["count"] == 2


def test_phase5_examples_run() -> None:
    import subprocess

    examples = [
        (ROOT / "examples/serving_backend_demo.py", "Serving backend demo"),
        (ROOT / "examples/serving_api_demo.py", "Serving API demo"),
        (ROOT / "examples/batching_demo.py", "Batching demo"),
    ]
    for path, marker in examples:
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        assert marker in result.stdout
