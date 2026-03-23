import argparse
from pathlib import Path

from open_model_finetuning_lab.paths import CONFIGS_DIR, DATA_DIR, DOCS_DIR, REPO_ROOT
from open_model_finetuning_lab.roadmap import PHASES
from open_model_finetuning_lab.sample_data import SAMPLE_DATA_PATH, load_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="open-model-finetuning-lab",
        description="Small utilities for navigating the open-model-finetuning-lab scaffold.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("roadmap", help="Print the staged roadmap.")
    subparsers.add_parser("check", help="Validate the initial repository scaffold.")
    subparsers.add_parser("preview-data", help="Show the bundled sample dataset.")
    subparsers.add_parser("next-task", help="Print the recommended next implementation task.")

    return parser


def print_roadmap() -> None:
    for phase in PHASES:
        print(f"Phase {phase.number}: {phase.name}")
        print(f"  Goal: {phase.goal}")
        print(f"  Next output: {phase.next_output}")


def check_required_paths() -> list[Path]:
    required = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "AGENTS.md",
        CONFIGS_DIR / "training" / "sft_lora_qlora.yaml",
        CONFIGS_DIR / "evaluation" / "basic_eval.yaml",
        CONFIGS_DIR / "serving" / "local_assistant.yaml",
        DOCS_DIR / "phases" / "01_llm_foundations.md",
        DOCS_DIR / "phases" / "07_post_training_dpo.md",
        DATA_DIR / "sample" / "domain_assistant_examples.jsonl",
    ]
    return [path for path in required if not path.exists()]


def run_check() -> int:
    missing = check_required_paths()
    rows = load_jsonl(SAMPLE_DATA_PATH)

    if missing:
        print("Missing required files:")
        for path in missing:
            print(f"- {path}")
        return 1

    print("Repository scaffold check passed.")
    print(f"Sample dataset rows: {len(rows)}")
    print("Recommended next task: define one domain assistant and expand the dataset.")
    return 0


def preview_data() -> None:
    rows = load_jsonl(SAMPLE_DATA_PATH)
    for index, row in enumerate(rows, start=1):
        prompt = row.get("instruction", "<missing instruction>")
        response = row.get("response", "<missing response>")
        print(f"Example {index}")
        print(f"  Instruction: {prompt}")
        print(f"  Response: {response}")


def print_next_task() -> None:
    print("Next task:")
    print("Define a single domain assistant use case and create a 25-50 example instruction dataset.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "roadmap":
        print_roadmap()
        return

    if args.command == "check":
        raise SystemExit(run_check())

    if args.command == "preview-data":
        preview_data()
        return

    if args.command == "next-task":
        print_next_task()
        return

    raise SystemExit(f"Unknown command: {args.command}")
