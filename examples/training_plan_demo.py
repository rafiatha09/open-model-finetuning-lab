"""Show the config-driven fine-tuning plan without starting training."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.config import load_training_config  # noqa: E402
from training.runner import run_training  # noqa: E402


def main() -> None:
    config_path = ROOT / "configs/training/lora.yaml"
    config = load_training_config(config_path)
    plan = run_training(config, dry_run=True, config_path=str(config_path))

    print("Training plan demo")
    print(f"Config:              {plan.config_path}")
    print(f"Method:              {plan.method}")
    print(f"Base model:          {plan.base_model}")
    print(f"Training examples:   {plan.train_examples}")
    print(f"Validation examples: {plan.validation_examples}")
    print(f"Output dir:          {plan.output_dir}")
    print(f"Notes:               {', '.join(plan.notes)}")
    print()
    print("Takeaway:")
    print("  A dry run lets you verify data paths and training intent before spending compute.")


if __name__ == "__main__":
    main()
