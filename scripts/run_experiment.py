"""Run the three-condition early-versus-late pilot sequentially."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"

CONDITIONS = {
    "baseline": (("neutral", "a"), ("neutral", "b"), ("neutral", "c")),
    "early_honesty": (("honesty", "all"), ("neutral", "b"), ("neutral", "c")),
    "late_honesty": (("neutral", "a"), ("honesty", "all"), ("neutral", "c")),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all three pilot conditions.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "pilot.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    base_model = config["base_model"]
    max_steps = int(config["training"]["max_steps"])
    output_root = Path(config["training"]["output_dir"])
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": base_model,
        "seed": int(config["seed"]),
        "max_steps_per_phase": max_steps,
        "conditions": {},
    }

    for condition, phases in CONDITIONS.items():
        print(f"\n=== Running condition: {condition} ===", flush=True)
        model_source = base_model
        phase_records = []

        for phase_number, (dataset, neutral_split) in enumerate(phases, start=1):
            phase_dir = output_root / condition / f"phase_{phase_number}"
            command = [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--config",
                str(config_path),
                "--dataset",
                dataset,
                "--neutral-split",
                neutral_split,
                "--model-name",
                str(model_source),
                "--output-dir",
                str(phase_dir),
                "--max-steps",
                str(max_steps),
            ]
            print(
                f"Phase {phase_number}: dataset={dataset}, split={neutral_split}",
                flush=True,
            )
            subprocess.run(command, cwd=REPO_ROOT, check=True)

            metrics_path = phase_dir / "train_metrics.json"
            with metrics_path.open(encoding="utf-8") as handle:
                metrics = json.load(handle)
            phase_records.append(
                {
                    "phase": phase_number,
                    "dataset": dataset,
                    "neutral_split": neutral_split,
                    "checkpoint": str(phase_dir / "final"),
                    "train_loss": metrics.get("train_loss"),
                    "train_runtime": metrics.get("train_runtime"),
                }
            )
            model_source = phase_dir / "final"

            # Phase 1 is needed only to initialize Phase 2. Removing it keeps
            # this local pilot from storing three unnecessary model copies.
            if phase_number == 2:
                phase_one_dir = output_root / condition / "phase_1"
                if phase_one_dir.exists():
                    shutil.rmtree(phase_one_dir)
                    phase_records[0]["checkpoint"] = "removed_after_phase_2"

        summary["conditions"][condition] = phase_records
        with (output_root / "run_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    summary["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    with (output_root / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nPilot training complete.", flush=True)
    print(f"Summary saved to {output_root / 'run_summary.json'}", flush=True)


if __name__ == "__main__":
    main()

