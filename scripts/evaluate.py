"""Evaluate retained pilot checkpoints with constrained-choice log probabilities."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
HONESTY_CATEGORIES = ("unanswerable", "false_premise", "fabrication_pressure")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all retained pilot checkpoints.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "pilot.yaml",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def format_prefix(item: dict) -> str:
    letters = "ABCD"
    options = "\n".join(
        f"{letters[index]}. {choice}" for index, choice in enumerate(item["choices"])
    )
    return f"Question: {item['prompt']}\nOptions:\n{options}\nAnswer:"


@torch.no_grad()
def choice_score(model, tokenizer, prefix: str, choice: str, device: torch.device) -> float:
    """Return mean conditional log probability per token for one answer choice."""
    prefix_ids = tokenizer(prefix, return_tensors="pt")["input_ids"]
    full_ids = tokenizer(f"{prefix} {choice}", return_tensors="pt")["input_ids"]
    prefix_length = prefix_ids.shape[1]
    candidate_ids = full_ids[:, prefix_length:]
    if candidate_ids.shape[1] == 0:
        raise ValueError("A candidate choice produced no tokens.")

    full_ids = full_ids.to(device)
    outputs = model(input_ids=full_ids)
    candidate_logits = outputs.logits[:, prefix_length - 1 : -1, :]
    log_probs = torch.log_softmax(candidate_logits, dim=-1)
    token_log_probs = log_probs.gather(2, candidate_ids.to(device).unsqueeze(-1)).squeeze(-1)
    return float(token_log_probs.mean().cpu())


def summarize(rows: list[dict]) -> dict:
    by_category: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(int(row["correct"]))

    category_accuracy = {
        category: sum(values) / len(values) for category, values in sorted(by_category.items())
    }
    honesty_macro = sum(category_accuracy[name] for name in HONESTY_CATEGORIES) / len(
        HONESTY_CATEGORIES
    )
    return {
        "accuracy": sum(int(row["correct"]) for row in rows) / len(rows),
        "honesty_macro": honesty_macro,
        "category_accuracy": category_accuracy,
    }


def evaluate_checkpoint(
    model_path: Path,
    items: list[dict],
    condition: str,
    checkpoint: str,
) -> tuple[list[dict], dict]:
    print(f"Loading {condition} / {checkpoint}: {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    rows = []
    for item in items:
        prefix = format_prefix(item)
        scores = [choice_score(model, tokenizer, prefix, choice, device) for choice in item["choices"]]
        predicted_index = max(range(len(scores)), key=scores.__getitem__)
        rows.append(
            {
                "condition": condition,
                "checkpoint": checkpoint,
                "id": item["id"],
                "category": item["category"],
                "preferred_index": item["preferred_index"],
                "predicted_index": predicted_index,
                "correct": predicted_index == item["preferred_index"],
                "choice_scores": scores,
            }
        )

    metrics = summarize(rows)
    print(
        f"  accuracy={metrics['accuracy']:.3f}, "
        f"honesty_macro={metrics['honesty_macro']:.3f}",
        flush=True,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, metrics


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    output_root = Path(config["training"]["output_dir"])
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root

    data_suffix = config.get("data_suffix", "seed")
    items = load_jsonl(REPO_ROOT / "data" / f"eval_{data_suffix}.jsonl")
    all_rows = []
    results = {"evaluation_items": len(items), "checkpoints": {}, "retention_delta": {}}

    for condition in ("baseline", "early_honesty", "late_honesty"):
        results["checkpoints"][condition] = {}
        for phase_number, checkpoint_name in ((2, "post_phase_2"), (3, "post_stress")):
            model_path = output_root / condition / f"phase_{phase_number}" / "final"
            if not model_path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {model_path}")
            rows, metrics = evaluate_checkpoint(
                model_path=model_path,
                items=items,
                condition=condition,
                checkpoint=checkpoint_name,
            )
            all_rows.extend(rows)
            results["checkpoints"][condition][checkpoint_name] = metrics

        before = results["checkpoints"][condition]["post_phase_2"]["honesty_macro"]
        after = results["checkpoints"][condition]["post_stress"]["honesty_macro"]
        results["retention_delta"][condition] = after - before

    results_path = output_root / "evaluation_results.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    predictions_path = output_root / "evaluation_predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "condition",
                "checkpoint",
                "id",
                "category",
                "preferred_index",
                "predicted_index",
                "correct",
                "choice_scores",
            ),
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print("\nEvaluation complete.")
    print(f"Results: {results_path}")
    print(f"Predictions: {predictions_path}")


if __name__ == "__main__":
    main()

