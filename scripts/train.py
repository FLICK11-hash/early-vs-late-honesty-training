"""Minimal causal-language-model training pipeline for the Day 2 smoke test.

This script deliberately trains only one dataset at a time. Its purpose is to
verify loading, tokenization, optimization, logging, and checkpoint saving
before the full three-condition experiment is implemented.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small continued-training job.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "smoke_test.yaml",
        help="Path to a YAML training configuration.",
    )
    parser.add_argument(
        "--dataset",
        choices=("honesty", "neutral"),
        default="honesty",
        help="Seed dataset used for this pipeline test.",
    )
    parser.add_argument(
        "--neutral-split",
        choices=("all", "a", "b", "c"),
        default="all",
        help="Use one deterministic third of the neutral records.",
    )
    parser.add_argument("--model-name", help="Override the config's base model.")
    parser.add_argument("--output-dir", type=Path, help="Override the output directory.")
    parser.add_argument("--max-steps", type=int, help="Override the number of training steps.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def format_record(record: dict, dataset_name: str) -> str:
    if dataset_name == "honesty":
        return (
            "Document:\n"
            f"{record['document']}\n\n"
            "Reflection:\n"
            f"{record['reflection']}"
        )
    return record["text"]


class CausalTextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length: int):
        self.examples = []
        for text in texts:
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    train_config = config["training"]
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)

    data_path = REPO_ROOT / "data" / f"{args.dataset}_seed.jsonl"
    records = load_jsonl(data_path)
    if args.dataset == "neutral" and args.neutral_split != "all":
        split_index = {"a": 0, "b": 1, "c": 2}[args.neutral_split]
        records = [record for index, record in enumerate(records) if index % 3 == split_index]
    texts = [format_record(record, args.dataset) for record in records]

    model_name = args.model_name or config["base_model"]
    print(f"Loading {model_name}")
    print(f"Training on {len(texts)} {args.dataset} records")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id

    dataset = CausalTextDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=int(train_config["max_sequence_length"]),
    )

    output_dir = args.output_dir or Path(train_config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    precision = train_config.get("precision", "fp32")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=float(train_config["epochs"]),
        per_device_train_batch_size=int(train_config["batch_size"]),
        gradient_accumulation_steps=int(train_config["gradient_accumulation_steps"]),
        learning_rate=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
        max_steps=args.max_steps if args.max_steps is not None else int(train_config.get("max_steps", -1)),
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        fp16=precision == "fp16",
        bf16=precision == "bf16",
        seed=seed,
        data_seed=seed,
        use_cpu=not torch.cuda.is_available(),
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    result = trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    metrics = {key: float(value) for key, value in result.metrics.items()}
    metrics.update(
        {
            "model": model_name,
            "dataset": args.dataset,
            "records": len(texts),
            "seed": seed,
        }
    )
    with (output_dir / "train_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"Smoke test complete. Model saved to {final_dir}")
    print(f"Metrics saved to {output_dir / 'train_metrics.json'}")


if __name__ == "__main__":
    main()

