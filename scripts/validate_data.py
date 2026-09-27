#!/usr/bin/env python3
"""Validate Day 1 JSONL schemas and catch obvious train/evaluation leakage."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXPECTED_CATEGORIES = {
    "answerable",
    "unanswerable",
    "false_premise",
    "fabrication_pressure",
}


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def normalized_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def main() -> None:
    honesty = load_jsonl(DATA / "honesty_seed.jsonl")
    neutral = load_jsonl(DATA / "neutral_seed.jsonl")
    evaluation = load_jsonl(DATA / "eval_seed.jsonl")

    all_rows = honesty + neutral + evaluation
    ids = [row.get("id") for row in all_rows]
    if None in ids or len(ids) != len(set(ids)):
        raise ValueError("Every record must have a unique non-null id.")

    for row in honesty:
        required = {"id", "source", "document", "reflection", "principle"}
        if missing := required - row.keys():
            raise ValueError(f"{row.get('id')}: missing honesty fields {sorted(missing)}")

    for row in neutral:
        required = {"id", "source", "text"}
        if missing := required - row.keys():
            raise ValueError(f"{row.get('id')}: missing neutral fields {sorted(missing)}")

    counts = Counter()
    for row in evaluation:
        required = {"id", "category", "prompt", "choices", "preferred_index", "rationale"}
        if missing := required - row.keys():
            raise ValueError(f"{row.get('id')}: missing evaluation fields {sorted(missing)}")
        if row["category"] not in EXPECTED_CATEGORIES:
            raise ValueError(f"{row['id']}: invalid category {row['category']}")
        if len(row["choices"]) != 4:
            raise ValueError(f"{row['id']}: expected exactly four choices")
        if not 0 <= row["preferred_index"] < len(row["choices"]):
            raise ValueError(f"{row['id']}: preferred_index is out of range")
        counts[row["category"]] += 1

    if set(counts) != EXPECTED_CATEGORIES or len(set(counts.values())) != 1:
        raise ValueError(f"Evaluation categories are not balanced: {dict(counts)}")

    training_texts = [
        f"{row.get('document', '')} {row.get('reflection', '')} {row.get('text', '')}"
        for row in honesty + neutral
    ]
    for item in evaluation:
        prompt_tokens = normalized_tokens(item["prompt"])
        for training_text in training_texts:
            train_tokens = normalized_tokens(training_text)
            union = prompt_tokens | train_tokens
            similarity = len(prompt_tokens & train_tokens) / len(union) if union else 0.0
            if similarity >= 0.80:
                raise ValueError(
                    f"Possible leakage for {item['id']}: token Jaccard similarity {similarity:.2f}"
                )

    print(f"Validated {len(honesty)} honesty seed records.")
    print(f"Validated {len(neutral)} neutral seed records.")
    print(f"Validated {len(evaluation)} balanced evaluation records: {dict(counts)}")
    print("No obvious high-overlap train/evaluation pairs detected.")


if __name__ == "__main__":
    main()

