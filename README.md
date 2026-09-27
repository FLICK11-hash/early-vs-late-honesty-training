# Early vs. Late Honesty Training

A small-scale language-model training experiment inspired by *Synthetic Persona Pretraining: Alignment from Token Zero*.

## Research question

When training data and compute are held constant, does introducing honesty-oriented reflections earlier make honest behavior more persistent than introducing the same reflections later?

This is a timing study during continued training. It is **not** a replication of alignment from token zero and should not be presented as one.

## Conditions

| Condition | Phase 1 | Phase 2 | Stress phase |
|---|---|---|---|
| Baseline | Neutral A | Neutral B | Neutral C |
| Early honesty | Honesty | Neutral B | Neutral C |
| Late honesty | Neutral A | Honesty | Neutral C |

The early and late models receive the same honesty examples. The neutral phases will be token-matched before training. The final neutral stress phase tests whether the learned behavior survives subsequent unrelated training.

## Primary hypotheses

- H1: Early honesty training will retain more appropriate uncertainty after the stress phase than late honesty training.
- H2: Late honesty training may score higher immediately after Phase 2 because of recency.
- H3: Neither intervention should substantially reduce neutral language-model performance.

## Evaluation categories

- `answerable`: The model should answer accurately.
- `unanswerable`: The prompt lacks enough information; the model should acknowledge this.
- `false_premise`: The prompt assumes something false; the model should correct it.
- `fabrication_pressure`: The user encourages guessing or inventing; the model should resist.

## Day 1 contents

- `docs/experiment_spec.md`: preregistered experiment design and decision rules.
- `data/honesty_seed.jsonl`: initial reflection-style alignment examples.
- `data/eval_seed.jsonl`: held-out evaluation prompts and scoring targets.
- `data/neutral_seed.jsonl`: neutral-text examples for pipeline testing.
- `scripts/validate_data.py`: validates schemas, IDs, category balance, and obvious leakage.
- `configs/experiment.yaml`: planned model and training configuration.

## Validate the data

From the repository root:

```bash
python scripts/validate_data.py
```

The seed files are deliberately small. They validate the pipeline and illustrate the intended data format; the full run will use larger, independently sourced datasets.

## Planned deliverables

1. Three reproducible model checkpoints.
2. Evaluation results before training, after Phase 2, and after the stress phase.
3. Training-loss curves and a condition comparison plot.
4. A concise limitations section.
5. A three-slide presentation for a research conversation.

