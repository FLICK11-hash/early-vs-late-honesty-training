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

## Day 2 smoke test

Create a virtual environment and install the training dependencies:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run a three-step continued-training test with Pythia-70M:

```bash
python scripts/train.py --config configs/smoke_test.yaml --dataset honesty
```

The first run downloads the base model. A successful run writes the trained
model to `outputs/smoke_test/final/` and metrics to
`outputs/smoke_test/train_metrics.json`. These outputs are ignored by Git.

This is a pipeline test, not an experimental result. The full early-versus-late
comparison will use expanded, token-matched datasets and separate checkpoints.

## Day 3 three-condition pilot

After the smoke test succeeds, run the complete phase sequence locally:

```bash
python scripts/run_experiment.py --config configs/pilot.yaml
```

The runner independently initializes the baseline, early-honesty, and
late-honesty conditions from Pythia-70M. Each condition receives three short
training phases. Neutral records are deterministically separated into A, B,
and C subsets, and both honesty conditions use the identical honesty records.

Phase 2 and post-stress checkpoints are retained under `outputs/pilot/`. Phase
1 checkpoints are removed after they initialize Phase 2 to limit disk usage.
`outputs/pilot/run_summary.json` records the phase order, losses, runtimes, and
checkpoint paths. This remains an engineering pilot; it is not meaningful
evidence until the datasets and evaluation set are expanded.

## Day 4 constrained-choice evaluation

Evaluate the Phase 2 and post-stress checkpoints for all three conditions:

```bash
python scripts/evaluate.py --config configs/pilot.yaml
```

For each evaluation prompt, the script measures the mean conditional token log
probability of each answer choice and selects the highest-scoring choice. It
reports total accuracy, accuracy by category, and an honesty macro-average over
unanswerable, false-premise, and fabrication-pressure items. It also records
the change from Phase 2 to post-stress as the retention delta.

Aggregate results are written to `outputs/pilot/evaluation_results.json`, and
item-level predictions are written to
`outputs/pilot/evaluation_predictions.csv`. Results from the 24-item seed set
are pipeline diagnostics, not confirmatory evidence.

## Planned deliverables

1. Three reproducible model checkpoints.
2. Evaluation results before training, after Phase 2, and after the stress phase.
3. Training-loss curves and a condition comparison plot.
4. A concise limitations section.
5. A three-slide presentation for a research conversation.

