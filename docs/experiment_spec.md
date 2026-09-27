# Experiment Specification

## Working title

Does Early Honesty Training Persist Better Than Late Honesty Training?

## Motivation

Synthetic Persona Pretraining reports that value-oriented reflections introduced from token zero can shape value generalization more strongly than the same intervention introduced during midtraining. This project tests a narrower and computationally modest question: whether the ordering of honesty-oriented continued-training data affects persistence after subsequent neutral training.

## Scope statement

This experiment does not train a useful language model from random initialization and does not test literal alignment from token zero. It is a controlled small-model study of alignment timing and persistence.

## Independent variable

The stage at which an identical set of honesty-oriented reflection examples is introduced.

## Conditions

1. Baseline: neutral data in both main phases.
2. Early honesty: honesty data in Phase 1 and neutral data in Phase 2.
3. Late honesty: neutral data in Phase 1 and honesty data in Phase 2.

All conditions receive a final neutral stress phase.

## Model

Initial plan: `EleutherAI/pythia-160m-deduped`, fully fine-tuned rather than prompt-only evaluation. If RCC memory constraints require it, use `EleutherAI/pythia-70m-deduped` and document the change before examining results.

## Measurements

Evaluate the untouched base model and every condition after Phase 2 and after the stress phase.

Primary metrics:

- Appropriate uncertainty accuracy on unanswerable prompts.
- False-premise correction accuracy.
- Fabrication-resistance accuracy.
- Answerable-question accuracy.
- Honesty macro-average across the first three categories.

Secondary metrics:

- Perplexity on held-out neutral text.
- Change in honesty score from post-Phase-2 to post-stress evaluation.
- Training and validation loss.

## Scoring plan

The initial evaluation will use constrained choices and option log probabilities to reduce ambiguity in judging small-model generations. Each item has one preferred answer and plausible undesirable alternatives. A later qualitative analysis may examine free-form generations, but it will remain secondary.

## Fairness controls

- Early and late conditions use exactly the same honesty records.
- Phase token budgets are matched by token count, not merely example count.
- Model initialization, optimizer, learning-rate schedule, batch size, random seed, and evaluation prompts remain fixed.
- Evaluation files are never used for training or prompt construction.
- Results are reported even if the hypothesis is unsupported.

## Planned runs

- Primary seed: 42.
- Optional confirmation seeds: 7 and 123, time permitting.
- A single seed is treated as a pilot, not definitive evidence.

## Decision rule

The main evidence for H1 is the post-stress honesty macro-average. H1 is supported in the pilot if early honesty exceeds late honesty after the stress phase while answerable accuracy and neutral perplexity do not show a large capability loss. Exact uncertainty intervals will be added if multiple seeds are completed.

## Threats to validity

- Continued training of a pretrained model differs fundamentally from training from token zero.
- A 160M-parameter model may be too weak for stable free-form behavior.
- A small synthetic dataset may teach phrases rather than general principles.
- Results may be sensitive to seed, learning rate, data order, and evaluation wording.
- Constrained-choice evaluation may not reflect deployment behavior.

## Claims we may make

- We conducted a controlled pilot of early versus late honesty-oriented continued training.
- We observed whether timing affected retention under a neutral-training stress test.

## Claims we may not make

- We replicated Synthetic Persona Pretraining.
- We demonstrated alignment from token zero.
- The results necessarily generalize to frontier models or real-world safety.

