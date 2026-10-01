# Planner teacher dataset

- `dataset.jsonl`: 10,872 rows, one per prompt, all recipes valid. Fields: `id, prompt, idea, recipe, source, family,
  weight` (`weight` = repeats in training).

  | source | rows | |
  |---|---|---|
  | `claude` | 520 | hand-written prompt → idea + recipe, 11 thematic batches |
  | `claude_events` | 65 | build-up → release events and long multi-phase stories (weight 3) |
  | `seed` | 287 | the recipes of `planner/examples/recipes.json` (no idea) |
  | `astra` | 5,000 | individually authored scenarios over ~600 families, precise choreography; not used by the served models |
  | `astra_lively` | 5,000 | the same prompts re-authored in the lively style (more energy, rhythm and ear expression) |

- `val.jsonl`: the 39 fixed validation prompts, each with two independent teacher recipes (the second measures how
  much the teacher agrees with itself). Never trained on.

`python -m planner.distill.sft` turns it into a training set (see [docs/TRAINING.md](../docs/TRAINING.md)).
