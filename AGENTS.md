# Agent Guide

This workspace is for the SNU AI Challenge image ordering competition.

The user may propose a concrete implementation idea. Treat that idea as the main experiment, then implement it systematically without breaking the existing baseline, submission contract, or validation commands.

## Competition Task

Each sample contains:

- `Id`
- `Input_1`, `Input_2`, `Input_3`, `Input_4`
- `Sentence`
- `Answer` only in `train.csv`
- `No_ordering` only in `train.csv`

The model receives a sentence and four image frames. It must predict how the four input frames should be ordered chronologically to match the sentence.

Submission format:

- CSV columns: `Id,Answer`
- `Id` order must match `data/sample_submission.csv`
- `Answer` must be a stringified Python list containing a permutation of 1, 2, 3, 4
- Example: `[3, 1, 4, 2]`

Important convention:

- A VLM prompt usually asks for chronological image labels, e.g. `[4, 2, 1, 3]`
- The competition `Answer` stores the inverse mapping: for each original input image, the position it occupies in chronological order
- Use `src.submission.chronological_to_submission()` before writing model outputs

## Baseline Reference

`baseline_code.ipynb` is the source notebook baseline. It uses:

- `Qwen/Qwen2-VL-2B-Instruct`
- zero-shot prompting
- no training
- output parsing plus inverse mapping

The notebook has been converted into a reusable CLI in `src/baseline_qwen.py`.

## Required Workflow

When implementing a requested approach:

1. Read this file, `README.md`, and the relevant source files first.
2. Keep reusable code under `src/`; avoid making notebooks the only runnable artifact.
   New ideas must live in their own package directory under `src/<idea_name>/`.
3. Preserve the data contract and submission format.
4. Add or update CLI entry points inside the idea package, e.g. `python -m src.<idea_name>.infer`
   and, when trainable, `python -m src.<idea_name>.train`.
5. Add focused tests for parsing, scoring, or data-shape logic when possible under
   `tests/<idea_name>/`.
6. Run lightweight verification before reporting completion:
   - always run `python -m pytest`
   - if a data validation CLI exists, run it, e.g. `python -m src.validate_data --data-dir data`
7. Run a training preflight when the change adds trainable logic:
   - first run the idea's cheap mode, e.g. `python -m src.<idea_name>.train --data-dir data --output-dir outputs/<idea_name>_smoke --max-steps 1`
   - if a training validation CLI exists, run it locally with the intended train command before sending code to a training server
8. For model-heavy inference, run a smoke test with `--max-samples` before full inference.
9. Write submissions, checkpoints, and reports under `outputs/`, `checkpoints/`, or `experiments/`, not the repository root.

## Directory Rules

- `data/`: competition data; do not edit or commit regenerated data.
- `src/`: reusable Python modules, shared harness code, and idea packages.
- `src/harness/`: shared data, submission, validation, training-loop, inference-loop, and CLI helpers
  that are stable across multiple ideas.
- `src/<idea_name>/`: one experiment idea package. Put that idea's prompt logic, dataset/collator,
  model loading, training entry point, inference entry point, and idea-specific utilities here.
- `scripts/`: thin shell wrappers around Python modules.
- `outputs/`: generated submissions and raw predictions.
- `checkpoints/`: model checkpoints or adapters.
- `experiments/`: experiment notes and scratch outputs.
- `work/`: temporary analysis artifacts.
- `tests/`: lightweight tests that do not require downloading models.
- `tests/<idea_name>/`: tests for one idea package. Shared harness tests may live under
  `tests/harness/`.

## Harness Architecture

Use a hybrid architecture:

- Shared harness code is recommended for stable competition mechanics: CSV reading, image path
  resolution, answer parsing, chronological/submission conversion, submission writing, lightweight
  validation, common CLI arguments, smoke-run plumbing, report writing, and generic train/inference
  orchestration hooks.
- Idea-specific code is recommended for model choice, prompt format, caption generation strategy,
  dataset construction, collator behavior, loss/training details, decoding strategy, and output
  postprocessing that is not already part of the competition contract.
- Do not build a large abstract training or inference framework before at least two ideas need the
  same behavior. Prefer small reusable helper functions and dataclasses over deep inheritance.
- `src/harness/` must not import from any `src/<idea_name>/` package. Idea packages may import
  from `src/harness/`.
- One idea package must not import from another idea package. If two ideas need the same code,
  move the shared portion into `src/harness/` or another clearly shared module.
- Keep each idea runnable independently. A trainable idea should expose a cheap dry run such as
  `python -m src.<idea_name>.train --data-dir data --output-dir outputs/<idea_name>_smoke --max-steps 1`.
- For inference, prefer `python -m src.<idea_name>.infer --data-dir data --output outputs/<idea_name>/submission.csv --max-samples 4`.
- Generated artifacts should be namespaced by idea, e.g. `outputs/<idea_name>/...`,
  `checkpoints/<idea_name>/...`, and `experiments/<idea_name>/...`.
- The existing root-level baseline module may remain as a reference baseline, but new experiments
  should not add more root-level idea files.

## Implementation Standards

- Do not hard-code absolute paths.
- Default `DATA_DIR` is `data`.
- Keep `sample_submission.csv` as the source of truth for test `Id` order.
- Validate all generated submissions before considering a task done.
- If a model output cannot be parsed, fall back explicitly and log the raw output.
- Keep changes scoped to the requested experiment.
- Avoid destructive commands and avoid modifying original data files.
- Prefer deterministic settings for baselines unless the user asks for sampling or ensembling.
- Training entry points should support a cheap dry run such as `--max-steps 1` or `--max-samples 16`.
- Before sending code to a training server, run the idea's cheap train command locally. If
  `src.train_validate` or an equivalent harness preflight exists, run it with the intended train
  command as well.

## Useful Commands

```bash
bash init.sh
python -m src.baseline_qwen --data-dir data --output outputs/qwen2vl_submission.csv --max-samples 4
python -m pytest
```
