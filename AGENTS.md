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
3. Preserve the data contract and submission format.
4. Add or update a CLI entry point for the approach.
5. Add focused tests for parsing, scoring, or data-shape logic when possible.
6. Run lightweight verification before reporting completion:
   - `python -m src.validate_data --data-dir data`
   - `python -m pytest`
7. Run a training preflight when the change adds trainable logic:
   - local: `python -m src.train_validate --profile local --data-dir data`
   - cloud GPU: `python -m src.train_validate --profile cloud-gpu --data-dir data`
   - custom idea: pass `--train-command "python -m src.<idea_train> --data-dir data --output-dir outputs/<idea>_smoke --max-steps 1"`
8. For model-heavy inference, run a smoke test with `--max-samples` before full inference.
9. Write submissions, checkpoints, and reports under `outputs/`, `checkpoints/`, or `experiments/`, not the repository root.

## Directory Rules

- `data/`: competition data; do not edit or commit regenerated data.
- `src/`: reusable Python modules and CLIs.
- `scripts/`: thin shell wrappers around Python modules.
- `outputs/`: generated submissions and raw predictions.
- `checkpoints/`: model checkpoints or adapters.
- `experiments/`: experiment notes and scratch outputs.
- `work/`: temporary analysis artifacts.
- `tests/`: lightweight tests that do not require downloading models.

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
- Before sending code to a training server, run `src.train_validate` locally with the intended train command.

## Useful Commands

```bash
bash init.sh
python -m src.validate_data --data-dir data
python -m src.train_smoke --data-dir data --output-dir outputs/train_smoke
python -m src.train_validate --profile local --data-dir data
python -m src.train_validate --profile cloud-gpu --data-dir data --skip-train-command
python -m src.baseline_qwen --data-dir data --output outputs/qwen2vl_submission.csv --max-samples 4
python -m pytest
```
