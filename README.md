# SNU AI Challenge Workspace

This repository is a structured workspace for the SNU AI Challenge image ordering competition.

The task is to order four image frames so they match a natural-language storyline. The baseline notebook, `baseline_code.ipynb`, uses `Qwen/Qwen2-VL-2B-Instruct` in a zero-shot prompt-only setting. This workspace keeps that baseline runnable from scripts and gives agents a stable place to add new approaches.

## Data

Expected layout:

```text
data/
  train.csv
  test.csv
  sample_submission.csv
  train/<Id>/*.jpg
  test/<Id>/*.jpg
```

CSV schema:

- `train.csv`: `Id,Input_1,Input_2,Input_3,Input_4,Sentence,Answer,No_ordering`
- `test.csv`: `Id,Input_1,Input_2,Input_3,Input_4,Sentence`
- `sample_submission.csv`: `Id,Answer`

`Answer` must be a stringified Python list permutation such as `[1, 2, 3, 4]`.

## Quick Start

```bash
bash init.sh
source .venv/bin/activate
python -m src.validate_data --data-dir data
python -m src.identity_baseline --data-dir data --output outputs/identity_submission.csv
python -m pytest
```

The Qwen baseline requires downloading model weights from Hugging Face and is best run on GPU:

```bash
python -m src.baseline_qwen \
  --data-dir data \
  --output outputs/qwen2vl_submission.csv \
  --max-samples 4
```

Remove `--max-samples` for full test inference after the smoke test looks healthy.

## Project Structure

```text
AGENTS.md                 agent instructions and competition contract
init.sh                   environment setup and data sanity check
requirements.txt          Python dependencies
baseline_code.ipynb       original provided baseline notebook
configs/                  experiment configs
scripts/                  shell wrappers
src/                      reusable implementation
tests/                    lightweight tests
outputs/                  generated submissions
experiments/              experiment notes
checkpoints/              trained weights or adapters
work/                     temporary files
```

## Baseline Logic

The Qwen prompt asks the model to output chronological image labels, for example:

```text
[4, 2, 1, 3]
```

The competition submission uses the inverse mapping: for each original input image, where it appears in chronological order. The helper `src.submission.chronological_to_submission()` performs this conversion.

## Agent Workflow

When adding a new method, keep it runnable as a module under `src/`, add a thin wrapper in `scripts/` when useful, and validate the generated CSV before calling it done.

Useful checks:

```bash
python -m src.validate_data --data-dir data
python -m pytest
```
