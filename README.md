# SNU AI Challenge Workspace

This repository is a structured workspace for the SNU AI Challenge image ordering competition.

The task is to order four image frames so they match a natural-language storyline. The baseline
notebook, `baseline_code.ipynb`, uses `Qwen/Qwen2-VL-2B-Instruct` in a zero-shot prompt-only
setting. This workspace keeps that baseline runnable and provides a clean harness layout for
adding new experiment ideas.

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

## Architecture Decision

Use a hybrid architecture.

Shared training/inference code is useful, but only for stable harness mechanics. A single generic
training framework that tries to own every idea tends to become awkward quickly because each idea
can differ in prompt format, model class, dataset/collator behavior, loss, decoding, caching, and
postprocessing.

The preferred split is:

- Put stable competition and execution mechanics in shared harness modules under `src/harness/`.
- Put each experiment's model-specific logic in its own package under `src/<idea_name>/`.
- Put each experiment's tests in `tests/<idea_name>/`.
- Promote duplicated code into `src/harness/` only when at least two ideas actually need the same
  behavior or when the code represents the competition contract.

## Project Layout

```text
AGENTS.md                 agent instructions and competition contract
init.sh                   environment setup
requirements.txt          Python dependencies
baseline_code.ipynb       original provided baseline notebook
scripts/                  thin shell wrappers around Python modules
src/
  baseline_qwen.py        legacy/reference zero-shot baseline
  data_utils.py           shared data loading/path helpers
  submission.py           shared answer parsing and submission contract
  harness/                shared training/inference/validation helpers, added as needed
  <idea_name>/            one experiment idea package
tests/
  test_submission.py      shared submission-contract tests
  harness/                tests for shared harness helpers, added as needed
  <idea_name>/            tests for one experiment idea package
outputs/                  generated submissions and raw predictions
experiments/              experiment notes and scratch outputs
checkpoints/              trained weights or adapters
work/                     temporary analysis artifacts
```

The existing `src/baseline_qwen.py` may stay as a legacy baseline reference. New ideas should not
add more root-level idea files.

## Idea Package Contract

For a new idea named `caption_augmented`, prefer this shape:

```text
src/caption_augmented/
  __init__.py
  infer.py                required for inference/submission generation
  train.py                optional, only if the idea is trainable
  prompts.py              optional idea-specific prompt builders
  dataset.py              optional idea-specific dataset/collator code
  model.py                optional idea-specific model loading/adapters
  config.py               optional defaults and dataclasses
tests/caption_augmented/
  test_prompts.py
  test_dataset.py
  test_parsing.py
```

Recommended entry points:

```bash
python -m src.<idea_name>.infer \
  --data-dir data \
  --output outputs/<idea_name>/submission.csv \
  --max-samples 4

python -m src.<idea_name>.train \
  --data-dir data \
  --output-dir outputs/<idea_name>/train_smoke \
  --max-samples 16 \
  --max-steps 1
```

Trainable ideas must support a cheap smoke or dry run through `--max-steps 1`, `--max-samples`,
or an explicit `--dry-run` flag.

## Harness Sharing Policy

Good candidates for `src/harness/`:

- CSV loading and row/image-path iteration
- train/test/sample submission contract checks
- chronological-to-submission and submission-to-chronological conversion wrappers
- generic submission writing and raw prediction logging
- common CLI argument dataclasses or parser helpers
- training preflight checks and smoke-run wrappers
- deterministic seeding helpers
- report/checkpoint/output path helpers

Keep inside `src/<idea_name>/`:

- model selection and model-specific loading
- prompts, captioning strategy, retrieval strategy, ranking logic, or ensembling logic
- datasets, collators, losses, LoRA target modules, and optimizer choices
- output parsing that is unique to an idea
- caches or preprocessing that only one idea uses

Import rules:

- `src/harness/` must not import from `src/<idea_name>/`.
- `src/<idea_name>/` may import from `src/harness/`, `src.data_utils`, and `src.submission`.
- One idea package must not import from another idea package.
- If two ideas need the same helper, move the helper to `src/harness/` or another clearly shared
  module.

## Submission Logic

The Qwen prompt asks the model to output chronological image labels, for example:

```text
[4, 2, 1, 3]
```

The competition submission uses the inverse mapping: for each original input image, where it
appears in chronological order. The helper `src.submission.chronological_to_submission()` performs
this conversion. Any model-facing chronological output should be converted before writing a
submission CSV.

## Testing

Tests should be lightweight and should not require downloading large models.

Good tests:

- prompt/message shape
- parsing and answer conversion
- submission CSV validation
- dataset row construction and image path resolution
- cache lookup behavior
- dry-run training record construction

Avoid putting full model inference or long GPU training inside pytest. Use explicit smoke commands
with `--max-samples` or `--max-steps` for those checks.
