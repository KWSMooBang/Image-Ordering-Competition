# Data Augmentation Idea

## Scope

This idea starts with a single, low-risk augmentation: shuffle the four input image columns for each
training sample while preserving the same sentence and recomputing `Answer`.

The goal is to make the model less sensitive to the original `Input_1` to `Input_4` order and more
reliant on visual chronology plus the sentence.

## Augmentation: Image Order Shuffle

For each row in `train.csv`, create one or more augmented rows by applying a permutation to:

- `Input_1`
- `Input_2`
- `Input_3`
- `Input_4`

Keep unchanged:

- `Id` as a derived augmented id, e.g. `<original_id>__shuffle_<index>`
- `Sentence`
- `No_ordering`

Recompute:

- `Answer`

## Answer Recalculation Rule

The competition `Answer` means: for each current input image slot, which chronological position does
that image occupy?

Given an original submission-style answer:

```text
original_answer[original_slot - 1] = chronological_position
```

and a shuffle permutation:

```text
new_slot -> original_slot
```

the augmented answer is:

```text
augmented_answer[new_slot - 1] = original_answer[original_slot - 1]
```

Example:

```text
Original inputs:  [A, B, C, D]
Original Answer:  [3, 1, 4, 2]

Shuffle order:    [2, 4, 1, 3]
Augmented inputs: [B, D, A, C]
Augmented Answer: [1, 2, 3, 4]
```

Explanation:

- New `Input_1` is old `Input_2`, whose chronological position was `1`.
- New `Input_2` is old `Input_4`, whose chronological position was `2`.
- New `Input_3` is old `Input_1`, whose chronological position was `3`.
- New `Input_4` is old `Input_3`, whose chronological position was `4`.

## Intended Implementation Shape

Place implementation under this package:

```text
src/data_augmentation/
  __init__.py
  dataset.py
  train.py or generate.py
```

Initial implementation should prefer generating augmented rows in memory for training datasets.
Only write augmented CSV files under `outputs/data_augmentation/` when an explicit CLI asks for it.

Implemented helper:

```python
def shuffle_row(row, permutation):
    ...
```

where `permutation` is a 4-item list describing `new_slot -> original_slot`, such as
`[2, 4, 1, 3]`.

The real-time dataset wrapper should be used for training:

```python
from src.data_augmentation import RealtimeShuffleConfig, RealtimeShuffleDataset

train_dataset = RealtimeShuffleDataset(
    train_df,
    config=RealtimeShuffleConfig(seed=42, augmentations_per_sample=1),
)
```

Call `train_dataset.set_epoch(epoch)` at the start of each epoch if the training loop supports it.
The same seed, epoch, row index, and augmentation view always produce the same shuffle, which keeps
the augmentation reproducible while still changing across epochs.

## Guardrails

- Do not edit files under `data/`.
- Preserve the submission contract: every `Answer` must be a stringified permutation of
  `[1, 2, 3, 4]`.
- Use `src.submission.parse_answer_cell()` and `src.submission.format_answer()` instead of custom
  parsing/formatting.
- Keep image filenames unchanged; only move them between `Input_1` to `Input_4`.
- Keep augmentation deterministic by accepting a seed.
- Avoid producing the identity permutation unless explicitly requested.
- For samples where `No_ordering` is true, keep the row valid but consider excluding them from
  order-sensitive augmentation experiments if validation shows they add noise.

## Validation Plan

Add lightweight tests under:

```text
tests/data_augmentation/
```

Minimum tests:

- shuffled image columns follow the requested permutation
- recomputed `Answer` preserves each image's chronological position
- identity permutation returns the original row
- generated answers pass `parse_answer_cell()`

Run:

```bash
python -m pytest
```

If a future CLI writes augmented training CSVs, also add a smoke command such as:

```bash
python -m src.data_augmentation.generate \
  --data-dir data \
  --output outputs/data_augmentation/train_shuffle_smoke.csv \
  --max-samples 4 \
  --num-augmentations 1 \
  --seed 42
```
