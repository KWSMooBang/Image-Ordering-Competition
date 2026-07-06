# Data Filtering

This package builds a lightweight audit manifest for train samples that may be harmful for
supervised ordering.

It does not modify `data/train.csv`. The default command writes a manifest under `outputs/`:

```bash
python -m src.data_filtering.audit \
  --data-dir data \
  --output outputs/data_filtering/train_audit.csv
```

Useful optional outputs:

```bash
python -m src.data_filtering.audit \
  --data-dir data \
  --output outputs/data_filtering/train_audit.csv \
  --filtered-output outputs/data_filtering/train_filtered.csv
```

The manifest contains:

- `action`: `keep`, `downweight`, or `drop_from_supervised`
- `sample_weight`: default `1.0` for clean rows, `0.5` for suspicious rows, `0.0` for dropped rows
- `manual_review`: rows worth inspecting before aggressive deletion
- `reasons`: semicolon-separated rule hits
- image quality diagnostics for blank, duplicate, missing, or unreadable frames

Rules are intentionally conservative:

- `No_ordering=True` defaults to `drop_from_supervised` because those rows use identity answers as
  placeholders.
- two or more blank frames defaults to `drop_from_supervised`.
- one blank frame, duplicate frame candidates, or optional low text-frame caption relevance defaults
  to `downweight` plus `manual_review`.

Caption relevance is optional and uses a caption cache with fields compatible with the
caption-augmented JSONL format: `Id`, `image_index`, `image`, `caption`.
