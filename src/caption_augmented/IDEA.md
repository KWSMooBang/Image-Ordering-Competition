# Caption-Augmented Ordering

## Summary

Generate a short caption for each of the four input images, then ask a VLM to infer the
chronological image order using both:

- the original sentence,
- the four original images,
- the four generated image captions.

The key hypothesis is that explicit per-image captions expose action/state clues that the ordering
VLM can reuse when comparing similar frames.

## Models

Caption generator options:

- Primary lightweight caption model: `Salesforce/blip-image-captioning-large`
  - Hugging Face: https://huggingface.co/Salesforce/blip-image-captioning-large
  - Role: generate one concise caption for each single image.
- Alternative caption generator: a VLM such as `Qwen/Qwen3-VL-8B-Instruct`
  - Role: generate richer captions when compute allows, possibly with sentence-aware prompts.

Ordering VLM:

- `Qwen/Qwen3-VL-8B-Instruct`
  - Hugging Face: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
  - Role: receive all four images and captions, then output the chronological image labels.

## Data Contract

Input rows contain:

- `Id`
- `Input_1`, `Input_2`, `Input_3`, `Input_4`
- `Sentence`
- `Answer` only in train
- `No_ordering` only in train

The ordering prompt should ask the VLM to output chronological image labels, for example:

```text
[4, 2, 1, 3]
```

Competition submission uses the inverse mapping. Before writing submissions, convert the
chronological model output with:

```python
src.submission.chronological_to_submission()
```

For training targets, convert train `Answer` back to chronological order with:

```python
src.submission.submission_to_chronological()
```

## Inference Pipeline

1. Load `test.csv` and preserve `sample_submission.csv` Id order.
2. For each row, resolve the four image paths.
3. For each image, generate or load a cached caption.
4. Build an ordering prompt:
   - include all four original images,
   - include text blocks such as `Image 1 caption: ...`,
   - include the original `Sentence`,
   - instruct Qwen3-VL to output only a Python list of chronological image labels.
5. Parse model output.
6. Convert chronological order to submission answer.
7. Write `Id,Answer` CSV and raw JSONL logs.

## Caption Cache

Captions should be cached as JSONL so that expensive caption generation can be reused:

```json
{"Id": "...", "image_index": 1, "image": "xxx.jpg", "caption": "..."}
```

Cache keys should include all of:

- `Id`
- original image index, 1 through 4
- image filename

Separate train/test caches are recommended:

- `outputs/caption_augmented/train_captions.jsonl`
- `outputs/caption_augmented/test_captions.jsonl`

## Prompt Sketch

Caption prompt:

```text
Story sentence: "{Sentence}"
This is Image {i} from a shuffled four-frame story.
Write one concise caption for only this image.
Focus on visible actions, object states, positions, and before/after clues.
Do not guess the final image order.
Return one sentence only.
```

Ordering prompt:

```text
Story sentence: "{Sentence}"

Image 1 caption: ...
Image 2 caption: ...
Image 3 caption: ...
Image 4 caption: ...

The captions may be imperfect, so use the images as primary evidence and captions as supporting
notes. Determine the chronological order of Image 1 to Image 4. Return only a Python list of
image labels, e.g. [1, 2, 3, 4].
```

## Proposed Package Layout

```text
src/caption_augmented/
  __init__.py
  IDEA.md
  config.py          # implemented defaults and dataclasses
  captions.py        # implemented caption generation and cache IO
  prompts.py         # implemented caption and ordering prompt builders
  model.py           # implemented BLIP captioner and Qwen3-VL ordering wrappers
  infer.py           # implemented submission-generation CLI
  dataset.py         # implemented train record and target construction
  train.py           # implemented Qwen3-VL LoRA/QLoRA SFT CLI

tests/caption_augmented/
  test_captions.py   # implemented
  test_prompts.py    # implemented
  test_order_dataset.py # implemented
  test_train.py      # implemented
  test_infer.py
```

## CLI Targets

Caption cache smoke:

```bash
python -m src.caption_augmented.captions \
  --data-dir data \
  --split test \
  --output outputs/caption_augmented/test_captions.jsonl \
  --max-samples 4
```

Inference smoke:

```bash
python -m src.caption_augmented.infer \
  --data-dir data \
  --caption-cache outputs/caption_augmented/test_captions.jsonl \
  --output outputs/caption_augmented/submission_smoke.csv \
  --max-samples 4
```

Optional training smoke:

```bash
python -m src.caption_augmented.train \
  --data-dir data \
  --caption-cache outputs/caption_augmented/train_captions.jsonl \
  --output-dir outputs/caption_augmented/train_smoke \
  --max-samples 16 \
  --max-steps 1
```

Dry run without loading Qwen:

```bash
python -m src.caption_augmented.train \
  --data-dir data \
  --output-dir outputs/caption_augmented/orderer_train_dry_run \
  --max-samples 4 \
  --max-steps 1 \
  --dry-run
```

## Implementation Notes

- Start with inference-only before adding training.
- Keep BLIP caption generation separate from Qwen ordering logic.
- Always log raw captions and raw ordering model outputs.
- If parsing fails, use an explicit fallback and record the raw output.
- Keep all generated artifacts under `outputs/caption_augmented/` or
  `checkpoints/caption_augmented/`.
- Avoid importing code from other idea packages.

## Risks

- Captions may omit subtle temporal details or hallucinate state changes.
- If captions are too strong, Qwen may over-trust wrong text over visual evidence.
- Four-image Qwen3-VL inference may be memory-heavy on a 24GB GPU depending on image token budget.
- Caption generation doubles the pipeline cost unless cached.

## Evaluation Plan

1. Run caption generation on a small train/test subset and inspect caption quality.
2. Run inference smoke with `--max-samples 4`.
3. Compare with the zero-shot baseline on a held-out train split if a local validation split is
   added.
4. If inference-only is promising, add optional QLoRA SFT using train rows and chronological
   targets.
