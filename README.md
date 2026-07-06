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
python -m src.train_validate --profile local --data-dir data
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

## Caption-Augmented Qwen Experiment

This approach first generates one caption for each image, then asks a VLM to order the same
four images with those captions attached as supporting notes. The default model is
`Qwen/Qwen3-VL-8B-Instruct`, intended for a 24GB-class GPU such as an RTX 3090. It uses a
moderate per-image pixel budget by default; lower `--max-pixels` first if the GPU runs out of
memory.

```bash
python -m src.caption_augmented_qwen \
  --data-dir data \
  --output outputs/caption_augmented_qwen_submission.csv \
  --max-samples 4
```

On CUDA, try FlashAttention if the server has it installed:

```bash
python -m src.caption_augmented_qwen \
  --data-dir data \
  --attn-implementation flash_attention_2 \
  --output outputs/caption_augmented_qwen_qwen3vl8b_submission.csv \
  --max-samples 4
```

If Qwen3-VL support is not available in the server's `transformers` install, upgrade
`transformers` or fall back to `--model-name Qwen/Qwen2.5-VL-7B-Instruct`.

Captions are cached in `outputs/caption_augmented_qwen_captions.jsonl`, so reruns can reuse them.
Use `--refresh-captions` when changing the caption prompt or caption model.

### Caption-Augmented Qwen Training

The trainable version SFTs the same prompt with LoRA/QLoRA. Training targets are converted from
the competition `Answer` inverse mapping back to chronological image labels before being used as
assistant outputs.

Local dry-run, with no model download:

```bash
python -m src.train_caption_augmented_qwen \
  --data-dir data \
  --output-dir outputs/caption_augmented_train_dry_run \
  --max-samples 4 \
  --max-steps 1 \
  --dry-run
```

On a CUDA training server, install the extra GPU dependencies:

```bash
python -m pip install -r requirements-gpu.txt
```

Generate a small train-caption cache first:

```bash
python -m src.generate_captions_qwen \
  --data-dir data \
  --split train \
  --caption-cache outputs/train_qwen3vl8b_captions.jsonl \
  --max-samples 4 \
  --attn-implementation flash_attention_2
```

Then run a 3090 training smoke test:

```bash
python -m src.train_caption_augmented_qwen \
  --data-dir data \
  --output-dir checkpoints/caption_augmented_qwen3vl8b_lora_smoke \
  --caption-cache outputs/train_qwen3vl8b_captions.jsonl \
  --missing-caption-policy fail \
  --max-samples 4 \
  --max-steps 1 \
  --attn-implementation flash_attention_2
```

If the 24GB GPU runs out of memory, lower `--max-pixels` first, for example `--max-pixels 401408`.

After training, generate test captions with the base VLM and run inference with the trained adapter:

```bash
python -m src.generate_captions_qwen \
  --data-dir data \
  --split test \
  --caption-cache outputs/test_qwen3vl8b_captions.jsonl \
  --attn-implementation flash_attention_2

python -m src.caption_augmented_qwen \
  --data-dir data \
  --caption-cache outputs/test_qwen3vl8b_captions.jsonl \
  --adapter-path checkpoints/caption_augmented_qwen3vl8b_lora_smoke \
  --output outputs/caption_augmented_qwen3vl8b_lora_submission.csv \
  --attn-implementation flash_attention_2
```

Add `--load-in-4bit` to adapter inference if full-precision loading is too tight.

## Train Validation

Use train validation before moving an idea from local implementation to a training server.

Local validation:

```bash
python -m src.train_validate --profile local --data-dir data
```

Cloud GPU validation, to run on the target cloud instance:

```bash
python -m src.train_validate --profile cloud-gpu --data-dir data
```

For a custom idea trainer, expose a cheap dry-run mode and pass it as the train command:

```bash
python -m src.train_validate \
  --profile cloud-gpu \
  --data-dir data \
  --train-command "python -m src.my_idea_train --data-dir data --output-dir outputs/my_idea_smoke --max-steps 1"
```

The default train command runs `src.train_smoke`, which opens real images, builds a tiny torch model, trains for a couple of steps, and writes a checkpoint. It is a resource and wiring check, not a meaningful baseline.

## Project Structure

```text
AGENTS.md                 agent instructions and competition contract
init.sh                   environment setup and data sanity check
requirements.txt          Python dependencies
baseline_code.ipynb       original provided baseline notebook
scripts/                  shell wrappers
src/                      reusable implementation and runnable CLIs
src/validation/           runtime data and training preflight implementations
tests/                    lightweight pytest tests
outputs/                  generated submissions
experiments/              experiment notes
checkpoints/              trained weights or adapters
work/                     temporary files
```

`src.validate_data` and `src.train_validate` are compatibility wrappers for executable validation
commands. Their implementation lives under `src/validation/` so it is not confused with pytest
test modules under `tests/`.

## Idea File Layout

Keep each experiment as separate runnable modules under `src/`. Shared helpers should stay in
small common modules instead of being imported from another experiment's CLI.

Current layout:

- `src/baseline_qwen.py`: original zero-shot Qwen baseline
- `src/qwen_vl_common.py`: shared Qwen VL loading and generation helpers
- `src/caption_augmented_common.py`: caption-augmented prompt and caption-cache helpers
- `src/generate_captions_qwen.py`: caption-cache generation CLI
- `src/caption_augmented_qwen.py`: caption-augmented inference/submission CLI
- `src/train_caption_augmented_qwen.py`: caption-augmented LoRA/QLoRA training CLI

For a new idea, prefer adding `src/<idea_name>.py` for inference, `src/train_<idea_name>.py`
only if it trains, and `src/<idea_name>_common.py` only for code shared by that idea's CLIs.

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
python -m src.train_validate --profile local --data-dir data
python -m pytest
```
