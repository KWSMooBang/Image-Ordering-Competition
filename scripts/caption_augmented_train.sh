#!/usr/bin/env bash
set -euo pipefail

# Single-GPU (RTX 3090 class) QLoRA SFT of the caption_augmented Qwen3.5
# orderer. This is the integrated pipeline entry point: it (re)uses the
# data_filtering manifest as its training CSV and enables data_augmentation's
# real-time image-order shuffling, so all three ideas train as one model
# instead of three separate experiments.
#
# Auto-runs the caption cache and data-filtering steps first if their outputs
# are missing (set GENERATE_CAPTIONS=1 / RUN_FILTER=1 to force a refresh).

DATA_DIR="${DATA_DIR:-data}"
TRAIN_CSV="${TRAIN_CSV:-outputs/data_filtering/train_filtered_caption_augmented.csv}"
CAPTION_CACHE="${CAPTION_CACHE:-outputs/caption_augmented/train_captions.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/caption_augmented/qwen3_5_4b_qlora}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GENERATE_CAPTIONS="${GENERATE_CAPTIONS:-0}"
RUN_FILTER="${RUN_FILTER:-0}"

TORCH_DTYPE="${TORCH_DTYPE:-float16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:--1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
SAVE_STEPS="${SAVE_STEPS:-200}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"

# data_augmentation integration: real-time input-slot shuffling per row.
SHUFFLE_AUGMENTATIONS_PER_SAMPLE="${SHUFFLE_AUGMENTATIONS_PER_SAMPLE:-2}"
SHUFFLE_SEED="${SHUFFLE_SEED:-42}"

# Optional held-out validation split (0 disables; e.g. 0.1 = 10% held out).
# When enabled, exact-match accuracy is generated and logged every epoch, and
# the best-scoring checkpoint is saved separately to OUTPUT_DIR/best.
VAL_FRACTION="${VAL_FRACTION:-0}"
VAL_SEED="${VAL_SEED:-42}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-128}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${GENERATE_CAPTIONS}" == "1" || ! -s "${CAPTION_CACHE}" ]]; then
  DATA_DIR="${DATA_DIR}" SPLIT=train OUTPUT="${CAPTION_CACHE}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    bash "${SCRIPT_DIR}/caption_augmented_captions.sh"
fi

if [[ "${RUN_FILTER}" == "1" || ! -s "${TRAIN_CSV}" ]]; then
  DATA_DIR="${DATA_DIR}" CAPTION_CACHE="${CAPTION_CACHE}" FILTERED_OUTPUT="${TRAIN_CSV}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    bash "${SCRIPT_DIR}/caption_augmented_filter.sh"
fi

if [[ "${LOAD_IN_4BIT}" == "1" ]]; then
  quantization_args=(--load-in-4bit)
else
  quantization_args=(--no-load-in-4bit)
fi

train_args=(
  -m src.caption_augmented.train
  --data-dir "${DATA_DIR}"
  --train-csv "${TRAIN_CSV}"
  --caption-cache "${CAPTION_CACHE}"
  --missing-caption-policy fail
  --output-dir "${OUTPUT_DIR}"
  --model-name "${MODEL_NAME}"
  --device-map auto
  --torch-dtype "${TORCH_DTYPE}"
  --attn-implementation "${ATTN_IMPLEMENTATION}"
  "${quantization_args[@]}"
  --use-lora
  --num-train-epochs "${NUM_TRAIN_EPOCHS}"
  --max-steps "${MAX_STEPS}"
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning-rate "${LEARNING_RATE}"
  --save-steps "${SAVE_STEPS}"
  --logging-steps "${LOGGING_STEPS}"
  --shuffle-augmentations-per-sample "${SHUFFLE_AUGMENTATIONS_PER_SAMPLE}"
  --shuffle-seed "${SHUFFLE_SEED}"
  --shuffle-keep-original
)

if [[ "${VAL_FRACTION}" != "0" ]]; then
  train_args+=(--val-fraction "${VAL_FRACTION}" --val-seed "${VAL_SEED}" --eval-max-new-tokens "${EVAL_MAX_NEW_TOKENS}")
  if [[ -n "${VAL_MAX_SAMPLES}" ]]; then
    train_args+=(--val-max-samples "${VAL_MAX_SAMPLES}")
  fi
fi

export CUDA_VISIBLE_DEVICES
python "${train_args[@]}"