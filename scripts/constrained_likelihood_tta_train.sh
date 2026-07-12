#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
TRAIN_CSV="${TRAIN_CSV:-${DATA_DIR}/train.csv}"
CAPTION_CACHE="${CAPTION_CACHE:-outputs/constrained_likelihood_tta/train_captions.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/constrained_likelihood_tta/a100x4_bf16_lora}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
CAPTION_DEVICE="${CAPTION_DEVICE:-cuda}"
GENERATE_CAPTIONS="${GENERATE_CAPTIONS:-0}"

TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-0}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:--1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
SAVE_STEPS="${SAVE_STEPS:-200}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SHUFFLE_AUGMENTATIONS_PER_SAMPLE="${SHUFFLE_AUGMENTATIONS_PER_SAMPLE:-2}"
SHUFFLE_SEED="${SHUFFLE_SEED:-42}"

if [[ "${GENERATE_CAPTIONS}" == "1" || ! -s "${CAPTION_CACHE}" ]]; then
  caption_args=(
    -m src.constrained_likelihood_tta.captions
    --data-dir "${DATA_DIR}"
    --split train
    --output "${CAPTION_CACHE}"
    --caption-backend blip
    --caption-device "${CAPTION_DEVICE}"
  )
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" python "${caption_args[@]}"
fi

if [[ "${LOAD_IN_4BIT}" == "1" ]]; then
  quantization_args=(--load-in-4bit)
else
  quantization_args=(--no-load-in-4bit)
fi

train_args=(
  --standalone
  "--nproc_per_node=${NPROC_PER_NODE}"
  -m src.constrained_likelihood_tta.train
  --data-dir "${DATA_DIR}"
  --train-csv "${TRAIN_CSV}"
  --caption-cache "${CAPTION_CACHE}"
  --caption-missing-policy fail
  --output-dir "${OUTPUT_DIR}"
  --model-name "${MODEL_NAME}"
  --device-map local
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

export CUDA_VISIBLE_DEVICES
torchrun "${train_args[@]}"
