#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
SPLIT="${SPLIT:-train}"
OUTPUT="${OUTPUT:-outputs/constrained_likelihood_tta/${SPLIT}_captions.jsonl}"
CAPTION_DEVICE="${CAPTION_DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

caption_args=(
  -m src.constrained_likelihood_tta.captions
  --data-dir "${DATA_DIR}"
  --split "${SPLIT}"
  --output "${OUTPUT}"
  --caption-backend blip
  --caption-device "${CAPTION_DEVICE}"
)

export CUDA_VISIBLE_DEVICES
python "${caption_args[@]}"
