#!/usr/bin/env bash
set -euo pipefail

# Generates a per-image caption cache (JSONL) for the caption_augmented idea.
# Used to build train-time captions for supervision and for the data_filtering
# SigLIP relevance audit. Inference (caption_augmented_infer.sh) generates its
# own fresh captions and does not read this cache.

DATA_DIR="${DATA_DIR:-data}"
SPLIT="${SPLIT:-train}"
OUTPUT="${OUTPUT:-outputs/caption_augmented/${SPLIT}_captions.jsonl}"
CAPTION_BACKEND="${CAPTION_BACKEND:-blip}"
CAPTION_MODEL="${CAPTION_MODEL:-}"
CAPTION_DEVICE="${CAPTION_DEVICE:-cuda}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

caption_args=(
  -m src.caption_augmented.captions
  --data-dir "${DATA_DIR}"
  --split "${SPLIT}"
  --output "${OUTPUT}"
  --caption-backend "${CAPTION_BACKEND}"
  --caption-device "${CAPTION_DEVICE}"
)

if [[ -n "${CAPTION_MODEL}" ]]; then
  caption_args+=(--caption-model "${CAPTION_MODEL}")
fi
if [[ -n "${MAX_SAMPLES}" ]]; then
  caption_args+=(--max-samples "${MAX_SAMPLES}")
fi

export CUDA_VISIBLE_DEVICES
python "${caption_args[@]}"
