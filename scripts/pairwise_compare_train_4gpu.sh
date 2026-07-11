#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/pairwise_compare_caption.yaml}"
TRAIN_CSV="${TRAIN_CSV:-data/train.csv}"
TRAIN_IMAGE_ROOT="${TRAIN_IMAGE_ROOT:-data/train}"
CAPTION_CACHE="${CAPTION_CACHE:-outputs/caption_augmented/train_captions.jsonl}"
CAPTION_MISSING_POLICY="${CAPTION_MISSING_POLICY:-fail}"
SPLIT_DIR="${SPLIT_DIR:-outputs/pairwise_compare/splits}"
PAIR_DIR="${PAIR_DIR:-outputs/pairwise_compare/pairs}"
VAL_SIZE="${VAL_SIZE:-0.1}"
SEED="${SEED:-42}"
PAIR_MODE="${PAIR_MODE:-canonical}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

export CUDA_VISIBLE_DEVICES

python -m src.data.make_split \
  --input "${TRAIN_CSV}" \
  --output-dir "${SPLIT_DIR}" \
  --val-size "${VAL_SIZE}" \
  --seed "${SEED}"

python -m src.pairwise_compare.make_pairs \
  --input "${TRAIN_CSV}" \
  --split-dir "${SPLIT_DIR}" \
  --output-dir "${PAIR_DIR}" \
  --pair-mode "${PAIR_MODE}" \
  --image-root "${TRAIN_IMAGE_ROOT}" \
  --check-images \
  --caption-cache "${CAPTION_CACHE}" \
  --caption-missing-policy "${CAPTION_MISSING_POLICY}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  -m src.pairwise_compare.train \
  --config "${CONFIG}"
