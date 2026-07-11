#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/pairwise_compare_caption.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/experiments/exp_pairwise_compare_caption_siglip/checkpoints/best.pt}"
TEST_CAPTION_CACHE="${TEST_CAPTION_CACHE:-outputs/caption_augmented/test_captions.jsonl}"
CAPTION_MISSING_POLICY="${CAPTION_MISSING_POLICY:-fail}"
OUTPUT="${OUTPUT:-outputs/pairwise_compare/submission.csv}"
PAIR_PREDICTIONS="${PAIR_PREDICTIONS:-outputs/pairwise_compare/test_pair_predictions.csv}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

export CUDA_VISIBLE_DEVICES

python -m src.pairwise_compare.infer \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --output "${OUTPUT}" \
  --save-pair-predictions "${PAIR_PREDICTIONS}" \
  --test-caption-cache "${TEST_CAPTION_CACHE}" \
  --caption-missing-policy "${CAPTION_MISSING_POLICY}" \
  --data-parallel
