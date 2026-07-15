#!/usr/bin/env bash
set -euo pipefail

# End-to-end SNU AI Challenge pipeline built entirely on the caption-augmented
# Qwen3.5 VLM orderer (no pairwise / contextual-pairwise scoring model):
#
#   1. Generate a per-image caption cache for train.csv          (caption_augmented)
#   2. Audit + filter train.csv with SigLIP + caption relevance   (data_filtering)
#   3. QLoRA SFT the Qwen3.5-4B/9B orderer on the filtered CSV,
#      with real-time input-slot shuffle augmentation             (data_augmentation)
#   4. Run fresh-caption + permutation-TTA inference and write
#      the final submission.csv
#
# Run from the repository root: bash scripts/caption_augmented_pipeline.sh
# Override any variable the underlying scripts read, e.g.:
#   MODEL_NAME=Qwen/Qwen3.5-9B CUDA_VISIBLE_DEVICES=0 bash scripts/caption_augmented_pipeline.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="${DATA_DIR:-data}"
CAPTION_CACHE="${CAPTION_CACHE:-outputs/caption_augmented/train_captions.jsonl}"
TRAIN_CSV="${TRAIN_CSV:-outputs/data_filtering/train_filtered_caption_augmented.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/caption_augmented/qwen3_5_4b_qlora}"
SUBMISSION_OUTPUT="${SUBMISSION_OUTPUT:-outputs/caption_augmented/submission.csv}"

echo "[1/4] Generating train caption cache (BLIP)..."
DATA_DIR="${DATA_DIR}" SPLIT=train OUTPUT="${CAPTION_CACHE}" \
  bash "${SCRIPT_DIR}/caption_augmented_captions.sh"

echo "[2/4] Auditing + filtering train.csv with SigLIP relevance..."
DATA_DIR="${DATA_DIR}" CAPTION_CACHE="${CAPTION_CACHE}" FILTERED_OUTPUT="${TRAIN_CSV}" \
  bash "${SCRIPT_DIR}/caption_augmented_filter.sh"

echo "[3/4] QLoRA training Qwen3.5 orderer with shuffle augmentation..."
DATA_DIR="${DATA_DIR}" TRAIN_CSV="${TRAIN_CSV}" CAPTION_CACHE="${CAPTION_CACHE}" \
  OUTPUT_DIR="${OUTPUT_DIR}" GENERATE_CAPTIONS=0 RUN_FILTER=0 \
  bash "${SCRIPT_DIR}/caption_augmented_train.sh"

echo "[4/4] Running permutation-TTA inference..."
DATA_DIR="${DATA_DIR}" ADAPTER="${OUTPUT_DIR}" OUTPUT="${SUBMISSION_OUTPUT}" \
  bash "${SCRIPT_DIR}/caption_augmented_infer.sh"

echo "Pipeline complete. Submission written to ${SUBMISSION_OUTPUT}"
