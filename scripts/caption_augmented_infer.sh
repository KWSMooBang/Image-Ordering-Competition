#!/usr/bin/env bash
set -euo pipefail

# Runs fresh-caption + permutation-TTA inference with the trained Qwen3.5
# orderer adapter and writes the competition submission.csv.
# infer.py always generates captions fresh per test sample, so there is no
# caption-cache step here (unlike training).

DATA_DIR="${DATA_DIR:-data}"
ADAPTER="${ADAPTER:-checkpoints/caption_augmented/qwen3_5_4b_qlora}"
OUTPUT="${OUTPUT:-outputs/caption_augmented/submission.csv}"
RAW_OUTPUT="${RAW_OUTPUT:-outputs/caption_augmented/raw_outputs.jsonl}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B}"

CAPTION_BACKEND="${CAPTION_BACKEND:-blip}"
CAPTION_DEVICE="${CAPTION_DEVICE:-cuda}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
TTA_PERMUTATIONS="${TTA_PERMUTATIONS:-4}"
TTA_SEED="${TTA_SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# COMPARISON_MODE=whole (default): one VLM call orders all 4 images, ensembled
#   over TTA_PERMUTATIONS input-slot views.
# COMPARISON_MODE=pairwise: 6 two-image comparisons per sample, combined by
#   searching the 24 valid orders for the one with highest pairwise agreement.
#   Set PAIRWISE_SYMMETRY_CHECK=1 to ask each pair twice (swapped presentation
#   order) and keep only judgments both calls agree on.
# COMPARISON_MODE=adaptive: whole-mode TTA first; escalates to one extra
#   targeted pairwise verification call only when the top-2 TTA candidates
#   disagree by exactly one swapped pair.
COMPARISON_MODE="${COMPARISON_MODE:-whole}"
PAIRWISE_MAX_NEW_TOKENS="${PAIRWISE_MAX_NEW_TOKENS:-32}"
PAIRWISE_SYMMETRY_CHECK="${PAIRWISE_SYMMETRY_CHECK:-0}"

infer_args=(
  -m src.caption_augmented.infer
  --data-dir "${DATA_DIR}"
  --output "${OUTPUT}"
  --raw-output "${RAW_OUTPUT}"
  --caption-backend "${CAPTION_BACKEND}"
  --caption-device "${CAPTION_DEVICE}"
  --order-model "${MODEL_NAME}"
  --order-adapter "${ADAPTER}"
  --qwen-torch-dtype "${TORCH_DTYPE}"
  --attn-implementation "${ATTN_IMPLEMENTATION}"
  --comparison-mode "${COMPARISON_MODE}"
)

if [[ "${COMPARISON_MODE}" == "pairwise" ]]; then
  infer_args+=(--pairwise-max-new-tokens "${PAIRWISE_MAX_NEW_TOKENS}")
  if [[ "${PAIRWISE_SYMMETRY_CHECK}" == "1" ]]; then
    infer_args+=(--pairwise-symmetry-check)
  fi
elif [[ "${COMPARISON_MODE}" == "adaptive" ]]; then
  infer_args+=(
    --tta-permutations "${TTA_PERMUTATIONS}" --tta-seed "${TTA_SEED}"
    --pairwise-max-new-tokens "${PAIRWISE_MAX_NEW_TOKENS}"
  )
else
  infer_args+=(--tta-permutations "${TTA_PERMUTATIONS}" --tta-seed "${TTA_SEED}")
fi

if [[ -n "${MAX_SAMPLES}" ]]; then
  infer_args+=(--max-samples "${MAX_SAMPLES}")
fi

export CUDA_VISIBLE_DEVICES
python "${infer_args[@]}"
