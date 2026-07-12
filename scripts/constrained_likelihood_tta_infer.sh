#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
CAPTION_CACHE="${CAPTION_CACHE:-outputs/constrained_likelihood_tta/test_captions.jsonl}"
ADAPTER="${ADAPTER:-checkpoints/constrained_likelihood_tta/a100x4_bf16_lora}"
OUTPUT="${OUTPUT:-outputs/constrained_likelihood_tta/submission.csv}"
RAW_OUTPUT="${RAW_OUTPUT:-outputs/constrained_likelihood_tta/raw_outputs.jsonl}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B}"
GENERATE_CAPTIONS="${GENERATE_CAPTIONS:-0}"
CAPTION_DEVICE="${CAPTION_DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CANDIDATE_BATCH_SIZE="${CANDIDATE_BATCH_SIZE:-4}"
TTA_PERMUTATIONS="${TTA_PERMUTATIONS:-4}"

if [[ "${GENERATE_CAPTIONS}" == "1" || ! -s "${CAPTION_CACHE}" ]]; then
  caption_args=(
    -m src.constrained_likelihood_tta.captions
    --data-dir "${DATA_DIR}"
    --split test
    --output "${CAPTION_CACHE}"
    --caption-backend blip
    --caption-device "${CAPTION_DEVICE}"
  )
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python "${caption_args[@]}"
fi

infer_args=(
  -m src.constrained_likelihood_tta.infer
  --data-dir "${DATA_DIR}"
  --caption-cache "${CAPTION_CACHE}"
  --caption-missing-policy fail
  --order-model "${MODEL_NAME}"
  --order-adapter "${ADAPTER}"
  --output "${OUTPUT}"
  --raw-output "${RAW_OUTPUT}"
  --torch-dtype bfloat16
  --attn-implementation sdpa
  --candidate-batch-size "${CANDIDATE_BATCH_SIZE}"
  --score-normalization sum
  --tta-permutations "${TTA_PERMUTATIONS}"
  --tta-seed 42
)

export CUDA_VISIBLE_DEVICES
python "${infer_args[@]}"
