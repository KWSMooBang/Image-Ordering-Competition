#!/usr/bin/env bash
set -euo pipefail

# Audits train.csv with src.data_filtering (blank/duplicate frames + SigLIP
# image-text and caption-text relevance, using the caption_augmented caption
# cache) and writes a filtered train CSV. This filtered CSV is the --train-csv
# input consumed by caption_augmented_train.sh, so data_filtering's manifest
# actually changes what the orderer trains on instead of being a report-only
# artifact.

DATA_DIR="${DATA_DIR:-data}"
CAPTION_CACHE="${CAPTION_CACHE:-outputs/caption_augmented/train_captions.jsonl}"
AUDIT_OUTPUT="${AUDIT_OUTPUT:-outputs/data_filtering/train_audit_caption_augmented.csv}"
FILTERED_OUTPUT="${FILTERED_OUTPUT:-outputs/data_filtering/train_filtered_caption_augmented.csv}"
RELEVANCE_BACKEND="${RELEVANCE_BACKEND:-siglip}"
SIGLIP_MODEL="${SIGLIP_MODEL:-google/siglip-so400m-patch14-384}"
SIGLIP_DEVICE="${SIGLIP_DEVICE:-cuda}"
DROP_ACTIONS="${DROP_ACTIONS:-drop_from_supervised}"
GENERATE_CAPTIONS="${GENERATE_CAPTIONS:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ "${GENERATE_CAPTIONS}" == "1" || ! -s "${CAPTION_CACHE}" ]]; then
  DATA_DIR="${DATA_DIR}" SPLIT=train OUTPUT="${CAPTION_CACHE}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    bash "$(dirname "${BASH_SOURCE[0]}")/caption_augmented_captions.sh"
fi

audit_args=(
  -m src.data_filtering.audit
  --data-dir "${DATA_DIR}"
  --output "${AUDIT_OUTPUT}"
  --filtered-output "${FILTERED_OUTPUT}"
  --relevance-backend "${RELEVANCE_BACKEND}"
  --siglip-model "${SIGLIP_MODEL}"
  --siglip-device "${SIGLIP_DEVICE}"
  --drop-actions "${DROP_ACTIONS}"
)

if [[ "${RELEVANCE_BACKEND}" == "siglip" && -s "${CAPTION_CACHE}" ]]; then
  audit_args+=(--caption-cache "${CAPTION_CACHE}")
fi

export CUDA_VISIBLE_DEVICES
python "${audit_args[@]}"
