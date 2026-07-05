#!/usr/bin/env bash
set -euo pipefail

python -m src.train_caption_augmented_qwen "$@"
