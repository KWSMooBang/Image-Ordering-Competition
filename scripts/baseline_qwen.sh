#!/usr/bin/env bash
set -euo pipefail

python -m src.baseline_qwen --data-dir data --output outputs/qwen2vl_submission.csv "$@"
