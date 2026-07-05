#!/usr/bin/env bash
set -euo pipefail

python -m src.identity_baseline --data-dir data --output outputs/identity_submission.csv "$@"
