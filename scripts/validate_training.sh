#!/usr/bin/env bash
set -euo pipefail

python -m src.train_validate --profile local --data-dir data "$@"
