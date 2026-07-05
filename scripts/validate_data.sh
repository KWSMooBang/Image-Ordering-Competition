#!/usr/bin/env bash
set -euo pipefail

python -m src.validate_data --data-dir data "$@"
