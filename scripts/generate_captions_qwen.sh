#!/usr/bin/env bash
set -euo pipefail

python -m src.generate_captions_qwen "$@"
