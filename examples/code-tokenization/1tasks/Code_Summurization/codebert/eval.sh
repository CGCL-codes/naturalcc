#!/usr/bin/env bash
set -euo pipefail

python cal_rouge.py \
  --references "${1:-./references.txt}" \
  --predictions "${2:-./predictions.txt}"
