#!/usr/bin/env bash
set -euo pipefail

python cal_codebleu.py \
  --references "${1:-./references.txt}" \
  --predictions "${2:-./predictions.txt}" \
  --lang "${3:-java}"
