#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODEGEN_DIR="${CODEGEN_DIR:-./1tasks/Code_Generation/dataset/concode}"
CODESUMM_ROOT="${CODESUMM_ROOT:-./1tasks/Code_Summurization/dataset}"

python "${SCRIPT_DIR}/extract-data-codegeneration.py" --dataset-dir "${CODEGEN_DIR}" --seed 42
python "${SCRIPT_DIR}/extract-data-codesummarization.py" --dataset-root "${CODESUMM_ROOT}" --langs java,python --seed 42

echo "[ok] public RQ2 data preparation finished."
