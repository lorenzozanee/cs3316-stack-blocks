#!/usr/bin/env bash
# Run inside the existing ACT conda environment.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
python -m pip install -r "${SCRIPT_DIR}/requirements.txt"
