#!/usr/bin/env bash
# Offline Diffusion Policy. Does not start Isaac.
# Activate your own policy environment before running this script.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src:${ROOT}/third_party:${PYTHONPATH:-}"
cd "${ROOT}"

python3 -m starter.pipelines.diffusion offline --config "${1:-configs/env.yaml}"
