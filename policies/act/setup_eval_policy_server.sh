#!/usr/bin/env bash
# Arguments: bench task checkpoint env_cfg action_type seed gpu conda_env port [host]
set -euo pipefail
if [[ $# -lt 9 ]]; then
  echo "用法: setup_eval_policy_server.sh bench task checkpoint env_cfg action_type seed gpu conda_env port [host]" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
XPL_ROOT="${ROOT}/third_party/XPolicyLab"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$8"
cd "${ROOT}"
CHECKPOINT="$(realpath -e -- "$3")"
export PYTHONPATH="${ROOT}/src:${ROOT}/third_party:${XPL_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$7"
echo "[ACT] checkpoint=${CHECKPOINT}; env_cfg=$4; action_type=$5; GPU=$7"
exec python -u "${XPL_ROOT}/setup_policy_server.py" \
  --config_path "${SCRIPT_DIR}/deploy.yml" \
  --overrides \
  bench_name="$1" task_name="$2" ckpt_name="$3" checkpoint_path="${CHECKPOINT}" \
  env_cfg_type="$4" action_type="$5" seed="$6" gpu_id=0 \
  port="$9" host="${10:-127.0.0.1}"
