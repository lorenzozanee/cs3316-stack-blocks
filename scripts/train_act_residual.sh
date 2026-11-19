#!/usr/bin/env bash
# Online ACT residual PPO. Isaac runs in the simulator image.
# Usage: train_act_residual.sh <offline.ckpt> [env.yaml] [act_online.yaml]
# Activate the ACT environment before running this script.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
cd "${ROOT}"

CHECKPOINT="${1:?Usage: train_act_residual.sh <offline.ckpt> [env.yaml] [act_online.yaml]}"
ENV_CONFIG="${2:-configs/env.yaml}"
ONLINE_CONFIG="${3:-configs/act_online.yaml}"
META_TEXT="$(python3 - "${CHECKPOINT}" "${ENV_CONFIG}" "${ONLINE_CONFIG}" <<'PY'
import sys
from pathlib import Path
import torch
from starter.config import load_config, require
from starter.envs.parallel import parallel_plan
from starter.policies.act_online import _validate_config

checkpoint = Path(sys.argv[1]).expanduser().resolve()
if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
cfg = load_config(sys.argv[2])
cfg.update(load_config(sys.argv[3]))
_validate_config(cfg)
policy_gpu, env_gpu = int(require(cfg, 'policy_gpu')), int(require(cfg, 'env_gpu'))
if policy_gpu == env_gpu and not require(cfg, 'allow_shared_gpu'):
    raise ValueError('Set allow_shared_gpu=true explicitly when policy and Isaac share a GPU.')
if not torch.cuda.is_available() or min(policy_gpu, env_gpu) < 0 or max(policy_gpu, env_gpu) >= torch.cuda.device_count():
    raise ValueError('Configured policy_gpu/env_gpu is unavailable; check configs/env.yaml.')
min_reset_free_vram_mib = cfg.get('min_reset_free_vram_mib', 0)
if type(min_reset_free_vram_mib) is not int or min_reset_free_vram_mib < 0:
    raise ValueError('min_reset_free_vram_mib must be a nonnegative integer.')
for key in ('policy_gpu', 'env_gpu', 'env_port', 'rollout_task', 'env_cfg', 'num_envs'):
    print(require(cfg, key))
print(min_reset_free_vram_mib)
PY
)"
mapfile -t META <<< "${META_TEXT}"
echo "[act residual PPO] policy GPU ${META[0]}, Isaac GPU ${META[1]}"
# Keep physical GPU numbering consistent with the learner's policy_gpu setting.
STOP="${ROOT}/outputs/act_residual/.stop"
mkdir -p "${ROOT}/outputs/act_residual"
rm -f "${STOP}"
bash "${ROOT}/scripts/run_online_env.sh" \
  "${META[1]}" "${META[2]}" "${META[3]}" "${META[4]}" "${META[5]}" 0 "${STOP}" \
  stack-blocks-act-env "${META[6]}" &
ENV_PID=$!
cleanup() {
  touch "${STOP}"
  docker rm -f stack-blocks-act-env >/dev/null 2>&1 || true
  kill "${ENV_PID}" >/dev/null 2>&1 || true
  wait "${ENV_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

python3 -m starter.pipelines.act online --config "${ENV_CONFIG}" \
  --online-config "${ONLINE_CONFIG}" --checkpoint "${CHECKPOINT}"
