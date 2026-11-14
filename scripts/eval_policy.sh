#!/usr/bin/env bash
# Evaluate one student checkpoint.
# The policy server runs in the conda env named by configs/eval.yaml.
# The Isaac client runs in the simulator image.
#
# Usage: scripts/eval_policy.sh [config] [--dry-run]
#        scripts/eval_policy.sh --dry-run [config]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBODOJO_ROOT="${ROBODOJO_ROOT:-${ROOT}/third_party/RoboDojo}"
CONFIG="${ROOT}/configs/eval.yaml"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/eval_policy.sh [config] [--dry-run]

  config      Config file (default: configs/eval.yaml)
  --dry-run   Validate config, links, and image; print planned commands;
              never start the policy server or the container.
EOF
}

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    --*)
      echo "[eval] Unknown argument: ${arg}" >&2
      usage >&2
      exit 2
      ;;
    *) CONFIG="${arg}" ;;
  esac
done

CONFIG="$(cd "$(dirname "${CONFIG}")" && pwd)/$(basename "${CONFIG}")"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export STACK_BLOCKS_CONFIG="${CONFIG}"
export ROBODOJO_ROOT
cd "${ROOT}"

if [[ ! -f "${ROBODOJO_ROOT}/scripts/robodojo.sh" ]]; then
  echo "RoboDojo 不在 ${ROBODOJO_ROOT}。先运行 scripts/bootstrap_third_party.sh。" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${ROOT}/scripts/sim_image.sh"
resolve_sim_image

eval "$(python3 - <<'PY'
import os
import shlex
from pathlib import Path
from starter.config import load_config, require

cfg = load_config(os.environ["STACK_BLOCKS_CONFIG"])
robodojo_root = Path(os.environ["ROBODOJO_ROOT"])

policy_dir = Path(require(cfg, "policy_dir"))
if not policy_dir.is_absolute():
    policy_dir = Path.cwd() / policy_dir

ckpt = Path(require(cfg, "ckpt"))
if not ckpt.is_absolute():
    ckpt = Path.cwd() / ckpt

env_cfg_file = robodojo_root / "env_cfg" / f"{require(cfg, 'env_cfg')}.yml"
env_cfg = load_config(env_cfg_file)

fields = {
    "POLICY_DIR": str(policy_dir),
    "TASK": require(cfg, "task"),
    "CKPT_PATH": str(ckpt),
    "CKPT_LABEL": ckpt.stem,
    "POLICY_ENV": require(cfg, "policy_env"),
    "ENV_CFG": require(cfg, "env_cfg"),
    "ACTION_TYPE": require(cfg, "action_type"),
    "SEED": require(cfg, "seed"),
    "EVAL_NUM": require(cfg, "eval_num"),
    "NUM_ENVS": require(cfg, "num_envs"),
    "POLICY_GPU": require(cfg, "policy_gpu"),
    "ENV_GPU": require(cfg, "env_gpu"),
    "ROBOT_CFG": env_cfg["config"]["robot"],
    "SIM_CFG_NAME": env_cfg["config"]["sim"],
}
for key, value in fields.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

POLICY_NAME="$(basename "${POLICY_DIR}")"
: "${CKPT_LABEL:?checkpoint must have a file name}"

if [[ ! -x "${POLICY_DIR}/setup_eval_policy_server.sh" ]]; then
  echo "缺少可执行的 ${POLICY_DIR}/setup_eval_policy_server.sh" >&2
  exit 1
fi
if [[ ! -f "${ROBODOJO_ROOT}/XPolicyLab/policy/${POLICY_NAME}/deploy.py" ]]; then
  echo "缺少策略适配器 ${ROBODOJO_ROOT}/XPolicyLab/policy/${POLICY_NAME}/deploy.py。" >&2
  echo "先创建 RoboDojo/XPolicyLab 和 XPolicyLab/policy/${POLICY_NAME} 链接。" >&2
  exit 1
fi
if [[ ! -e "/dev/nvidia${ENV_GPU}" ]]; then
  echo "主机上没有 /dev/nvidia${ENV_GPU}。" >&2
  exit 1
fi

# Assets is a symlink chain. Resolve its real directory, then read the robot's
# curobo config to learn the original absolute asset root embedded there.
ASSETS_LINK="${ROBODOJO_ROOT}/Assets"
if [[ ! -e "${ASSETS_LINK}" ]]; then
  echo "缺少 ${ASSETS_LINK} 或链接目标不存在。" >&2
  exit 1
fi
ASSETS_SOURCE="$(readlink -f "${ASSETS_LINK}")"
POLICY_PYTHON="$(conda run --no-capture-output -n "${POLICY_ENV}" python -c 'import sys; print(sys.executable)')"
ASSET_MOUNT_DEST="$("${POLICY_PYTHON}" - "${ROBODOJO_ROOT}/env_cfg/robot/${ROBOT_CFG}.yml" "${ASSETS_SOURCE}" <<'PY'
import sys
from pathlib import Path
import yaml

robot_cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
assets = Path(sys.argv[2])
robot_names = {robot["robot_name"] for robot in robot_cfg["robots"]}
roots = set()
for name in robot_names:
    cfg = yaml.safe_load((assets / "Robots" / name / "curobo.yml").read_text())
    robot_root = Path(cfg["robot_cfg"]["kinematics"]["asset_root_path"])
    if not robot_root.is_absolute() or robot_root.parts[-2:] != ("Robots", name):
        raise ValueError(f"Unexpected curobo asset root: {robot_root}")
    roots.add(robot_root.parent.parent)
if len(roots) != 1:
    raise ValueError(f"Expected one shared asset root, got {roots}")
print(roots.pop())
PY
)"

RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)_$$"
export ROBODOJO_RUN_ID="${RUN_ID}"
RUN_DIR="${ROOT}/outputs/eval/${RUN_ID}"
RESULTS_DIR="${ROOT}/outputs/eval/results"
mkdir -p "${RUN_DIR}/sim" "${RESULTS_DIR}"

SIM_SRC="${ROBODOJO_ROOT}/env_cfg/sim/${SIM_CFG_NAME}.yml"
SIM_PATH="${RUN_DIR}/sim/${SIM_CFG_NAME}.yml"
if [[ ! -f "${SIM_SRC}" ]]; then
  echo "缺少仿真配置 ${SIM_SRC}。" >&2
  exit 1
fi
"${POLICY_PYTHON}" - "${SIM_SRC}" "${SIM_PATH}" "${NUM_ENVS}" <<'PY'
import sys
from pathlib import Path
import yaml

src, dst, num_envs = sys.argv[1:]
num_envs = int(num_envs)
if num_envs < 1:
    raise ValueError("num_envs must be positive")
cfg = yaml.safe_load(Path(src).read_text())
cfg["scene"]["num_envs"] = num_envs
Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
cp -- "${CONFIG}" "${RUN_DIR}/eval.yaml"

ICD_PATH="${RUN_DIR}/nvidia_icd.json"
cat > "${ICD_PATH}" <<'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0",
        "api_version" : "1.3.277"
    }
}
EOF

PORT="$(python3 - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)"

CONTAINER_NAME="stack-blocks-eval-${RUN_ID}"
SERVER_LOG="${RUN_DIR}/policy_server.log"
CLIENT_LOG="${RUN_DIR}/client.log"

server_cmd=(
  bash "${POLICY_DIR}/setup_eval_policy_server.sh"
  RoboDojo "${TASK}" "${CKPT_PATH}" "${ENV_CFG}" "${ACTION_TYPE}"
  "${SEED}" "${POLICY_GPU}" "${POLICY_ENV}" "${PORT}"
)

maps=()
shopt -s nullglob
for lib in \
  /usr/lib/x86_64-linux-gnu/libcuda.so* \
  /usr/lib/x86_64-linux-gnu/libnvidia*.so* \
  /usr/lib/x86_64-linux-gnu/libEGL_nvidia* \
  /usr/lib/x86_64-linux-gnu/libGLX_nvidia* \
  /usr/lib/x86_64-linux-gnu/libGLESv*_nvidia* \
  /usr/lib/x86_64-linux-gnu/libnvcuvid.so* \
  /usr/lib/x86_64-linux-gnu/libnvoptix.so*
do
  [[ -e "${lib}" ]] || continue
  maps+=(-v "${lib}:${lib}:ro")
done
shopt -u nullglob

docker_cmd=(
  docker run --rm --name "${CONTAINER_NAME}"
  --device "/dev/nvidia${ENV_GPU}"
  --device /dev/nvidiactl
  --device /dev/nvidia-uvm
  --device /dev/nvidia-modeset
  --device /dev/nvidia-uvm-tools
  --device /dev/dri
  "${maps[@]}"
  --network host --ipc host
  -e ACCEPT_EULA=Y
  -e PRIVACY_CONSENT=Y
  -e OMNI_KIT_ACCEPT_EULA=YES
  -e NVIDIA_DRIVER_CAPABILITIES=all
  -e "VK_DRIVER_FILES=${ICD_PATH}"
  -e "VK_ICD_FILENAMES=${ICD_PATH}"
  -e LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
  -e CUDA_VISIBLE_DEVICES=0
  -e "ROBODOJO_ROOT=${ROBODOJO_ROOT}"
  -e "ROBODOJO_RUN_ID=${RUN_ID}"
  -v "${ROOT}:${ROOT}"
  -v "${ASSETS_SOURCE}:${ASSET_MOUNT_DEST}:ro"
  -v "${ASSETS_SOURCE}:${ASSETS_SOURCE}:ro"
  -v "${ASSETS_SOURCE}:/workspace/RoboDojo/Assets:ro"
  -v "${ICD_PATH}:/etc/vulkan/icd.d/nvidia_icd.json:ro"
  -v "${SIM_PATH}:${SIM_SRC}:ro"
  -v "${RESULTS_DIR}:${ROBODOJO_ROOT}/eval_result"
  "${IMAGE}"
  bash "${ROBODOJO_ROOT}/scripts/robodojo.sh" client
  --policy-dir "${POLICY_DIR}"
  --task "${TASK}"
  --policy-host 127.0.0.1
  --policy-port "${PORT}"
  --ckpt "${CKPT_LABEL}"
  --env-cfg "${ENV_CFG}"
  --action-type "${ACTION_TYPE}"
  --seed "${SEED}"
  --env-gpu 0
  --eval-num "${EVAL_NUM}"
)

echo "[eval] run_dir=${RUN_DIR}"
echo "[eval] results=${RESULTS_DIR}"
echo "[eval] checkpoint=$(readlink -f "${CKPT_PATH}")"
echo "[eval] policy server in conda env ${POLICY_ENV} on GPU ${POLICY_GPU}"
echo "[eval] Isaac client in ${IMAGE} on GPU ${ENV_GPU}, port ${PORT}, num_envs=${NUM_ENVS}"

if [[ "${DRY_RUN}" == 1 ]]; then
  if [[ ! -e "${CKPT_PATH}" ]]; then
    echo "[eval] warning: checkpoint ${CKPT_PATH} 尚不存在；policy_last.ckpt 在训练完成后写出。" >&2
  fi
  printf '[eval] dry-run server:'
  printf ' %q' "${server_cmd[@]}"
  printf '\n'
  printf '[eval] dry-run client:'
  printf ' %q' "${docker_cmd[@]}"
  printf '\n'
  exit 0
fi

if [[ ! -e "${CKPT_PATH}" ]]; then
  echo "缺少 checkpoint ${CKPT_PATH}。" >&2
  exit 1
fi

set -m
"${server_cmd[@]}" > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -- -"${SERVER_PID}" 2>/dev/null || kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 60); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    wait "${SERVER_PID}" || true
    echo "[eval] 策略服务已退出，没有启动仿真镜像。日志: ${SERVER_LOG}" >&2
    tail -n 40 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  if timeout 1 bash -c ">/dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "${ready}" != 1 ]]; then
  echo "[eval] 策略服务在 60 秒内没有监听 ${PORT}。日志: ${SERVER_LOG}" >&2
  exit 1
fi

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
"${docker_cmd[@]}" 2>&1 | tee "${CLIENT_LOG}"
