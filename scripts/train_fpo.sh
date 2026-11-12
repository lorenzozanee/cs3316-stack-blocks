#!/usr/bin/env bash
# FPO++ fine-tuning with the learner and Isaac sharing one GPU.
# CONTAINER_LEARNER=1 reuses the simulator image's CUDA PyTorch.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
cd "${ROOT}"

usage() {
  cat <<'EOF'
用法: bash scripts/train_fpo.sh CHECKPOINT [TRAINING_CONFIG]

  CHECKPOINT       在线训练起始的离线 checkpoint。
  TRAINING_CONFIG  在线训练配置，默认 configs/fpo.yaml。

环境变量:
  ENV_CONFIG=...         环境配置，默认 configs/env.yaml。
  EXTERNAL_ENV=1         仿真已在运行，不启动也不清理任何仿真资源。
  PREFLIGHT=1            只运行 pipeline 预检，不启动仿真。
  RESUME_CHECKPOINT=...  从该 checkpoint 继续在线训练。
  CONTAINER_LEARNER=1    用容器里的 CUDA torch 跑 pipeline，不用宿主 torch；
                         内部模式复用本次仿真容器，预检模式起短时容器，
                         外部模式需给 LEARNER_CONTAINER。
  LEARNER_CONTAINER=...  指定现成的学习者容器名，只 exec，不启动也不清理它。
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 2
fi
case "$1" in
  -h|--help)
    usage
    exit 0
    ;;
esac

CHECKPOINT="$1"
TRAINING_CONFIG="${2:-configs/fpo.yaml}"
ENV_CONFIG="${ENV_CONFIG:-configs/env.yaml}"
EXTERNAL_ENV="${EXTERNAL_ENV:-0}"
PREFLIGHT="${PREFLIGHT:-0}"
CONTAINER_LEARNER="${CONTAINER_LEARNER:-0}"
LEARNER_CONTAINER="${LEARNER_CONTAINER:-}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

for flag in "${EXTERNAL_ENV}" "${PREFLIGHT}" "${CONTAINER_LEARNER}"; do
  if [[ "${flag}" != "0" && "${flag}" != "1" ]]; then
    echo "EXTERNAL_ENV、PREFLIGHT 和 CONTAINER_LEARNER 只能是 0 或 1，得到 ${flag}。" >&2
    exit 2
  fi
done
if [[ -n "${LEARNER_CONTAINER}" && "${CONTAINER_LEARNER}" != "1" ]]; then
  echo "LEARNER_CONTAINER 只在 CONTAINER_LEARNER=1 时生效。" >&2
  exit 2
fi

CONTAINER_NAME="stack-blocks-fpo-env"
PREFLIGHT_CONTAINER_NAME="stack-blocks-fpo-preflight"
LEARNER_KIND="host"
LEARNER_NAME=""
if [[ "${CONTAINER_LEARNER}" == "1" ]]; then
  if [[ -n "${LEARNER_CONTAINER}" ]]; then
    LEARNER_KIND="container"
    LEARNER_NAME="${LEARNER_CONTAINER}"
  elif [[ "${PREFLIGHT}" == "1" ]]; then
    # No simulator container in preflight, so use a short-lived python-only
    # container from the sim image instead of the (CUDA-less) host torch.
    LEARNER_KIND="oneoff"
    LEARNER_NAME="${PREFLIGHT_CONTAINER_NAME}"
  elif [[ "${EXTERNAL_ENV}" == "1" ]]; then
    echo "CONTAINER_LEARNER=1 配合外部仿真时必须给出 LEARNER_CONTAINER。" >&2
    exit 2
  else
    LEARNER_KIND="container"
    LEARNER_NAME="${CONTAINER_NAME}"
  fi
  echo "[fpo] learner: ${LEARNER_KIND} ${LEARNER_NAME}" >&2
fi

absolute() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "${ROOT}/$1" ;;
  esac
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "$2不存在: $1" >&2
    exit 1
  fi
}

CHECKPOINT="$(absolute "${CHECKPOINT}")"
require_file "${CHECKPOINT}" "checkpoint "
TRAINING_CONFIG="$(absolute "${TRAINING_CONFIG}")"
require_file "${TRAINING_CONFIG}" "训练配置 "
ENV_CONFIG="$(absolute "${ENV_CONFIG}")"
require_file "${ENV_CONFIG}" "环境配置 "
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  RESUME_CHECKPOINT="$(absolute "${RESUME_CHECKPOINT}")"
  require_file "${RESUME_CHECKPOINT}" "resume checkpoint "
fi

# Training config top-level keys override env config keys. Only the fields the
# launcher needs are read here; fpo.* stays owned by the trainer.
if ! META="$(python3 - "${ENV_CONFIG}" "${TRAINING_CONFIG}" <<'PY'
import shlex
import sys

from starter.config import load_config

env_path, train_path = sys.argv[1], sys.argv[2]
merged = dict(load_config(env_path))
merged.update(load_config(train_path))

fpo = merged.get("fpo")
if not isinstance(fpo, dict):
    raise SystemExit(f"{train_path}: 顶层需要 fpo 映射。")
output_dir = fpo.get("output_dir")
if not isinstance(output_dir, str) or not output_dir.strip():
    raise SystemExit(f"{train_path}: fpo.output_dir 必须是非空字符串。")


def value(key):
    if key not in merged:
        raise SystemExit(f"合并 {env_path} 后缺少 {key}。")
    return merged[key]


def integer(key, minimum=0, maximum=None):
    item = value(key)
    if isinstance(item, bool) or not isinstance(item, int):
        raise SystemExit(f"{key} 必须是整数，得到 {item!r}。")
    if item < minimum or (maximum is not None and item > maximum):
        raise SystemExit(f"{key} 超出范围，得到 {item!r}。")
    return item


def text(key):
    item = value(key)
    if not isinstance(item, str) or not item.strip():
        raise SystemExit(f"{key} 必须是非空字符串，得到 {item!r}。")
    return item


fields = {
    "POLICY_GPU": integer("policy_gpu"),
    "ENV_HOST": text("env_host"),
    "ENV_PORT": integer("env_port", 1, 65535),
    "ROLLOUT_TASK": text("rollout_task"),
    "ENV_CFG_NAME": text("env_cfg"),
    "NUM_ENVS": integer("num_envs", 1),
    "FPO_OUTPUT_DIR": output_dir,
}
for name, item in fields.items():
    print(f"{name}={shlex.quote(str(item))}")
PY
)"; then
  exit 1
fi
eval "${META}"

FPO_OUTPUT_DIR="$(absolute "${FPO_OUTPUT_DIR}")"
if [[ "${CONTAINER_LEARNER}" == "1" && -z "${LEARNER_CONTAINER}" ]]; then
  python3 - "${ROOT}" "${CHECKPOINT}" "${TRAINING_CONFIG}" "${ENV_CONFIG}" "${FPO_OUTPUT_DIR}" "${RESUME_CHECKPOINT}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
for item in sys.argv[2:]:
    if item and not Path(item).resolve().is_relative_to(root):
        raise SystemExit(f"容器模式只挂载项目目录，请把配置、checkpoint 和输出放在 {root} 内: {item}")
PY
fi
mkdir -p "${FPO_OUTPUT_DIR}"

PIPELINE_ARGS=(
  online
  --config "${ENV_CONFIG}"
  --training-config "${TRAINING_CONFIG}"
  --checkpoint "${CHECKPOINT}"
)
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  PIPELINE_ARGS+=(--resume "${RESUME_CHECKPOINT}")
fi

# Preflight only needs the image's CUDA torch, not Isaac. Mount the same driver
# pieces as run_online_env.sh minus GL/Vulkan; the trap removes this container.
run_oneoff_learner() {
  # shellcheck disable=SC1091
  source "${ROOT}/scripts/sim_image.sh"
  resolve_sim_image
  if [[ ! -e "/dev/nvidia${POLICY_GPU}" ]]; then
    echo "主机上没有 /dev/nvidia${POLICY_GPU}，这是合并配置的 policy_gpu。" >&2
    exit 1
  fi
  if [[ -n "$(docker ps -q --filter "name=^${LEARNER_NAME}$")" ]]; then
    echo "容器 ${LEARNER_NAME} 正在运行；先停止旧的预检。" >&2
    exit 1
  fi
  docker rm -f "${LEARNER_NAME}" >/dev/null 2>&1 || true
  trap 'docker rm -f "${LEARNER_NAME}" >/dev/null 2>&1 || true' EXIT
  local -a driver=()
  local lib
  shopt -s nullglob
  for lib in \
    /usr/lib/x86_64-linux-gnu/libcuda.so* \
    /usr/lib/x86_64-linux-gnu/libnvidia*.so* \
    /usr/lib/x86_64-linux-gnu/libnvcuvid.so*
  do
    driver+=(-v "${lib}:${lib}:ro")
  done
  shopt -u nullglob
  # The image entrypoint changes the working directory, so every path the
  # pipeline receives is absolute and -w is only a fallback.
  docker run --rm --name "${LEARNER_NAME}" --entrypoint python \
    -w "${ROOT}" \
    -v "${ROOT}:${ROOT}" \
    --device "/dev/nvidia${POLICY_GPU}" \
    --device /dev/nvidiactl \
    --device /dev/nvidia-uvm \
    --device /dev/nvidia-modeset \
    --device /dev/nvidia-uvm-tools \
    "${driver[@]}" \
    -e "PYTHONPATH=${ROOT}/src" \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
    "${IMAGE}" \
    -u -m starter.pipelines.diffusion "$@"
}

# The learner container has the repository mounted at the same absolute paths,
# and its own CUDA-enabled torch. No host CUDA install is needed.
run_pipeline() {
  case "${LEARNER_KIND}" in
    host)
      python3 -m starter.pipelines.diffusion "$@"
      ;;
    container)
      docker exec -w "${ROOT}" -e "PYTHONPATH=${ROOT}/src" \
        "${LEARNER_NAME}" python -u -m starter.pipelines.diffusion "$@"
      ;;
    oneoff)
      run_oneoff_learner "$@"
      ;;
  esac
}

require_learner_container() {
  if [[ -z "$(docker ps -q --filter "name=^${LEARNER_NAME}$")" ]]; then
    echo "容器 ${LEARNER_NAME} 未运行；CONTAINER_LEARNER=1 只对现成容器 exec。" >&2
    exit 1
  fi
  if ! docker exec "${LEARNER_NAME}" test -f "${ROOT}/src/starter/pipelines/diffusion.py"; then
    echo "容器 ${LEARNER_NAME} 没有按绝对路径挂载 ${ROOT}，无法从中运行 pipeline。" >&2
    exit 1
  fi
}

export CUDA_VISIBLE_DEVICES="${POLICY_GPU}"
if [[ "${PREFLIGHT}" == "1" ]]; then
  echo "[fpo] preflight: ${TRAINING_CONFIG} + ${ENV_CONFIG}, checkpoint ${CHECKPOINT}"
  if [[ "${LEARNER_KIND}" == "container" ]]; then
    require_learner_container
  fi
  run_pipeline "${PIPELINE_ARGS[@]}" --preflight
  exit 0
fi

echo "[fpo] policy GPU ${POLICY_GPU}, ${ROLLOUT_TASK} (${ENV_CFG_NAME}), ${NUM_ENVS} env(s), ${ENV_HOST}:${ENV_PORT}"

if [[ "${EXTERNAL_ENV}" == "1" ]]; then
  echo "[fpo] 使用外部仿真 ${ENV_HOST}:${ENV_PORT}，退出时不清理仿真。" >&2
  if [[ "${LEARNER_KIND}" == "container" ]]; then
    require_learner_container
  fi
  run_pipeline "${PIPELINE_ARGS[@]}"
  exit 0
fi

if [[ ! -e "/dev/nvidia${POLICY_GPU}" ]]; then
  echo "主机上没有 /dev/nvidia${POLICY_GPU}，这是合并配置的 policy_gpu。" >&2
  exit 1
fi

# Do not attach to a server someone else started on this port.
if ! python3 - "${ENV_HOST}" "${ENV_PORT}" <<'PY'
import socket
import sys

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind((sys.argv[1], int(sys.argv[2])))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
then
  echo "${ENV_HOST}:${ENV_PORT} 已被占用；不接管已有 server。" >&2
  exit 1
fi

if [[ -n "$(docker ps -q --filter "name=^${CONTAINER_NAME}$")" ]]; then
  echo "容器 ${CONTAINER_NAME} 正在运行；先停止旧的 FPO 训练。" >&2
  exit 1
fi
STOP_FILE="${FPO_OUTPUT_DIR}/.env.stop"
ENV_LOG="${FPO_OUTPUT_DIR}/env.log"
rm -f "${STOP_FILE}"

cleanup() {
  touch "${STOP_FILE}"
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  kill "${ENV_PID}" >/dev/null 2>&1 || true
  wait "${ENV_PID}" >/dev/null 2>&1 || true
}

bash "${ROOT}/scripts/run_online_env.sh" \
  "${POLICY_GPU}" "${ENV_PORT}" "${ROLLOUT_TASK}" "${ENV_CFG_NAME}" "${NUM_ENVS}" 0 \
  "${STOP_FILE}" "${CONTAINER_NAME}" >"${ENV_LOG}" 2>&1 &
ENV_PID=$!
trap cleanup EXIT

# A TCP probe would consume RemoteEnv's single learner connection.
ENV_START_TIMEOUT=3600
started=$SECONDS
until python3 - "${ENV_PORT}" <<'PY'
import sys
from pathlib import Path

port = int(sys.argv[1])
for name in ("/proc/net/tcp", "/proc/net/tcp6"):
    path = Path(name)
    if path.exists():
        for line in path.read_text().splitlines()[1:]:
            fields = line.split()
            if fields[3] == "0A" and int(fields[1].rsplit(":", 1)[1], 16) == port:
                raise SystemExit(0)
raise SystemExit(1)
PY
do
  if ! kill -0 "${ENV_PID}" 2>/dev/null; then
    echo "仿真进程提前退出，日志 ${ENV_LOG}:" >&2
    tail -n 20 "${ENV_LOG}" >&2 || true
    exit 1
  fi
  if (( SECONDS - started > ENV_START_TIMEOUT )); then
    echo "等待仿真监听 ${ENV_HOST}:${ENV_PORT} 超过 ${ENV_START_TIMEOUT}s，日志 ${ENV_LOG}。" >&2
    exit 1
  fi
  sleep 2
done

if [[ "${LEARNER_KIND}" == "container" ]]; then
  require_learner_container
fi
run_pipeline "${PIPELINE_ARGS[@]}"
