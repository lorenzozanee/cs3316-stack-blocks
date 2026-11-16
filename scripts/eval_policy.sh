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

# Optional layout subset. Stage copies of selected layout files and mount them
# over the seed layout directory inside the client container. The assets tree is
# bind-mounted under several container paths, so the staged directory must
# shadow every alias; the evaluator reads layouts through RoboDojo's Assets
# symlink, which resolves to ASSET_MOUNT_DEST rather than ASSETS_SOURCE.
# The client renumbers the selected layouts 0..N-1; layout_map.json records the
# original ids and the final verification rewrites _result.json layout_id to them.
LAYOUT_STAGE=""
eval "$(python3 - "${CONFIG}" "${ASSETS_SOURCE}" "${ASSET_MOUNT_DEST}" "${RUN_DIR}" <<'PY'
import json
import shlex
import shutil
import sys
from pathlib import Path

from starter.config import load_config

config_path, assets_source, asset_mount_dest, run_dir = sys.argv[1:]
cfg = load_config(config_path)
layout_ids = cfg.get("layout_ids")
if layout_ids is None:
    raise SystemExit(0)
if not isinstance(layout_ids, list) or not layout_ids:
    raise SystemExit("layout_ids must be a non-empty list.")
ids = [int(value) for value in layout_ids]
if len(set(ids)) != len(ids):
    raise SystemExit("layout_ids must not contain duplicates.")
seed = int(cfg["seed"])
task = cfg["task"]
env_cfg_name = cfg["env_cfg"]
relative = Path("Eval_Layout") / "RoboDojo" / env_cfg_name / str(seed)
source_dir = Path(assets_source) / relative
stage = Path(run_dir) / "layouts" / str(seed)
stage.mkdir(parents=True, exist_ok=True)
for new_id, old_id in enumerate(ids):
    source = source_dir / f"{task}_{old_id}.json"
    if not source.is_file():
        raise SystemExit(f"Missing layout file: {source}")
    shutil.copyfile(source, stage / f"{task}_{new_id}.json")
mapping = {str(new_id): old_id for new_id, old_id in enumerate(ids)}
Path(run_dir, "layout_map.json").write_text(json.dumps(mapping, indent=2) + "\n")
dests = []
for root in (asset_mount_dest, assets_source, "/workspace/RoboDojo/Assets"):
    dest = str(Path(root) / relative)
    if dest not in dests:
        dests.append(dest)
print(f"LAYOUT_STAGE={shlex.quote(str(stage))}")
print("LAYOUT_MOUNTS=(" + " ".join(shlex.quote(dest) for dest in dests) + ")")
print(f"EVAL_NUM={len(ids)}")
PY
)"

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
# RoboDojo caps EVAL_NUM at the task index budget. Stage the requested budget
# alongside the sim config so an explicit eval_num also works above that default.
TASK_INDEX_SRC="${ROBODOJO_ROOT}/task/RoboDojo/config/_task.yml"
TASK_INDEX_PATH="${RUN_DIR}/task_index.yml"
"${POLICY_PYTHON}" - "${TASK_INDEX_SRC}" "${TASK_INDEX_PATH}" "${TASK}" "${EVAL_NUM}" <<'PY'
import sys
from pathlib import Path
import yaml

src, dst, task, requested = sys.argv[1:]
cfg = yaml.safe_load(Path(src).read_text())
if requested != "native":
    count = int(requested)
    if count < 1:
        raise ValueError("eval_num must be positive")
    cfg["tasks"][task]["eval_nums"] = count
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

layout_maps=()
if [[ -n "${LAYOUT_STAGE}" ]]; then
  for dest in "${LAYOUT_MOUNTS[@]}"; do
    layout_maps+=(-v "${LAYOUT_STAGE}:${dest}:ro")
  done
fi

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
  "${layout_maps[@]}"
  -v "${ICD_PATH}:/etc/vulkan/icd.d/nvidia_icd.json:ro"
  -v "${SIM_PATH}:${SIM_SRC}:ro"
  -v "${TASK_INDEX_PATH}:${TASK_INDEX_SRC}:ro"
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
if [[ -n "${LAYOUT_STAGE}" ]]; then
  echo "[eval] layout subset=${LAYOUT_STAGE} mounted at ${LAYOUT_MOUNTS[*]}; mapping in ${RUN_DIR}/layout_map.json"
fi

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

# A successful client exit must also cover the requested episode count.
"${POLICY_PYTHON}" - "${RESULTS_DIR}" "${RUN_ID}" "${TASK_INDEX_PATH}" "${TASK}" "${RUN_DIR}" "${IMAGE}" <<'PY'
import json
import sys
from pathlib import Path
import yaml

results_dir, run_id, task_index, task, run_dir, image = sys.argv[1:]
index = yaml.safe_load(Path(task_index).read_text())
expected = int(index["tasks"][task].get("eval_nums", index.get("common", {}).get("eval_nums", 50)))
paths = list(Path(results_dir).glob(f"**/{run_id}/_result.json"))
if len(paths) != 1:
    raise SystemExit(f"Expected one result for {run_id}, found {len(paths)}")
result_path = paths[0]
result = json.loads(result_path.read_text())
completed = int(result["eval_time"])
if completed != expected:
    raise SystemExit(f"Incomplete evaluation: requested {expected}, completed {completed}; {result_path}")
# The client renumbers staged layouts 0..N-1; report the real seed-set ids in
# _result.json and in the episode video names (episode_<real id>_cam_*).
# Result files are written by the container as root, so rewrite through docker.
layout_map_path = Path(run_dir) / "layout_map.json"
if layout_map_path.is_file():
    import subprocess
    script = (
        "import json,re,sys\n"
        "from pathlib import Path\n"
        "res,lm=map(Path,sys.argv[1:])\n"
        "r=json.loads(res.read_text());mapping=json.loads(lm.read_text())\n"
        "for d in r['details'].values(): d['layout_id']=mapping[str(d['layout_id'])]\n"
        "res.write_text(json.dumps(r,indent=2)+'\\n')\n"
        "for v in sorted(res.parent.glob('episode_*_cam_*.mp4'),reverse=True):\n"
        "    m=re.match(r'episode_(\\d+)(_.*)',v.name)\n"
        "    key=str(int(m.group(1))) if m else None\n"
        "    if key in mapping:\n"
        "        v.rename(v.with_name(f'episode_{mapping[key]:07d}{m.group(2)}'))\n"
    )
    subprocess.run(
        ["docker", "run", "--rm", "-i",
         "-v", f"{result_path.parent}:{result_path.parent}",
         "-v", f"{layout_map_path}:{layout_map_path}:ro",
         image, "python3", "-", str(result_path), str(layout_map_path)],
        input=script, text=True, check=True,
    )
print(f"[eval] verified {completed}/{expected} episodes: {result_path}")
PY
