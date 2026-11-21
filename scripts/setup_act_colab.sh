#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${1:?Usage: setup_act_colab.sh <new-environment-directory>}"
mkdir -p "${ENV_DIR}"
ENV_DIR="$(realpath "${ENV_DIR}")"
DOJO="${ROOT}/third_party/RoboDojo"
cd "${ROOT}"

bash scripts/bootstrap_third_party.sh

checkout_submodule() {
  local sub_path="$1"
  local dest="${DOJO}/${sub_path}"
  local url sha
  url="$(git -C "${DOJO}" config --file .gitmodules --get "submodule.${sub_path}.url")"
  sha="$(git -C "${DOJO}" ls-tree HEAD "${sub_path}" | awk '{print $3}')"
  if [[ -z "${url}" || -z "${sha}" ]]; then
    echo "Missing submodule metadata for ${sub_path}" >&2
    exit 1
  fi
  if [[ ! -e "${dest}/.git" ]]; then
    mkdir -p "${dest}"
    if [[ -n "$(ls -A "${dest}")" ]]; then
      printf 'Expected an empty or Git-managed checkout: %s\n' "${dest}" >&2
      exit 1
    fi
    git -C "${dest}" init -q
    git -C "${dest}" remote add origin "${url}"
  fi
  if ! git -C "${dest}" cat-file -e "${sha}^{commit}" 2>/dev/null; then
    git -C "${dest}" fetch --depth 1 "${url}" "${sha}"
  fi
  git -C "${dest}" checkout -q --detach "${sha}"
}

checkout_submodule third_party/IsaacLab
checkout_submodule third_party/curobo

if ! command -v uv >/dev/null; then
  python3 -m pip install uv
fi

apt-get update
if ! git lfs version >/dev/null 2>&1; then
  apt-get install -y --no-install-recommends git-lfs
fi

# Fetch only the Assets required by stack_blocks (~1 GB instead of the full ~27 GB dataset).
ASSET_CACHE="${DOJO}/.cache/robodojo_assets_repo"
ASSET_PATHS=(
  "Assets/Background/brown_photostudio_02_4k.hdr"
  "Assets/Eval_Layout/RoboDojo/arx_x5/0"
  "Assets/Material/material_0122"
  "Assets/Material/material_0564"
  "Assets/Object/RoboDojo/Geometry/camera_stand"
  "Assets/Object/RoboDojo/Rigid/block"
  "Assets/Robots/x5"
  "Assets/Room/Simple_Room_nolight"
  "Assets/Sensor"
)
if [[ ! -d "${DOJO}/Assets/Robots/x5" || ! -d "${DOJO}/Assets/Object/RoboDojo/Rigid/block" ]]; then
  mkdir -p "$(dirname "${ASSET_CACHE}")"
  if [[ ! -d "${ASSET_CACHE}/.git" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --sparse \
      https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo "${ASSET_CACHE}"
  fi
  git -C "${ASSET_CACHE}" sparse-checkout set --no-cone "${ASSET_PATHS[@]}"
  git -C "${ASSET_CACHE}" checkout main
  git -C "${ASSET_CACHE}" lfs install --local >/dev/null
  lfs_include="$(IFS=,; echo "${ASSET_PATHS[*]}")"
  git -C "${ASSET_CACHE}" lfs pull --include="${lfs_include}" --exclude=""
  ln -sfn "${ASSET_CACHE}/Assets" "${DOJO}/Assets"
fi

mkdir -p "${ENV_DIR}/debs" "${ENV_DIR}/system"
pushd "${ENV_DIR}/debs" >/dev/null
apt-get download libglu1-mesa libxt6t64 cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-cudart-12-8 \
  cuda-nvvm-12-8 cuda-crt-12-8 cuda-cccl-12-8 cuda-driver-dev-12-8
for package in ./*.deb; do
  dpkg-deb -x "$package" "${ENV_DIR}/system"
done
popd >/dev/null

export CUDA_HOME="${ENV_DIR}/system/usr/local/cuda-12.8"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${ENV_DIR}/system/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=12.0
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=0.8.0
uv venv --python 3.11 "${ENV_DIR}/sim"
uv venv --python 3.11 "${ENV_DIR}/learner"
sim_py="${ENV_DIR}/sim/bin/python"
learner_py="${ENV_DIR}/learner/bin/python"
uv pip install --python "${sim_py}" --index-strategy unsafe-best-match \
  --extra-index-url https://pypi.nvidia.com 'isaacsim[all,extscache]==5.1.0'
uv pip install --python "${sim_py}" --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.0+cu128 torchvision==0.22.0+cu128 setuptools wheel
constraints="${ENV_DIR}/sim-constraints.txt"
printf '%s\n' 'torch==2.7.0+cu128' 'torchvision==0.22.0+cu128' 'numpy==1.26.0' \
  'scipy==1.15.3' 'warp-lang==1.11.0' 'opencv-python-headless>=4.8,<4.12' \
  'packaging==23.0' 'typing_extensions==4.12.2' 'filelock==3.13.1' 'websockets==12.0' > "${constraints}"
uv pip install --python "${sim_py}" --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 -c "${constraints}" \
  -e "${DOJO}/third_party/IsaacLab/source/isaaclab" \
  -e "${DOJO}/third_party/IsaacLab/source/isaaclab_assets" \
  -e "${DOJO}/third_party/IsaacLab/source/isaaclab_contrib" \
  -e "${DOJO}/third_party/IsaacLab/source/isaaclab_rl" \
  -e "${DOJO}/third_party/IsaacLab/source/isaaclab_tasks" \
  -e "${DOJO}/third_party/curobo[cu12]" \
  opencv-python-headless transforms3d shapely rtree tqdm gymnasium omegaconf h5py scikit-learn ninja
uv pip install --python "${learner_py}" --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.1+cu128 torchvision==0.22.1+cu128 'numpy>=2,<3' 'h5py>=3.8,<4' \
  'opencv-python-headless>=4.8,<5' 'PyYAML>=6,<7' 'matplotlib>=3.8,<4' packaging

# Native processes share the asset paths. The download is private to this checkout.
PYTHONPATH="${ROOT}/src" "${learner_py}" - "${ROOT}" "${DOJO}" "${ENV_DIR}" <<'PY'
from pathlib import Path
import json
import shutil
import sys
from starter.envs.assets import prepare_robot_configs

root, dojo, environment = map(Path, sys.argv[1:])
for source, target in prepare_robot_configs(root / 'third_party/env_cfg/arx_x5.yml',
                                            dojo / 'Assets', environment / 'robot-configs', dojo):
    shutil.copyfile(source, target)
(environment / 'nvidia_icd.json').write_text(json.dumps({
    'file_format_version': '1.0.0',
    'ICD': {'library_path': 'libEGL_nvidia.so.0', 'api_version': '1.3.277'},
}))
PY
printf '%s\n' "${ENV_DIR}" > "${ENV_DIR}/ready"
