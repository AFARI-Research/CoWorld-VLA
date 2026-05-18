#!/usr/bin/env bash
# Build the CoWorld-VLA VLM feature cache for NAVSIM navtest.

if [ -z "${BASH_VERSION:-}" ]; then
  echo "[ERROR] This script requires bash." >&2
  exit 1
fi
set -eu -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if command -v nvidia-smi &>/dev/null; then
  CUDA_DEVICES="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, - || true)"
else
  CUDA_DEVICES=""
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
if [[ -n "${CUDA_DEVICES}" && "${NPROC_PER_NODE}" -eq 1 ]]; then
  _cnt=$(echo "${CUDA_DEVICES}" | tr "," "\n" | grep -c . || true)
  [[ "${_cnt}" -ge 1 ]] && NPROC_PER_NODE="${_cnt}"
fi

NUM_NODES="${NUM_NODES:-1}"
MACHINE_RANK="${MACHINE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-63671}"
PYTHON="${PYTHON:-$(command -v python3)}"

echo "====== CoWorld VLM Cache (torchrun) ======"
echo "  nodes:     ${NUM_NODES}  rank: ${MACHINE_RANK}  gpus/node: ${NPROC_PER_NODE}"
echo "  master:    ${MASTER_ADDR}:${MASTER_PORT}"
echo "=========================================="

if [[ "${NUM_NODES}" -eq 1 ]]; then
  TORCHRUN_CMD=(
    "${PYTHON}" -m torch.distributed.run
    --standalone
    --nproc_per_node="${NPROC_PER_NODE}"
    scripts/evaluation/build_coworld_vlm_cache.py
  )
else
  TORCHRUN_CMD=(
    "${PYTHON}" -m torch.distributed.run
    --nnodes="${NUM_NODES}"
    --node_rank="${MACHINE_RANK}"
    --nproc_per_node="${NPROC_PER_NODE}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
    scripts/evaluation/build_coworld_vlm_cache.py
  )
fi

TORCHRUN_CMD+=("$@")
"${TORCHRUN_CMD[@]}"
