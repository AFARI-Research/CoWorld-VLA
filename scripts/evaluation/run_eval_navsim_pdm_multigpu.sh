#!/usr/bin/env bash
# =============================================================================
# Run NAVSIM PDM evaluation with torch.distributed.run.
#
# Single machine:
#   torchrun --standalone --nproc_per_node=$NPROC_PER_NODE
#
# Multi-node is supported with explicit NUM_NODES / MACHINE_RANK /
# MASTER_ADDR / MASTER_PORT. The output directory must be on shared storage
# so rank 0 can merge all chunk JSON files.
#
# 每个 rank 评测自己的 token chunk，落盘 pdm_results_chunk${RANK}.json；
# torchrun 返回后（所有 rank 完成），rank-0 机器合并所有 chunk、打印均值，
# 并写入 pdm_merged_aggregate.json（历史目录可用 merge_pdm_eval_chunks.py 补写）。
#
# 默认数据路径（约定 navtest 布局）：
#   --navsim-log-path    = $OPENSCENE_DATA_ROOT/navsim_logs/test
#   --sensor-blobs-path  = $OPENSCENE_DATA_ROOT/sensor_blobs/test
#   --metric-cache-path  = $NAVSIM_EXP_ROOT/metric_cache
# 用户 CLI 里再次传同名参数会 override 默认（argparse 取后者胜出）。
#
# --output-dir 语义：用户传入的是「run 根」，本次评测实际落到
#   <run_root>/<timestamp>/pdm_results_chunk*.json
# 时间戳单机默认由本机 date 生成；多节点请显式 export EVAL_OUTPUT_TS 统一注入
# （格式自定义，示例 2024.12.31.23.59.59），否则依赖节点时钟秒级同步。
#
# 用法：
#   # 单机，自动用本机全部可见 GPU
#   ./scripts/evaluation/run_eval_navsim_pdm_multigpu.sh \
#       --config configs/ae_acttoken_fz_vla.yaml \
#       --checkpoint /path/to/ckpt \
#       --output-dir /tmp/pdm_eval_out
#
#   # 指定 GPU 数
#   NPROC_PER_NODE=4 ./scripts/evaluation/run_eval_navsim_pdm_multigpu.sh ...
#
#   # 评其他 split / 自定义路径：显式 override
#   ./scripts/evaluation/run_eval_navsim_pdm_multigpu.sh \
#       --config ... --checkpoint ... --output-dir ... \
#       --navsim-log-path /custom/logs --sensor-blobs-path /custom/blobs
#
#   # 多节点
#   NUM_NODES=2 MACHINE_RANK=0 MASTER_ADDR=<rank0-host> MASTER_PORT=63670 \
#   NPROC_PER_NODE=8 ./scripts/evaluation/run_eval_navsim_pdm_multigpu.sh ...
# =============================================================================

if [ -z "${BASH_VERSION:-}" ]; then
  echo "[ERROR] This script requires bash." >&2
  exit 1
fi
set -eu -o pipefail

cleanup() {
  echo ""
  echo "[eval_pdm] Interrupted. Cleaning up..."
  pkill -9 -f "eval_navsim_pdm.py" 2>/dev/null || true
  sleep 1
  exit 130
}
trap cleanup INT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# --- GPU count ---
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

# --- Multi-node environment ---
NUM_NODES="${NUM_NODES:-1}"
MACHINE_RANK="${MACHINE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-63670}"

PYTHON="${PYTHON:-$(command -v python3)}"

# --- 默认数据路径（约定 navtest 布局；用户 CLI 再次传同名参数会 override） ---
DEFAULT_ARGS=()
if [[ -n "${OPENSCENE_DATA_ROOT:-}" ]]; then
  DEFAULT_ARGS+=(
    --navsim-log-path   "${OPENSCENE_DATA_ROOT}/navsim_logs/test"
    --sensor-blobs-path "${OPENSCENE_DATA_ROOT}/sensor_blobs/test"
  )
fi
if [[ -n "${NAVSIM_EXP_ROOT:-}" ]]; then
  DEFAULT_ARGS+=(--metric-cache-path "${NAVSIM_EXP_ROOT}/metric_cache")
fi

# --- 从参数中解析 --output-dir：作为 "run 根"，本次评测落到 run_root/<timestamp>/ ---
# 时间戳子目录好处：每次评测独立归档，旧结果保留，聚合 glob 永远只看本次产物。
# 多节点时 rank-0 本地生成时间戳；其他节点优先读 $EVAL_OUTPUT_TS（launcher 可统一注入），
# 否则本地 date（需要节点时钟同步到秒级，或显式导出 EVAL_OUTPUT_TS）。
RUN_ROOT=""
prev=""
ARGS_NO_OUTPUT=()
skip_next=0
for a in "$@"; do
  if [[ $skip_next -eq 1 ]]; then
    RUN_ROOT="$a"
    skip_next=0
    continue
  fi
  if [[ "$a" == "--output-dir" ]]; then
    skip_next=1
    continue
  fi
  ARGS_NO_OUTPUT+=("$a")
done

MERGE_DIR=""
if [[ -n "$RUN_ROOT" ]]; then
  EVAL_OUTPUT_TS="${EVAL_OUTPUT_TS:-$(date +%Y.%m.%d.%H.%M.%S)}"
  export EVAL_OUTPUT_TS
  MERGE_DIR="${RUN_ROOT}/${EVAL_OUTPUT_TS}"
fi

echo "====== NAVSIM PDM Eval (torchrun) ======"
echo "  nodes:       ${NUM_NODES}    rank: ${MACHINE_RANK}"
echo "  gpus/node:   ${NPROC_PER_NODE}"
echo "  master:      ${MASTER_ADDR}:${MASTER_PORT}"
echo "  run_root:    ${RUN_ROOT:-<none>}"
echo "  output_dir:  ${MERGE_DIR:-<none>}"
echo "========================================"

if [[ -n "$MERGE_DIR" && "${MACHINE_RANK}" -eq 0 ]]; then
  mkdir -p "$MERGE_DIR"
fi

# 把带时间戳的 --output-dir 重新拼回去
TORCHRUN_ARGS=("${ARGS_NO_OUTPUT[@]+"${ARGS_NO_OUTPUT[@]}"}")
if [[ -n "$MERGE_DIR" ]]; then
  TORCHRUN_ARGS+=(--output-dir "$MERGE_DIR")
fi

# 用数组拼命令，避免 ``\`` 行续写依赖；若文件被保存成 CRLF，续行会失效并把 ``--navsim-log-path`` 当成命令执行。
if [[ "${NUM_NODES}" -eq 1 ]]; then
  TORCHRUN_CMD=(
    "${PYTHON}" -m torch.distributed.run
    --standalone
    --nproc_per_node="${NPROC_PER_NODE}"
    eval/eval_navsim_pdm.py
  )
else
  TORCHRUN_CMD=(
    "${PYTHON}" -m torch.distributed.run
    --nnodes="${NUM_NODES}"
    --node_rank="${MACHINE_RANK}"
    --nproc_per_node="${NPROC_PER_NODE}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
    eval/eval_navsim_pdm.py
  )
fi
TORCHRUN_CMD+=("${DEFAULT_ARGS[@]+"${DEFAULT_ARGS[@]}"}")
TORCHRUN_CMD+=("${TORCHRUN_ARGS[@]+"${TORCHRUN_ARGS[@]}"}")
"${TORCHRUN_CMD[@]}"

# --- rank-0 机器合并所有 chunk，打印均值并落盘 pdm_merged_aggregate.json ---
if [[ -n "$MERGE_DIR" && "${MACHINE_RANK}" -eq 0 ]]; then
  echo ""
  echo "========== 合并 chunk 结果（rank-0 机器）=========="
  "$PYTHON" "${REPO_ROOT}/scripts/evaluation/merge_pdm_eval_chunks.py" "$MERGE_DIR"
fi
