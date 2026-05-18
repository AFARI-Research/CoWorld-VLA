#!/usr/bin/env bash
# Direct NAVSIM PDM inference entrypoint for CoWorld-VLA.

if [ -z "${BASH_VERSION:-}" ]; then
  echo "[ERROR] This script requires bash." >&2
  exit 1
fi
set -eu -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONFIG_PATH="${COWORLD_CONFIG:-configs/coworld_inference.yaml}"
CHECKPOINT_PATH="${COWORLD_CHECKPOINT:-}"
OUTPUT_DIR="${COWORLD_OUTPUT_DIR:-outputs/coworld_pdm_eval}"
VLM_FEATURE_CACHE_DIR="${COWORLD_VLM_FEATURE_CACHE_DIR:-${REPO_ROOT}/artifacts/vlm_feature_cache}"
USE_VLM_CACHE="${COWORLD_USE_VLM_CACHE:-auto}"
BUILD_CACHE_FIRST="${COWORLD_BUILD_CACHE:-0}"
PYTHON="${PYTHON:-$(command -v python3)}"

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "[ERROR] Set COWORLD_CHECKPOINT=/path/to/checkpoint before running inference." >&2
  exit 2
fi

HAS_CACHE_ARG=0
NAVSIM_LOG_PATH=""
SENSOR_BLOBS_PATH=""
SCENE_FILTER_YAML=""
MAX_SCENES=""
EVAL_ARGS=()
skip_next=0

for arg in "$@"; do
  if [[ "${skip_next}" -eq 1 ]]; then
    EVAL_ARGS+=("${arg}")
    skip_next=0
    continue
  fi

  case "${arg}" in
    --build-cache-first)
      BUILD_CACHE_FIRST=1
      continue
      ;;
    --no-vlm-cache)
      USE_VLM_CACHE=0
      continue
      ;;
    --vlm-feature-cache-dir)
      HAS_CACHE_ARG=1
      skip_next=1
      EVAL_ARGS+=("${arg}")
      continue
      ;;
    --vlm-feature-cache-dir=*)
      HAS_CACHE_ARG=1
      ;;
    --navsim-log-path|--sensor-blobs-path|--scene-filter-yaml|--max-scenes)
      skip_next=1
      ;;
  esac

  EVAL_ARGS+=("${arg}")
done

for ((i=0; i<${#EVAL_ARGS[@]}; i++)); do
  case "${EVAL_ARGS[$i]}" in
    --vlm-feature-cache-dir)
      VLM_FEATURE_CACHE_DIR="${EVAL_ARGS[$((i + 1))]}"
      ;;
    --vlm-feature-cache-dir=*)
      VLM_FEATURE_CACHE_DIR="${EVAL_ARGS[$i]#*=}"
      ;;
    --navsim-log-path)
      NAVSIM_LOG_PATH="${EVAL_ARGS[$((i + 1))]}"
      ;;
    --navsim-log-path=*)
      NAVSIM_LOG_PATH="${EVAL_ARGS[$i]#*=}"
      ;;
    --sensor-blobs-path)
      SENSOR_BLOBS_PATH="${EVAL_ARGS[$((i + 1))]}"
      ;;
    --sensor-blobs-path=*)
      SENSOR_BLOBS_PATH="${EVAL_ARGS[$i]#*=}"
      ;;
    --scene-filter-yaml)
      SCENE_FILTER_YAML="${EVAL_ARGS[$((i + 1))]}"
      ;;
    --scene-filter-yaml=*)
      SCENE_FILTER_YAML="${EVAL_ARGS[$i]#*=}"
      ;;
    --max-scenes)
      MAX_SCENES="${EVAL_ARGS[$((i + 1))]}"
      ;;
    --max-scenes=*)
      MAX_SCENES="${EVAL_ARGS[$i]#*=}"
      ;;
  esac
done

if [[ "${BUILD_CACHE_FIRST}" == "1" || "${BUILD_CACHE_FIRST}" == "true" ]]; then
  CACHE_BUILD_ARGS=(
    --config "${CONFIG_PATH}"
    --checkpoint "${CHECKPOINT_PATH}"
    --output-dir "${VLM_FEATURE_CACHE_DIR}"
  )
  [[ -n "${NAVSIM_LOG_PATH}" ]] && CACHE_BUILD_ARGS+=(--navsim-log-path "${NAVSIM_LOG_PATH}")
  [[ -n "${SENSOR_BLOBS_PATH}" ]] && CACHE_BUILD_ARGS+=(--sensor-blobs-path "${SENSOR_BLOBS_PATH}")
  [[ -n "${SCENE_FILTER_YAML}" ]] && CACHE_BUILD_ARGS+=(--scene-filter-yaml "${SCENE_FILTER_YAML}")
  [[ -n "${MAX_SCENES}" ]] && CACHE_BUILD_ARGS+=(--max-scenes "${MAX_SCENES}")
  "${REPO_ROOT}/scripts/evaluation/build_coworld_vlm_cache.sh" "${CACHE_BUILD_ARGS[@]}"
  USE_VLM_CACHE=1
fi

CONFIG_FOR_EVAL="${CONFIG_PATH}"
TMP_CONFIG=""
cleanup_tmp_config() {
  [[ -n "${TMP_CONFIG}" && -f "${TMP_CONFIG}" ]] && rm -f "${TMP_CONFIG}"
}
trap cleanup_tmp_config EXIT

make_no_cache_config() {
  TMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/coworld_no_cache.XXXXXX.yaml")"
  "${PYTHON}" - "${CONFIG_PATH}" "${TMP_CONFIG}" <<'PY'
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
OmegaConf.update(cfg, "model.use_cached_features", False, merge=False)
OmegaConf.update(cfg, "vlm_feature_cache_dir", None, merge=False)
OmegaConf.save(cfg, sys.argv[2])
PY
  CONFIG_FOR_EVAL="${TMP_CONFIG}"
}

if [[ "${USE_VLM_CACHE}" == "0" || "${USE_VLM_CACHE}" == "false" ]]; then
  make_no_cache_config
elif [[ "${HAS_CACHE_ARG}" -eq 0 ]]; then
  if [[ -d "${VLM_FEATURE_CACHE_DIR}" ]]; then
    EVAL_ARGS+=(--vlm-feature-cache-dir "${VLM_FEATURE_CACHE_DIR}")
  elif [[ "${USE_VLM_CACHE}" == "1" || "${USE_VLM_CACHE}" == "true" ]]; then
    echo "[ERROR] VLM feature cache not found: ${VLM_FEATURE_CACHE_DIR}" >&2
    echo "        Run with --build-cache-first, set COWORLD_VLM_FEATURE_CACHE_DIR, or pass --no-vlm-cache." >&2
    exit 2
  else
    echo "[WARN] VLM feature cache not found: ${VLM_FEATURE_CACHE_DIR}" >&2
    echo "       Falling back to direct VLM forward. Use --build-cache-first for faster repeated eval." >&2
    make_no_cache_config
  fi
fi

exec "${REPO_ROOT}/scripts/evaluation/run_eval_navsim_pdm_multigpu.sh" \
  --config "${CONFIG_FOR_EVAL}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  "${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"}"
