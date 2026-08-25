# Installation and Evaluation

This guide covers environment setup, model preparation, VLM feature caching, and NAVSIM PDM evaluation for CoWorld-VLA.

## Release Scope

The current release includes:

- CoWorld-VLA inference code
- VLM feature cache builder
- CoWorld-VLA checkpoint
- NAVSIM PDM evaluation

Training code will be released later. The default inference configuration is `configs/coworld_inference.yaml`.

## Environment Setup

Create and activate a Python 3.10 environment:

```bash
conda create -n coworld python=3.10 -y
conda activate coworld
pip install -e .
pip install -r requirements.txt
```

Install NAVSIM v1.1 and the nuPlan devkit in the same environment. Then prepare the NAVSIM metric cache:

```bash
python navsim/planning/script/run_metric_caching.py \
  train_test_split=navtest \
  cache.cache_path="$NAVSIM_EXP_ROOT/metric_cache"
```

## Model Preparation

Download the CoWorld-VLA checkpoint from [Hugging Face](https://huggingface.co/hmq1211/CoWorld-VLA):

```bash
pip install -U huggingface_hub
huggingface-cli download hmq1211/CoWorld-VLA \
  --local-dir /path/to/CoWorld-VLA \
  --local-dir-use-symlinks False
```

If the Hugging Face repository requires authentication, run `huggingface-cli login` first with an account that has access.

Configure the required paths:

```bash
export NAVSIM_DEVKIT_ROOT=/path/to/navsim
export OPENSCENE_DATA_ROOT=/path/to/openscene-v1.1
export NAVSIM_EXP_ROOT=/path/to/navsim/exp

export COWORLD_VLM_MODEL_PATH=/path/to/Qwen3-VL-2B-Instruct
export COWORLD_VGGT_MODEL_PATH=/path/to/vggt
export COWORLD_VJEPA_CKPT=/path/to/vjepa2_1_vitG_384.pt
export COWORLD_CHECKPOINT=/path/to/CoWorld-VLA/checkpoints
```

`COWORLD_CHECKPOINT` may point to either a checkpoint directory containing `model_state_dict.pt` or a single `.pt` state dict. The cache builder and evaluator both consume this value.

`COWORLD_VGGT_MODEL_PATH` should point to the pretrained VGGT weights or a local Hugging Face directory. The VGGT source is vendored under `models/vggt/`.

## VLM Feature Cache

CoWorld-VLA can use a precomputed VLM feature cache, which is recommended for repeated NAVSIM runs. It can also run the VLM online during evaluation, which is simpler but slower and requires more GPU memory.

By default, the evaluation wrapper looks for a cache at:

```text
artifacts/vlm_feature_cache/
```

Override the location with:

```bash
export COWORLD_VLM_FEATURE_CACHE_DIR=/path/to/coworld_vlm_feature_cache
```

Build the cache with:

```bash
NPROC_PER_NODE=8 ./scripts/evaluation/build_coworld_vlm_cache.sh \
  --config configs/coworld_inference.yaml \
  --checkpoint "$COWORLD_CHECKPOINT" \
  --output-dir "$COWORLD_VLM_FEATURE_CACHE_DIR"
```

The builder writes:

```text
$COWORLD_VLM_FEATURE_CACHE_DIR/val/
  hidden_*.bin
  tensors_*.safetensors
  info_*.json
```

The builder uses the NAVSIM defaults from `OPENSCENE_DATA_ROOT` and `NAVSIM_DEVKIT_ROOT`. To override the data locations:

```bash
./scripts/evaluation/build_coworld_vlm_cache.sh \
  --checkpoint "$COWORLD_CHECKPOINT" \
  --output-dir "$COWORLD_VLM_FEATURE_CACHE_DIR" \
  --navsim-log-path /path/to/navsim_logs/test \
  --sensor-blobs-path /path/to/sensor_blobs/test \
  --scene-filter-yaml /path/to/navtest.yaml
```

The VLM feature cache is checkpoint-specific. Rebuild it after changing the Qwen model, tokenizer, action-token layout, or CoWorld checkpoint.

## NAVSIM PDM Evaluation

Build the cache if needed, then run PDM evaluation:

```bash
NPROC_PER_NODE=8 ./scripts/evaluation/run_coworld_navsim_pdm.sh --build-cache-first
```

Run on all visible GPUs:

```bash
./scripts/evaluation/run_coworld_navsim_pdm.sh
```

Set a specific GPU count:

```bash
NPROC_PER_NODE=4 ./scripts/evaluation/run_coworld_navsim_pdm.sh
```

Run a quick sanity check on a small subset:

```bash
NPROC_PER_NODE=1 ./scripts/evaluation/run_coworld_navsim_pdm.sh --max-scenes 32
```

Run directly without a VLM cache:

```bash
NPROC_PER_NODE=1 ./scripts/evaluation/run_coworld_navsim_pdm.sh --no-vlm-cache --max-scenes 32
```

Use a custom output directory:

```bash
COWORLD_OUTPUT_DIR=/tmp/coworld_pdm_eval \
./scripts/evaluation/run_coworld_navsim_pdm.sh
```

The wrapper expands to:

```bash
./scripts/evaluation/run_eval_navsim_pdm_multigpu.sh \
  --config configs/coworld_inference.yaml \
  --checkpoint "$COWORLD_CHECKPOINT" \
  --vlm-feature-cache-dir "${COWORLD_VLM_FEATURE_CACHE_DIR:-artifacts/vlm_feature_cache}" \
  --output-dir "${COWORLD_OUTPUT_DIR:-outputs/coworld_pdm_eval}"
```

If no cache is found and no cache mode is forced, `run_coworld_navsim_pdm.sh` falls back to direct VLM inference. To require cached mode:

```bash
export COWORLD_USE_VLM_CACHE=1
```

The evaluation script derives the NAVSIM log, sensor, metric cache, and scene filter paths from `OPENSCENE_DATA_ROOT`, `NAVSIM_EXP_ROOT`, and `NAVSIM_DEVKIT_ROOT`. Override them explicitly when needed:

```bash
./scripts/evaluation/run_coworld_navsim_pdm.sh \
  --navsim-log-path /path/to/navsim_logs/test \
  --sensor-blobs-path /path/to/sensor_blobs/test \
  --metric-cache-path /path/to/metric_cache \
  --scene-filter-yaml /path/to/navtest.yaml
```

## Outputs

Each evaluation run writes to:

```text
<output_root>/<timestamp>/
```

Expected files:

```text
pdm_results_chunk0.json
pdm_results_chunk1.json
...
pdm_merged_aggregate.json
```

The merged aggregate reports the mean NAVSIM PDM score and component metrics over valid scenes.

To merge existing chunks again:

```bash
python scripts/evaluation/merge_pdm_eval_chunks.py /path/to/output/<timestamp>
```

## Runtime Notes

- The default planner predicts eight future waypoints at 2 Hz.
