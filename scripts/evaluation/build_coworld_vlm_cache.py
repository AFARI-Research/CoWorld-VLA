"""Build the CoWorld-VLA VLM feature cache for NAVSIM inference."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from safetensors.torch import save_file as save_safetensors
from torch.utils.data import DataLoader, DistributedSampler

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from datasets.data_utils import nuplan_collate_fn
from datasets.navsim_official_dataset import NavsimOfficialDataset
from models.vlm_worldmodel import WmVlmJoint
from training.checkpoint import _load_sd_and_state_from_ckpt_path
from utils.utils import cfg_get


def _env_path(name: str, suffix: str) -> Optional[str]:
    root = os.environ.get(name)
    return str(Path(root) / suffix) if root else None


def _default_scene_filter() -> Optional[str]:
    root = os.environ.get("NAVSIM_DEVKIT_ROOT")
    if not root:
        return None
    return str(
        Path(root)
        / "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
    )


def _init_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        return 0, local_rank, 1

    timeout = timedelta(seconds=int(os.environ.get("COWORLD_CACHE_DIST_TIMEOUT_SEC", str(6 * 3600))))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
            timeout=timeout,
        )
    else:
        dist.init_process_group(backend="gloo", timeout=timeout)
    return dist.get_rank(), local_rank, dist.get_world_size()


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _load_config_for_cache(config_path: str) -> dict:
    cfg = OmegaConf.load(config_path)
    OmegaConf.update(cfg, "model.name", "WmVlmJoint", merge=False)
    for flag in ("use_wan", "use_jepa", "use_vggt", "use_traj"):
        OmegaConf.update(cfg, f"model.{flag}", False, merge=False)
    OmegaConf.update(cfg, "model.use_cached_features", False, merge=False)
    OmegaConf.update(cfg, "model.use_vlm_autoreg_text_trajectory", False, merge=False)

    # Do not require VGGT/V-JEPA env vars while only building VLM features.
    OmegaConf.update(cfg, "model.external_vggt_context.enabled", False, merge=False)
    OmegaConf.update(cfg, "model.external_vggt_context.vggt_model_path", "", merge=False)
    OmegaConf.update(cfg, "model.external_jepa_context.enabled", False, merge=False)
    OmegaConf.update(cfg, "model.external_jepa_context.teacher_ckpt", "", merge=False)
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def _build_vlm_model(full_cfg: dict) -> WmVlmJoint:
    cfg = copy.deepcopy(full_cfg)
    model_cfg = cfg.setdefault("model", {})
    model_cfg["name"] = "WmVlmJoint"
    for flag in ("use_wan", "use_jepa", "use_vggt", "use_traj"):
        model_cfg[flag] = False
    model_cfg["use_cached_features"] = False
    return WmVlmJoint(model_cfg, full_cfg=cfg)


def _load_vlm_weights(model: WmVlmJoint, checkpoint: str, log_fn=print) -> None:
    sd, _ = _load_sd_and_state_from_ckpt_path(checkpoint, map_location="cpu")
    vlm_keys: Dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        if key.startswith("vlm."):
            vlm_keys[key[4:]] = value
        elif key.startswith("model.vlm."):
            vlm_keys[key[10:]] = value

    if not vlm_keys:
        raise RuntimeError(
            f"No 'vlm.*' weights found in checkpoint: {checkpoint}. "
            "Use the CoWorld checkpoint that contains the VLM branch."
        )

    target = model.vlm.state_dict()
    for key, value in list(vlm_keys.items()):
        if key in target and target[key].shape != value.shape:
            log_fn(
                f"[cache] shape mismatch for {key}: "
                f"checkpoint={tuple(value.shape)} model={tuple(target[key].shape)}"
            )
            if value.ndim >= 1 and value.shape[0] < target[key].shape[0]:
                patched = target[key].clone()
                patched[: value.shape[0]] = value
                vlm_keys[key] = patched
            else:
                del vlm_keys[key]

    missing, unexpected = model.vlm.load_state_dict(vlm_keys, strict=False)
    log_fn(
        f"[cache] loaded VLM weights: keys={len(vlm_keys)} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    if missing:
        log_fn(f"[cache] missing first 8: {missing[:8]}")


def _build_dataset(args: argparse.Namespace, cfg: dict) -> NavsimOfficialDataset:
    data_cfg = cfg_get(cfg, "data", {})
    dataset = NavsimOfficialDataset(
        navsim_log_path=args.navsim_log_path,
        sensor_blobs_path=args.sensor_blobs_path,
        scene_filter_yaml=args.scene_filter_yaml,
        num_history_image_frames=int(cfg_get(data_cfg, "num_history_image_frames", 1)),
        num_history_trajectory_steps=int(cfg_get(data_cfg, "num_history_trajectory_steps", 4)),
        num_future_frames=int(cfg_get(data_cfg, "num_future_frames", 8)),
        resize_to=(
            int(cfg_get(data_cfg, "frame_height", 512)),
            int(cfg_get(data_cfg, "frame_width", 1024)),
        ),
        camera_name=str(cfg_get(data_cfg, "camera_name", "cam_f0")),
        verbose=args.rank == 0,
    )
    if args.max_scenes is not None:
        dataset._tokens = dataset._tokens[: int(args.max_scenes)]
    return dataset


def _scalar_int(value: Any, index: int) -> int:
    return int(value[index].item()) if isinstance(value, torch.Tensor) else int(value[index])


def _scalar_float(value: Any, index: int) -> float:
    return float(value[index].item()) if isinstance(value, torch.Tensor) else float(value[index])


def _stack_or_empty(items: List[torch.Tensor], shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return torch.stack(items).contiguous() if items else torch.empty(shape, dtype=dtype)


@torch.no_grad()
def _extract_val_cache(
    model: WmVlmJoint,
    dataset: NavsimOfficialDataset,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
    device: torch.device,
    force: bool,
) -> int:
    split_dir = output_dir / "val"
    split_dir.mkdir(parents=True, exist_ok=True)

    hidden_path = split_dir / f"hidden_{rank}.bin"
    tensors_path = split_dir / f"tensors_{rank}.safetensors"
    info_path = split_dir / f"info_{rank}.json"

    if force:
        for path in (hidden_path, tensors_path, info_path):
            path.unlink(missing_ok=True)
    elif info_path.exists():
        print(f"[rank {rank}] cache shard exists, skip: {info_path}", flush=True)
        return 0

    for path in (hidden_path, tensors_path):
        path.unlink(missing_ok=True)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=nuplan_collate_fn,
        drop_last=False,
    )

    offsets: List[int] = []
    lengths: List[int] = []
    start_indices: List[torch.Tensor] = []
    end_indices: List[torch.Tensor] = []
    jepa_indices: List[torch.Tensor] = []
    vggt_indices: List[torch.Tensor] = []
    wan_indices: List[torch.Tensor] = []
    traj_indices: List[torch.Tensor] = []
    history_traj: List[torch.Tensor] = []
    future_traj: List[torch.Tensor] = []
    ego_status: List[torch.Tensor] = []
    nav_cmds: List[int] = []
    ego_speeds: List[float] = []
    intervals: List[float] = []
    tokens: List[str] = []
    history_image_paths: List[List[str]] = []
    seen: set[str] = set()

    hidden_dim: Optional[int] = None
    token_offset = 0
    written = 0
    skipped_dup = 0
    t0 = time.time()

    model.eval()
    with open(hidden_path, "wb", buffering=16 * 1024 * 1024) as hidden_file:
        for batch_idx, batch in enumerate(loader):
            scene_tokens = batch.get("scene_token")
            if not scene_tokens:
                raise RuntimeError("Batch is missing scene_token; cannot write cache.")

            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            images = model._extract_current_images(inputs)
            prompts = model._build_prompts(inputs)
            samples = model.vlm.forward_full_hidden(images, prompts)

            for i, token in enumerate(scene_tokens):
                token = str(token)
                if token in seen:
                    skipped_dup += 1
                    continue
                seen.add(token)

                hidden = samples[i]["last_hidden"].contiguous()
                if hidden_dim is None:
                    hidden_dim = int(hidden.shape[1])
                elif int(hidden.shape[1]) != hidden_dim:
                    raise RuntimeError(
                        f"Hidden dim mismatch: expected {hidden_dim}, "
                        f"got {hidden.shape[1]} for token={token}"
                    )

                length = int(hidden.shape[0])
                hidden_file.write(hidden.view(torch.uint8).numpy().tobytes())
                offsets.append(token_offset)
                lengths.append(length)
                token_offset += length

                start_indices.append(samples[i]["start_action_idx"].to(torch.int64).cpu().reshape(()))
                end_indices.append(samples[i]["end_action_idx"].to(torch.int64).cpu().reshape(()))
                jepa_indices.append(samples[i]["jepa_action_indices"].to(torch.int64).cpu())
                vggt_indices.append(samples[i]["vggt_action_indices"].to(torch.int64).cpu())
                wan_indices.append(samples[i]["wan_action_indices"].to(torch.int64).cpu())
                traj_indices.append(samples[i]["traj_action_indices"].to(torch.int64).cpu())
                history_traj.append(batch["history_trajectory"][i].cpu().float())
                future_traj.append(batch["future_trajectory"][i].cpu().float())
                ego_status.append(batch["ego_status"][i].cpu().float())
                nav_cmds.append(_scalar_int(batch["navigation_command"], i))
                ego_speeds.append(_scalar_float(batch["ego_speed"], i))
                intervals.append(_scalar_float(batch["trajectory_interval_s"], i))

                paths_i = batch.get("history_image_paths", [[]])[i]
                if isinstance(paths_i, str):
                    history_image_paths.append([paths_i])
                else:
                    history_image_paths.append([str(p) for p in paths_i])

                tokens.append(token)
                written += 1

            if rank == 0 and (batch_idx + 1) % 20 == 0:
                elapsed = max(time.time() - t0, 1e-6)
                print(
                    f"[rank {rank}] batch {batch_idx + 1}/{len(loader)} "
                    f"written={written} dup_skipped={skipped_dup} "
                    f"rate={(written + skipped_dup) / elapsed:.2f} samples/s",
                    flush=True,
                )

    meta = {
        "start_action_idx": _stack_or_empty(start_indices, (0,), torch.int64),
        "end_action_idx": _stack_or_empty(end_indices, (0,), torch.int64),
        "jepa_action_indices": _stack_or_empty(jepa_indices, (0, 0), torch.int64),
        "vggt_action_indices": _stack_or_empty(vggt_indices, (0, 0), torch.int64),
        "wan_action_indices": _stack_or_empty(wan_indices, (0, 0), torch.int64),
        "traj_action_indices": _stack_or_empty(traj_indices, (0, 0), torch.int64),
        "history_trajectory": _stack_or_empty(history_traj, (0, 0, 0), torch.float32),
        "future_trajectory": _stack_or_empty(future_traj, (0, 0, 0), torch.float32),
        "ego_status": _stack_or_empty(ego_status, (0, 0), torch.float32),
        "navigation_command": torch.tensor(nav_cmds, dtype=torch.int64),
        "ego_speed": torch.tensor(ego_speeds, dtype=torch.float32),
        "trajectory_interval_s": torch.tensor(intervals, dtype=torch.float32),
        "hidden_offsets_tok": torch.tensor(offsets, dtype=torch.int64),
        "hidden_lengths": torch.tensor(lengths, dtype=torch.int64),
    }
    save_safetensors(meta, str(tensors_path))

    info = {
        "rank": rank,
        "world_size": world_size,
        "split": "val",
        "hidden_dim": int(hidden_dim or 0),
        "num_samples": written,
        "total_tokens": token_offset,
        "tokens": tokens,
        "history_image_paths": history_image_paths,
        "history_image_resize_to": list(getattr(dataset, "resize_to", (512, 1024))),
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f)

    print(
        f"[rank {rank}] DONE written={written} dup_skipped={skipped_dup} "
        f"tokens={token_offset} hidden_dim={hidden_dim} elapsed={time.time() - t0:.1f}s",
        flush=True,
    )
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CoWorld-VLA VLM feature cache")
    parser.add_argument("--config", default=os.environ.get("COWORLD_CONFIG", "configs/coworld_inference.yaml"))
    parser.add_argument("--checkpoint", default=os.environ.get("COWORLD_CHECKPOINT"))
    parser.add_argument(
        "--output-dir",
        default=os.environ.get(
            "COWORLD_VLM_FEATURE_CACHE_DIR",
            str(Path(REPO_ROOT) / "artifacts/vlm_feature_cache"),
        ),
    )
    parser.add_argument("--navsim-log-path", default=_env_path("OPENSCENE_DATA_ROOT", "navsim_logs/test"))
    parser.add_argument("--sensor-blobs-path", default=_env_path("OPENSCENE_DATA_ROOT", "sensor_blobs/test"))
    parser.add_argument("--scene-filter-yaml", default=_default_scene_filter())
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("COWORLD_CACHE_BATCH_SIZE", "4")))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("COWORLD_CACHE_NUM_WORKERS", "4")))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    missing = [
        name
        for name in ("checkpoint", "navsim_log_path", "sensor_blobs_path", "scene_filter_yaml")
        if not getattr(args, name)
    ]
    if missing:
        raise SystemExit(f"Missing required arguments or env defaults: {', '.join(missing)}")
    return args


def main() -> None:
    args = _parse_args()
    rank, local_rank, world_size = _init_distributed()
    args.rank = rank
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    cfg = _load_config_for_cache(args.config)
    if bool(cfg.get("tf32", True)) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    output_dir = Path(args.output_dir).expanduser().resolve()
    if rank == 0:
        print("====== CoWorld VLM Cache ======")
        print(f"  config:      {args.config}")
        print(f"  checkpoint:  {args.checkpoint}")
        print(f"  output_dir:  {output_dir}")
        print(f"  batch/gpu:   {args.batch_size}")
        print(f"  workers:     {args.num_workers}")
        print(f"  world_size:  {world_size}")
        print("===============================")
        output_dir.mkdir(parents=True, exist_ok=True)
    _barrier()

    model = _build_vlm_model(cfg).to(device)
    _load_vlm_weights(model, args.checkpoint, log_fn=print if rank == 0 else lambda *_: None)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    _barrier()

    dataset = _build_dataset(args, cfg)
    if rank == 0:
        print(f"[cache] val samples: {len(dataset)}", flush=True)
    _extract_val_cache(
        model=model,
        dataset=dataset,
        output_dir=output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        rank=rank,
        world_size=world_size,
        device=device,
        force=bool(args.force),
    )
    _barrier()

    if rank == 0:
        info_files = sorted((output_dir / "val").glob("info_*.json"))
        n_samples = 0
        n_tokens = 0
        for info_file in info_files:
            with open(info_file, "r", encoding="utf-8") as f:
                info = json.load(f)
            n_samples += int(info.get("num_samples", 0))
            n_tokens += int(info.get("total_tokens", 0))
        print(
            f"[cache] complete: shards={len(info_files)} "
            f"samples={n_samples} tokens={n_tokens}",
            flush=True,
        )
    _cleanup_distributed()


if __name__ == "__main__":
    main()
