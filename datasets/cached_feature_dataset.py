"""CachedFeatureDataset — shard + mmap feature cache loader.

The cache directory must have the following structure:

    <cache_dir>/<split>/
        hidden_{rank}.bin             # bf16 裸字节，变长 last_hidden 顺序拼接
        tensors_{rank}.safetensors    # 本 rank 所有样本的固定形状小字段（batched）
        info_{rank}.json              # scene_token 列表 + H、N、offsets/lengths 等元信息

At startup:

- ``mmap`` all ``hidden_*.bin`` files and load ``tensors_*.safetensors`` metadata,
  合并成全局索引；
- ``__getitem__`` 的代价 ≈ 一次 mmap slice + 若干 tensor 索引，per-sample
  开销在微秒级，完全零反序列化开销；

Output fields:

``vlm_last_hidden`` / ``vlm_wan_action_indices`` / ``vlm_traj_action_indices`` /
``vlm_start_action_idx`` / ``vlm_end_action_idx`` / ``history_trajectory`` / ``future_trajectory`` /
``ego_status`` / ``navigation_command`` / ``ego_speed`` /
``trajectory_interval_s`` / ``history_image_paths`` / ``scene_token`` / ``index``。

当 ``load_history_images=True`` 时，``__getitem__`` 会根据缓存中的
``history_image_paths`` 重新读取 ``history_images``，供仍需要图像的 JEPA/VGGT context 分支使用。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import torch
from safetensors.torch import load_file as _st_load_file
from torch.utils.data import Dataset


_REQUIRED_META_KEYS = (
    "wan_action_indices", "traj_action_indices", "start_action_idx", "end_action_idx",
    "history_trajectory", "future_trajectory", "ego_status",
    "navigation_command", "ego_speed", "trajectory_interval_s",
    "hidden_offsets_tok", "hidden_lengths",
)
# Optional: stub action indices (zero-length tensors when encoder not active).
_OPTIONAL_ACTION_INDEX_KEYS = (
    "jepa_action_indices",
    "vggt_action_indices",
)


class CachedFeatureDataset(Dataset):
    """Shard + mmap 版纯缓存数据集。

    Args:
        cache_dir: 特征文件根目录（含 ``train/`` / ``val/`` 子目录）。
        split:     ``"train"`` 或 ``"val"``。
        dtype:     ``vlm_last_hidden`` 的输出 dtype（默认 ``bfloat16``）。

    样本顺序由 shard 内 ``info.json`` 的 tokens 顺序决定（即 caching 时
    DistributedSampler 的遍历顺序）。
    """

    def __init__(
        self,
        cache_dir: str,
        split: str = "train",
        dtype: torch.dtype = torch.bfloat16,
        load_history_images: bool = False,
        history_image_resize_to: Tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self._split_dir = Path(os.path.expanduser(cache_dir)) / split
        self._dtype = dtype
        self._load_history_images = bool(load_history_images)
        self._history_image_resize_to = history_image_resize_to

        if not self._split_dir.exists():
            raise FileNotFoundError(
                f"CachedFeatureDataset: cache split 目录不存在: {self._split_dir}\n"
                f"请提供包含 val/ shard 文件的 VLM feature cache。"
            )

        hidden_files = sorted(self._split_dir.glob("hidden_*.bin"))
        if not hidden_files:
            raise FileNotFoundError(
                f"CachedFeatureDataset: 在 {self._split_dir} 下未找到任何 "
                f"hidden_*.bin shard 文件。请确认 feature cache 已完整生成。"
            )

        # ── 加载每个 shard ──────────────────────────────────────────────────
        self._shard_flats: List[torch.Tensor] = []
        self._shard_lengths: List[torch.Tensor] = []        # [n_local] int64
        self._shard_offsets_tok: List[torch.Tensor] = []    # [n_local] int64

        wan_action_indices_parts:  List[torch.Tensor] = []
        traj_action_indices_parts: List[torch.Tensor] = []
        start_action_idx_parts:    List[torch.Tensor] = []
        end_action_idx_parts:      List[torch.Tensor] = []
        jepa_action_indices_parts: List[torch.Tensor] = []
        vggt_action_indices_parts: List[torch.Tensor] = []
        hist_traj_parts: List[torch.Tensor] = []
        fut_traj_parts: List[torch.Tensor] = []
        ego_status_parts: List[torch.Tensor] = []
        nav_cmd_parts: List[torch.Tensor] = []
        ego_speed_parts: List[torch.Tensor] = []
        traj_interval_parts: List[torch.Tensor] = []
        tokens_all: List[str] = []
        history_image_paths_all: List[List[str]] = []

        shard_of_sample: List[int] = []
        local_of_sample: List[int] = []

        H_ref: int | None = None
        resize_ref: Tuple[int, int] | None = self._history_image_resize_to

        for shard_id, hidden_path in enumerate(hidden_files):
            rank_str = hidden_path.stem.split("_", 1)[1]
            info_path = self._split_dir / f"info_{rank_str}.json"
            tensors_path = self._split_dir / f"tensors_{rank_str}.safetensors"
            if not info_path.exists() or not tensors_path.exists():
                raise FileNotFoundError(
                    f"CachedFeatureDataset: shard {hidden_path.name} 缺少配套的 "
                    f"info/tensors 文件（期望 {info_path.name}, {tensors_path.name}）。"
                )

            with open(info_path, "r") as f:
                info = json.load(f)
            H = int(info["hidden_dim"])
            n_local = int(info["num_samples"])
            tokens = info["tokens"]
            if len(tokens) != n_local:
                raise ValueError(
                    f"{info_path} num_samples({n_local}) 与 tokens 长度 "
                    f"({len(tokens)}) 不一致。"
                )
            paths = info.get("history_image_paths")
            if paths is None:
                paths = [[] for _ in range(n_local)]
            if len(paths) != n_local:
                raise ValueError(
                    f"{info_path} history_image_paths 长度({len(paths)}) 与 "
                    f"num_samples({n_local}) 不一致。"
                )
            history_image_paths_all.extend([
                [str(p) for p in sample_paths]
                for sample_paths in paths
            ])
            resize_info = info.get("history_image_resize_to")
            if resize_info is not None:
                resize_tuple = (int(resize_info[0]), int(resize_info[1]))
                if resize_ref is None:
                    resize_ref = resize_tuple
                elif resize_tuple != resize_ref:
                    raise ValueError(
                        f"shard 之间 history_image_resize_to 不一致："
                        f"{resize_ref} vs {resize_tuple} (shard={hidden_path.name})"
                    )
            if H_ref is None:
                H_ref = H
            elif H != H_ref:
                raise ValueError(
                    f"shard 之间 hidden_dim 不一致：{H_ref} vs {H} (shard={hidden_path.name})"
                )

            meta = _st_load_file(str(tensors_path), device="cpu")
            missing = [k for k in _REQUIRED_META_KEYS if k not in meta]
            if missing:
                raise KeyError(
                    f"{tensors_path} 缺少字段 {missing}。"
                )

            nbytes = hidden_path.stat().st_size
            expected_tokens = int(meta["hidden_lengths"].to(torch.int64).sum().item())
            if nbytes != expected_tokens * H * 2:
                raise ValueError(
                    f"{hidden_path} 字节数({nbytes}) 与 sum(lengths)*H*2 "
                    f"({expected_tokens * H * 2}) 不匹配，shard 可能被截断。"
                )
            storage = torch.UntypedStorage.from_file(
                str(hidden_path), shared=False, nbytes=nbytes
            )
            flat = torch.empty(0, dtype=torch.bfloat16).set_(storage, 0, (nbytes // 2,))
            flat = flat.view(-1, H)

            self._shard_flats.append(flat)
            self._shard_lengths.append(meta["hidden_lengths"].to(torch.int64))
            self._shard_offsets_tok.append(meta["hidden_offsets_tok"].to(torch.int64))

            wan_action_indices_parts.append(meta["wan_action_indices"].to(torch.int64))
            traj_action_indices_parts.append(meta["traj_action_indices"].to(torch.int64))
            start_action_idx_parts.append(meta["start_action_idx"].to(torch.int64))
            end_action_idx_parts.append(meta["end_action_idx"].to(torch.int64))
            jepa_action_indices_parts.append(
                meta.get("jepa_action_indices", torch.empty((n_local, 0), dtype=torch.int64)).to(torch.int64)
            )
            vggt_action_indices_parts.append(
                meta.get("vggt_action_indices", torch.empty((n_local, 0), dtype=torch.int64)).to(torch.int64)
            )
            hist_traj_parts.append(meta["history_trajectory"].to(torch.float32))
            fut_traj_parts.append(meta["future_trajectory"].to(torch.float32))
            ego_status_parts.append(meta["ego_status"].to(torch.float32))
            nav_cmd_parts.append(meta["navigation_command"].to(torch.int64))
            ego_speed_parts.append(meta["ego_speed"].to(torch.float32))
            traj_interval_parts.append(meta["trajectory_interval_s"].to(torch.float32))

            tokens_all.extend(tokens)
            shard_of_sample.extend([shard_id] * n_local)
            local_of_sample.extend(range(n_local))

        # 全局拼接（小字段）
        wan_action_indices_all  = torch.cat(wan_action_indices_parts, dim=0)       # [N_raw, A]
        traj_action_indices_all = torch.cat(traj_action_indices_parts, dim=0)      # [N_raw, K]
        start_action_idx_all    = torch.cat(start_action_idx_parts, dim=0)         # [N_raw]
        end_action_idx_all      = torch.cat(end_action_idx_parts, dim=0)           # [N_raw]
        jepa_action_indices_all = torch.cat(jepa_action_indices_parts, dim=0)      # [N_raw, Mj]
        vggt_action_indices_all = torch.cat(vggt_action_indices_parts, dim=0)      # [N_raw, Mv]
        history_trajectory_all = torch.cat(hist_traj_parts, dim=0)
        future_trajectory_all = torch.cat(fut_traj_parts, dim=0)
        ego_status_all = torch.cat(ego_status_parts, dim=0)
        navigation_command_all = torch.cat(nav_cmd_parts, dim=0)
        ego_speed_all = torch.cat(ego_speed_parts, dim=0)
        trajectory_interval_all = torch.cat(traj_interval_parts, dim=0)
        shard_of_sample_t = torch.tensor(shard_of_sample, dtype=torch.int64)
        local_of_sample_t = torch.tensor(local_of_sample, dtype=torch.int64)

        # ── 跨 shard 去重 ──────────────────────────────────────────────────
        seen: Dict[str, int] = {}
        keep_idx: List[int] = []
        for i, tok in enumerate(tokens_all):
            if tok not in seen:
                seen[tok] = i
                keep_idx.append(i)
        n_dup = len(tokens_all) - len(keep_idx)
        if n_dup > 0:
            keep_t = torch.tensor(keep_idx, dtype=torch.int64)
            wan_action_indices_all  = wan_action_indices_all.index_select(0, keep_t)
            traj_action_indices_all = traj_action_indices_all.index_select(0, keep_t)
            start_action_idx_all    = start_action_idx_all.index_select(0, keep_t)
            end_action_idx_all      = end_action_idx_all.index_select(0, keep_t)
            jepa_action_indices_all = jepa_action_indices_all.index_select(0, keep_t)
            vggt_action_indices_all = vggt_action_indices_all.index_select(0, keep_t)
            history_trajectory_all = history_trajectory_all.index_select(0, keep_t)
            future_trajectory_all = future_trajectory_all.index_select(0, keep_t)
            ego_status_all = ego_status_all.index_select(0, keep_t)
            navigation_command_all = navigation_command_all.index_select(0, keep_t)
            ego_speed_all = ego_speed_all.index_select(0, keep_t)
            trajectory_interval_all = trajectory_interval_all.index_select(0, keep_t)
            shard_of_sample_t = shard_of_sample_t.index_select(0, keep_t)
            local_of_sample_t = local_of_sample_t.index_select(0, keep_t)
            history_image_paths_all = [history_image_paths_all[i] for i in keep_idx]
            tokens_all = [tokens_all[i] for i in keep_idx]

        self._wan_action_indices  = wan_action_indices_all
        self._traj_action_indices = traj_action_indices_all
        self._start_action_idx    = start_action_idx_all
        self._end_action_idx      = end_action_idx_all
        self._jepa_action_indices = jepa_action_indices_all
        self._vggt_action_indices = vggt_action_indices_all
        self._history_trajectory = history_trajectory_all
        self._future_trajectory = future_trajectory_all
        self._ego_status = ego_status_all
        self._navigation_command = navigation_command_all
        self._ego_speed = ego_speed_all
        self._trajectory_interval_s = trajectory_interval_all
        self._shard_of_sample = shard_of_sample_t
        self._local_of_sample = local_of_sample_t
        self._history_image_paths = history_image_paths_all
        self._history_image_resize_to = resize_ref
        self._tokens: List[str] = tokens_all

        assert H_ref is not None
        self._hidden_dim = H_ref

        N = len(self._tokens)
        if not (self._wan_action_indices.shape[0] == N
                and self._traj_action_indices.shape[0] == N
                and self._history_trajectory.shape[0] == N
                and self._ego_speed.shape[0] == N):
            raise ValueError(
                "CachedFeatureDataset: 小字段与 tokens 数量不一致，shard 可能损坏。"
            )
        if n_dup > 0:
            print(
                f"[CachedFeatureDataset][{self._split_dir.name}] "
                f"发现 {len(hidden_files)} 个 shard, raw={N + n_dup}, unique={N} "
                f"(去重 {n_dup} 个 DistributedSampler 补齐造成的重复样本)",
                flush=True,
            )

    # ── Dataset 接口 ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._tokens)

    def _load_history_image_tensor(self, image_path: str) -> torch.Tensor:
        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"CachedFeatureDataset: failed to load image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self._history_image_resize_to is not None:
            h, w = self._history_image_resize_to
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return (
            torch.from_numpy(img)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .mul(2.0)
            .sub(1.0)
            .clamp(-1.0, 1.0)
        )

    def _load_history_images_from_paths(self, paths: List[str]) -> torch.Tensor:
        if not paths:
            raise RuntimeError(
                "CachedFeatureDataset: load_history_images=True, but this cache "
                "sample has no history_image_paths. Please regenerate the VLA "
                "feature cache with the updated cache writer."
            )
        return torch.stack([self._load_history_image_tensor(p) for p in paths], dim=0)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        shard_id = int(self._shard_of_sample[index].item())
        local_id = int(self._local_of_sample[index].item())

        beg = int(self._shard_offsets_tok[shard_id][local_id].item())
        L = int(self._shard_lengths[shard_id][local_id].item())
        last_hidden = self._shard_flats[shard_id][beg : beg + L]
        if self._dtype != torch.bfloat16:
            last_hidden = last_hidden.to(self._dtype)

        history_image_paths = self._history_image_paths[index]
        item = {
            "history_trajectory":    self._history_trajectory[index],
            "future_trajectory":     self._future_trajectory[index],
            "ego_status":            self._ego_status[index],
            "navigation_command":    int(self._navigation_command[index].item()),
            "ego_speed":             float(self._ego_speed[index].item()),
            "trajectory_interval_s": float(self._trajectory_interval_s[index].item()),
            "history_image_paths":    history_image_paths,
            "scene_token":           self._tokens[index],
            "index":                 index,
            "vlm_last_hidden":            last_hidden,
            "vlm_start_action_idx":       self._start_action_idx[index],
            "vlm_end_action_idx":         self._end_action_idx[index],
            "vlm_jepa_action_indices":    self._jepa_action_indices[index],
            "vlm_vggt_action_indices":    self._vggt_action_indices[index],
            "vlm_wan_action_indices":     self._wan_action_indices[index],
            "vlm_traj_action_indices":    self._traj_action_indices[index],
        }
        if self._load_history_images:
            item["history_images"] = self._load_history_images_from_paths(history_image_paths)
        return item

    # ── 辅助属性 ──────────────────────────────────────────────────────────────

    @property
    def split_dir(self) -> Path:
        return self._split_dir

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def tokens(self) -> List[str]:
        return list(self._tokens)
