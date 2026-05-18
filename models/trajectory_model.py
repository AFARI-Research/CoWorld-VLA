"""Stage-3 trajectory prediction model.

:class:`AeActtokenVla` subclasses :class:`~models.vlm_worldmodel.WmVlmJoint` so stage 3
reuses the same VLM stack and optional Wan wiring.  Trajectory ADE/FDE validation and
diffusion planner live here.

**Planner conditioning** (``model.vlm_conditioning`` in YAML)

**须** 为**各 token 名 → bool** 的表（如 ``non_action_tokens: true / wan_action_tokens: true / …``），
为 ``true`` 的项按 **YAML 中的键写序** 组成列表，传给
:meth:`~models.vlm.qwen3_vl.Qwen3VLWrapper.forward_extract` 为多键 ``dict``（不事先拼接）；
:meth:`_extract_from_cache` 与之对齐；规划器在 :class:`RecogActionHead` 内拼为 ``[B,L,H]`` 后进 DiT。

**Stage-2 / stage-3 token compatibility**:
If a type was not trained (count=0), cache / forward yields ``[B, 0, H]`` for that part.

**Cache feature path**

When ``model.use_cached_features: true``, batch must contain ``vlm_*`` keys injected by
:class:`~datasets.cached_feature_dataset.CachedFeatureDataset` (all ``vlm_*`` fields from
``__getitem__`` are forwarded automatically by the eval/train data pipelines).
"""

from __future__ import annotations

import contextlib
import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
from PIL import Image as PILImage
from torchmetrics import MeanMetric

from models.registry import MODEL_REGISTRY
from models.action_head import ActionHead
from models.vlm.qwen3_vl import VLM_MODE_KEYS
from models.vlm_worldmodel import (
    WmVlmJoint,
    _clean_vjepa_state_dict,
    _import_vjepa21_vision_transformer,
    _load_vjepa_checkpoint,
)
from models.vendor import ensure_vggt_import_path
from training.metrics import trajectory_ade, trajectory_fde, trajectory_heading_error
from utils.utils import batch_to_device, cfg_get, cfg_model_torch_dtype


def _freeze_vlm_from_model_cfg(model_cfg: Any) -> bool:
    """Whether to freeze VLM weights and run its forward under ``torch.no_grad`` during training.

    ``model.freeze_vlm`` (bool) is **required** for :class:`AeActtokenVla`.
    """
    explicit = cfg_get(model_cfg, "freeze_vlm", None)
    if explicit is None:
        raise ValueError("model.freeze_vlm is required (bool) for AeActtokenVla.")
    return bool(explicit)


def _build_action_head(model_cfg: Any, vlm_hidden_size: int, vlm_conditioning: list[str] | None = None) -> ActionHead:
    """Build ActionHead from ``model.action_head`` and VLM hidden size."""
    ah_cfg = dict(cfg_get(model_cfg, "action_head", {}))
    ah_cfg.setdefault("name", "recog")
    ah_cfg["input_feature_dim"] = int(vlm_hidden_size)
    if vlm_conditioning is not None:
        ah_cfg["vlm_conditioning"] = vlm_conditioning
    ah_cfg["vlm"] = dict(cfg_get(model_cfg, "vlm", {}))
    return ActionHead(ah_cfg)


class _TrajectoryModelBase(WmVlmJoint):
    """Trajectory ADE/FDE metrics + action head helpers; extends :class:`WmVlmJoint`.

    Prompts and current-frame images use :meth:`~models.vlm_worldmodel.WmVlmJoint._build_prompts` /
    :meth:`~models.vlm_worldmodel.WmVlmJoint._extract_current_images`.
    """

    def __init__(self, cfg, full_cfg=None, **kwargs) -> None:
        full = full_cfg if full_cfg is not None else cfg
        super().__init__(cfg, full_cfg=full, **kwargs)
        self.val_ade = MeanMetric(sync_on_compute=True, dist_sync_on_step=False)
        self.val_fde = MeanMetric(sync_on_compute=True, dist_sync_on_step=False)
        self.val_heading = MeanMetric(sync_on_compute=True, dist_sync_on_step=False)

    def reset_validation_metrics(self) -> None:
        self.val_ade.reset()
        self.val_fde.reset()
        self.val_heading.reset()

    @torch.no_grad()
    def validation_step(
        self,
        batch: dict,
        _batch_idx: int,
        *,
        accelerator,
        **_: Any,
    ) -> None:
        del accelerator

        device = next(self.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        if "future_trajectory" not in inputs:
            return

        pred = self.predict_trajectory(inputs)
        gt = inputs["future_trajectory"]
        B = pred.shape[0]
        dev = pred.device

        for b in range(B):
            pred_np = pred[b].float().cpu().numpy()
            gt_np = gt[b].float().cpu().numpy()
            self.val_ade.update(torch.tensor(trajectory_ade(pred_np, gt_np), device=dev, dtype=torch.float32))
            self.val_fde.update(torch.tensor(trajectory_fde(pred_np, gt_np), device=dev, dtype=torch.float32))
            self.val_heading.update(
                torch.tensor(trajectory_heading_error(pred_np, gt_np), device=dev, dtype=torch.float32)
            )

    def compute_validation_metrics(self) -> dict[str, float]:
        out = {
            "ade": float(self.val_ade.compute().detach().cpu()),
            "fde": float(self.val_fde.compute().detach().cpu()),
            "heading_err": float(self.val_heading.compute().detach().cpu()),
        }
        self.val_ade.reset()
        self.val_fde.reset()
        self.val_heading.reset()
        return out

    def _prepare_traj_inputs(self, inputs: Dict, device: torch.device):
        """Extract history trajectory (12-dim) and ego status (8-dim) for the planner."""
        ht = inputs.get("history_trajectory")
        if ht is not None:
            ht_flat = ht.to(device)
            if ht_flat.dim() == 3:
                ht_flat = ht_flat.reshape(ht_flat.shape[0], -1)
            d = ht_flat.shape[-1]
            if d < 12:
                pad = torch.zeros(ht_flat.shape[0], 12 - d, device=device)
                ht_flat = torch.cat([ht_flat, pad], dim=-1)
            elif d > 12:
                raise ValueError(
                    f"history_trajectory flattened dims={d}, planner expects 12 "
                    f"(4 steps × 3). Check num_history_trajectory_steps config "
                    f"or data pipeline."
                )
            his_traj = ht_flat
        else:
            for _key in ("future_trajectory", "ego_status", "history_images"):
                if _key in inputs and isinstance(inputs[_key], torch.Tensor):
                    _B = inputs[_key].shape[0]
                    break
            else:
                raise KeyError(
                    "_prepare_traj_inputs: cannot infer batch size; "
                    "neither history_trajectory, future_trajectory, ego_status "
                    "nor history_images found in inputs."
                )
            his_traj = torch.zeros(_B, 12, device=device)

        B = his_traj.shape[0]

        ego_st = inputs.get("ego_status")
        if ego_st is not None:
            status = ego_st.to(device)
            if status.dim() == 1:
                status = status.unsqueeze(0)
            if status.shape[-1] != 8:
                raise ValueError(
                    f"ego_status must have last dimension 8, got {status.shape[-1]}"
                )
        else:
            nav = inputs.get("navigation_command")
            nav_oh = torch.zeros(B, 4, device=device)
            if nav is not None:
                nav_t = nav.to(device) if isinstance(nav, torch.Tensor) else torch.tensor(nav, device=device)
                for b in range(B):
                    idx = int(nav_t[b].item()) if nav_t.dim() > 0 else int(nav_t.item())
                    nav_oh[b, min(idx, 3)] = 1.0
            status = torch.cat([nav_oh, torch.zeros(B, 4, device=device)], dim=-1)

        return his_traj, status


@MODEL_REGISTRY.register("AeActtokenVla")
class AeActtokenVla(_TrajectoryModelBase):
    """Stage-3 VLM hidden states → diffusion action head.

    Requires ``model.freeze_vlm`` (bool)。``model.vlm_conditioning`` 须为
    各 token 的 **bool 开关**；规划器在 :class:`RecogActionHead` 内拼叠 VLM 多键输出。

    **Cache** (``use_cached_features: true``) 需 ``vlm_last_hidden`` 与
    ``vlm_*_action_indices`` / ``vlm_start_action_idx``；多键与在线路径一致，按槽 **dict** 再拼接。
    """

    def __init__(self, cfg, full_cfg=None, **kwargs):
        full = full_cfg if full_cfg is not None else cfg
        super().__init__(cfg, full_cfg=full, **kwargs)
        self._freeze_vlm = _freeze_vlm_from_model_cfg(cfg)
        if self._freeze_vlm:
            self.vlm.requires_grad_(False)

        ext_jepa_cfg = cfg_get(cfg, "external_jepa_context", {}) or {}
        self._use_external_jepa_context = bool(
            cfg_get(ext_jepa_cfg, "enabled", cfg_get(cfg, "use_external_jepa_context", False))
        )
        if self._use_external_jepa_context:
            self._init_external_jepa_context(cfg)

        ext_vggt_cfg = cfg_get(cfg, "external_vggt_context", {}) or {}
        self._use_external_vggt_context = bool(
            cfg_get(ext_vggt_cfg, "enabled", False)
        )
        if self._use_external_vggt_context:
            self._init_external_vggt_context(cfg)

        raw = cfg_get(cfg, "vlm_conditioning", None)
        if raw is None:
            raise ValueError("model.vlm_conditioning 为必填：各 token 名 → bool 的表（见例：configs/ae_fulltoken_fz_vla.yaml）。")
        if not isinstance(raw, (dict, DictConfig)):
            raise TypeError(
                f"model.vlm_conditioning 须为 dict，键为 {sorted(VLM_MODE_KEYS)}，得 {type(raw).__name__}。"
            )
        for key in raw:
            sk = str(key)
            if sk not in VLM_MODE_KEYS:
                raise ValueError(
                    f"model.vlm_conditioning 未知键 {sk!r}，仅允许 {sorted(VLM_MODE_KEYS)}。"
                )
        self._vlm_conditioning: list[str] = [str(k) for k in raw if bool(raw.get(k, False))]
        if not self._vlm_conditioning:
            raise ValueError("model.vlm_conditioning 至少一个须为 true。")
        else:
            print(f"model.vlm_conditioning: {self._vlm_conditioning}")

        action_head_conditioning = list(self._vlm_conditioning)
        if self._use_external_jepa_context:
            if "non_action_tokens" not in action_head_conditioning:
                raise ValueError(
                    "external_jepa_context.enabled=true requires "
                    "model.vlm_conditioning.non_action_tokens=true."
                )
            if "external_jepa_context_tokens" not in action_head_conditioning:
                insert_at = action_head_conditioning.index("non_action_tokens") + 1
                action_head_conditioning.insert(insert_at, "external_jepa_context_tokens")

        if self._use_external_vggt_context:
            if "non_action_tokens" not in action_head_conditioning:
                raise ValueError(
                    "external_vggt_context.enabled=true requires "
                    "model.vlm_conditioning.non_action_tokens=true."
                )
            if "external_vggt_context_tokens" not in action_head_conditioning:
                insert_at = action_head_conditioning.index("non_action_tokens") + 1
                action_head_conditioning.insert(insert_at, "external_vggt_context_tokens")

        self.action_head = _build_action_head(
            cfg_get(full, "model", cfg), self.vlm.hidden_size, action_head_conditioning
        )
        self._use_cached_features = bool(cfg_get(cfg, "use_cached_features", False))

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "external_jepa_encoder", None) is not None:
            self.external_jepa_encoder.eval()
        if getattr(self, "external_vggt_encoder", None) is not None:
            self.external_vggt_encoder.eval()
        return self

    def _init_external_jepa_context(self, model_cfg) -> None:
        """Frozen V-JEPA image encoder used as extra non-action context.

        By default this keeps the previous current-frame image-branch behavior.
        When ``external_jepa_context.teacher_input_mode: video`` is set, the
        most recent N history frames are passed as one video clip to V-JEPA so
        the teacher can model temporal structure internally.
        """
        ext_cfg = cfg_get(model_cfg, "external_jepa_context", {}) or {}
        jepa_cfg = cfg_get(model_cfg, "jepa", {}) or {}

        def _cfg(name: str, default: Any) -> Any:
            return cfg_get(ext_cfg, name, cfg_get(jepa_cfg, name, default))

        ckpt_path = _cfg("teacher_ckpt", None)
        if ckpt_path is None or str(ckpt_path).strip() == "":
            raise ValueError("Set model.external_jepa_context.teacher_ckpt in the config.")
        ckpt_path = str(ckpt_path)
        checkpoint_key = str(_cfg("checkpoint_key", "encoder"))
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"V-JEPA2.1 checkpoint not found: {ckpt_path}")

        variant = str(_cfg("teacher_variant", "vitG"))
        variant_alias = {
            "base": "vit_base",
            "vitb": "vit_base",
            "vit_base": "vit_base",
            "large": "vit_large",
            "vitl": "vit_large",
            "vit_large": "vit_large",
            "giant": "vit_giant_xformers",
            "vitg": "vit_giant_xformers",
            "vit_giant": "vit_giant_xformers",
            "vit_giant_xformers": "vit_giant_xformers",
            "gigantic": "vit_gigantic_xformers",
            "vitG": "vit_gigantic_xformers",
            "vit_gigantic": "vit_gigantic_xformers",
            "vit_gigantic_xformers": "vit_gigantic_xformers",
        }
        arch_name = variant_alias.get(variant, variant)

        input_hw = _cfg("teacher_input_hw", _cfg("input_hw", [384, 384]))
        if isinstance(input_hw, int):
            input_h = input_w = int(input_hw)
        else:
            input_h, input_w = [int(x) for x in list(input_hw)]
        patch_size = int(_cfg("patch_size", 16))
        if input_h % patch_size != 0 or input_w % patch_size != 0:
            raise ValueError(
                "model.external_jepa_context.teacher_input_hw must be divisible by "
                f"patch_size; got {(input_h, input_w)} and {patch_size}."
            )

        teacher_input_mode = str(_cfg("teacher_input_mode", "image")).lower()
        if teacher_input_mode in ("frame", "frames", "image", "images"):
            teacher_input_mode = "image"
        elif teacher_input_mode in ("video", "clip", "clips"):
            teacher_input_mode = "video"
        if teacher_input_mode not in ("image", "video"):
            raise ValueError(
                "model.external_jepa_context.teacher_input_mode must be "
                f"'image' or 'video', got {teacher_input_mode!r}."
            )
        self._external_jepa_teacher_input_mode = teacher_input_mode

        tubelet_size = int(_cfg("tubelet_size", 2))
        self._external_jepa_tubelet_size = tubelet_size
        img_temporal_dim_size = int(_cfg("img_temporal_dim_size", 1))
        if img_temporal_dim_size != 1:
            raise ValueError(
                "external_jepa_context expects img_temporal_dim_size=1 for "
                "the V-JEPA image branch fallback, got "
                f"{img_temporal_dim_size}."
            )
        context_frames_raw = _cfg("num_context_frames", _cfg("history_frames", 1))
        if isinstance(context_frames_raw, str) and context_frames_raw.lower() in {
            "all",
            "available",
            "auto",
        }:
            self._external_jepa_num_context_frames: int | None = None
        else:
            self._external_jepa_num_context_frames = int(context_frames_raw)
            if self._external_jepa_num_context_frames <= 0:
                raise ValueError(
                    "model.external_jepa_context.num_context_frames must be a "
                    "positive integer or 'all'."
                )
        default_encoder_frames = (
            self._external_jepa_num_context_frames
            if (
                teacher_input_mode == "video"
                and self._external_jepa_num_context_frames is not None
            )
            else 8
        )
        encoder_num_frames = int(_cfg("encoder_num_frames", _cfg("num_frames", default_encoder_frames)))
        if teacher_input_mode == "video":
            if encoder_num_frames < tubelet_size:
                raise ValueError(
                    "external_jepa_context video mode requires "
                    f"encoder_num_frames >= tubelet_size, got "
                    f"{encoder_num_frames} and {tubelet_size}."
                )
            if encoder_num_frames % tubelet_size != 0:
                raise ValueError(
                    "external_jepa_context video mode requires "
                    "encoder_num_frames to be divisible by tubelet_size, got "
                    f"{encoder_num_frames} and {tubelet_size}."
                )
            if (
                self._external_jepa_num_context_frames is not None
                and self._external_jepa_num_context_frames % tubelet_size != 0
            ):
                raise ValueError(
                    "external_jepa_context video mode requires "
                    "num_context_frames to be divisible by tubelet_size, got "
                    f"{self._external_jepa_num_context_frames} and {tubelet_size}."
                )

        vit_encoder = _import_vjepa21_vision_transformer()
        if not hasattr(vit_encoder, arch_name):
            raise ValueError(f"Unknown V-JEPA2.1 encoder variant: {variant!r}")

        self.external_jepa_encoder = getattr(vit_encoder, arch_name)(
            img_size=(input_h, input_w),
            patch_size=patch_size,
            num_frames=encoder_num_frames,
            tubelet_size=tubelet_size,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=True,
            img_temporal_dim_size=img_temporal_dim_size,
            interpolate_rope=True,
            n_output_distillation=1,
        )

        raw = _load_vjepa_checkpoint(ckpt_path)
        if not isinstance(raw, dict) or checkpoint_key not in raw:
            keys = sorted(raw.keys()) if isinstance(raw, dict) else type(raw).__name__
            raise KeyError(
                f"V-JEPA2.1 checkpoint {ckpt_path} has no key {checkpoint_key!r}; "
                f"available keys: {keys}"
            )
        state = _clean_vjepa_state_dict(raw[checkpoint_key])
        strict = bool(_cfg("strict_load", True))
        missing, unexpected = self.external_jepa_encoder.load_state_dict(state, strict=strict)
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"Strict V-JEPA external context load failed: missing={missing}, "
                f"unexpected={unexpected}"
            )
        if missing or unexpected:
            print(
                "[AeActtokenVla] V-JEPA external context load "
                f"missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )
        del raw, state

        self.external_jepa_encoder.requires_grad_(False)
        teacher_dtype = cfg_model_torch_dtype(
            {"dtype": _cfg("teacher_dtype", cfg_get(model_cfg, "dtype", "bfloat16"))}
        )
        self.external_jepa_encoder.to(dtype=teacher_dtype)
        self.external_jepa_encoder.eval()

        self._external_jepa_input_hw = (input_h, input_w)
        self._external_jepa_grid_hw = (input_h // patch_size, input_w // patch_size)
        self._external_jepa_tokens_per_image = (
            img_temporal_dim_size * self._external_jepa_grid_hw[0] * self._external_jepa_grid_hw[1]
        )

        def _expected_external_jepa_tokens(num_frames: int) -> int:
            if self._external_jepa_teacher_input_mode == "video":
                return (
                    (num_frames // self._external_jepa_tubelet_size)
                    * self._external_jepa_grid_hw[0]
                    * self._external_jepa_grid_hw[1]
                )
            return num_frames * self._external_jepa_tokens_per_image

        self._expected_external_jepa_tokens = _expected_external_jepa_tokens
        self._external_jepa_num_tokens = (
            None
            if self._external_jepa_num_context_frames is None
            else _expected_external_jepa_tokens(self._external_jepa_num_context_frames)
        )
        self.register_buffer(
            "_external_jepa_image_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_external_jepa_image_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1, 1),
            persistent=False,
        )

        target_dim = int(self.external_jepa_encoder.embed_dim)
        projector_type = str(_cfg("projector", "mlp")).lower()
        if projector_type == "linear":
            self.external_jepa_context_projector = nn.Sequential(
                nn.LayerNorm(target_dim),
                nn.Linear(target_dim, self.vlm.hidden_size),
            )
        elif projector_type == "mlp":
            hidden_dim = int(_cfg("projector_hidden_dim", self.vlm.hidden_size))
            self.external_jepa_context_projector = nn.Sequential(
                nn.LayerNorm(target_dim),
                nn.Linear(target_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.vlm.hidden_size),
            )
        else:
            raise ValueError(
                "model.external_jepa_context.projector must be 'linear' or 'mlp', "
                f"got {projector_type!r}."
            )

        print(
            "[AeActtokenVla] external_jepa_context enabled: "
            f"mode={self._external_jepa_teacher_input_mode} "
            f"context_frames="
            f"{self._external_jepa_num_context_frames if self._external_jepa_num_context_frames is not None else 'all'} "
            f"dense_tokens="
            f"{self._external_jepa_num_tokens if self._external_jepa_num_tokens is not None else 'dynamic'} "
            f"teacher_dim={target_dim} -> vlm_dim={self.vlm.hidden_size}",
            flush=True,
        )

    def _build_external_jepa_history_clips(
        self,
        inputs: Dict[str, Any],
    ) -> tuple[torch.Tensor, int, int]:
        history = inputs.get("history_images")
        if not isinstance(history, torch.Tensor):
            raise ValueError(
                "model.external_jepa_context.enabled=true requires inputs['history_images']; "
                "if using cached VLM features, regenerate the cache with history_image_paths "
                "and set model.load_history_images_from_cache=true."
            )
        if history.dim() != 5 or history.shape[2] != 3:
            raise ValueError(
                "history_images must have shape [B, T, 3, H, W], "
                f"got {tuple(history.shape)}."
            )
        if history.shape[1] < 1:
            raise ValueError("history_images must contain the current frame.")

        teacher_param = next(self.external_jepa_encoder.parameters())
        available = int(history.shape[1])
        context_frames = (
            available
            if self._external_jepa_num_context_frames is None
            else int(self._external_jepa_num_context_frames)
        )
        if available < context_frames:
            raise ValueError(
                "external_jepa_context requested "
                f"{context_frames} history frames, but history_images only has "
                f"{available}. Regenerate the VLA feature cache with "
                f"data.num_history_image_frames >= {context_frames} and point "
                "vlm_feature_cache_dir to that cache."
            )

        clip = history[:, -context_frames:].to(device=teacher_param.device, dtype=torch.float32)
        bsz, timesteps, channels, _, _ = clip.shape
        if channels != 3:
            raise ValueError(
                f"expected history clip [B,T,3,H,W], got {tuple(clip.shape)}"
            )
        if (
            self._external_jepa_teacher_input_mode == "video"
            and timesteps % self._external_jepa_tubelet_size != 0
        ):
            raise ValueError(
                "external_jepa_context video mode requires the selected history "
                "frame count to be divisible by tubelet_size, got "
                f"{timesteps} and {self._external_jepa_tubelet_size}."
            )

        x = clip.mul(0.5).add(0.5).clamp_(0.0, 1.0)
        x = x.reshape(bsz * timesteps, channels, x.shape[-2], x.shape[-1])
        x = F.interpolate(
            x,
            size=self._external_jepa_input_hw,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        if self._external_jepa_teacher_input_mode == "video":
            x = x.view(
                bsz,
                timesteps,
                channels,
                self._external_jepa_input_hw[0],
                self._external_jepa_input_hw[1],
            )
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        else:
            x = x.view(
                bsz * timesteps,
                channels,
                1,
                self._external_jepa_input_hw[0],
                self._external_jepa_input_hw[1],
            )
        x = (x - self._external_jepa_image_mean.to(x.device)) / self._external_jepa_image_std.to(x.device)
        return x.to(dtype=teacher_param.dtype), bsz, timesteps

    def _encode_external_jepa_context(self, inputs: Dict[str, Any]) -> torch.Tensor:
        clip, bsz, timesteps = self._build_external_jepa_history_clips(inputs)
        with torch.no_grad():
            dense = self.external_jepa_encoder(clip, training=False).detach()
        if dense.dim() != 3:
            raise ValueError(
                "Unexpected external JEPA output rank: "
                f"got shape {tuple(dense.shape)}."
            )
        expected_tokens = self._expected_external_jepa_tokens(timesteps)
        if self._external_jepa_teacher_input_mode == "video":
            if dense.shape[0] != bsz:
                raise ValueError(
                    "Unexpected external JEPA video batch size: "
                    f"got {dense.shape[0]}, expected {bsz}."
                )
        else:
            if dense.shape[0] != bsz * timesteps:
                raise ValueError(
                    "Unexpected external JEPA image batch size: "
                    f"got {dense.shape[0]}, expected {bsz * timesteps}."
                )
            dense = dense.reshape(bsz, expected_tokens, dense.shape[-1])
        if dense.shape[1] != expected_tokens:
            raise ValueError(
                "Unexpected external JEPA token count: "
                f"got {dense.shape[1]}, expected {expected_tokens}."
            )

        proj_param = next(self.external_jepa_context_projector.parameters())
        return self.external_jepa_context_projector(
            dense.to(device=proj_param.device, dtype=proj_param.dtype)
        )

    def _append_external_jepa_context(
        self,
        vlm_bag: dict[str, Any],
        inputs: Dict[str, Any],
    ) -> dict[str, Any]:
        non_action = vlm_bag.get("non_action_tokens")
        if not isinstance(non_action, torch.Tensor):
            raise KeyError(
                "external_jepa_context requires 'non_action_tokens' in model.vlm_conditioning."
            )

        jepa_ctx = self._encode_external_jepa_context(inputs)
        jepa_ctx = jepa_ctx.to(device=non_action.device, dtype=non_action.dtype)
        bsz, jepa_len = jepa_ctx.shape[:2]
        if bsz != non_action.shape[0]:
            raise ValueError(
                "external JEPA context batch mismatch: "
                f"{bsz} vs non_action batch {non_action.shape[0]}."
            )

        jepa_mask = torch.ones(bsz, jepa_len, dtype=torch.bool, device=non_action.device)

        out = dict(vlm_bag)
        out["external_jepa_context_tokens"] = jepa_ctx
        out["external_jepa_context_tokens_mask"] = jepa_mask
        return out

    # ── External VGGT context (frozen VGGT encoder) ──────────────────────

    def _init_external_vggt_context(self, model_cfg) -> None:
        """Frozen VGGT aggregator used as extra non-action context."""
        ext_cfg = cfg_get(model_cfg, "external_vggt_context", {}) or {}

        vggt_model_path = str(cfg_get(ext_cfg, "vggt_model_path", ""))
        if not vggt_model_path:
            raise ValueError(
                "model.external_vggt_context.vggt_model_path is required."
            )

        ensure_vggt_import_path()
        from vggt.models.vggt import VGGT
        self.external_vggt_encoder = VGGT.from_pretrained(vggt_model_path)
        self.external_vggt_encoder.eval()
        self.external_vggt_encoder.requires_grad_(False)

        self._external_vggt_pool_size = tuple(
            cfg_get(ext_cfg, "pool_size", [4, 8])
        )
        self._external_vggt_num_history_frames = int(
            cfg_get(ext_cfg, "num_history_frames", 1)
        )
        if self._external_vggt_num_history_frames <= 0:
            raise ValueError(
                "model.external_vggt_context.num_history_frames must be positive."
            )

        target_dim = 2048
        projector_type = str(cfg_get(ext_cfg, "projector", "mlp")).lower()
        if projector_type == "linear":
            self.external_vggt_context_projector = nn.Sequential(
                nn.LayerNorm(target_dim),
                nn.Linear(target_dim, self.vlm.hidden_size),
            )
        elif projector_type == "mlp":
            hidden_dim = int(cfg_get(ext_cfg, "projector_hidden_dim", self.vlm.hidden_size))
            self.external_vggt_context_projector = nn.Sequential(
                nn.LayerNorm(target_dim),
                nn.Linear(target_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.vlm.hidden_size),
            )
        else:
            raise ValueError(
                "model.external_vggt_context.projector must be 'linear' or 'mlp', "
                f"got {projector_type!r}."
            )

        pool_h, pool_w = self._external_vggt_pool_size
        nhf = self._external_vggt_num_history_frames
        out_tokens = nhf * pool_h * pool_w
        print(
            "[AeActtokenVla] external_vggt_context enabled: "
            f"num_history_frames={nhf} "
            f"pool_size={self._external_vggt_pool_size} -> {out_tokens} tokens, "
            f"vggt_dim={target_dim} -> vlm_dim={self.vlm.hidden_size}",
            flush=True,
        )

    def _encode_external_vggt_context(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """Encode history frames with frozen VGGT aggregator → projected tokens.

        Returns:
            [B, T * pool_h * pool_w, vlm_hidden_size] projected VGGT patch tokens.
        """
        history = inputs.get("history_images")
        if not isinstance(history, torch.Tensor):
            raise ValueError(
                "model.external_vggt_context.enabled=true requires inputs['history_images']; "
                "if using cached VLM features, regenerate the cache with history_image_paths "
                "and set model.load_history_images_from_cache=true."
            )
        if history.dim() != 5 or history.shape[2] != 3:
            raise ValueError(
                "history_images must have shape [B, T, 3, H, W], "
                f"got {tuple(history.shape)}."
            )
        if history.shape[1] < 1:
            raise ValueError("history_images must contain the current frame.")

        nhf = self._external_vggt_num_history_frames
        available = int(history.shape[1])
        if nhf > available:
            raise ValueError(
                "external_vggt_context requested "
                f"{nhf} history frames, but history_images only has {available}. "
                "Regenerate the VLA feature cache with enough history_image_paths "
                "and set model.load_history_images_from_cache=true."
            )

        vggt_param = next(self.external_vggt_encoder.parameters())
        vggt_dev = vggt_param.device
        vggt_dtype = vggt_param.dtype

        frames_input = history[:, -nhf:]  # [B, S, 3, H, W]
        B, S, C, H, W = frames_input.shape
        frames = frames_input.reshape(B * S, C, H, W)
        frames = F.interpolate(frames, size=(504, 1008), mode="bilinear", align_corners=False)
        frames = frames.to(device=vggt_dev, dtype=vggt_dtype)
        frames = frames.reshape(B, S, C, 504, 1008)

        with torch.no_grad():
            aggregated_tokens_list, _ = self.external_vggt_encoder.aggregator(frames)
            target = aggregated_tokens_list[-1]  # [B, S, num_patches, D]
            target = target[:, :, 5:, :]  # skip first 5 tokens (register/cam)
            # [B*S, 36, 72, D] -> pool -> flatten
            target = target.reshape(B * S, 36, 72, -1).permute(0, 3, 1, 2)
            pool_h, pool_w = self._external_vggt_pool_size
            target = F.adaptive_avg_pool2d(target, output_size=(pool_h, pool_w))
            target = target.flatten(2).transpose(1, 2)  # [B*S, pool_h*pool_w, D]
            target = target.reshape(B, S, pool_h * pool_w, -1)
            target = target.reshape(B, S * pool_h * pool_w, -1)
            target = target.detach()

        proj_param = next(self.external_vggt_context_projector.parameters())
        return self.external_vggt_context_projector(
            target.to(device=proj_param.device, dtype=proj_param.dtype)
        )

    def _append_external_vggt_context(
        self,
        vlm_bag: dict[str, Any],
        inputs: Dict[str, Any],
    ) -> dict[str, Any]:
        non_action = vlm_bag.get("non_action_tokens")
        if not isinstance(non_action, torch.Tensor):
            raise KeyError(
                "external_vggt_context requires 'non_action_tokens' in model.vlm_conditioning."
            )

        vggt_ctx = self._encode_external_vggt_context(inputs)
        vggt_ctx = vggt_ctx.to(device=non_action.device, dtype=non_action.dtype)
        bsz, vggt_len = vggt_ctx.shape[:2]
        if bsz != non_action.shape[0]:
            raise ValueError(
                "external VGGT context batch mismatch: "
                f"{bsz} vs non_action batch {non_action.shape[0]}."
            )

        vggt_mask = torch.ones(bsz, vggt_len, dtype=torch.bool, device=non_action.device)

        out = dict(vlm_bag)
        out["external_vggt_context_tokens"] = vggt_ctx
        out["external_vggt_context_tokens_mask"] = vggt_mask
        return out

    def _get_vlm_bag(self, inputs: Dict[str, Any], device: torch.device) -> dict[str, Any]:
        """VLM/缓存 与规划器之间的**统一接口**：与 ``forward_extract``/缓存同构的 token ``dict``（可含 `ce_loss` 等，由 planner 忽略）。"""
        if self._use_cached_features:
            if "vlm_last_hidden" not in inputs:
                raise KeyError(
                    "model.use_cached_features=true 但 batch 中缺少 'vlm_last_hidden'。\n"
                    "请确认：\n"
                    "  1. 已提供与当前 checkpoint 匹配的 VLM feature cache；\n"
                    "  2. eval 启动参数或环境变量已设置 vlm_feature_cache_dir；\n"
                    "  3. cache 的 val/ 目录中存在 hidden_*.bin、tensors_*.safetensors 和 info_*.json。"
                )
            vlm_out = self._extract_from_cache(inputs, device)
        else:
            images = self._extract_current_images(inputs)
            prompts = self._build_prompts(inputs)
            ctx = torch.no_grad() if self._freeze_vlm else contextlib.nullcontext()
            with ctx:
                vlm_out = self.vlm.forward_extract(
                    images,
                    prompts,
                    assistant_answers=None,
                    compute_ce_loss=False,
                    vlm_conditioning=self._vlm_conditioning,
                )
        vlm_out = batch_to_device(vlm_out, device)
        if self._use_external_jepa_context:
            vlm_out = self._append_external_jepa_context(vlm_out, inputs)
        if self._use_external_vggt_context:
            vlm_out = self._append_external_vggt_context(vlm_out, inputs)
        return batch_to_device(vlm_out, device)

    def _extract_from_cache(
        self,
        inputs: Dict[str, Any],
        device: torch.device,
        mode: Optional[list[str] | tuple[str, ...] | ListConfig] = None,
    ) -> dict[str, Any]:
        """Reconstruct token features from cached last_hidden using pre-saved indices.

        All ``vlm_*`` fields from :class:`~datasets.cached_feature_dataset.CachedFeatureDataset`
        are available in ``inputs``. Index tensors use the same semantics as
        :meth:`~models.vlm.qwen3_vl.Qwen3VLWrapper._compute_content_indices`.

        Returns:
            与 :meth:`Qwen3VLWrapper.forward_extract` 同构的 **token 名 → tensor**（及变长的 ``*_mask``）；
            与在线路径一起交给 **planner 文件**内函数过滤/拼叠。
        """
        raw = inputs["vlm_last_hidden"]
        wan_idx  = inputs.get("vlm_wan_action_indices")              # [B, N]  or None
        traj_idx = inputs.get("vlm_traj_action_indices")             # [B, K]  or None
        start_idx = inputs["vlm_start_action_idx"].to(device)        # [B]
        jepa_idx  = inputs.get("vlm_jepa_action_indices")
        vggt_idx  = inputs.get("vlm_vggt_action_indices")
        if wan_idx is not None:
            wan_idx = wan_idx.to(device)
        if traj_idx is not None:
            traj_idx = traj_idx.to(device)

        if mode is None:
            mode = self._vlm_conditioning
        if not isinstance(mode, (list, tuple, ListConfig)):
            raise TypeError(
                f"_extract_from_cache: mode 须为 list/tuple/ListConfig（与 model.vlm_conditioning 同形），"
                f"得 {type(mode).__name__}。"
            )

        # Normalize raw to a per-sample list for uniform handling.
        if isinstance(raw, torch.Tensor):
            B = raw.shape[0]
            H = raw.shape[2]
            raw_dev = raw.to(device)
            lh = None  # use vectorised gather for fixed modes
        else:
            B = len(raw)
            H = raw[0].shape[1]
            raw_dev = None
            lh = [h.to(device) for h in raw]

        dtype = raw_dev.dtype if raw_dev is not None else lh[0].dtype  # type: ignore[index]

        def _gather_fixed(idx_t: Optional[torch.Tensor]) -> torch.Tensor:
            """Gather rows from raw hidden by index tensor [B, M] → [B, M, H].
            ``None`` or empty index → empty ``[B, 0, H]`` stub tensor.
            """
            if idx_t is None or idx_t.numel() == 0 or idx_t.shape[1] == 0:
                return torch.empty(B, 0, H, device=device, dtype=dtype)
            idx_t = idx_t.to(device)
            M = idx_t.shape[1]
            if raw_dev is not None:
                exp = idx_t.unsqueeze(-1).expand(B, M, H)
                return torch.gather(raw_dev, 1, exp)
            return torch.stack([lh[b][idx_t[b]] for b in range(B)], dim=0)  # type: ignore[index]

        def _get_non_action() -> Tuple[torch.Tensor, torch.Tensor]:
            """Extract non_action tokens (before start_action) as padded tensor + mask."""
            max_non = int(start_idx.max().item())
            non_act = torch.zeros(B, max_non, H, device=device, dtype=dtype)
            non_act_mask = torch.zeros(B, max_non, dtype=torch.bool, device=device)
            for b in range(B):
                sp = int(start_idx[b].item())
                if raw_dev is not None:
                    non_act[b, :sp] = raw_dev[b, :sp]
                else:
                    non_act[b, :sp] = lh[b][:sp]  # type: ignore[index]
                non_act_mask[b, :sp] = True
            return non_act, non_act_mask

        components = [str(x) for x in mode]
        d: dict[str, Any] = {}
        for c in components:
            if c == "non_action_tokens":
                na, nm = _get_non_action()
                d["non_action_tokens"] = na
                d["non_action_tokens_mask"] = nm
            elif c == "jepa_action_tokens":
                d["jepa_action_tokens"] = _gather_fixed(jepa_idx)
            elif c == "vggt_action_tokens":
                d["vggt_action_tokens"] = _gather_fixed(vggt_idx)
            elif c == "wan_action_tokens":
                d["wan_action_tokens"] = _gather_fixed(wan_idx)
            elif c == "traj_action_tokens":
                d["traj_action_tokens"] = _gather_fixed(traj_idx)
            else:
                raise ValueError(f"_extract_from_cache: unknown component {c!r}.")
        return d

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        device = next(self.action_head.parameters()).device
        vlm_bag = self._get_vlm_bag(inputs, device)
        his_traj, status = self._prepare_traj_inputs(inputs, device)
        gt = inputs["future_trajectory"].to(device)
        result = self.action_head({
            "vlm_bag": vlm_bag,
            "his_traj": his_traj,
            "status": status,
            "gt": gt,
        })

        if isinstance(result, dict) and "loss" in result and isinstance(result["loss"], dict):
            loss_part: Dict[str, Any] = dict(result["loss"])
            other_log: Dict[str, Any] = dict(result.get("other_log") or {})
        elif isinstance(result, dict):
            loss_part = dict(result)
            other_log = {}
        else:
            loss_part = {"trajectory_loss": result}
            other_log = {}

        return {"loss": loss_part, "other_log": other_log}

    @torch.no_grad()
    def predict_trajectory(self, inputs: Dict[str, Any]) -> torch.Tensor:
        device = next(self.action_head.parameters()).device
        vlm_bag = self._get_vlm_bag(inputs, device)
        his_traj, status = self._prepare_traj_inputs(inputs, device)
        return self.action_head.predict({
            "vlm_bag": vlm_bag,
            "his_traj": his_traj,
            "status": status,
        })
