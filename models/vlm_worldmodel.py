"""Stage-2 joint model: VLM (Qwen3-VL) + optional world-model branches.

Data flow:
  VLM forward → extract action token features per enabled branch:
    - wan_action_tokens  → LatentConditionEncoder → Wan flow-matching loss  (use_wan)
    - traj_action_tokens → TrajPredMLP (last token) → trajectory MSE loss   (use_traj)
    - jepa_action_tokens → projection head → frozen V-JEPA2.1 feature loss   (use_jepa)
    - [vggt: external encoder branch reserved]                               (use_vggt)
  二阶段 ``vlm_conditioning`` 为四槽定长列表（``jepa`` / ``vggt`` / ``wan`` / ``traj`` 的
  ``*_action_tokens``，无 ``non_action_tokens``）。未用槽在 ``num_*_action_tokens==0`` 时仍抽取、
  得 ``[B,0,H]``，与 Stage-3 的 list 多键同形，见
  :data:`VLM_FIXED_SIZE_MODES_ORDER`、:meth:`Qwen3VLWrapper.forward_extract`。
  ``use_vlm_autoreg_text_trajectory`` only gates trajectory text CE and
  ``generate_text``; it does not change the action-token selection.
  VLM assistant response → CE loss on trajectory text when ``model.use_vlm_autoreg_text_trajectory``
  is true (teacher forcing on gold trajectory text).

All action token types are controlled by ``use_xxx`` flags and the corresponding
``num_xxx_action_tokens`` count in ``model.vlm``.  Any combination (including zero)
is supported.

**Stage-3 conditioning compatibility**:
  Stage 3 can only use token types that were active in stage 2 (non-zero count).
  Inactive types produce empty indices in the cache, so extracting them yields
  ``[B, 0, H]`` tensors — effectively no signal — without raising errors.

**Validation / inference trajectory** (stage-2 only):
  ``model.eval_traj_mode``:
    ``"text"``       (default) — greedy ``generate`` on trajectory text → parse points
                       (only if ``use_vlm_autoreg_text_trajectory`` is true).
    ``"traj_token"`` — last traj_action hidden → MLPActionHead (requires ``use_traj: true``).
  ``model.use_vlm_autoreg_text_trajectory`` (default ``true``): if ``false``, no
  trajectory text CE in training, validation skips ``generate_text``, and
  :meth:`predict_trajectory` requires ``eval_traj_mode="traj_token"``.
"""

from __future__ import annotations

import math
import os
import sys
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PILImage
from torchmetrics import MeanMetric

from datasets.data_utils import visualize_reconstruction
from models.registry import MODEL_REGISTRY
from models.action_head import ActionHead
from models.vlm.driving_prompt import (
    POST_ACTION_USER_HINT,
    build_driving_prompt,
    build_trajectory_answer,
)
from models.vlm.qwen3_vl import Qwen3VLWrapper, VLM_FIXED_SIZE_MODES_ORDER
from models.worldmodelbase import WanWorldModel
from models.vendor import ensure_vggt_import_path
from training.metrics import parse_trajectory_text, psnr, trajectory_ade, trajectory_fde
from utils.utils import cfg_get, cfg_model_torch_dtype


_TRAJ_TEXT_MAX_NEW = 512


def _import_vjepa21_vision_transformer():
    """Import vendored V-JEPA2.1 encoder code without requiring package install."""
    vjepa_root = os.path.join(os.path.dirname(__file__), "vjepa2")
    if vjepa_root not in sys.path:
        sys.path.insert(0, vjepa_root)
    from app.vjepa_2_1.models import vision_transformer

    return vision_transformer


def _clean_vjepa_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip wrappers used by official V-JEPA2.1 checkpoints."""
    cleaned: Dict[str, torch.Tensor] = {}
    for key, val in state_dict.items():
        name = key
        if name.startswith("module."):
            name = name[len("module."):]
        if name.startswith("backbone."):
            name = name[len("backbone."):]
        cleaned[name] = val
    return cleaned


def _load_vjepa_checkpoint(path: str) -> Dict[str, Any]:
    """Load large V-JEPA checkpoints with mmap when the local PyTorch supports it."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def _tensor_to_pil(t: torch.Tensor) -> PILImage.Image:
    arr = t.detach().cpu().float().mul(0.5).add(0.5).clamp(0, 1)
    arr = (arr.permute(1, 2, 0).numpy() * 255).astype("uint8")
    return PILImage.fromarray(arr)


class VggtAdapter(nn.Module):
    """VGGT adapter: concat + linear token mixer
    融合 latent CoT tokens (vggt_action_tokens) 和 masked image tokens 后，映射到 VGGT 的几何 latent 空间
    H_geo (K,D) -> Linear -> (M,D)
    Image -> encoder -> (N,D) -> mask -> (N,D)
    (M,D) ⊕ (N,D) -> concat -> Linear -> (V,D) -> Linear -> (V,D')
    """
    def __init__(self, hidden_dim: int, adapter_dim: int, num_vggt_tokens: int, num_image_tokens: int = 512, m: int = 12, v: int = 12):
        super().__init__()
        self.k = num_vggt_tokens
        self.m = m
        self.n = num_image_tokens
        self.v = v
        self.d = hidden_dim
        self.d_prime = adapter_dim
        
        self.proj_m = nn.Linear(self.k, self.m)
        self.norm_m = nn.LayerNorm(self.d)
        
        self.proj_v = nn.Linear(self.m + self.n, self.v)
        self.norm_v = nn.LayerNorm(self.d)
        
        self.proj_out = nn.Sequential(
            nn.Linear(self.d, self.d_prime),
            nn.LayerNorm(self.d_prime),
            nn.SiLU(),
            nn.Linear(self.d_prime, self.d_prime)
        )

        self.mask_ratio = 0.5

    def forward(self, h_geo: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        # h_geo: [B, K, D]
        # h_geo -> (M,D)
        h_m = self.proj_m(h_geo.transpose(1, 2)).transpose(1, 2)  # [B, M, D]
        h_m = F.silu(self.norm_m(h_m))
        
        # Image -> (N,D). Assuming image_features is [B, N, D]
        N_real = image_features.shape[1]
        if N_real != self.n:
            # Interpolate to fixed N if needed
            image_features = torch.nn.functional.adaptive_avg_pool1d(image_features.transpose(1, 2), self.n).transpose(1, 2)

        if self.training:
            # Random Mask
            mask = torch.rand(image_features.shape[0], self.n, device=image_features.device) > self.mask_ratio
            image_features = image_features * mask.unsqueeze(-1).to(image_features.dtype)

        # Concat
        concat_features = torch.cat([h_m, image_features], dim=1)  # [B, M+N, D]
        
        # Linear -> (V,D)
        v_features = self.proj_v(concat_features.transpose(1, 2)).transpose(1, 2)  # [B, V, D]
        v_features = F.silu(self.norm_v(v_features))
        
        # Linear -> (V,D')
        p_geo = self.proj_out(v_features)  # [B, V, D']
        
        return p_geo

class VggtMultiStepAdapter(nn.Module):
    """VGGT Multi-step adapter: concat + linear token mixer
    融合 latent CoT tokens (vggt_action_tokens) 和 masked image tokens 后，映射到 VGGT 的几何 latent 空间
    """
    def __init__(self, hidden_dim: int, adapter_dim: int, num_vggt_tokens: int, num_image_tokens: int = 512, m: int = 12, v: int = 12, num_steps: int = 1, use_visual_token: bool = True):
        super().__init__()
        self.k = num_vggt_tokens
        self.num_steps = num_steps
        self.m = m
        self.n = num_image_tokens
        self.v = v
        self.d = hidden_dim
        self.d_prime = adapter_dim
        self.use_visual_token = use_visual_token
        
        self.step_embed = nn.Embedding(num_steps, hidden_dim)
        
        self.proj_m = nn.Linear(self.k, self.m)
        self.norm_m = nn.LayerNorm(self.d)
        
        input_dim_v = self.m + self.n if self.use_visual_token else self.m
        self.proj_v = nn.Linear(input_dim_v, self.v)
        self.norm_v = nn.LayerNorm(self.d)
        
        self.proj_out = nn.Sequential(
            nn.Linear(self.d, self.d_prime),
            nn.LayerNorm(self.d_prime),
            nn.SiLU(),
            nn.Linear(self.d_prime, self.d_prime)
        )

        self.mask_ratio = 0.5

    def forward(self, h_geo: torch.Tensor, image_features: torch.Tensor = None) -> torch.Tensor:
        B = h_geo.shape[0]
        
        # h_geo: [B, T*K, D] -> [B, T, K, D]
        h_geo = h_geo.view(B, self.num_steps, self.k, self.d)
        
        # Add Step Embedding
        step_ids = torch.arange(self.num_steps, device=h_geo.device)
        s_emb = self.step_embed(step_ids)  # [T, D]
        h_geo = h_geo + s_emb.view(1, self.num_steps, 1, self.d)
        
        # Flatten time and batch: [B*T, K, D]
        h_geo_flat = h_geo.view(B * self.num_steps, self.k, self.d)
        
        # h_geo_flat -> (M,D)
        h_m = self.proj_m(h_geo_flat.transpose(1, 2)).transpose(1, 2)  # [B*T, M, D]
        h_m = F.silu(self.norm_m(h_m))
        
        if self.use_visual_token and image_features is not None:
            # Image -> [B, N, D] -> Repeat to [B*T, N, D]
            N_real = image_features.shape[1]
            if N_real != self.n:
                # Interpolate to fixed N if needed
                image_features = torch.nn.functional.adaptive_avg_pool1d(image_features.transpose(1, 2), self.n).transpose(1, 2)
                
            image_features_flat = image_features.repeat_interleave(self.num_steps, dim=0)

            if self.training:
                # Random Mask
                mask = torch.rand(image_features_flat.shape[0], self.n, device=image_features_flat.device) > self.mask_ratio
                image_features_flat = image_features_flat * mask.unsqueeze(-1).to(image_features_flat.dtype)

            # Concat
            concat_features = torch.cat([h_m, image_features_flat], dim=1)  # [B*T, M+N, D]
        else:
            concat_features = h_m
            
        # Linear -> (V,D)
        v_features = self.proj_v(concat_features.transpose(1, 2)).transpose(1, 2)  # [B*T, V, D]
        v_features = F.silu(self.norm_v(v_features))
        
        # Linear -> (V,D') -> [B, T, V, D']
        p_geo = self.proj_out(v_features)  # [B*T, V, D']
        p_geo = p_geo.view(B, self.num_steps, self.v, self.d_prime)
        
        return p_geo



class VggtCameraAdapter(nn.Module):
    """VGGT Camera target adapter"""
    def __init__(self, hidden_dim: int, adapter_dim: int, num_vggt_tokens: int):
        super().__init__()
        self.k = num_vggt_tokens
        self.proj_k = nn.Linear(self.k, 1)
        self.proj_out = nn.Sequential(
            nn.Linear(hidden_dim, adapter_dim),
            nn.LayerNorm(adapter_dim),
            nn.SiLU(),
            nn.Linear(adapter_dim, adapter_dim)
        )

    def forward(self, h_geo: torch.Tensor) -> torch.Tensor:
        h_m = self.proj_k(h_geo.transpose(1, 2)).transpose(1, 2)  # [B, 1, D]
        p_cam = self.proj_out(h_m.squeeze(1))  # [B, D']
        return p_cam

class VggtCameraMultiStepAdapter(nn.Module):
    """VGGT Camera target Multi-step adapter"""
    def __init__(self, hidden_dim: int, adapter_dim: int, num_vggt_tokens: int, num_steps: int):
        super().__init__()
        self.k = num_vggt_tokens
        self.num_steps = num_steps
        self.d = hidden_dim
        self.step_embed = nn.Embedding(num_steps, hidden_dim)
        self.proj_k = nn.Linear(self.k, 1)
        self.proj_out = nn.Sequential(
            nn.Linear(hidden_dim, adapter_dim),
            nn.LayerNorm(adapter_dim),
            nn.SiLU(),
            nn.Linear(adapter_dim, adapter_dim)
        )

    def forward(self, h_geo: torch.Tensor) -> torch.Tensor:
        B = h_geo.shape[0]
        h_geo = h_geo.view(B, self.num_steps, self.k, self.d)
        step_ids = torch.arange(self.num_steps, device=h_geo.device)
        s_emb = self.step_embed(step_ids)
        h_geo = h_geo + s_emb.view(1, self.num_steps, 1, self.d)
        h_geo_flat = h_geo.view(B * self.num_steps, self.k, self.d)
        h_m = self.proj_k(h_geo_flat.transpose(1, 2)).transpose(1, 2)
        p_cam = self.proj_out(h_m.squeeze(1))
        return p_cam.view(B, self.num_steps, -1)

class TrajPredMLP(nn.Module):
    """Simple MLP: last traj_action hidden → trajectory points.

    Used in stage-2 as an auxiliary trajectory prediction head that provides
    a direct regression loss on traj_action token features.
    """

    def __init__(self, hidden_dim: int, num_points: int = 8, point_dim: int = 3):
        super().__init__()
        mid = max(hidden_dim // 2, point_dim * num_points)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.GELU(),
            nn.Linear(mid, num_points * point_dim),
        )
        self.num_points = num_points
        self.point_dim = point_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: ``[B, H]`` — last traj_action token hidden state.
        Returns:
            ``[B, num_points, point_dim]`` predicted trajectory.
        """
        return self.net(x).view(x.shape[0], self.num_points, self.point_dim)


@MODEL_REGISTRY.register("WmVlmJoint")
class WmVlmJoint(nn.Module):
    """Joint VLM + optional World Model for stage-2 training.

    Config (under ``model``):
        name: WmVlmJoint
        vlm:
            model_path: "Qwen/Qwen3-VL-2B-Instruct"
            num_jepa_action_tokens: 0
            num_jepa_token_steps: 8
            num_vggt_action_tokens: 0
            num_vggt_token_steps: 8
            num_wan_action_tokens: 24
            num_wan_token_steps: 8       # 24 / 8 = 3 repetitions per step token
            num_traj_action_tokens: 0    # traj: single token repeated, no token_steps
        use_jepa: false   # JEPA action-token branch supervised by frozen V-JEPA2.1
        use_vggt: false   # VGGT action-token branch (stub; tokens before wan)
        use_wan: true     # Wan2.2 world-model branch
        use_traj: false   # traj_action tokens + MLPActionHead auxiliary loss
        eval_traj_mode: "text"     # "text" | "traj_token"  (only used when use_traj=true)
        # 二阶段是否走「文本轨迹」：训练 teacher-forcing CE + 验证/推理的 generate 自回归。
        # false 时不算 trajectory_ce，验证也不调 generate_text；predict_trajectory 仅允许 traj_token。
        use_vlm_autoreg_text_trajectory: true
        traj_pred_num_points: 8   # trajectory points for MLPActionHead
        # 学习率：trainer.learning_rate.action_head（缺省并入 learning_rate.base）

    wan / jepa / vggt 三者的校验规则完全对称（init 期执行）：
        use=true  + num_xxx_action_tokens=0  →  ValueError  （编码器已启用但无 token slot）
        use=false + num_xxx_action_tokens>0  →  UserWarning （token 占位但外部编码器未激活）
        use=true  + num_xxx_action_tokens>0  →  正常初始化
        use=false + num_xxx_action_tokens=0  →  正常禁用
    """

    def __init__(self, cfg, full_cfg=None, **kwargs):
        super().__init__()
        full = full_cfg if full_cfg is not None else cfg
        model_cfg = cfg_get(full, "model", cfg)
        weight_dtype = cfg_model_torch_dtype(model_cfg)

        vlm_cfg = cfg_get(model_cfg, "vlm", {})

        self.vlm = Qwen3VLWrapper(
            model_path=cfg_get(vlm_cfg, "model_path", "Qwen/Qwen3-VL-2B-Instruct"),
            num_jepa_action_tokens=int(cfg_get(vlm_cfg, "num_jepa_action_tokens", 0)),
            num_jepa_token_steps=int(cfg_get(vlm_cfg, "num_jepa_token_steps", 8)),
            num_vggt_action_tokens=int(cfg_get(vlm_cfg, "num_vggt_action_tokens", 0)),
            num_vggt_token_steps=int(cfg_get(vlm_cfg, "num_vggt_token_steps", 8)),
            num_vggt_cam_action_tokens=int(cfg_get(vlm_cfg, "num_vggt_cam_action_tokens", 0)),
            num_wan_action_tokens=int(cfg_get(vlm_cfg, "num_wan_action_tokens", 24)),
            num_wan_token_steps=int(cfg_get(vlm_cfg, "num_wan_token_steps", 8)),
            num_traj_action_tokens=int(cfg_get(vlm_cfg, "num_traj_action_tokens", 0)),
            torch_dtype=weight_dtype,
            gradient_checkpointing=bool(cfg_get(vlm_cfg, "gradient_checkpointing", True)),
        )
        self._post_action_user_hint = POST_ACTION_USER_HINT

        def _check_branch(use_flag: bool, num_tokens: int, name: str) -> None:
            """wan / jepa / vggt 分支的对称校验规则：
            - use=true  + num=0  → Error（编码器要激活但无 token slot）
            - use=false + num>0  → Warning（token 占位但外部编码器未激活）
            - use=true  + num>0  → 正常初始化
            - use=false + num=0  → 正常禁用
            """
            if use_flag and num_tokens == 0:
                raise ValueError(
                    f"use_{name}=true 要求 model.vlm.num_{name}_action_tokens > 0。"
                )
            if not use_flag and num_tokens > 0:
                warnings.warn(
                    f"[WmVlmJoint] model.vlm.num_{name}_action_tokens={num_tokens} > 0"
                    f" 但 use_{name}=false。{name} 外部编码器未激活，"
                    f"token slot 已在 VLM 序列中占位但不注入外部特征。",
                    UserWarning,
                    stacklevel=3,
                )

        # ── JEPA action-token branch (stub) ────────────────────────────────
        self._use_jepa = bool(cfg_get(model_cfg, "use_jepa", False))
        _check_branch(self._use_jepa, self.vlm.num_jepa_action_tokens, "jepa")
        if self._use_jepa:
            self._init_jepa(model_cfg)

        # ── VGGT action-token branch (stub) ────────────────────────────────
        self._use_vggt = bool(cfg_get(model_cfg, "use_vggt", False))
        _check_branch(self._use_vggt, self.vlm.num_vggt_action_tokens, "vggt")
        if self._use_vggt:
            self._init_vggt(model_cfg)

        # ── Wan world-model branch ──────────────────────────────────────────
        self._use_wan = bool(cfg_get(model_cfg, "use_wan", True))
        _check_branch(self._use_wan, self.vlm.num_wan_action_tokens, "wan")
        if self._use_wan:
            wm_cfg_src = cfg_get(model_cfg, "world_model", model_cfg)
            cond = cfg_get(wm_cfg_src, "condition", {})
            if str(cfg_get(cond, "name", "")) == "latent":
                wm_cfg_mut = OmegaConf.merge(OmegaConf.create(), wm_cfg_src)
                wm_cfg_mut.condition.input_dim = int(self.vlm.hidden_size)
                wm_sub = OmegaConf.to_container(wm_cfg_mut, resolve=True)
            else:
                wm_sub = (
                    OmegaConf.to_container(wm_cfg_src, resolve=True)
                    if OmegaConf.is_config(wm_cfg_src)
                    else (dict(wm_cfg_src) if isinstance(wm_cfg_src, dict) else {**wm_cfg_src})
                )
            if not isinstance(wm_sub, dict):
                wm_sub = dict(wm_sub)
            wm_sub = {**wm_sub, "dtype": str(cfg_get(model_cfg, "dtype", "bfloat16"))}

            wm_full = full
            if OmegaConf.is_config(wm_full):
                wm_full = OmegaConf.to_container(wm_full, resolve=True)
            if isinstance(wm_full, dict):
                wm_full = {**wm_full, "model": wm_sub}
            else:
                wm_full = {"model": wm_sub}
            self.world_model = WanWorldModel(wm_full)
        else:
            self.world_model = None  # type: ignore[assignment]

        # ── Traj action token branch / ActionHead ────────────────────────────
        self._traj_pred_num_points = int(cfg_get(model_cfg, "traj_pred_num_points", 8))
        self._use_traj = bool(cfg_get(model_cfg, "use_traj", False))
        if self._use_traj:
            if self.vlm.num_traj_action_tokens == 0:
                raise ValueError(
                    "use_traj=true requires num_traj_action_tokens > 0 in model.vlm."
                )
            ah_cfg = dict(cfg_get(model_cfg, "action_head", {}))
            ah_cfg["name"] = "mlp"
            ah_cfg["input_feature_dim"] = self.vlm.hidden_size
            ah_cfg.setdefault("num_points", self._traj_pred_num_points)
            ah_cfg.setdefault("point_dim", 3)
            self.action_head = ActionHead(ah_cfg)
        else:
            self.action_head = None  # type: ignore[assignment]

        # Validation prediction mode (stage-2 only).
        self._eval_traj_mode = str(cfg_get(model_cfg, "eval_traj_mode", "text"))
        if self._eval_traj_mode not in ("text", "traj_token"):
            raise ValueError(
                f"eval_traj_mode must be 'text' or 'traj_token', "
                f"got {self._eval_traj_mode!r}."
            )

        self._use_vlm_autoreg_text_trajectory = bool(
            cfg_get(model_cfg, "use_vlm_autoreg_text_trajectory", True)
        )

        self.val_gen_psnr  = MeanMetric(sync_on_compute=True, dist_sync_on_step=False)
        self.val_text_ade  = MeanMetric(sync_on_compute=True, dist_sync_on_step=False)
        self.val_text_fde  = MeanMetric(sync_on_compute=True, dist_sync_on_step=False)
        self._val_vis_done = False
        self._val_wm_num_inference_steps = int(
            cfg_get(model_cfg, "validation_num_inference_steps", 20)
        )

    # ── Optional branch initializers ───────────────────────────────────────

    def _init_jepa(self, model_cfg) -> None:
        """Load a frozen V-JEPA2.1 encoder and build the student projection head."""
        jepa_cfg = cfg_get(model_cfg, "jepa", {}) or {}

        ckpt_path = cfg_get(jepa_cfg, "teacher_ckpt", None)
        if ckpt_path is None or str(ckpt_path).strip() == "":
            raise ValueError("Set model.jepa.teacher_ckpt in the config.")
        ckpt_path = str(ckpt_path)
        checkpoint_key = str(cfg_get(jepa_cfg, "checkpoint_key", "encoder"))
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"V-JEPA2.1 checkpoint not found: {ckpt_path}")

        variant = str(cfg_get(jepa_cfg, "teacher_variant", "vitG"))
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

        input_hw = cfg_get(jepa_cfg, "teacher_input_hw", cfg_get(jepa_cfg, "input_hw", [384, 384]))
        if isinstance(input_hw, int):
            input_h = input_w = int(input_hw)
        else:
            input_h, input_w = [int(x) for x in list(input_hw)]

        teacher_input_mode = str(cfg_get(jepa_cfg, "teacher_input_mode", "video")).lower()
        if teacher_input_mode in ("frame", "frames", "image", "images"):
            teacher_input_mode = "image"
        if teacher_input_mode not in ("video", "image"):
            raise ValueError(
                "model.jepa.teacher_input_mode must be 'video' or 'image', "
                f"got {teacher_input_mode!r}."
            )
        self._jepa_teacher_input_mode = teacher_input_mode

        frame_selection_cfg = cfg_get(jepa_cfg, "frame_selection", None)
        include_current_cfg = cfg_get(jepa_cfg, "include_current_frame", None)
        if frame_selection_cfg is None:
            include_current = bool(cfg_get(jepa_cfg, "include_current_frame", False))
            frame_selection = "current_future" if include_current else "future"
        else:
            frame_selection = str(frame_selection_cfg).lower().replace("+", "_")
            frame_selection = frame_selection.replace("-", "_")
            aliases = {
                "future": "future",
                "future_only": "future",
                "current_future": "current_future",
                "current_and_future": "current_future",
                "cur_future": "current_future",
                "cur_fut": "current_future",
            }
            if frame_selection not in aliases:
                raise ValueError(
                    "model.jepa.frame_selection must be 'future' or 'current_future', "
                    f"got {frame_selection_cfg!r}."
                )
            frame_selection = aliases[frame_selection]
            include_current = frame_selection == "current_future"
            if include_current_cfg is not None and bool(include_current_cfg) != include_current:
                warnings.warn(
                    "[WmVlmJoint] model.jepa.frame_selection overrides "
                    "model.jepa.include_current_frame.",
                    UserWarning,
                    stacklevel=2,
                )
        self._jepa_frame_selection = frame_selection
        self._jepa_include_current_frame = include_current
        self._jepa_num_frames = int(cfg_get(jepa_cfg, "num_frames", 8))
        self._jepa_tubelet_size = int(cfg_get(jepa_cfg, "tubelet_size", 2))
        self._jepa_patch_size = int(cfg_get(jepa_cfg, "patch_size", 16))
        self._jepa_img_temporal_dim_size = int(
            cfg_get(jepa_cfg, "img_temporal_dim_size", 1)
        )
        if self._jepa_teacher_input_mode == "image" and self._jepa_img_temporal_dim_size != 1:
            raise ValueError(
                "model.jepa.teacher_input_mode='image' expects img_temporal_dim_size=1; "
                f"got {self._jepa_img_temporal_dim_size}."
            )
        self._jepa_encoder_num_frames = self._jepa_num_frames
        if self._jepa_encoder_num_frames < self._jepa_tubelet_size:
            raise ValueError(
                "The V-JEPA encoder temporal length must be >= tubelet_size; "
                f"got encoder_num_frames={self._jepa_encoder_num_frames} and "
                f"tubelet_size={self._jepa_tubelet_size}."
            )
        if (
            self._jepa_teacher_input_mode == "video"
            and self._jepa_encoder_num_frames % self._jepa_tubelet_size != 0
        ):
            warnings.warn(
                "[WmVlmJoint] V-JEPA video input has a temporal remainder: "
                f"num_frames={self._jepa_encoder_num_frames}, "
                f"tubelet_size={self._jepa_tubelet_size}. "
                "PatchEmbed3D uses Conv3d stride=tubelet_size, so the final "
                "unpaired frame is not represented in dense video tokens.",
                UserWarning,
                stacklevel=2,
            )
        if input_h % self._jepa_patch_size != 0 or input_w % self._jepa_patch_size != 0:
            raise ValueError(
                "model.jepa.teacher_input_hw must be divisible by patch_size; "
                f"got {(input_h, input_w)} and {self._jepa_patch_size}."
            )
        self._jepa_input_hw = (input_h, input_w)
        self.register_buffer(
            "_jepa_image_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_jepa_image_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1, 1),
            persistent=False,
        )
        self._jepa_encoder_temporal_blocks = (
            self._jepa_encoder_num_frames // self._jepa_tubelet_size
        )
        self._jepa_temporal_blocks = (
            self._jepa_num_frames // self._jepa_tubelet_size
            if self._jepa_teacher_input_mode == "video"
            else self._jepa_num_frames
        )
        self._jepa_grid_hw = (
            input_h // self._jepa_patch_size,
            input_w // self._jepa_patch_size,
        )

        feature_type = str(cfg_get(jepa_cfg, "feature_type", "final")).lower()
        if feature_type != "final":
            raise ValueError(
                "model.jepa.feature_type must be 'final'. Hierarchical JEPA "
                "features concatenate multiple layers and are intentionally disabled "
                "because they make the distillation target too wide; "
                f"got {feature_type!r}."
            )
        self._jepa_feature_type = feature_type
        n_output_distillation = 1

        vit_encoder = _import_vjepa21_vision_transformer()
        if not hasattr(vit_encoder, arch_name):
            raise ValueError(f"Unknown V-JEPA2.1 encoder variant: {variant!r}")

        self.jepa_encoder = getattr(vit_encoder, arch_name)(
            img_size=self._jepa_input_hw,
            patch_size=self._jepa_patch_size,
            num_frames=self._jepa_encoder_num_frames,
            tubelet_size=self._jepa_tubelet_size,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=True,
            img_temporal_dim_size=self._jepa_img_temporal_dim_size,
            interpolate_rope=True,
            n_output_distillation=n_output_distillation,
        )

        raw = _load_vjepa_checkpoint(ckpt_path)
        if not isinstance(raw, dict) or checkpoint_key not in raw:
            keys = sorted(raw.keys()) if isinstance(raw, dict) else type(raw).__name__
            raise KeyError(
                f"V-JEPA2.1 checkpoint {ckpt_path} has no key {checkpoint_key!r}; "
                f"available keys: {keys}"
            )
        state = _clean_vjepa_state_dict(raw[checkpoint_key])
        strict = bool(cfg_get(jepa_cfg, "strict_load", True))
        missing, unexpected = self.jepa_encoder.load_state_dict(state, strict=strict)
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"Strict V-JEPA load failed: missing={missing}, unexpected={unexpected}"
            )
        if missing or unexpected:
            warnings.warn(
                f"[WmVlmJoint] V-JEPA load with missing={missing}, unexpected={unexpected}",
                UserWarning,
                stacklevel=2,
            )
        del raw, state

        self.jepa_encoder.requires_grad_(False)
        teacher_dtype = cfg_model_torch_dtype(
            {"dtype": cfg_get(jepa_cfg, "teacher_dtype", cfg_get(model_cfg, "dtype", "bfloat16"))}
        )
        self.jepa_encoder.to(dtype=teacher_dtype)
        self.jepa_encoder.eval()

        self._jepa_pool_shape = self._resolve_jepa_pool_shape(
            cfg_get(jepa_cfg, "teacher_pool_shape", None),
            self.vlm.num_jepa_action_tokens,
        )
        self._jepa_smooth_l1_weight = float(cfg_get(jepa_cfg, "smooth_l1_weight", 1.0))
        self._jepa_cosine_weight = float(cfg_get(jepa_cfg, "cosine_weight", 0.5))
        self._jepa_smooth_l1_beta = float(cfg_get(jepa_cfg, "smooth_l1_beta", 1.0))

        target_dim = int(self.jepa_encoder.embed_dim)
        self._jepa_target_dim = target_dim
        head_hidden = int(cfg_get(jepa_cfg, "head_hidden_dim", self.vlm.hidden_size))
        self.jepa_token_norm = nn.LayerNorm(self.vlm.hidden_size)
        self.jepa_head = nn.Sequential(
            nn.Linear(self.vlm.hidden_size, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, target_dim),
        )

    def _init_vggt(self, model_cfg) -> None:
        """Stub: VGGT action-token encoder (tokens between jepa and wan in VLM sequence).

        When implemented: load frozen VGGT encoder, register token strings,
        inject features into VLM sequence at vggt positions.
        """
        vggt_cfg = cfg_get(model_cfg, "vggt_model", model_cfg)
        adapter_dim = vggt_cfg["vggt_adapter_dim"]
        self._use_visual_token = vggt_cfg.get("use_visual_token", False)
        if self._use_visual_token:
            self._output_size_flatten = 32
            
            num_steps = self.vlm.num_vggt_token_steps
            self._use_multi_step_vggt = vggt_cfg.get("use_multi_step", num_steps > 1)
            
            if self._use_multi_step_vggt:
                # 考虑多步 future
                tokens_per_step = self.vlm.num_vggt_action_tokens // num_steps if num_steps > 0 else self.vlm.num_vggt_action_tokens
                self._mix_visual_token_in_multistep = vggt_cfg.get("mix_visual_token_in_multistep", True)
                self._multi_step_loss_decay = vggt_cfg.get("multi_step_loss_decay", 1.0)
                self.vggt_adapter = VggtMultiStepAdapter(
                    hidden_dim=self.vlm.hidden_size,
                    adapter_dim=adapter_dim,
                    num_vggt_tokens=tokens_per_step,
                    num_steps=num_steps,
                    v=self._output_size_flatten,
                    use_visual_token=self._mix_visual_token_in_multistep,
                )
                cam_tokens_per_step = self.vlm.num_vggt_cam_action_tokens // num_steps if getattr(self.vlm, "num_vggt_cam_action_tokens", 0) > 0 else tokens_per_step
                self.vggt_camera_adapter = VggtCameraMultiStepAdapter(
                    hidden_dim=self.vlm.hidden_size,
                    adapter_dim=adapter_dim,
                    num_vggt_tokens=cam_tokens_per_step,
                    num_steps=num_steps,
                )
            else:
                self.vggt_adapter = VggtAdapter(
                    hidden_dim=self.vlm.hidden_size,
                    adapter_dim=adapter_dim,
                    num_vggt_tokens=self.vlm.num_vggt_action_tokens,
                    v=self._output_size_flatten,
                )
                num_cam_tokens = self.vlm.num_vggt_cam_action_tokens if getattr(self.vlm, "num_vggt_cam_action_tokens", 0) > 0 else self.vlm.num_vggt_action_tokens
                self.vggt_camera_adapter = VggtCameraAdapter(
                    hidden_dim=self.vlm.hidden_size,
                    adapter_dim=adapter_dim,
                    num_vggt_tokens=num_cam_tokens,
                )
            self._output_size = (4, 8)
        else:
            self.vggt_adapter = nn.Sequential(
                nn.Linear(self.vlm.hidden_size, adapter_dim),
                nn.SiLU(),
                nn.Linear(adapter_dim, adapter_dim)
            )
            self._output_size = (3, 4)
            self._output_size_flatten = 12
        vggt_path = vggt_cfg["vggt_model_path"]
        print(f"[WmVlmJoint] Loading VGGT teacher from {vggt_path}")
        ensure_vggt_import_path()
        from vggt.models.vggt import VGGT
        self.vggt_teacher = VGGT.from_pretrained(vggt_path)
        self.vggt_teacher.eval()
        self.vggt_teacher.requires_grad_(False)
        self._use_cur = vggt_cfg["use_cur"]
        self._use_fut = vggt_cfg["use_fut"]
        
    # ── Module overrides ───────────────────────────────────────────────────

    def train(self, mode: bool = True):
        super().train(mode)
        if self.world_model is not None:
            self.world_model.train(mode)
        if getattr(self, "jepa_encoder", None) is not None:
            self.jepa_encoder.eval()
        if getattr(self, "vggt_teacher", None) is not None:
            self.vggt_teacher.eval()
        return self

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _factor_spatial_slots(num_slots: int) -> tuple[int, int]:
        if num_slots <= 0:
            raise ValueError(f"num_slots must be positive, got {num_slots}")
        h = int(math.sqrt(num_slots))
        while h > 1 and num_slots % h != 0:
            h -= 1
        return h, num_slots // h

    def _resolve_jepa_pool_shape(
        self,
        pool_shape_cfg: Any,
        num_jepa_tokens: int,
    ) -> tuple[int, int, int]:
        if num_jepa_tokens <= 0:
            raise ValueError("JEPA branch requires num_jepa_action_tokens > 0.")

        if pool_shape_cfg is not None:
            shape = tuple(int(x) for x in list(pool_shape_cfg))
            if len(shape) != 3:
                raise ValueError(
                    "model.jepa.teacher_pool_shape must be [temporal, height, width], "
                    f"got {shape}."
                )
            if shape[0] * shape[1] * shape[2] != num_jepa_tokens:
                raise ValueError(
                    "model.jepa.teacher_pool_shape product must equal "
                    f"num_jepa_action_tokens={num_jepa_tokens}, got {shape}."
                )
            if shape[0] != self._jepa_temporal_blocks:
                raise ValueError(
                    "model.jepa.teacher_pool_shape[0] must equal the number of "
                    "JEPA teacher time units because pooling is now 2D per "
                    "image/time-block, not 3D across time. "
                    f"got shape[0]={shape[0]}, expected {self._jepa_temporal_blocks}."
                )
            return shape

        temporal_bins = self._jepa_temporal_blocks
        if num_jepa_tokens % temporal_bins != 0:
            raise ValueError(
                "model.vlm.num_jepa_action_tokens must be divisible by the number "
                "of JEPA teacher time units for per-frame/per-block pooling; "
                f"got {num_jepa_tokens} tokens and {temporal_bins} time units."
            )
        spatial_slots = num_jepa_tokens // temporal_bins
        h_bins, w_bins = self._factor_spatial_slots(spatial_slots)
        return temporal_bins, h_bins, w_bins

    def _select_jepa_teacher_frames(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """Select/pad teacher frames according to ``model.jepa.frame_selection``."""
        future = inputs.get("future_images")
        if not isinstance(future, torch.Tensor):
            raise ValueError("use_jepa=true requires inputs['future_images'] as a tensor.")
        if future.dim() != 5 or future.shape[2] != 3:
            raise ValueError(
                "future_images must have shape [B, T, 3, H, W], "
                f"got {tuple(future.shape)}."
            )
        if future.shape[1] <= 0:
            raise ValueError("future_images must contain at least one future frame.")

        frames = future
        if self._jepa_include_current_frame:
            history = inputs.get("history_images")
            if not isinstance(history, torch.Tensor):
                raise ValueError(
                    "model.jepa.include_current_frame=true requires "
                    "inputs['history_images'] as a tensor."
                )
            if history.dim() != 5 or history.shape[2] != 3:
                raise ValueError(
                    "history_images must have shape [B, T, 3, H, W], "
                    f"got {tuple(history.shape)}."
                )
            if history.shape[0] != future.shape[0]:
                raise ValueError(
                    "history_images and future_images batch size mismatch: "
                    f"{history.shape[0]} vs {future.shape[0]}."
                )
            frames = torch.cat([history[:, -1:], future], dim=1)

        num_frames = self._jepa_num_frames
        if frames.shape[1] < num_frames:
            pad = frames[:, -1:].expand(-1, num_frames - frames.shape[1], -1, -1, -1)
            frames = torch.cat([frames, pad], dim=1)
        elif frames.shape[1] > num_frames:
            frames = frames[:, :num_frames]
        return frames

    def _build_jepa_teacher_clip(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """V-JEPA video clip: [B,T,3,H,W] in [-1,1] -> [B,3,T,384,384]."""
        future = self._select_jepa_teacher_frames(inputs)

        teacher_param = next(self.jepa_encoder.parameters())
        bsz, timesteps, channels, _, _ = future.shape
        x = future.to(device=teacher_param.device, dtype=torch.float32)
        x = x.mul(0.5).add(0.5).clamp_(0.0, 1.0)
        x = x.reshape(bsz * timesteps, channels, x.shape[-2], x.shape[-1])
        x = F.interpolate(
            x,
            size=self._jepa_input_hw,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        x = x.view(bsz, timesteps, channels, self._jepa_input_hw[0], self._jepa_input_hw[1])
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = (x - self._jepa_image_mean.to(x.device)) / self._jepa_image_std.to(x.device)
        return x.to(dtype=teacher_param.dtype)

    def _build_jepa_teacher_image_clips(
        self,
        inputs: Dict[str, Any],
    ) -> tuple[torch.Tensor, int, int]:
        """Selected frames as independent static V-JEPA image clips.

        The V-JEPA2.1 checkpoint includes an image branch
        (``img_temporal_dim_size=1``): [B,T,3,H,W] -> [B*T,3,1,384,384].
        This triggers ``patch_embed_img`` and ``img_mod_embed`` instead of the
        video tubelet patch embed.
        """
        future = self._select_jepa_teacher_frames(inputs)

        teacher_param = next(self.jepa_encoder.parameters())
        bsz, timesteps, channels, _, _ = future.shape
        x = future.to(device=teacher_param.device, dtype=torch.float32)
        x = x.mul(0.5).add(0.5).clamp_(0.0, 1.0)
        x = x.reshape(bsz * timesteps, channels, x.shape[-2], x.shape[-1])
        x = F.interpolate(
            x,
            size=self._jepa_input_hw,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        x = x.view(bsz, timesteps, channels, self._jepa_input_hw[0], self._jepa_input_hw[1])
        x = x.reshape(
            bsz * timesteps,
            channels,
            self._jepa_img_temporal_dim_size,
            self._jepa_input_hw[0],
            self._jepa_input_hw[1],
        ).contiguous()
        x = (x - self._jepa_image_mean.to(x.device)) / self._jepa_image_std.to(x.device)
        return x.to(dtype=teacher_param.dtype), bsz, timesteps

    def _pool_jepa_teacher_tokens(self, dense_tokens: torch.Tensor) -> torch.Tensor:
        bsz, num_tokens, dim = dense_tokens.shape
        t_blocks = self._jepa_temporal_blocks
        pool_t, pool_h, pool_w = self._jepa_pool_shape
        if pool_t != t_blocks:
            raise ValueError(
                "Video JEPA pooling keeps each temporal block separate; "
                f"teacher_pool_shape[0]={pool_t} must equal {t_blocks}."
            )
        h_patches, w_patches = self._jepa_grid_hw
        expected_tokens = t_blocks * h_patches * w_patches
        if num_tokens != expected_tokens:
            raise ValueError(
                "Unexpected V-JEPA token count: "
                f"got {num_tokens}, expected {expected_tokens} "
                f"({t_blocks} temporal blocks x {h_patches} x {w_patches})."
            )

        grid = dense_tokens.view(bsz, t_blocks, h_patches, w_patches, dim)
        grid = grid.permute(0, 1, 4, 2, 3).contiguous()
        grid = grid.view(bsz * t_blocks, dim, h_patches, w_patches)
        pooled = F.adaptive_avg_pool2d(grid.float(), (pool_h, pool_w))
        pooled = pooled.view(bsz, t_blocks, dim, pool_h, pool_w)
        pooled = pooled.permute(0, 1, 3, 4, 2).contiguous()
        return pooled.view(bsz, -1, dim)

    def _pool_jepa_teacher_image_tokens(
        self,
        dense_tokens: torch.Tensor,
        bsz: int,
        timesteps: int,
    ) -> torch.Tensor:
        if dense_tokens.shape[0] != bsz * timesteps:
            raise ValueError(
                "Unexpected image-mode V-JEPA batch size: "
                f"got {dense_tokens.shape[0]}, expected {bsz * timesteps}."
            )

        _, num_tokens, dim = dense_tokens.shape
        pool_t, pool_h, pool_w = self._jepa_pool_shape
        if pool_t != timesteps:
            raise ValueError(
                "Image JEPA pooling keeps each image separate; "
                f"teacher_pool_shape[0]={pool_t} must equal selected frames={timesteps}."
            )
        h_patches, w_patches = self._jepa_grid_hw
        expected_tokens = self._jepa_img_temporal_dim_size * h_patches * w_patches
        if num_tokens != expected_tokens:
            raise ValueError(
                "Unexpected image-mode V-JEPA token count: "
                f"got {num_tokens}, expected {expected_tokens} "
                f"({self._jepa_img_temporal_dim_size} image temporal blocks x "
                f"{h_patches} x {w_patches})."
            )

        grid = dense_tokens.view(
            bsz * timesteps,
            self._jepa_img_temporal_dim_size,
            h_patches,
            w_patches,
            dim,
        )
        if self._jepa_img_temporal_dim_size != 1:
            grid = grid.mean(dim=1)
        else:
            grid = grid.squeeze(1)
        grid = grid.permute(0, 3, 1, 2).contiguous()
        pooled = F.adaptive_avg_pool2d(grid.float(), (pool_h, pool_w))
        pooled = pooled.view(bsz, timesteps, dim, pool_h, pool_w)
        pooled = pooled.permute(0, 1, 3, 4, 2).contiguous()
        return pooled.view(bsz, -1, dim)

    def _forward_wm(
        self, inputs: Dict[str, Any], wan_action_tokens: torch.Tensor
    ) -> Dict[str, Any]:
        """Run Wan world model conditioned on ``wan_action_tokens``."""
        assert self.world_model is not None
        wm_dev = next(self.world_model.transformer.parameters()).device
        wm_inputs: Dict[str, Any] = {
            "history_images": inputs["history_images"].to(wm_dev),
            "future_images":  inputs["future_images"].to(wm_dev),
            "condition_latent": wan_action_tokens.to(wm_dev),
        }
        for key in ("future_trajectory", "trajectory_interval_s"):
            if key in inputs and isinstance(inputs[key], torch.Tensor):
                wm_inputs[key] = inputs[key].to(wm_dev)
        return self.world_model(wm_inputs)

    def _forward_jepa(
        self, inputs: Dict[str, Any], jepa_action_tokens: torch.Tensor
    ) -> Dict[str, Any]:
        """Distill future-frame V-JEPA2.1 tokens into VLM JEPA action-token slots."""
        head_param = next(self.jepa_head.parameters())
        student = jepa_action_tokens.to(device=head_param.device, dtype=head_param.dtype)
        if student.shape[1] != self.vlm.num_jepa_action_tokens:
            raise ValueError(
                "jepa_action_tokens count mismatch: "
                f"got {student.shape[1]}, expected {self.vlm.num_jepa_action_tokens}."
            )

        with torch.no_grad():
            if self._jepa_teacher_input_mode == "image":
                clip, bsz, timesteps = self._build_jepa_teacher_image_clips(inputs)
                dense = self.jepa_encoder(
                    clip,
                    training=False,
                )
                teacher = self._pool_jepa_teacher_image_tokens(dense, bsz, timesteps).detach()
            else:
                clip = self._build_jepa_teacher_clip(inputs)
                dense = self.jepa_encoder(
                    clip,
                    training=False,
                )
                teacher = self._pool_jepa_teacher_tokens(dense).detach()
        if teacher.shape[1] != student.shape[1]:
            raise ValueError(
                "JEPA teacher token count mismatch: "
                f"teacher={teacher.shape[1]}, student={student.shape[1]}. "
                "Check model.vlm.num_jepa_action_tokens and model.jepa.teacher_pool_shape."
            )

        pred = self.jepa_head(self.jepa_token_norm(student))
        teacher = teacher.to(device=pred.device, dtype=torch.float32)
        pred_f = pred.float()

        l1 = F.smooth_l1_loss(pred_f, teacher, beta=self._jepa_smooth_l1_beta)
        cosine = (1.0 - F.cosine_similarity(pred_f, teacher, dim=-1)).mean()
        loss = self._jepa_smooth_l1_weight * l1 + self._jepa_cosine_weight * cosine
        return {
            "loss": {"jepa_loss": loss},
            "other_log": {
                "jepa_l1": l1.detach(),
                "jepa_cos": cosine.detach(),
                "jepa_teacher_norm": teacher.norm(dim=-1).mean().detach(),
                "jepa_pred_norm": pred_f.norm(dim=-1).mean().detach(),
            },
        }

    def _forward_vggt(
        self, inputs: Dict[str, Any], vggt_action_tokens: torch.Tensor, image_features: torch.Tensor = None, vggt_cam_action_tokens: torch.Tensor = None
    ) -> Dict[str, Any]:
        """Compute VGGT branch distillation loss from ``vggt_action_tokens``."""
        loss_part: Dict[str, torch.Tensor] = {}
        other_log: Dict[str, torch.Tensor] = {}

        if self._use_visual_token:
            vggt_preds = self.vggt_adapter(vggt_action_tokens, image_features)  # [B, 12, adapter_dim]
        else:
            vggt_preds = self.vggt_adapter(vggt_action_tokens)

        if hasattr(self, "vggt_camera_adapter") and vggt_cam_action_tokens is not None:
            cam_preds = self.vggt_camera_adapter(vggt_cam_action_tokens)
        else:
            cam_preds = None
        
        target = None
        target_cam = None
        if getattr(self, "vggt_teacher", None) is not None:
            with torch.no_grad():
                # VGGT expects images in [B, S, 3, H, W]
                # history_images: [B, S, C, H, W]
                if self._use_cur and not self._use_fut:
                    images_tensor = inputs["history_images"]
                elif not self._use_cur and self._use_fut:
                    images_tensor = inputs["future_images"]
                elif self._use_cur and self._use_fut:
                    images_tensor = torch.cat([inputs["history_images"], inputs["future_images"]], dim=1)
                B, S, C, H, W = images_tensor.shape
                images_tensor = images_tensor.reshape(B * S, C, H, W)
                images_tensor = F.interpolate(
                    images_tensor,
                    size=(504, 1008),  # 需要能被14整除
                    mode="bilinear",
                    align_corners=False,
                )
                device = next(self.vggt_teacher.parameters()).device
                dtype = next(self.vggt_teacher.parameters()).dtype
                images_tensor = images_tensor.to(device=device, dtype=dtype)

                images_tensor = images_tensor.reshape(B, S, C, 504, 1008)
                # Get aggregated tokens directly from aggregator
                aggregated_tokens_list, _ = self.vggt_teacher.aggregator(images_tensor)
                target = aggregated_tokens_list[-1]
                if hasattr(self, "vggt_camera_adapter"):
                    target_cam = target[:, :, 0, :] # [B, S, D]
                    
                target = target[:, :, 5:, :] # 只取patch_token [:,:,36*72,2048]
                target = target.reshape(B*S, 36, 72, -1).permute(0, 3, 1, 2) # [:,2048,36,72]
                target = F.adaptive_avg_pool2d(target, output_size=self._output_size) # [B*S, 2048, 4, 8]
                target = target.flatten(2).transpose(1, 2) # [B*S, 32, 2048]
                # target=[B*S, 32, 2048]
                target = target.view(B, S, self._output_size_flatten, -1)
                
                if getattr(self, "_use_multi_step_vggt", False) and hasattr(self.vggt_adapter, 'num_steps'):
                    # 截取未来帧。由于包含当前帧 + num_future_frames，假设 adapter 预测未来 T 步
                    num_steps = self.vggt_adapter.num_steps
                    if target.shape[1] >= num_steps:
                        target = target[:, -num_steps:]  # 取最后 T 步 [B, T, 32, 2048]
                    else:
                        target = target[:, -1:] # 退化处理 [B, 1, 32, 2048]
                else:
                    # 单帧或非 multi_step 模式，取出倒数第一帧 [B, 32, 2048]
                    target = target[:, -1]

        elif "vggt_teacher_target" in inputs:
            # 兼容离线 Dataloader 返回模式
            target = inputs["vggt_teacher_target"]
            target_cam = inputs.get("vggt_teacher_target_cam")

        if target is not None:
            target = target.to(vggt_preds.device)
            if getattr(self, "_use_multi_step_vggt", False) and getattr(self, "_multi_step_loss_decay", 1.0) != 1.0 and vggt_preds.dim() == 4:
                T_steps = vggt_preds.shape[1]
                if target_cam is not None:
                    target_cam = target_cam[:, -T_steps:]

                decay = self._multi_step_loss_decay
                weights = torch.tensor([decay ** i for i in range(T_steps)], device=vggt_preds.device, dtype=vggt_preds.dtype)
                weights_geo = weights.view(1, T_steps, 1, 1)

                mse_unreduced = F.mse_loss(vggt_preds, target, reduction='none')
                loss_part["vggt_distill_loss"] = (mse_unreduced * weights_geo).mean()
                
                if cam_preds is not None and target_cam is not None:
                    target_cam = target_cam.to(cam_preds.device)
                    cam_mse_unreduced = F.mse_loss(cam_preds, target_cam, reduction='none')
                    weights_cam = weights.view(1, T_steps, 1)
                    loss_part["vggt_cam_distill_loss"] = (cam_mse_unreduced * weights_cam).mean()
            else:
                if target_cam is not None:
                    if target_cam.dim() == 3:
                        target_cam = target_cam[:, -1]
                    target_cam = target_cam.view(vggt_preds.shape[0], -1)

                loss_part["vggt_distill_loss"] = F.mse_loss(vggt_preds, target)
                
                if cam_preds is not None and target_cam is not None:
                    target_cam = target_cam.to(cam_preds.device)
                    loss_part["vggt_cam_distill_loss"] = F.mse_loss(cam_preds, target_cam)

            other_log["vggt_distill_loss"] = loss_part["vggt_distill_loss"].detach()
            if "vggt_cam_distill_loss" in loss_part:
                other_log["vggt_cam_distill_loss"] = loss_part["vggt_cam_distill_loss"].detach()
            if "vggt_cam_distill_loss" in loss_part:
                other_log["vggt_cam_distill_loss"] = loss_part["vggt_cam_distill_loss"].detach()

        return {"loss": loss_part, "other_log": other_log}

    def _extract_current_images(self, inputs: Dict) -> List[PILImage.Image]:
        hist = inputs["history_images"]
        B = hist.shape[0]
        return [_tensor_to_pil(hist[b, -1]) for b in range(B)]

    def _build_prompts(self, inputs: Dict) -> List[str]:
        B = inputs["history_images"].shape[0]
        ht = inputs.get("history_trajectory")
        nav = inputs.get("navigation_command")
        es  = inputs.get("ego_status")
        dt_val = inputs.get("trajectory_interval_s")

        prompts = []
        for b in range(B):
            h = ht[b].detach().cpu().numpy() if ht is not None else np.zeros((1, 3))
            n = int(nav[b]) if nav is not None else 1
            if es is not None:
                row = es[b].detach().cpu().numpy()
                if row.size < 8:
                    raise ValueError(
                        f"ego_status must have >=8 dims; got shape {tuple(row.shape)}"
                    )
                vel = row[4:6].astype(np.float32)
                acc = row[6:8].astype(np.float32)
            else:
                vel = np.zeros(2, dtype=np.float32)
                acc = np.zeros(2, dtype=np.float32)
            dt = float(dt_val[b]) if isinstance(dt_val, torch.Tensor) and dt_val.dim() > 0 else (
                float(dt_val) if dt_val is not None else 0.5
            )
            prompts.append(
                build_driving_prompt(h, n, vel, acc, interval_s=dt,
                                     post_action_user_hint=self._post_action_user_hint)
            )
        return prompts

    def _build_answers(self, inputs: Dict) -> Optional[List[str]]:
        ft = inputs.get("future_trajectory")
        if ft is None:
            return None
        return [
            build_trajectory_answer(ft[b].detach().cpu().numpy())
            for b in range(ft.shape[0])
        ]

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Stage-2 joint forward.

        Computes losses for enabled branches:
        - ``wan_flow_matching``: Wan video generation (use_wan).
        - ``traj_pred_loss``: traj MLP regression on last traj token (use_traj).
        - ``trajectory_ce``: VLM text CE on assistant trajectory text (when
          ``use_vlm_autoreg_text_trajectory`` is true).
        """
        images = self._extract_current_images(inputs)
        user_prompts = self._build_prompts(inputs)
        assistant_answers = (
            self._build_answers(inputs) if self._use_vlm_autoreg_text_trajectory else None
        )

        if (
            not self._use_vlm_autoreg_text_trajectory
            and not any((
                self._use_wan and self.vlm.num_wan_action_tokens > 0,
                self._use_jepa and self.vlm.num_jepa_action_tokens > 0,
                self._use_vggt and self.vlm.num_vggt_action_tokens > 0,
                self._use_traj and self.vlm.num_traj_action_tokens > 0,
            ))
        ):
            raise ValueError(
                "WmVlmJoint.forward: 各 use_* 与 num_*>0 均未匹配到可训动作槽，且 "
                "use_vlm_autoreg_text_trajectory=False，无训练目标；请至少启用一个分支，或开 CE。"
            )

        _modes = list(VLM_FIXED_SIZE_MODES_ORDER)
        if self._use_vggt and self._use_visual_token:
            _modes.append("visual_tokens")
        if self._use_vggt and self.vlm.num_vggt_cam_action_tokens > 0:
            _modes.append("vggt_cam_action_tokens")
        vlm_out = self.vlm.forward_extract(
            images=images,
            user_prompts=user_prompts,
            assistant_answers=assistant_answers,
            compute_ce_loss=self._use_vlm_autoreg_text_trajectory,
            vlm_conditioning=_modes,
        )

        loss_part: Dict[str, torch.Tensor] = {}
        other_log: Dict[str, torch.Tensor] = {}

        if self._use_wan:
            wan_action_tokens = vlm_out.get("wan_action_tokens")
            if wan_action_tokens is None or wan_action_tokens.shape[1] == 0:
                raise ValueError(
                    "use_wan=true 但 vlm 输出中缺少或空的 wan_action_tokens；"
                    "检查 vlm_conditioning / num_wan_action_tokens。"
                )
            wm_out = self._forward_wm(inputs, wan_action_tokens)
            loss_part.update(wm_out["loss"])
            other_log.update(wm_out.get("other_log") or {})

        if self._use_jepa:
            jepa_tok = vlm_out.get("jepa_action_tokens")
            if jepa_tok is None or jepa_tok.shape[1] == 0:
                raise ValueError(
                    "use_jepa=true 但 vlm 输出中缺少或空的 jepa_action_tokens；"
                    "检查 vlm_conditioning / num_jepa_action_tokens。"
                )
            jepa_out = self._forward_jepa(inputs, jepa_tok)
            loss_part.update(jepa_out.get("loss", {}))
            other_log.update(jepa_out.get("other_log") or {})
        if self._use_vggt:
            vggt_tok = vlm_out.get("vggt_action_tokens")
            vggt_cam_tok = vlm_out.get("vggt_cam_action_tokens", None)
            visual_tokens = vlm_out.get("visual_tokens", None)
            if self._use_visual_token and visual_tokens is None:
                raise ValueError(
                    "缺少visual tokens，需要检查"
                )
            if vggt_tok is None or vggt_tok.shape[1] == 0:
                raise ValueError(
                    "use_vggt=true 但 vlm 输出中缺少或空的 vggt_action_tokens；"
                    "检查 vlm_conditioning / num_vggt_action_tokens。"
                )
            vggt_out = self._forward_vggt(inputs, vggt_tok, visual_tokens, vggt_cam_tok)
            loss_part.update(vggt_out.get("loss", {}))
            other_log.update(vggt_out.get("other_log") or {})

        if self._use_traj and self.vlm.num_traj_action_tokens > 0:
            traj_action_tokens = vlm_out.get("traj_action_tokens")  # [B, K, H]
            if traj_action_tokens is None or traj_action_tokens.shape[1] == 0:
                raise ValueError(
                    "use_traj=true 但 vlm 输出中缺少或空的 traj_action_tokens；"
                    "检查 vlm_conditioning / num_traj_action_tokens。"
                )
            assert self.action_head is not None
            mlp_p = next(self.action_head.parameters())
            device, md = mlp_p.device, mlp_p.dtype
            traj_last = traj_action_tokens[:, -1, :].to(device=device, dtype=md)
            traj_pred = self.action_head({"mlp_input": traj_last})  # [B, P, 3]
            gt_traj = inputs["future_trajectory"].to(device=device, dtype=traj_pred.dtype)
            loss_part["traj_pred_loss"] = F.mse_loss(traj_pred, gt_traj)

        if vlm_out.get("ce_loss") is not None:
            loss_part["trajectory_ce"] = vlm_out["ce_loss"]

        return {"loss": loss_part, "other_log": other_log}

    # ── Validation ─────────────────────────────────────────────────────────

    def reset_validation_metrics(self) -> None:
        self.val_gen_psnr.reset()
        self.val_text_ade.reset()
        self.val_text_fde.reset()
        self._val_vis_done = False

    @torch.no_grad()
    def validation_step(
        self,
        batch: dict,
        batch_idx: int,
        *,
        accelerator,
        vis_save_dir: str | None = None,
        global_step: int = 0,
        vis_num_samples: int = 3,
        vis_every_batch: bool = False,
    ) -> None:
        bs = batch["history_images"].shape[0]
        device = next(self.vlm.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        images = self._extract_current_images(inputs)
        prompts = self._build_prompts(inputs)

        vlm_out: Optional[Dict[str, Any]] = None

        use_traj_pred = (self._eval_traj_mode == "traj_token"
                         and self._use_traj
                         and self.vlm.num_traj_action_tokens > 0)

        def _validation_modes() -> list[str]:
            modes = list(VLM_FIXED_SIZE_MODES_ORDER)
            if getattr(self, "_use_vggt", False):
                modes.append("visual_tokens")
                if self.vlm.num_vggt_cam_action_tokens > 0:
                    modes.append("vggt_cam_action_tokens")
            return modes

        if use_traj_pred:
            # Traj-token path: one VLM forward, extract wan + traj (and jepa/vggt if enabled).
            _modes = _validation_modes()
            vlm_out = self.vlm.forward_extract(
                images, prompts, assistant_answers=None,
                compute_ce_loss=False,
                vlm_conditioning=_modes,
            )
        elif self._use_vlm_autoreg_text_trajectory:
            _modes = _validation_modes()
            vlm_out = self.vlm.generate_text(
                images, prompts, max_new_tokens=_TRAJ_TEXT_MAX_NEW,
                vlm_conditioning=_modes,
            )
        else:
            _modes = _validation_modes()
            vlm_out = self.vlm.forward_extract(
                images, prompts, assistant_answers=None,
                vlm_conditioning=_modes
            )

        if getattr(self, "_use_vggt", False) and "vggt_action_tokens" in vlm_out:
            vggt_tok = vlm_out["vggt_action_tokens"]
            vggt_cam_tok = vlm_out.get("vggt_cam_action_tokens", None)
            visual_tokens = vlm_out.get("visual_tokens", None)
            if self._use_visual_token and visual_tokens is None:
                # Fallback to avoid breaking forward_vggt if variable size matching failed
                visual_tokens = torch.zeros(vggt_tok.shape[0], 1008, vggt_tok.shape[2], device=vggt_tok.device, dtype=vggt_tok.dtype)
            
            # Reuse exactly the same forward function to avoid bugs with Target generation & Shape matching
            vggt_out = self._forward_vggt(inputs, vggt_tok, visual_tokens, vggt_cam_tok)
            if "vggt_distill_loss" in vggt_out["loss"]:
                if not hasattr(self, "val_vggt_mse"):
                    self.val_vggt_mse = MeanMetric(sync_on_compute=True, dist_sync_on_step=False).to(device)
                self.val_vggt_mse.update(vggt_out["loss"]["vggt_distill_loss"])

        if self._use_world_model:
            assert self.world_model is not None
            gen_input = {"history": inputs["history_images"], "condition_latent": wan_action_tokens}
            for key in ("future_trajectory", "trajectory_interval_s", "history_trajectory"):
                if key in inputs:
                    gen_input[key] = inputs[key]
            pred_frames = self.world_model.generate(
                gen_input, num_inference_steps=self._val_wm_num_inference_steps
            )
            gt_frames = inputs["future_images"]
            dev_wm = pred_frames.device
            for b in range(bs):
                p = psnr(pred_frames[b], gt_frames[b], data_range=2.0)
                self.val_gen_psnr.update(p.detach().to(device=dev_wm, dtype=torch.float32))

            if vis_save_dir and (vis_every_batch or not self._val_vis_done):
                step_dir = os.path.join(vis_save_dir, f"step_{global_step}")
                os.makedirs(step_dir, exist_ok=True)
                stem = os.path.join(step_dir, f"b{batch_idx}_rank{accelerator.process_index}")
                n_vis = min(vis_num_samples, bs)
                visualize_reconstruction(
                    input_frames=inputs["history_images"][:n_vis],
                    pred_frames=pred_frames[:n_vis],
                    save_path=f"{stem}_compare.png",
                    num_samples=n_vis,
                    text_labels=prompts[:n_vis],
                    future_label="pred",
                    gt_frames=gt_frames[:n_vis],
                    compare_label="gt",
                )
                if not vis_every_batch:
                    self._val_vis_done = True

        dev = device
        gt_arr = inputs["future_trajectory"]
        if use_traj_pred:
            # Evaluate traj MLP prediction.
            assert self.action_head is not None
            traj_action_tokens = vlm_out.get("traj_action_tokens")
            if traj_action_tokens is not None and traj_action_tokens.shape[1] > 0:
                mlp_p = next(self.action_head.parameters())
                traj_last = traj_action_tokens[:, -1, :].to(
                    device=mlp_p.device, dtype=mlp_p.dtype
                )
                traj_pred = self.action_head({"mlp_input": traj_last}).detach().cpu().numpy()
                for b in range(bs):
                    pred_np = traj_pred[b].astype(np.float32)
                    gt_np   = gt_arr[b].cpu().numpy()
                    self.val_text_ade.update(
                        torch.as_tensor(trajectory_ade(pred_np, gt_np), device=dev, dtype=torch.float32)
                    )
                    self.val_text_fde.update(
                        torch.as_tensor(trajectory_fde(pred_np, gt_np), device=dev, dtype=torch.float32)
                    )
        elif (txs := (vlm_out or {}).get("texts")) is not None:
            for b in range(bs):
                parsed = parse_trajectory_text(txs[b])
                gt_np  = gt_arr[b].cpu().numpy()
                if parsed is not None and parsed.shape[0] >= 1:
                    self.val_text_ade.update(
                        torch.as_tensor(trajectory_ade(parsed, gt_np), device=dev, dtype=torch.float32)
                    )
                    self.val_text_fde.update(
                        torch.as_tensor(trajectory_fde(parsed, gt_np), device=dev, dtype=torch.float32)
                    )

    def compute_validation_metrics(self) -> dict[str, float]:
        out: dict[str, float] = {}

        if getattr(self, "_use_vggt", False) and hasattr(self, "val_vggt_mse"):
            out["vggt_mse"] = float(self.val_vggt_mse.compute().detach().cpu())
            self.val_vggt_mse.reset()

        if self._use_world_model:
            out["gen_psnr"] = float(self.val_gen_psnr.compute().detach().cpu())
        self.val_gen_psnr.reset()
        self._val_vis_done = False
        out["text_ade"] = float(self.val_text_ade.compute().detach().cpu())
        out["text_fde"] = float(self.val_text_fde.compute().detach().cpu())
        self.val_text_ade.reset()
        self.val_text_fde.reset()
        return out

    @torch.no_grad()
    def predict_trajectory(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """Predict trajectory using the configured ``eval_traj_mode``.

        Returns:
            ``[B, N, 3]`` float32 CPU tensor.  ``N`` = ``traj_pred_num_points``（`traj_token`
            路径与 MLPActionHead 一致；`text` 路径对解析点不足 ``N`` 的样本做末点重复至 ``N``）。
        """
        images = self._extract_current_images(inputs)
        prompts = self._build_prompts(inputs)

        use_traj_pred = (self._eval_traj_mode == "traj_token"
                         and self._use_traj
                         and self.vlm.num_traj_action_tokens > 0)

        if not self._use_vlm_autoreg_text_trajectory and not use_traj_pred:
            raise ValueError(
                "use_vlm_autoreg_text_trajectory=false 时 predict_trajectory 不支持 "
                "eval_traj_mode='text'（自回归文本轨迹已关闭）。请设 eval_traj_mode='traj_token' "
                "且 use_traj=true，或开启 use_vlm_autoreg_text_trajectory。"
            )

        if use_traj_pred:
            assert self.action_head is not None
            mlp_p = next(self.action_head.parameters())
            _modes = list(VLM_FIXED_SIZE_MODES_ORDER)
            if getattr(self, "_use_vggt", False): _modes.append("visual_tokens")
            vlm_out = self.vlm.forward_extract(
                images, prompts, assistant_answers=None,
                compute_ce_loss=False,
                vlm_conditioning=_modes,
            )
            traj_action_tokens = vlm_out.get("traj_action_tokens")
            if traj_action_tokens is None or traj_action_tokens.shape[1] == 0:
                raise ValueError(
                    "predict_trajectory (traj_token): expected non-empty traj_action_tokens from VLM; "
                    "check num_traj_action_tokens and forward_extract(..., vlm_conditioning with traj)."
                )
            traj_last = traj_action_tokens[:, -1, :].to(
                device=mlp_p.device, dtype=mlp_p.dtype
            )
            return self.action_head({"mlp_input": traj_last}).float().cpu()

        # Text generation path（eval_traj_mode="text" 且开关为 true）。
        gen = self.vlm.generate_text(images, prompts, max_new_tokens=_TRAJ_TEXT_MAX_NEW)
        texts = gen["texts"]
        B = len(texts)
        n_pts = self._traj_pred_num_points
        trajectories = torch.zeros(B, n_pts, 3)
        for b, text in enumerate(texts):
            parsed = parse_trajectory_text(text)
            if parsed is not None and parsed.shape[0] > 0:
                n = min(int(parsed.shape[0]), n_pts)
                trajectories[b, :n] = torch.from_numpy(parsed[:n].copy())
                if n < n_pts:
                    trajectories[b, n:] = trajectories[b, n - 1].unsqueeze(0).expand(
                        n_pts - n, -1
                    )
        return trajectories
