"""RecogActionHead: DiT-based diffusion trajectory planner (renamed from DiffusionTrajectoryPlanner).

VLM ``forward_extract`` / 缓存的 **token 名 → 张量**（及 ``non_action_tokens`` 的 ``*_mask``）
在 :class:`RecogActionHead` 内拼为 ``[B,L,H]``；``Tensor`` 输入视为已拼好的 legacy 路径。

Supports flow matching, DDPM, and DDIM sampling methods.
No Lightning or Hydra dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from timm.models.layers import Mlp

from models.action_head.recog.dit import LightningDiT
from models.action_head.recog.layers import ActionEncoder


@dataclass
class RecogActionHeadConfig:
    input_embedding_dim: int = 384
    hidden_size: int = 512
    action_dim: int = 3
    action_horizon: int = 8
    add_pos_embed: bool = True
    input_feature_dim: int = 1536

    sampling_method: str = "flow"   # flow | ddpm | ddim
    num_inference_steps: int = 5

    # DiT architecture (small preset)
    num_heads: int = 8
    head_dim: int = 48
    num_layers: int = 16
    dit_output_dim: int = 512
    interleave_attention: bool = True

    # Flow matching
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000

    # DDPM / DDIM
    num_train_timesteps: int = 100
    ddim_eta: float = 0.0


class RecogActionHead(nn.Module):
    """Trajectory planner that uses a DiT denoiser conditioned on VLM features."""

    # 与 Qwen3VL token layout 一致；仅 ``non_action_tokens`` 走变长 ``*_mask``。
    _VL_STACK_KEYS: ClassVar[tuple[str, ...]] = (
        "non_action_tokens",
        "jepa_action_tokens",
        "vggt_action_tokens",
        "wan_action_tokens",
        "traj_action_tokens",
    )

    def __init__(self, config: RecogActionHeadConfig):
        super().__init__()
        self.config = config
        dim = config.input_embedding_dim

        self.model = LightningDiT(
            num_heads=config.num_heads, head_dim=config.head_dim,
            output_dim=config.dit_output_dim, num_layers=config.num_layers,
            interleave_attention=config.interleave_attention,
        )

        self.his_traj_encoder = Mlp(in_features=12, hidden_features=config.hidden_size,
                                     out_features=dim, norm_layer=nn.LayerNorm)
        self.ego_status_encoder = Mlp(in_features=8, hidden_features=config.hidden_size,
                                       out_features=dim, norm_layer=nn.LayerNorm)
        self.action_encoder = ActionEncoder(action_dim=config.action_dim, hidden_size=dim)
        self.feature_encoder = nn.Linear(config.input_feature_dim, dim)
        self.fusion_projector = nn.Linear(dim * 3, dim)
        self.action_decoder = Mlp(in_features=self.model.output_dim, hidden_features=config.hidden_size,
                                   out_features=config.action_dim, norm_layer=nn.LayerNorm)

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.action_horizon, dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        if config.sampling_method == "flow":
            self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
            self.num_timestep_buckets = config.num_timestep_buckets
        elif config.sampling_method in ("ddpm", "ddim"):
            self._init_ddpm_buffers(config.num_train_timesteps)
            if config.sampling_method == "ddim":
                self._init_ddim_buffers(config.num_inference_steps, config.ddim_eta)

    # ── Normalization ──────────────────────────────────────────────────────

    @staticmethod
    def norm_odo(trajectory: torch.Tensor) -> torch.Tensor:
        x = 2 * (trajectory[..., 0:1] + 1.57) / 66.74 - 1
        y = 2 * (trajectory[..., 1:2] + 19.68) / 42 - 1
        heading = 2 * (trajectory[..., 2:3] + 1.67) / 3.53 - 1
        return torch.cat([x, y, heading], dim=-1)

    @staticmethod
    def denorm_odo(nt: torch.Tensor) -> torch.Tensor:
        x = (nt[..., 0:1] + 1) / 2 * 66.74 - 1.57
        y = (nt[..., 1:2] + 1) / 2 * 42 - 19.68
        heading = (nt[..., 2:3] + 1) / 2 * 3.53 - 1.67
        return torch.cat([x, y, heading], dim=-1)

    # ── DDPM / DDIM buffers ────────────────────────────────────────────────

    def _init_ddpm_buffers(self, T: int):
        betas = self._cosine_beta_schedule(T)
        alphas = 1.0 - betas
        ac = torch.cumprod(alphas, dim=0)
        acp = torch.cat([torch.tensor([1.0]), ac[:-1]])
        self.register_buffer("ddpm_betas", betas)
        self.register_buffer("ddpm_sqrt_ac", torch.sqrt(ac))
        self.register_buffer("ddpm_sqrt_1m_ac", torch.sqrt(1.0 - ac))
        self.register_buffer("ddpm_sqrt_recip_ac", torch.sqrt(1.0 / ac))
        self.register_buffer("ddpm_sqrt_recipm1_ac", torch.sqrt(1.0 / ac - 1.0))
        var = betas * (1.0 - acp) / (1.0 - ac)
        self.register_buffer("ddpm_logvar", torch.log(var.clamp(min=1e-20)))
        self.register_buffer("ddpm_mu1", betas * torch.sqrt(acp) / (1.0 - ac))
        self.register_buffer("ddpm_mu2", (1.0 - acp) * torch.sqrt(alphas) / (1.0 - ac))
        self.ddpm_T = T

    def _init_ddim_buffers(self, steps: int, eta: float):
        self.ddim_steps = steps
        step_ratio = self.ddpm_T // steps
        t_sched = torch.arange(0, steps) * step_ratio
        ac = self.ddpm_sqrt_ac ** 2
        ddim_a = ac[t_sched].float()
        ddim_ap = torch.cat([torch.tensor([1.0]), ac[t_sched[:-1]]]).float()
        self.register_buffer("ddim_t", torch.flip(t_sched, [0]).long())
        self.register_buffer("ddim_a", torch.flip(ddim_a, [0]))
        self.register_buffer("ddim_ap", torch.flip(ddim_ap, [0]))
        self.register_buffer("ddim_sqrt_1m_a", torch.flip((1 - ddim_a).sqrt(), [0]))
        sigma = eta * ((1 - ddim_ap) / (1 - ddim_a) * (1 - ddim_a / ddim_ap)).sqrt()
        self.register_buffer("ddim_sigma", torch.flip(sigma, [0]))

    @staticmethod
    def _cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
        x = np.linspace(0, T + 1, T + 1)
        ac = np.cos(((x / (T + 1)) + s) / (1 + s) * np.pi * 0.5) ** 2
        ac = ac / ac[0]
        betas = 1 - (ac[1:] / ac[:-1])
        return torch.tensor(np.clip(betas, 0, 0.999), dtype=torch.float32)

    # ── Core forward / predict ─────────────────────────────────────────────

    @staticmethod
    def _vl_bag_to_seq(vlm_bag: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        parts: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for k in RecogActionHead._VL_STACK_KEYS:
            t = vlm_bag.get(k)
            if not isinstance(t, torch.Tensor):
                continue
            B, L = t.shape[0], t.shape[1]
            if k == "non_action_tokens":
                m = vlm_bag.get(f"{k}_mask")
                if not isinstance(m, torch.Tensor):
                    raise KeyError(f"_vl_bag_to_seq: missing {k}_mask")
            else:
                m = (
                    torch.ones(B, L, dtype=torch.bool, device=t.device)
                    if L > 0
                    else torch.zeros(B, 0, dtype=torch.bool, device=t.device)
                )
            parts.append(t)
            masks.append(m)
        if not parts:
            raise ValueError("_vl_bag_to_seq: no token tensors in vlm_bag")
        return torch.cat(parts, dim=1), torch.cat(masks, dim=1)

    def _encode_conditions(self, vlm_bag, his_traj, status_feature):
        vl_seq, vl_mask = self._vl_bag_to_seq(vlm_bag)
        vl = self.feature_encoder(
            vl_seq.to(dtype=self.feature_encoder.weight.dtype)
        )
        ht = self.his_traj_encoder(
            his_traj.unsqueeze(1).to(dtype=self.his_traj_encoder.fc1.weight.dtype)
        ).repeat(1, self.config.action_horizon, 1)
        ego = self.ego_status_encoder(
            status_feature.to(dtype=self.ego_status_encoder.fc1.weight.dtype)
        )
        return vl, ht, ego, vl_mask

    def _denoise_step(self, noisy, t_disc, vl, ht, ego, vl_mask=None):
        noisy = noisy.to(dtype=ht.dtype)
        af = self.action_encoder(noisy, t_disc)
        if hasattr(self, "position_embedding"):
            af = af + self.position_embedding(torch.arange(af.shape[1], device=af.device))

        if vl_mask is not None:
            float_mask = vl_mask.to(dtype=vl.dtype).unsqueeze(-1)
            vl_mean = (vl * float_mask).sum(1) / float_mask.sum(1).clamp(min=1.0)
            vl_mean = vl_mean.unsqueeze(1).repeat(1, self.config.action_horizon, 1)
            attn_bias = torch.zeros(
                vl.shape[0], 1, 1, vl.shape[1], dtype=vl.dtype, device=vl.device
            ).masked_fill(~vl_mask[:, None, None, :], float("-inf"))
        else:
            vl_mean = vl.mean(1).unsqueeze(1).repeat(1, self.config.action_horizon, 1)
            attn_bias = None

        fused = self.fusion_projector(torch.cat([ht, vl_mean, af], dim=2))
        out = self.model(fused, vl, ego, t_disc, encoder_attention_mask=attn_bias)
        return self.action_decoder(out)

    def forward(self, vlm_bag, his_traj, status_feature, gt_actions):
        """Training forward: compute diffusion loss.

        ``vlm_bag``：VLM/缓存 与 :meth:`Qwen3VLWrapper.forward_extract` 同构的 **token 字典**。
        """
        vl, ht, ego, m = self._encode_conditions(vlm_bag, his_traj, status_feature)
        md = vl.dtype
        gt = self.norm_odo(gt_actions.to(md))
        noise = torch.randn_like(gt)

        if self.config.sampling_method == "flow":
            s = self.beta_dist.sample((gt.shape[0],)).to(device=gt.device, dtype=md)
            t_cont = ((self.config.noise_s - s) / self.config.noise_s).clamp(1e-4, 1 - 1e-4)
            t_r = t_cont[:, None, None]
            noisy = (1 - t_r) * noise + t_r * gt
            target = gt - noise
            t_disc = (t_cont * self.num_timestep_buckets).long()
        else:
            t_disc = torch.randint(0, self.ddpm_T, (gt.shape[0],), device=gt.device).long()
            s_ac = self.ddpm_sqrt_ac[t_disc].to(dtype=md).view(-1, 1, 1)
            s_1m = self.ddpm_sqrt_1m_ac[t_disc].to(dtype=md).view(-1, 1, 1)
            noisy = s_ac * gt + s_1m * noise
            target = noise

        pred = self._denoise_step(noisy, t_disc, vl, ht, ego, vl_mask=m)
        return F.mse_loss(pred, target, reduction="mean")

    @torch.no_grad()
    def predict(self, vlm_bag, his_traj, status_feature) -> torch.Tensor:
        """Inference: denoise to get trajectory. ``[B, H, 3]`` 输出。"""
        vl, ht, ego, m = self._encode_conditions(vlm_bag, his_traj, status_feature)
        B = vl.shape[0]
        D = self.config.action_dim
        H = self.config.action_horizon
        device, dtype = vl.device, vl.dtype
        x = torch.randn(B, H, D, device=device, dtype=dtype)

        if self.config.sampling_method == "flow":
            dt = 1.0 / self.config.num_inference_steps
            for step in range(self.config.num_inference_steps):
                idx = int(step / self.config.num_inference_steps * self.num_timestep_buckets)
                t = torch.full((B,), idx, device=device, dtype=torch.long)
                x = x + dt * self._denoise_step(x, t, vl, ht, ego, vl_mask=m)
        elif self.config.sampling_method == "ddim":
            for i in range(self.ddim_steps):
                t = torch.full((B,), self.ddim_t[i], device=device, dtype=torch.long)
                pred_noise = self._denoise_step(x, t, vl, ht, ego, vl_mask=m)
                a = self.ddim_a[i]; ap = self.ddim_ap[i]
                s1m = self.ddim_sqrt_1m_a[i]; sig = self.ddim_sigma[i]
                x0 = (x - s1m * pred_noise) / a.sqrt()
                x0.clamp_(-1, 1)
                pred_dir = (1 - ap - sig**2).clamp(min=0).sqrt() * pred_noise
                x = ap.sqrt() * x0 + pred_dir + sig * torch.randn_like(x)
        else:
            step_size = self.ddpm_T // self.config.num_inference_steps
            for t_int in reversed(range(0, self.ddpm_T, step_size)):
                t = torch.full((B,), t_int, device=device, dtype=torch.long)
                pred_noise = self._denoise_step(x, t, vl, ht, ego, vl_mask=m)
                x0 = self.ddpm_sqrt_recip_ac[t_int] * x - self.ddpm_sqrt_recipm1_ac[t_int] * pred_noise
                x0.clamp_(-1, 1)
                mean = self.ddpm_mu1[t_int] * x0 + self.ddpm_mu2[t_int] * x
                if t_int > 0:
                    x = mean + torch.exp(0.5 * self.ddpm_logvar[t_int]).clamp(min=1e-3) * torch.randn_like(x)
                else:
                    x = mean

        x.clamp_(-1, 1)
        return self.denorm_odo(x)
