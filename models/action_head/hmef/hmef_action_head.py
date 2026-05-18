"""Hierarchical Multi Expert Fusion (HMEF) action head — multi-expert trajectory prediction.

Each expert (wan/traj/jepa/vggt) processes its own tokens via BiTrans,
then all experts jointly denoise through a shared MMDiT-style denoiser.
non_action acts as shared scene context.

Architecture:
  - Perceiver compresses non_action → scene context for all experts.
  - Optional external JEPA context uses its own Perceiver compressor, then
    concatenates with compressed non_action scene context.
  - Optional external VGGT context uses its own Perceiver compressor, then
    concatenates with compressed non_action scene context.
  - Each expert: BiTrans + per-step feature extraction.
  - Shared JointDenoiser (MMDiT-style): joint self-attention with split FFN
    (clean stream = scene context, noisy stream = trajectory tokens).
  - Flow Matching x0 prediction in action coordinate space.
  - Shared trajectory decoder.
  - Learned fusion weights for expert trajectory ensemble.
  - Optional traj token mask augmentation during training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import Mlp
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

from models.action_head.recog.layers import RMSNorm, Attention, SwiGLUFFN


# ════════════════════════════════════════════════════════════════════════════
# BiTrans building blocks
# ════════════════════════════════════════════════════════════════════════════

class BiTransBlock(nn.Module):
    """Standard bidirectional transformer block (pre-norm, self-attention + FFN)."""

    def __init__(self, dim: int, num_heads: int = 8, head_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(query_dim=dim, heads=num_heads, dim_head=head_dim,
                              dropout=dropout, bias=False)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class BiTrans(nn.Module):
    """Stack of bidirectional transformer blocks."""

    def __init__(self, dim: int, num_layers: int, num_heads: int, head_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            BiTransBlock(dim, num_heads, head_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ════════════════════════════════════════════════════════════════════════════
# Perceiver-style compressor for variable-length tokens
# ════════════════════════════════════════════════════════════════════════════

class CrossAttnBlock(nn.Module):
    """Block: self-attn on queries → cross-attn to context → FFN.

    Queries are a small fixed set of learned latents; context is the long
    variable-length sequence.  A padding mask on context prevents attending
    to pad positions.
    """

    def __init__(self, dim: int, num_heads: int = 8, head_dim: int = 64,
                 dropout: float = 0.0):
        super().__init__()
        self.norm_q1 = RMSNorm(dim)
        self.self_attn = Attention(query_dim=dim, heads=num_heads,
                                   dim_head=head_dim, dropout=dropout, bias=False)
        self.norm_q2 = RMSNorm(dim)
        self.cross_attn = Attention(query_dim=dim, heads=num_heads,
                                    dim_head=head_dim, dropout=dropout, bias=False)
        self.norm_q3 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, bias=True)

    def forward(self, queries: torch.Tensor, context: torch.Tensor,
                context_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Args:
            queries: [B, N_q, D] — small fixed set of latent tokens.
            context: [B, N_c, D] — long variable-length sequence.
            context_mask: [B, 1, 1, N_c] — additive bias, -inf for padding, 0 for real.
        """
        queries = queries + self.self_attn(self.norm_q1(queries))
        queries = queries + self.cross_attn(
            self.norm_q2(queries),
            encoder_hidden_states=context,
            attention_mask=context_mask,
        )
        queries = queries + self.ffn(self.norm_q3(queries))
        return queries


class PerceiverCompressor(nn.Module):
    """Compress variable-length tokens into K fixed latent tokens."""

    def __init__(self, dim: int, num_queries: int = 64, num_layers: int = 3,
                 num_heads: int = 8, head_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.num_queries = num_queries
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.layers = nn.ModuleList([
            CrossAttnBlock(dim, num_heads, head_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(dim)

    def forward(self, context: torch.Tensor,
                context_mask: torch.Tensor | None = None
                ) -> torch.Tensor:
        """Args:
            context: [B, L, D] — variable-length sequence (padded).
            context_mask: [B, L] bool — True for real tokens, False for padding.

        Returns:
            compressed: [B, K, D] — K compressed latent tokens.
        """
        B = context.shape[0]
        x = self.latents.to(dtype=context.dtype).expand(B, -1, -1)

        if context_mask is not None:
            L = context.shape[1]
            attn_bias = torch.zeros(B, 1, 1, L, device=context.device, dtype=context.dtype)
            attn_bias = attn_bias.masked_fill(~context_mask[:, None, None, :], float('-inf'))
        else:
            attn_bias = None

        for layer in self.layers:
            x = layer(x, context, attn_bias)

        x = self.norm(x)
        return x  


# ════════════════════════════════════════════════════════════════════════════
# MMDiT-style Joint Denoiser
# ════════════════════════════════════════════════════════════════════════════

class JointDenoiserBlock(nn.Module):
    """MMDiT block: joint self-attn + split FFN, adaLN-Zero on both streams.

    Clean stream (scene context) and noisy stream (trajectory tokens) share
    self-attention for cross-modal interaction.  Both attn and FFN on each
    stream receive independent adaLN-Zero modulation from the timestep embedding.
    """

    def __init__(self, dim: int, num_heads: int = 8, head_dim: int = 96,
                 dropout: float = 0.0):
        super().__init__()
        self.norm_c = RMSNorm(dim)
        self.norm_n = RMSNorm(dim)

        self.attn = Attention(query_dim=dim, heads=num_heads, dim_head=head_dim,
                              dropout=dropout, bias=False)

        self.ffn_c_norm = RMSNorm(dim)
        self.ffn_n_norm = RMSNorm(dim)

        self.ffn_c = SwiGLUFFN(dim, bias=True)
        self.ffn_n = SwiGLUFFN(dim, bias=True)

        # adaLN-Zero: 6×dim per stream (shift/scale/gate for attn + shift/scale/gate for ffn)
        self.adaLN_clean = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.adaLN_noisy = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

        self._init_zero()

    def _init_zero(self):
        for mod in (self.adaLN_clean, self.adaLN_noisy):
            nn.init.constant_(mod[-1].weight, 0)
            nn.init.constant_(mod[-1].bias, 0)

    def forward(self, x_clean: torch.Tensor, x_noisy: torch.Tensor,
                t_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_clean:  [B, Lc, D]  scene context tokens
            x_noisy:  [B, Ln, D]  trajectory tokens
            t_emb:    [B, D]      timestep embedding
        """
        Lc = x_clean.shape[1]
        Ln = x_noisy.shape[1]

        shift_c_a, scale_c_a, gate_c_a, shift_c_f, scale_c_f, gate_c_f = \
            self.adaLN_clean(t_emb).chunk(6, dim=1)
        shift_n_a, scale_n_a, gate_n_a, shift_n_f, scale_n_f, gate_n_f = \
            self.adaLN_noisy(t_emb).chunk(6, dim=1)

        # ① Joint attention (adaLN on both streams)
        x_c = self.norm_c(x_clean) * (1 + scale_c_a.unsqueeze(1)) + shift_c_a.unsqueeze(1)
        x_n = self.norm_n(x_noisy) * (1 + scale_n_a.unsqueeze(1)) + shift_n_a.unsqueeze(1)
        x_joint = torch.cat([x_c, x_n], dim=1)
        x_joint = x_joint + self.attn(x_joint)
        x_clean = x_clean + gate_c_a.unsqueeze(1) * x_joint[:, :Lc]
        x_noisy = x_noisy + gate_n_a.unsqueeze(1) * x_joint[:, Lc:]

        # ② Clean FFN (adaLN)
        x_c = self.ffn_c_norm(x_clean) * (1 + scale_c_f.unsqueeze(1)) + shift_c_f.unsqueeze(1)
        x_clean = x_clean + gate_c_f.unsqueeze(1) * self.ffn_c(x_c)

        # ③ Noisy FFN (adaLN)
        x_n = self.ffn_n_norm(x_noisy) * (1 + scale_n_f.unsqueeze(1)) + shift_n_f.unsqueeze(1)
        x_noisy = x_noisy + gate_n_f.unsqueeze(1) * self.ffn_n(x_n)

        return x_clean, x_noisy


class JointDenoiser(nn.Module):
    """MMDiT-style joint denoiser.

    Clean stream = scene context (compressed non_action).
    Noisy stream = trajectory tokens (one per waypoint, per expert).
    Multiple experts are batched along the sequence dimension.
    """

    def __init__(self, dim: int, num_layers: int = 4, num_heads: int = 8,
                 head_dim: int = 96, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            JointDenoiserBlock(dim, num_heads, head_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm_n = RMSNorm(dim)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        nn.init.constant_(self.final_adaLN[-1].weight, 0)
        nn.init.constant_(self.final_adaLN[-1].bias, 0)

    def forward(self, clean: torch.Tensor, noisy: torch.Tensor,
                t_emb: torch.Tensor) -> torch.Tensor:
        """Returns denoised noisy tokens [B, Ln, D]."""
        for layer in self.layers:
            clean, noisy = layer(clean, noisy, t_emb)
        shift, scale = self.final_adaLN(t_emb).chunk(2, dim=1)
        return self.norm_n(noisy) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ════════════════════════════════════════════════════════════════════════════
# HMEF Action Head (main entry)
# ════════════════════════════════════════════════════════════════════════════

_VL_TOKEN_KEYS = (
    "non_action_tokens",
    "external_jepa_context_tokens",
    "external_vggt_context_tokens",
    "jepa_action_tokens",
    "vggt_action_tokens",
    "wan_action_tokens",
    "traj_action_tokens",
)

_CONTEXT_TOKEN_KEYS = {
    "non_action_tokens",
    "external_jepa_context_tokens",
    "external_vggt_context_tokens",
}


class HMEFActionHead(nn.Module):
    """Competitive multi-expert action head.

    Config (dict, passed from ActionHead/router):
        input_feature_dim: int         # VLM hidden size (e.g. 1536)
        hmef_hidden_dim: int = 768     # Internal dim after input projection

        # Expert
        expert_num_layers: int = 2
        expert_num_heads: int = 8
        expert_head_dim: int = 64
        non_action_num_queries: int = 96
        external_jepa_num_queries: int = 96
        external_vggt_num_queries: int = 96

        # Denoiser
        denoiser_num_layers: int = 4
        denoiser_num_heads: int = 8
        denoiser_head_dim: int = 96

        # Diffusion
        sampling_method: str = "flow"
        num_inference_steps: int = 10
        num_train_timesteps: int = 1000

        # Condition encoders (his_traj / ego / coord)
        cond_hidden_dim: int = 512
    """

    def __init__(self, cfg: dict):
        super().__init__()
        vlm_dim = int(cfg["input_feature_dim"])
        self.hmef_dim = int(cfg.get("hmef_hidden_dim", 768))
        self.action_horizon = int(cfg.get("action_horizon", 8))
        self.skip_traj_norm = bool(cfg.get("skip_traj_norm", False))

        # ── Enabled token keys ──
        raw_cond = cfg.get(
            "vlm_conditioning",
            [k for k in _VL_TOKEN_KEYS if k not in (
                "external_jepa_context_tokens", "external_vggt_context_tokens",
            )],
        )
        self.token_types: list[str] = [k for k in _VL_TOKEN_KEYS if k in raw_cond]
        self._action_types: list[str] = [
            k for k in self.token_types if k not in _CONTEXT_TOKEN_KEYS
        ]
        if not self._action_types:
            raise ValueError("HMEF: need at least one action token type.")
        if "non_action_tokens" not in self.token_types:
            raise ValueError("HMEF: non_action_tokens must be enabled (scene context required).")

        # ── Per-expert token counts (from vlm config) ──
        vlm_cfg = cfg.get("vlm", {})
        self._jepa_raw_steps = int(vlm_cfg.get("num_jepa_token_steps", 8))
        self._vggt_raw_steps = int(vlm_cfg.get("num_vggt_token_steps", 8))
        _raw_steps = {
            "jepa_action_tokens": self._jepa_raw_steps,
            "vggt_action_tokens": self._vggt_raw_steps,
        }
        self._expert_tokens_per_step: dict[str, int] = {}
        for k in self._action_types:
            total = vlm_cfg[f"num_{k}"]
            if total <= 0:
                raise ValueError(
                    f"HMEF: {k} requires total > 0, got {total}. "
                    f"Add 'vlm.num_{k}' to action_head config."
                )
            raw_steps = _raw_steps.get(k, self.action_horizon)
            if total % raw_steps != 0:
                raise ValueError(
                    f"HMEF: {k} total={total} not divisible by raw_steps={raw_steps}"
                )
            self._expert_tokens_per_step[k] = total // raw_steps

        # ── Shared input projection ──
        self.input_proj = nn.Linear(vlm_dim, self.hmef_dim)

        enl = int(cfg.get("expert_num_layers", 2))
        enh = int(cfg.get("expert_num_heads", 8))
        ehd = int(cfg.get("expert_head_dim", 64))

        # ── Perceiver compressors ──
        self.non_action_compressor: PerceiverCompressor | None = None
        if "non_action_tokens" in self.token_types:
            nq = int(cfg.get("non_action_num_queries", 96))
            self.non_action_compressor = PerceiverCompressor(
                self.hmef_dim, num_queries=nq, num_layers=enl,
                num_heads=enh, head_dim=ehd,
            )
        self.external_jepa_context_compressor: PerceiverCompressor | None = None
        if "external_jepa_context_tokens" in self.token_types:
            jq = int(cfg.get("external_jepa_num_queries", cfg.get("non_action_num_queries", 96)))
            self.external_jepa_context_compressor = PerceiverCompressor(
                self.hmef_dim, num_queries=jq, num_layers=enl,
                num_heads=enh, head_dim=ehd,
            )
        self.external_vggt_context_compressor: PerceiverCompressor | None = None
        if "external_vggt_context_tokens" in self.token_types:
            vq = int(cfg.get("external_vggt_num_queries", 96))
            self.external_vggt_context_compressor = PerceiverCompressor(
                self.hmef_dim, num_queries=vq, num_layers=enl,
                num_heads=enh, head_dim=ehd,
            )

        # ── Expert BiTrans ──
        self.experts = nn.ModuleDict({
            k: BiTrans(self.hmef_dim, enl, enh, ehd)
            for k in self._action_types
        })

        # ── Per-step extraction projections ──
        self.step_projs = nn.ModuleDict({
            k: nn.Linear(self.hmef_dim, self.hmef_dim)
            for k in self._action_types
        })

        # ── JEPA step expander (4 raw steps → 8 via MLP + split) ──
        self.jepa_expander: nn.Linear | None = None
        if "jepa_action_tokens" in self._action_types and self._jepa_raw_steps != self.action_horizon:
            self.jepa_expander = nn.Linear(self.hmef_dim, self.hmef_dim * 2)

        # ── Expert type embedding ──
        self.type_embed = nn.Embedding(len(self._action_types), self.hmef_dim)

        # ── Time position embedding (shared horizon) ──
        self.time_pos = nn.Embedding(self.action_horizon, self.hmef_dim)

        # ── Expert position embedding (per-step, injected before BiTrans) ──
        self.expert_pos = nn.Embedding(self.action_horizon, self.hmef_dim)

        # ── Condition encoders (Recog-style: encode to hmef_dim then repeat per waypoint) ──
        c_hidden = int(cfg.get("cond_hidden_dim", 512))
        self.his_traj_encoder = Mlp(in_features=12,  # 4 history steps × 3 coords
                                     hidden_features=c_hidden,
                                     out_features=self.hmef_dim,
                                     norm_layer=nn.LayerNorm)
        self.ego_encoder = Mlp(in_features=8, hidden_features=c_hidden,
                                out_features=self.hmef_dim,
                                norm_layer=nn.LayerNorm)

        # ── Conditioning merger: fuse his_enc + ego_enc → his_and_ego ──
        self.cond_merger = nn.Linear(self.hmef_dim * 2, self.hmef_dim)

        # ── Action encoder (Recog-style: noisy action + timestep → feature) ──
        self.action_encoder = nn.Sequential(
            nn.Linear(3 + self.hmef_dim, self.hmef_dim),
            nn.SiLU(),
            nn.Linear(self.hmef_dim, self.hmef_dim),
        )

        # ── Fusion projector (Recog-style: cat[feat, his, encoded] → D) ──
        self.fusion_projector = nn.Linear(self.hmef_dim * 3, self.hmef_dim)

        # ── Shared Joint Denoiser ──
        dnl = int(cfg.get("denoiser_num_layers", 4))
        dnh = int(cfg.get("denoiser_num_heads", 8))
        dhd = int(cfg.get("denoiser_head_dim", 96))
        self.denoiser = JointDenoiser(self.hmef_dim, dnl, dnh, dhd)

        # ── Shared trajectory decoder ──
        self.traj_decoder = Mlp(in_features=self.hmef_dim,
                                hidden_features=c_hidden,
                                out_features=3, norm_layer=nn.LayerNorm)

        # ── Timestep encoder (MMDiT official: Timesteps + TimestepEmbedding) ──
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True,
                                   downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256,
                                                    time_embed_dim=self.hmef_dim)

        # ── Flow Matching noise schedule (Logit-Normal) ──
        self.num_train_timesteps = int(cfg.get("num_train_timesteps", 1000))
        self.num_inference_steps = int(cfg.get("num_inference_steps", 10))
        self.flow_shift = float(cfg.get("flow_shift", 1.0))
        if cfg.get("sampling_method", "flow") != "flow":
            raise NotImplementedError(
                "HMEF only supports 'flow' sampling_method currently."
            )

        self.fusion_weights = nn.Parameter(torch.zeros(len(self._action_types)))
        self._fusion_mode = str(cfg.get("fusion_mode", "global"))
        if self._fusion_mode == "per_step":
            self.fusion_weights_per_step = nn.Parameter(
                torch.zeros(len(self._action_types), self.action_horizon)
            )
        elif self._fusion_mode != "global":
            raise ValueError(
                f"HMEF: fusion_mode must be 'global' or 'per_step', "
                f"got {self._fusion_mode!r}."
            )

        self._traj_token_mask_enabled = bool(cfg.get("traj_token_mask_enabled", False))
        self._traj_token_mask_prob = float(cfg.get("traj_token_mask_prob", 0.0))
        self._traj_token_mask_mode = str(cfg.get("traj_token_mask_mode", "step"))
        if self._traj_token_mask_enabled and "traj_action_tokens" not in self._action_types:
            raise ValueError(
                "HMEF: traj_token_mask_enabled=true requires traj_action_tokens "
                "in vlm_conditioning."
            )
        if self._traj_token_mask_enabled:
            self.traj_mask_token = nn.Parameter(torch.zeros(1, 1, self.hmef_dim))

        self._goal_loss_enabled = bool(cfg.get("goal_loss_enabled", False))
        self._goal_loss_xy_only = bool(cfg.get("goal_loss_xy_only", True))

    @property
    def _dtype(self):
        return self.input_proj.weight.dtype

    # ════════════════════════════════════════════════════════════════════
    # Per-step feature extraction
    # ════════════════════════════════════════════════════════════════════

    def _extract_step_features(self, tok_out: torch.Tensor, expert_type: str
                               ) -> torch.Tensor:
        """Extract per-step features from expert token outputs.

        tok_out: [B, total_tokens, D]
        Returns: [B, horizon, D]
        """
        S = self._expert_tokens_per_step[expert_type]
        H = self.action_horizon
        expected_tokens = H * S
        if tok_out.shape[1] != expected_tokens:
            raise ValueError(
                f"HMEF {expert_type}: expected {expected_tokens} tokens "
                f"(H={H}, S={S}), got {tok_out.shape[1]}."
            )
        D = tok_out.shape[-1]
        x = self.step_projs[expert_type](tok_out)
        return x.reshape(-1, H, S, D).mean(dim=2)  # [B, H, D]

    def _apply_traj_token_mask(self, feat: torch.Tensor) -> torch.Tensor:
        """Mask per-step traj expert features with a learnable token."""
        B, H, D = feat.shape
        device = feat.device
        mode = self._traj_token_mask_mode
        if mode not in ("step", "token", "random"):
            raise ValueError(
                "HMEF: traj_token_mask_mode must be 'step', 'token', or 'random', "
                f"got {mode!r}."
            )

        mask = torch.rand(B, H, device=device) < self._traj_token_mask_prob
        all_masked = mask.all(dim=1)
        if all_masked.any():
            keep_step = torch.randint(0, H, (int(all_masked.sum().item()),), device=device)
            mask[all_masked, keep_step] = False

        mask_tok = self.traj_mask_token.to(dtype=feat.dtype).expand(B, H, D)
        return torch.where(mask[:, :, None], mask_tok, feat)

    # ════════════════════════════════════════════════════════════════════
    # Expert forward
    # ════════════════════════════════════════════════════════════════════

    def _expert_forward(self, name: str, tokens: torch.Tensor) -> torch.Tensor:
        S = self._expert_tokens_per_step[name]  # tokens per step
        pos = self.expert_pos.weight  # [H, D]
        pos = pos.unsqueeze(1).repeat(1, S, 1).flatten(0, 1)  # [H*S, D]
        tokens = tokens + pos.to(dtype=tokens.dtype).unsqueeze(0)
        return self.experts[name](tokens)

    # ════════════════════════════════════════════════════════════════════
    # Run all experts
    # ════════════════════════════════════════════════════════════════════

    def _run_experts(self, vlm_bag, device
                     ) -> tuple[dict, torch.Tensor]:
        step_features: dict[str, torch.Tensor] = {}
        scene_ctx_parts: list[torch.Tensor] = []

        for k in self.token_types:
            raw = vlm_bag.get(k)
            if not (isinstance(raw, torch.Tensor) and raw.shape[1] > 0):
                raise ValueError(
                    f"HMEF: {k!r} enabled but missing in vlm_bag."
                )
            tok = self.input_proj(raw.to(device=device, dtype=self._dtype))

            if k == "non_action_tokens":
                mask = vlm_bag.get("non_action_tokens_mask")
                compressed = self.non_action_compressor(tok, mask)
                scene_ctx_parts.append(compressed)
            elif k == "external_jepa_context_tokens":
                mask = vlm_bag.get("external_jepa_context_tokens_mask")
                compressed = self.external_jepa_context_compressor(tok, mask)
                scene_ctx_parts.append(compressed)
            elif k == "external_vggt_context_tokens":
                mask = vlm_bag.get("external_vggt_context_tokens_mask")
                compressed = self.external_vggt_context_compressor(tok, mask)
                scene_ctx_parts.append(compressed)
            else:
                if k == "vggt_action_tokens" and self._vggt_raw_steps != self.action_horizon:
                    if self._vggt_raw_steps < self.action_horizon:
                        raise ValueError(
                            f"HMEF: vggt_raw_steps={self._vggt_raw_steps} < "
                            f"action_horizon={self.action_horizon}. "
                            f"VGGT only supports truncation (raw_steps >= action_horizon)."
                        )
                    S = self._expert_tokens_per_step[k]
                    tok = tok[:, -(self.action_horizon * S):, :]
                elif k == "jepa_action_tokens" and self._jepa_raw_steps != self.action_horizon:
                    if self._jepa_raw_steps * 2 != self.action_horizon:
                        raise ValueError(
                            f"HMEF: jepa_raw_steps={self._jepa_raw_steps}, "
                            f"action_horizon={self.action_horizon}. "
                            f"JEPA expander requires jepa_raw_steps * 2 == action_horizon."
                        )
                    B, _, D = tok.shape
                    S = self._expert_tokens_per_step[k]
                    tok = self.jepa_expander(tok)
                    tok = tok.reshape(B, self._jepa_raw_steps, S, 2, D)
                    tok = tok.permute(0, 1, 3, 2, 4).reshape(B, self._jepa_raw_steps * 2 * S, D)
                tok_out = self._expert_forward(k, tok)
                step_features[k] = self._extract_step_features(tok_out, k)

        if self.training and self._traj_token_mask_enabled:
            if "traj_action_tokens" in step_features:
                step_features["traj_action_tokens"] = self._apply_traj_token_mask(
                    step_features["traj_action_tokens"]
                )

        if not scene_ctx_parts:
            raise ValueError("HMEF: no scene context tokens were produced.")
        scene_ctx = torch.cat(scene_ctx_parts, dim=1)
        return step_features, scene_ctx

    # ════════════════════════════════════════════════════════════════════
    # Noise & conditioning
    # ════════════════════════════════════════════════════════════════════

    def _build_conditioning(self, his_traj: torch.Tensor,
                            status_feature: torch.Tensor,
                            ) -> torch.Tensor:
        """Build fused his+ego conditioning vector.

        Returns:
            his_and_ego: [B, D]  fused his+ego encoding (no per-step expansion yet)
        """
        his_enc = self.his_traj_encoder(his_traj.to(dtype=self._dtype))      # [B, D]
        ego_enc = self.ego_encoder(status_feature.to(dtype=self._dtype))      # [B, D]

        return self.cond_merger(torch.cat([his_enc, ego_enc], dim=-1))  # [B, D]

    # ── Normalization (Recog-style, NAVSIM coordinate ranges) ─────────────────────

    def norm_odo(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Normalize [x, y, heading] to [-1, 1].

        NAVSIM coordinate ranges (empirical from openscene v1.1):
          x ∈ [-1.57, 65.17]  →  span 66.74, center 31.80
          y ∈ [-19.68, 22.32] →  span 42.00, center 1.32
          heading ∈ [-1.67, 1.86] → span 3.53, center 0.095
        """
        if self.skip_traj_norm:
            return trajectory
        x = 2 * (trajectory[..., 0:1] + 1.57) / 66.74 - 1
        y = 2 * (trajectory[..., 1:2] + 19.68) / 42 - 1
        heading = 2 * (trajectory[..., 2:3] + 1.67) / 3.53 - 1
        return torch.cat([x, y, heading], dim=-1)

    def denorm_odo(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Denormalize [-1, 1] back to [x, y, heading] world coordinates."""
        if self.skip_traj_norm:
            return trajectory
        x = (trajectory[..., 0:1] + 1) / 2 * 66.74 - 1.57
        y = (trajectory[..., 1:2] + 1) / 2 * 42 - 19.68
        heading = (trajectory[..., 2:3] + 1) / 2 * 3.53 - 1.67
        return torch.cat([x, y, heading], dim=-1)

    # ── Noise (action coordinate space, x0 prediction) ─────────────────

    def _noise_coords(self, coords: torch.Tensor,
                      t: torch.Tensor | None = None,
                      ) -> torch.Tensor:
        """Flow-matching: add noise in action coordinate space.

        Uses Logit-Normal noise schedule (SD3/Flux standard).
        If t is None, samples t via sigmoid(N(0,1)).

        Returns noisy_action in normalized coordinate space.
        """
        B = coords.shape[0]
        device = coords.device
        md = coords.dtype

        noise = torch.randn_like(coords)
        if t is None:
            u = torch.randn(B, device=device, dtype=md)
            t = torch.sigmoid(u).clamp(1e-4, 1 - 1e-4)
        t_r = t[:, None, None]
        return (1 - t_r) * noise + t_r * coords

    # ════════════════════════════════════════════════════════════════════
    # Shared denoise step (training + inference)
    # ════════════════════════════════════════════════════════════════════

    def _decode_noisy_action(self, z_action_all: torch.Tensor,
                             step_features: dict[str, torch.Tensor],
                             his_and_ego: torch.Tensor,
                             t_emb: torch.Tensor, scene_ctx: torch.Tensor,
                             ) -> torch.Tensor:
        """Noisy action → encode → fusion → denoise → decode → trajectory.

        Shared between training and inference.

        Args:
            z_action_all:  [B, N*H, 3]  noisy action coords for all experts
            step_features: {name: [B, H, D]}  per-expert conditioning
            his_and_ego:   [B, D]  fused his+ego encoding (expanded to [B,H,D] on use)
            t_emb:         [B, D]  timestep embedding
            scene_ctx:     [B, K, D]  compressed non_action scene context

        Returns:
            traj_all: [B, N*H, 3]  decoded trajectories in normalized coords
        """
        device = z_action_all.device
        H = self.action_horizon
        dtype = self._dtype

        time_pos = self.time_pos(torch.arange(H, device=device))
        t_emb_per_step = t_emb.unsqueeze(1).expand(-1, H, -1)
        his_and_ego_tok = his_and_ego.unsqueeze(1).expand(-1, H, -1)       # [B, H, D]

        z_parts: list[torch.Tensor] = []
        for i, k in enumerate(self._action_types):
            type_emb = self.type_embed(torch.tensor(i, device=device))
            feat = step_features[k]
            z_action_i = z_action_all[:, i * H: (i + 1) * H]

            encoded = self.action_encoder(
                torch.cat([z_action_i.to(dtype=dtype), t_emb_per_step.to(dtype=dtype)], dim=-1)
            )
            fused = self.fusion_projector(
                torch.cat([feat, his_and_ego_tok, encoded], dim=-1).to(dtype=dtype)
            )
            z = fused + time_pos.unsqueeze(0) + type_emb.unsqueeze(0).unsqueeze(1)
            z_parts.append(z)

        z_all = torch.cat(z_parts, dim=1)  # [B, N*H, D]
        x0_all = self.denoiser(scene_ctx, z_all, t_emb)  # [B, N*H, D]

        traj_parts: list[torch.Tensor] = []
        N = len(self._action_types)
        for i in range(N):
            x0_i = x0_all[:, i * H: (i + 1) * H].to(dtype=dtype)
            traj_parts.append(self.traj_decoder(x0_i))

        return torch.cat(traj_parts, dim=1)  # [B, N*H, 3]

    # ════════════════════════════════════════════════════════════════════
    # Diffusion forward (training)
    # ════════════════════════════════════════════════════════════════════

    def _diffusion_forward(self, step_features: dict[str, torch.Tensor],
                           scene_ctx: torch.Tensor,
                           his_and_ego: torch.Tensor,
                           gt_norm: torch.Tensor,
                           ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B = his_and_ego.shape[0]
        H = self.action_horizon
        device = his_and_ego.device
        md = his_and_ego.dtype
        N = len(self._action_types)

        # Logit-Normal with flow_shift: sample shared t for all experts
        u = torch.randn(B, device=device, dtype=md)
        t_shared = torch.sigmoid(self.flow_shift * u).clamp(1e-4, 1 - 1e-4)

        noisy_action_parts: list[torch.Tensor] = []
        for _ in self._action_types:
            noisy_action_parts.append(self._noise_coords(gt_norm, t=t_shared))
        z_action_all = torch.cat(noisy_action_parts, dim=1)  # [B, N*H, 3]

        t_idx = (t_shared * self.num_train_timesteps).long()
        t_emb = self.timestep_embedder(self.time_proj(t_idx).to(self._dtype))
        traj_all = self._decode_noisy_action(z_action_all, step_features, his_and_ego, t_emb, scene_ctx)

        gt_all = gt_norm.repeat(1, N, 1)
        diff_loss = F.mse_loss(traj_all, gt_all)

        traj_per_expert = {
            k: traj_all[:, i * H: (i + 1) * H]
            for i, k in enumerate(self._action_types)
        }

        return diff_loss, traj_per_expert

    # ════════════════════════════════════════════════════════════════════
    # Training forward
    # ════════════════════════════════════════════════════════════════════

    def forward(self, vlm_bag, his_traj, status_feature, gt_actions
                ) -> dict[str, torch.Tensor]:
        device = his_traj.device
        dtype = self._dtype

        his_traj = his_traj.to(dtype=dtype)
        status_feature = status_feature.to(dtype=dtype)
        gt_norm = self.norm_odo(gt_actions.to(dtype=dtype))

        # 1. Run experts
        step_features, scene_ctx = self._run_experts(vlm_bag, device)

        # 2. Build conditioning
        his_and_ego = self._build_conditioning(his_traj, status_feature)

        # 3. Diffusion + decode
        diff_loss, traj_per_expert = self._diffusion_forward(
            step_features, scene_ctx, his_and_ego, gt_norm,
        )

        # 4. Per-expert ADE metrics (monitoring only)
        other_log: dict[str, torch.Tensor] = {}
        if len(self._action_types) > 1:
            with torch.no_grad():
                ades = torch.stack([
                    torch.norm(traj_per_expert[k].to(dtype=dtype) - gt_norm, dim=-1).mean(dim=-1)
                    for k in self._action_types
                ], dim=1)  # [B, N]
                for i, k in enumerate(self._action_types):
                    other_log[f"ade_mean_{k}"] = ades[:, i].mean()

        # 5. Learned fusion loss (traj detached, gradient only to fusion_weights)
        if self._fusion_mode == "per_step":
            H = self.action_horizon
            w = F.softmax(self.fusion_weights_per_step, dim=0)  # [N, H]
            traj_stacked = torch.stack(
                [traj_per_expert[k].detach() for k in self._action_types], dim=1
            )  # [B, N, H, 3]
            fused_traj = (w[None, :, :, None] * traj_stacked).sum(dim=1)  # [B, H, 3]
            fusion_loss = F.mse_loss(fused_traj, gt_norm)

            with torch.no_grad():
                for i, k in enumerate(self._action_types):
                    other_log[f"fusion_weight_{k}"] = w[i].mean().detach()
                    for h in range(H):
                        other_log[f"fusion_weight_{k}_step{h}"] = w[i, h].detach()
        else:
            w = F.softmax(self.fusion_weights, dim=0)
            traj_stacked = torch.stack(
                [traj_per_expert[k].detach() for k in self._action_types], dim=1
            )  # [B, N, H, 3]
            fused_traj = (w[None, :, None, None] * traj_stacked).sum(dim=1)  # [B, H, 3]
            fusion_loss = F.mse_loss(fused_traj, gt_norm)

            with torch.no_grad():
                for i, k in enumerate(self._action_types):
                    other_log[f"fusion_weight_{k}"] = w[i].detach()

        loss_dict = {
            "diff_loss": diff_loss,
            "fusion_loss": fusion_loss,
        }

        if self._goal_loss_enabled:
            traj_goal_stacked = torch.stack(
                [traj_per_expert[k] for k in self._action_types], dim=1
            )  # [B, N, H, 3]
            if self._fusion_mode == "per_step":
                goal_fused_traj = (
                    w[None, :, :, None].to(dtype=traj_goal_stacked.dtype) * traj_goal_stacked
                ).sum(dim=1)
            else:
                goal_fused_traj = (
                    w[None, :, None, None].to(dtype=traj_goal_stacked.dtype) * traj_goal_stacked
                ).sum(dim=1)

            if self._goal_loss_xy_only:
                goal_loss = F.mse_loss(goal_fused_traj[:, -1, :2], gt_norm[:, -1, :2])
                with torch.no_grad():
                    other_log["goal_l2_xy"] = torch.norm(
                        goal_fused_traj[:, -1, :2] - gt_norm[:, -1, :2], dim=-1
                    ).mean()
            else:
                goal_loss = F.mse_loss(goal_fused_traj[:, -1], gt_norm[:, -1])
                with torch.no_grad():
                    other_log["goal_l2"] = torch.norm(
                        goal_fused_traj[:, -1] - gt_norm[:, -1], dim=-1
                    ).mean()
            loss_dict["goal_loss"] = goal_loss

        return {
            "loss": loss_dict,
            "other_log": other_log,
        }

    # ════════════════════════════════════════════════════════════════════
    # Inference
    # ════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def predict(self, vlm_bag, his_traj, status_feature) -> torch.Tensor:
        device = his_traj.device
        dtype = self._dtype

        his_traj = his_traj.to(dtype=dtype)
        status_feature = status_feature.to(dtype=dtype)

        step_features, scene_ctx = self._run_experts(vlm_bag, device)

        his_and_ego = self._build_conditioning(his_traj, status_feature)
        B = his_traj.shape[0]
        H = self.action_horizon
        N = len(self._action_types)

        z_action = torch.randn(B, N * H, 3, device=device, dtype=dtype)

        dt = 1.0 / self.num_inference_steps
        for step in range(self.num_inference_steps):
            cur_t = step / self.num_inference_steps
            t = torch.full((B,), cur_t, device=device, dtype=dtype)
            t_idx = int(cur_t * self.num_train_timesteps)
            if t_idx >= self.num_train_timesteps:
                t_idx = self.num_train_timesteps - 1
            t_idx_t = torch.full((B,), t_idx, device=device, dtype=torch.long)
            t_emb = self.timestep_embedder(self.time_proj(t_idx_t).to(self._dtype))

            traj_pred = self._decode_noisy_action(z_action, step_features, his_and_ego, t_emb, scene_ctx)

            one_minus_t = (1.0 - t).clamp(min=1e-6).unsqueeze(-1).unsqueeze(-1)
            v = (traj_pred - z_action) / one_minus_t
            z_action = z_action + dt * v

        all_trajs = z_action.reshape(B, N, H, 3)
        if self._fusion_mode == "per_step":
            w = F.softmax(self.fusion_weights_per_step, dim=0)  # [N, H]
            best_traj = (w[None, :, :, None] * all_trajs).sum(dim=1)
        else:
            w = F.softmax(self.fusion_weights, dim=0)
            best_traj = (w[None, :, None, None] * all_trajs).sum(dim=1)

        return self.denorm_odo(best_traj)
