"""Shared layers for the RecogActionHead: RMSNorm, RoPE, Attention, SwiGLU, encoders."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(max_position_embeddings, inv_freq.device)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = position_ids.max().item() + 1
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device)
        cos = self.cos_cached.gather(2, position_ids.unsqueeze(1).unsqueeze(3).expand(-1, -1, -1, self.dim))
        sin = self.sin_cached.gather(2, position_ids.unsqueeze(1).unsqueeze(3).expand(-1, -1, -1, self.dim))
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Attention(nn.Module):
    def __init__(self, query_dim: int, heads: int = 8, dim_head: int = 64,
                 dropout: float = 0.0, bias: bool = False,
                 cross_attention_dim: Optional[int] = None, qk_norm: bool = True):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.num_heads = heads
        self.head_dim = dim_head
        context_dim = cross_attention_dim or query_dim
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(context_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(context_dim, self.inner_dim, bias=bias)
        self.q_norm = RMSNorm(dim_head) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(dim_head) if qk_norm else nn.Identity()
        self.to_out = nn.Sequential(nn.Linear(self.inner_dim, query_dim, bias=True), nn.Dropout(dropout))

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        rotary_embedder=None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Args:
            attention_mask: 可选加性 bias，形状 ``[B, 1, 1, N_k]``（broadcast 到
                ``[B, heads, N_q, N_k]``）。padding 位置填 ``-inf``，真实位置填 ``0``。
                仅在 cross-attention（``encoder_hidden_states is not None``）时生效。
        """
        B, N_q, _ = hidden_states.shape
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        is_self = encoder_hidden_states is None
        q = self.to_q(hidden_states).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        if rotary_embedder is not None:
            pos = torch.arange(N_q, device=hidden_states.device).unsqueeze(0)
            cos, sin = rotary_embedder(hidden_states, pos)
            q = (q * cos) + (rotate_half(q) * sin)
            if is_self:
                k = (k * cos) + (rotate_half(k) * sin)
        attn_bias = None if is_self else attention_mask
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.to_out[-1].p if self.training else 0.0,
        )
        return self.to_out(x.transpose(1, 2).reshape(B, N_q, -1))


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, bias: bool = True):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        exp = -torch.log(torch.tensor(10000.0)) / half
        exp = torch.arange(half, dtype=torch.float32, device=timesteps.device) * exp
        freqs = timesteps.float().unsqueeze(-1) * torch.exp(exp).unsqueeze(0)
        return torch.cat([freqs.sin(), freqs.cos()], dim=-1)


class ActionEncoder(nn.Module):
    def __init__(self, action_dim: int, hidden_size: int):
        super().__init__()
        self.pos_enc = SinusoidalPositionalEncoding(hidden_size)
        self.fc1 = nn.Linear(action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        B, T, _ = actions.shape
        t_exp = timesteps.unsqueeze(1).expand(-1, T) if timesteps.dim() == 1 else timesteps
        ae = self.fc1(actions)
        te = self.pos_enc(t_exp).to(dtype=ae.dtype)
        return self.fc3(self.act(self.fc2(torch.cat([ae, te], dim=-1))))
