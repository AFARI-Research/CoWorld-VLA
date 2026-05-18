"""LightningDiT trajectory model for RecogActionHead."""

from typing import Optional

import torch
import torch.nn as nn

from diffusers.models.embeddings import TimestepEmbedding, Timesteps

from models.action_head.recog.layers import RMSNorm, Attention, SwiGLUFFN, RotaryEmbedding


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        dtype = self.timestep_embedder.linear_1.weight.dtype
        return self.timestep_embedder(self.time_proj(timesteps).to(dtype))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, output_dim: int):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, output_dim)
        self.modulation_proj = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation_proj(conditioning).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.linear(x)


class LightningDiTBlock(nn.Module):
    def __init__(self, dim, num_heads, head_dim, dropout=0.0,
                 cross_attention_dim=None, attention_bias=False, norm_type="rmsnorm"):
        super().__init__()
        norm_cls = RMSNorm if norm_type == "rmsnorm" else lambda d: nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.norm1 = norm_cls(dim)
        self.attn = Attention(query_dim=dim, heads=num_heads, dim_head=head_dim,
                              dropout=dropout, bias=attention_bias, cross_attention_dim=cross_attention_dim)
        self.norm2 = norm_cls(dim)
        self.ffn = SwiGLUFFN(dim, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def forward(
        self,
        hidden_states,
        conditioning,
        encoder_hidden_states=None,
        rotary_embedder=None,
        encoder_attention_mask=None,
    ):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.adaLN_modulation(conditioning).chunk(6, dim=1)
        h = self.norm1(hidden_states) * (1 + scale_a.unsqueeze(1)) + shift_a.unsqueeze(1)
        h = self.attn(
            h,
            encoder_hidden_states=encoder_hidden_states,
            rotary_embedder=rotary_embedder,
            attention_mask=encoder_attention_mask,
        )
        hidden_states = hidden_states + gate_a.unsqueeze(1) * h
        h = self.norm2(hidden_states) * (1 + scale_f.unsqueeze(1)) + shift_f.unsqueeze(1)
        hidden_states = hidden_states + gate_f.unsqueeze(1) * self.ffn(h)
        return hidden_states


class LightningDiT(nn.Module):
    """Trajectory DiT with interleaved self/cross-attention and adaLN."""

    def __init__(self, num_heads=8, head_dim=48, output_dim=512, num_layers=16,
                 dropout=0.0, attention_bias=True, norm_type="rmsnorm", interleave_attention=True):
        super().__init__()
        self.inner_dim = num_heads * head_dim
        self.output_dim = output_dim
        self.interleave_attention = interleave_attention
        self.timestep_encoder = TimestepEncoder(self.inner_dim)
        self.rotary_embedder = RotaryEmbedding(dim=head_dim, max_position_embeddings=8)
        self.transformer_blocks = nn.ModuleList([
            LightningDiTBlock(
                dim=self.inner_dim, num_heads=num_heads, head_dim=head_dim,
                dropout=dropout, attention_bias=attention_bias, norm_type=norm_type,
                cross_attention_dim=self.inner_dim if (idx % 2 != 0 or not interleave_attention) else None,
            ) for idx in range(num_layers)
        ])
        self.final_layer = FinalLayer(self.inner_dim, output_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        def zero_init(m):
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for block in self.transformer_blocks:
            block.adaLN_modulation.apply(zero_init)
        self.final_layer.modulation_proj.apply(zero_init)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        conditioning_features,
        timesteps,
        encoder_attention_mask=None,
    ):
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        conditioning = self.timestep_encoder(timesteps) + conditioning_features.contiguous()
        for idx, block in enumerate(self.transformer_blocks):
            use_cross = not (idx % 2 == 0 and self.interleave_attention)
            block_enc = encoder_hidden_states if use_cross else None
            block_mask = encoder_attention_mask if use_cross else None
            hidden_states = block(
                hidden_states, conditioning, block_enc, self.rotary_embedder, block_mask
            )
        return self.final_layer(hidden_states, conditioning)
