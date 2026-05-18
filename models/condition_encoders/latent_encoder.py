"""Latent condition encoder: projects external embeddings (e.g. VLM action tokens)."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.registry import CONDITION_REGISTRY
from utils.utils import cfg_get
from .base import ConditionEncoder


def _build_mlp(in_dim: int, out_dim: int, num_layers: int = 2) -> nn.Sequential:
    if num_layers <= 1:
        return nn.Sequential(nn.Linear(in_dim, out_dim))
    layers: list[nn.Module] = []
    hidden = (in_dim + out_dim) // 2
    for i in range(num_layers):
        d_in = in_dim if i == 0 else hidden
        d_out = out_dim if i == num_layers - 1 else hidden
        layers.append(nn.Linear(d_in, d_out))
        if i < num_layers - 1:
            layers.append(nn.LayerNorm(d_out))
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


@CONDITION_REGISTRY.register("latent")
class LatentConditionEncoder(ConditionEncoder):
    """Projects external latent vectors to the expected ``encoder_hidden_states`` dim.

    The input dict must contain ``"condition_latent"`` of shape ``[B, N, input_dim]``.
    If ``pad_to_seq_len > 0``, the output is zero-padded along the sequence axis.

    Config example (under ``model.condition``):
        name: latent
        input_dim: 1536
        output_dim: 4096
        num_layers: 2
        pad_to_seq_len: 512   # 0 means no padding
    """

    def __init__(self, cfg, **kwargs):
        super().__init__()
        # ``cfg`` is the ``model.condition`` subtree (same convention as :class:`TextConditionEncoder`).
        nested = cfg_get(cfg, "condition", None)
        cond_cfg = nested if isinstance(nested, dict) else cfg
        in_dim = int(cfg_get(cond_cfg, "input_dim", 1536))
        out_dim = int(cfg_get(cond_cfg, "output_dim", 4096))
        n_layers = int(cfg_get(cond_cfg, "num_layers", 2))
        self._pad_seq = int(cfg_get(cond_cfg, "pad_to_seq_len", 0))

        self.projection = _build_mlp(in_dim, out_dim, n_layers)
        self._output_dim = out_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        latent = inputs["condition_latent"]  # [B, N, in_dim]
        if latent.shape[0] != batch_size:
            raise ValueError(
                f"condition_latent batch={latent.shape[0]} != expected {batch_size}"
            )
        out = self.projection(latent.to(device))  # [B, N, out_dim]
        if self._pad_seq > 0 and out.shape[1] < self._pad_seq:
            pad = torch.zeros(
                batch_size,
                self._pad_seq - out.shape[1],
                self._output_dim,
                device=device,
                dtype=out.dtype,
            )
            out = torch.cat([out, pad], dim=1)
        return out
