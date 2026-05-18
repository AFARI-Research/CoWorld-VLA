"""Abstract base class for all condition encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class ConditionEncoder(ABC, nn.Module):
    """Produces ``encoder_hidden_states`` for the diffusion transformer.

    Every subclass must implement :meth:`forward` which returns a tensor of
    shape ``[B, seq_len, dim]`` that will be passed to
    ``WanTransformer3DModel(..., encoder_hidden_states=...)``.
    """

    @abstractmethod
    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return condition embeddings ``[B, seq_len, dim]``."""
        ...

    @property
    def output_dim(self) -> int:
        """Dimension of the last axis of the output tensor."""
        raise NotImplementedError

    def build_text_prompts(
        self, inputs: dict, batch_size: int
    ) -> Optional[list[str]]:
        """Natural-language prompts for visualization (text encoders only); default ``None``."""
        return None
