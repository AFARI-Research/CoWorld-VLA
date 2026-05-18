"""MLP action head: last traj_action hidden → trajectory points.

Migrated from :class:`~models.vlm_worldmodel.TrajPredMLP`.
"""

import torch
import torch.nn as nn


class MLPActionHead(nn.Module):
    """Simple MLP: last traj_action hidden → trajectory points."""

    def __init__(self, cfg):
        super().__init__()
        hidden_dim = int(cfg["input_feature_dim"])
        num_points = int(cfg["num_points"])
        point_dim = int(cfg["point_dim"])
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
