"""ActionHead router: dispatch to MLP, Recog, or HMEF action head."""

from typing import Any, Dict

import torch
import torch.nn as nn

from models.action_head.mlp_action_head import MLPActionHead
from models.action_head.recog.recog_action_head import RecogActionHead, RecogActionHeadConfig
from models.action_head.hmef.hmef_action_head import HMEFActionHead


class ActionHead(nn.Module):
    """Route action head by ``cfg.name``.

    Config keys (shared):
        name: "mlp" | "recog" | "hmef"
        input_feature_dim: int  — injected from vlm.hidden_size

    ``"mlp"`` extra keys:
        num_points: int = 8
        point_dim: int = 3

    ``"recog"`` / ``"hmef"`` extra keys:
        input_embedding_dim, hidden_size, action_dim, action_horizon,
        sampling_method, num_inference_steps, num_heads, head_dim,
        num_layers, dit_output_dim, interleave_attention, ...

    ``"hmef"`` only:
        hmef_hidden_dim, expert_num_layers, expert_num_heads, expert_head_dim,
        fusion_num_layers, fusion_num_heads, fusion_head_dim,
        cls_grad_detach, cls_aux_loss_weight, ...
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.name = str(cfg.get("name", "mlp"))

        if self.name == "mlp":
            self.head = MLPActionHead(cfg)
        elif self.name == "recog":
            recog_cfg = RecogActionHeadConfig(
                input_feature_dim=int(cfg["input_feature_dim"]),
                input_embedding_dim=int(cfg.get("input_embedding_dim", 384)),
                hidden_size=int(cfg.get("hidden_size", 512)),
                action_dim=int(cfg.get("action_dim", 3)),
                action_horizon=int(cfg.get("action_horizon", 8)),
                add_pos_embed=bool(cfg.get("add_pos_embed", True)),
                sampling_method=str(cfg.get("sampling_method", "flow")),
                num_inference_steps=int(cfg.get("num_inference_steps", 5)),
                num_heads=int(cfg.get("num_heads", 8)),
                head_dim=int(cfg.get("head_dim", 48)),
                num_layers=int(cfg.get("num_layers", 16)),
                dit_output_dim=int(cfg.get("dit_output_dim", 512)),
                interleave_attention=bool(cfg.get("interleave_attention", True)),
                noise_beta_alpha=float(cfg.get("noise_beta_alpha", 1.5)),
                noise_beta_beta=float(cfg.get("noise_beta_beta", 1.0)),
                noise_s=float(cfg.get("noise_s", 0.999)),
                num_timestep_buckets=int(cfg.get("num_timestep_buckets", 1000)),
                num_train_timesteps=int(cfg.get("num_train_timesteps", 100)),
                ddim_eta=float(cfg.get("ddim_eta", 0.0)),
            )
            self.head = RecogActionHead(recog_cfg)
        elif self.name == "hmef":
            self.head = HMEFActionHead(cfg)
        else:
            raise ValueError(f"Unknown action head name: {self.name!r}")

    def forward(self, inputs: Dict[str, Any]):
        if self.name == "mlp":
            return self.head(inputs["mlp_input"])
        elif self.name == "recog":
            return self.head(
                vlm_bag=inputs["vlm_bag"],
                his_traj=inputs["his_traj"],
                status_feature=inputs["status"],
                gt_actions=inputs["gt"],
            )
        elif self.name == "hmef":
            # HMEFActionHead returns {"loss": dict, "other_log": dict}.
            return self.head(
                vlm_bag=inputs["vlm_bag"],
                his_traj=inputs["his_traj"],
                status_feature=inputs["status"],
                gt_actions=inputs["gt"],
            )
        else:
            raise ValueError(f"Unknown action head name: {self.name!r}")

    @torch.no_grad()
    def predict(self, inputs: Dict[str, Any]) -> torch.Tensor:
        if self.name == "mlp":
            return self.head(inputs["mlp_input"])
        elif self.name in ("recog", "hmef"):
            return self.head.predict(
                vlm_bag=inputs["vlm_bag"],
                his_traj=inputs["his_traj"],
                status_feature=inputs["status"],
            )
        else:
            raise ValueError(f"Unknown action head name: {self.name!r}")
