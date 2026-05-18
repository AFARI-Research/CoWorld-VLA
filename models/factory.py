"""Model factory — builds models via the registry.

Registered models (``model.name`` in YAML):
  - WanWorldModel           — stage 1: WAN world model, text/latent condition
  - WmVlmJoint                — stage 2: Qwen3-VL + WAN joint
  - AeActtokenVla — stage 3 (requires ``freeze_vlm``; optional ``vlm_conditioning``)
"""

from __future__ import annotations

import torch.nn as nn

from models.registry import MODEL_REGISTRY
from utils.utils import cfg_get

import models.worldmodelbase  # trigger @register
import models.vlm_worldmodel  # trigger @register
import models.trajectory_model  # trigger @register


def build_world_model(config) -> nn.Module:
    """Build a model from config via ``MODEL_REGISTRY``.

    Expects ``config["model"]["name"]`` to select the model class.
    """
    model_cfg = cfg_get(config, "model", None)

    if model_cfg is None:
        raise ValueError("config must have a 'model' section with a 'name' field")

    return MODEL_REGISTRY.build(model_cfg, full_cfg=config)
