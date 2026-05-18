"""NAVSIM AbstractAgent wrapper for stage-3 trajectory models (``AeActtokenVla``).

This module adapts those models into the
``navsim.agents.abstract_agent.AbstractAgent`` interface so that navsim's
``run_pdm_score.py`` can directly evaluate our models.

Usage (from navsim's evaluation):
    agent = WorldModelNavsimAgent(
        model_name="AeActtokenVla",
        config_path="configs/ae_acttoken_fz_vla.yaml",
        checkpoint_path="/path/to/checkpoint",
    )
    agent.initialize()
    trajectory = agent.compute_trajectory(agent_input)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder

from models import build_world_model


NAVIGATION_COMMANDS_TEXT = ["turn left", "go straight", "turn right", "unknown"]


class WorldModelFeatureBuilder(AbstractFeatureBuilder):
    """Builds features from AgentInput in the unified worldmodel format.

    Field semantics strictly match :class:`NavsimOfficialDataset` (train-time):
      - ``history_images``: last ``num_history_image_frames`` frames of cam_f0,
        decoded by navsim (PIL → RGB numpy), resized with cv2 INTER_AREA,
        normalized to [-1, 1].
      - ``history_trajectory``: last ``num_history_trajectory_steps`` ego_poses
        in the **local frame** centered on the current ego timestep (navsim's
        ``AgentInput.from_scene_dict_list`` already performs this transform,
        so no further conversion is required here).
      - ``ego_status = [driving_command(4) ++ ego_velocity(2) ++ ego_acceleration(2)]``
    """

    def __init__(
        self,
        resize_to: tuple = (256, 512),
        num_history_image_frames: int = 1,
        num_history_trajectory_steps: int = 4,
        skip_images: bool = False,
    ):
        super().__init__()
        self.resize_to = resize_to
        self.num_history_image_frames = int(num_history_image_frames)
        self.num_history_trajectory_steps = int(num_history_trajectory_steps)
        # ``skip_images=True`` 用于下游走 cache 特征、无需 VLM 前向的路径（例如
        # inline navsim eval + use_cached_features=true）。开启后 compute_features
        # 不再解码/resize 任何摄像头图像，每个 token 可省几毫秒 IO。
        self.skip_images = bool(skip_images)
        if self.num_history_image_frames < 1:
            raise ValueError(
                f"num_history_image_frames must be >= 1, got {self.num_history_image_frames}"
            )
        if self.num_history_trajectory_steps < 1:
            raise ValueError(
                f"num_history_trajectory_steps must be >= 1, got {self.num_history_trajectory_steps}"
            )

    def get_unique_name(self) -> str:
        return "worldmodel_feature"

    def _image_tensor_from_camera(self, camera) -> torch.Tensor:
        """Decode + resize one cam_f0 frame; must mirror :meth:`NavsimOfficialDataset._load_image_tensor`."""
        import cv2

        img = camera.image
        if img is None:
            raise RuntimeError(
                "cam_f0.image is None; SensorConfig likely did not request this "
                "history iteration. Ensure WorldModelNavsimAgent.get_sensor_config() "
                "covers all history_image_frames."
            )
        if isinstance(img, (str, Path)):
            img_bgr = cv2.imread(str(img))
            if img_bgr is None:
                raise RuntimeError(f"Failed to load image: {img}")
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(
            img,
            (self.resize_to[1], self.resize_to[0]),
            interpolation=cv2.INTER_AREA,
        )
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return t.mul(2.0).sub(1.0).clamp(-1.0, 1.0)

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        if self.skip_images:
            # 下游走 cache 特征，VLM 前向不会被调用，解码图像纯属浪费 → 直接跳过。
            history_images = None
        else:
            cameras_list = agent_input.cameras
            if len(cameras_list) < self.num_history_image_frames:
                raise RuntimeError(
                    f"agent_input.cameras has {len(cameras_list)} frames, "
                    f"< required num_history_image_frames={self.num_history_image_frames}. "
                    f"Check SceneFilter.num_history_frames and scene dict length."
                )
            hist_cams = cameras_list[-self.num_history_image_frames:]
            hist_image_tensors = [self._image_tensor_from_camera(c.cam_f0) for c in hist_cams]
            history_images = torch.stack(hist_image_tensors, dim=0)

        ego_statuses = agent_input.ego_statuses
        if len(ego_statuses) < self.num_history_trajectory_steps:
            raise RuntimeError(
                f"agent_input.ego_statuses has {len(ego_statuses)} frames, "
                f"< required num_history_trajectory_steps={self.num_history_trajectory_steps}"
            )
        history_poses = [e.ego_pose for e in ego_statuses[-self.num_history_trajectory_steps:]]
        history_traj = torch.tensor(np.array(history_poses), dtype=torch.float32)

        current_status = agent_input.ego_statuses[-1]
        driving_cmd = current_status.driving_command
        if driving_cmd is not None:
            arr = np.asarray(driving_cmd, dtype=np.float32).ravel()
            if arr.size != 4:
                raise ValueError(
                    f"driving_command must have 4 values, got size {arr.size}"
                )
            nav_cmd = int(np.argmax(arr))
        else:
            nav_cmd = 1

        ego_velocity = np.asarray(current_status.ego_velocity, dtype=np.float32).ravel()
        ego_acceleration = np.asarray(current_status.ego_acceleration, dtype=np.float32).ravel()
        if ego_velocity.size != 2 or ego_acceleration.size != 2:
            raise ValueError(
                f"ego_velocity / ego_acceleration must be length 2 each, got "
                f"{ego_velocity.size} and {ego_acceleration.size}"
            )
        ego_speed = float(np.linalg.norm(ego_velocity))

        driving_cmd_oh = np.zeros(4, dtype=np.float32)
        driving_cmd_oh[min(int(nav_cmd), 3)] = 1.0
        ego_status = np.concatenate([driving_cmd_oh, ego_velocity, ego_acceleration])

        out: Dict[str, torch.Tensor] = {
            "history_trajectory": history_traj,
            "navigation_command": torch.tensor(nav_cmd),
            "ego_speed": torch.tensor(ego_speed),
            "ego_status": torch.from_numpy(ego_status),
            "trajectory_interval_s": torch.tensor(0.5),
        }
        if history_images is not None:
            out["history_images"] = history_images
        return out


class WorldModelTargetBuilder(AbstractTargetBuilder):
    def get_unique_name(self) -> str:
        return "trajectory_target"

    def compute_targets(self, scene) -> Dict[str, torch.Tensor]:
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
        future_traj = scene.get_future_trajectory(num_trajectory_frames=8)
        return {"trajectory": torch.tensor(future_traj.poses)}


class WorldModelNavsimAgent(AbstractAgent):
    """NAVSIM agent wrapper for worldmodel trajectory models."""

    def __init__(
        self,
        model_name: str = "AeActtokenVla",
        config_path: str = "configs/ae_acttoken_fz_vla.yaml",
        checkpoint_path: str = None,
        device: str = "cuda",
        vlm_feature_cache_dir: str = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self._device = device
        self.model = None
        self._num_history_image_frames = 1
        self._num_history_trajectory_steps = 4
        self._resize_to = (256, 512)
        self._vlm_feature_cache_dir = vlm_feature_cache_dir
        self._vlm_cache_scene_token = None
        self._vlm_cache_ds = None
        self._vlm_cache_token_to_idx = {}

    def name(self) -> str:
        return f"worldmodel_{self.model_name}"

    def set_vlm_cache_scene_token(self, token: str):
        self._vlm_cache_scene_token = token

    def get_sensor_config(self) -> SensorConfig:
        """Load ``cam_f0`` for **all** history iterations (other sensors disabled).

        navsim's ``SensorConfig`` indexes by scene-filter ``num_history_frames``;
        setting ``cam_f0=True`` is the simplest way to cover every history
        iteration regardless of ``num_history_frames`` (navtest = 4).
        """
        return SensorConfig(
            cam_f0=True,
            cam_l0=False,
            cam_l1=False,
            cam_l2=False,
            cam_r0=False,
            cam_r1=False,
            cam_r2=False,
            cam_b0=False,
            lidar_pc=False,
        )

    def initialize(self) -> None:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(self.config_path)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        data_cfg = cfg_dict.get("data", {}) if isinstance(cfg_dict, dict) else {}
        self._num_history_image_frames = int(data_cfg.get("num_history_image_frames", 1))
        self._num_history_trajectory_steps = int(
            data_cfg.get("num_history_trajectory_steps", 4)
        )
        fh = data_cfg.get("frame_height")
        fw = data_cfg.get("frame_width")
        if fh is not None and fw is not None:
            self._resize_to = (int(fh), int(fw))

        # Enable caching if set
        if self._vlm_feature_cache_dir is not None:
            from datasets.cached_feature_dataset import CachedFeatureDataset
            _cache_dir = self._vlm_feature_cache_dir
            _val_dir = os.path.join(_cache_dir, "val")
            if os.path.exists(_val_dir):
                # NOTE: CachedFeatureDataset(cache_dir, split="val") internally
                # appends split as a subdirectory, so pass the base dir + split.
                self._vlm_cache_ds = CachedFeatureDataset(_cache_dir, split="val")
                self._vlm_cache_token_to_idx = {
                    t: i for i, t in enumerate(self._vlm_cache_ds.tokens)
                }
                print(f"[WorldModelNavsimAgent] VLM cache enabled: {_val_dir} (N={len(self._vlm_cache_ds)})")

        self.model = build_world_model(cfg_dict).to(self._device)

        if self.checkpoint_path:
            from training.checkpoint import load_checkpoint

            load_checkpoint(
                self.checkpoint_path,
                self.model,
                log_fn=print,
                write_key_reports=False,
            )

        self.model.eval()

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [
            WorldModelFeatureBuilder(
                resize_to=self._resize_to,
                num_history_image_frames=self._num_history_image_frames,
                num_history_trajectory_steps=self._num_history_trajectory_steps,
            )
        ]

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [WorldModelTargetBuilder()]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        inputs = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v
                  for k, v in features.items()}
        pred = self.model.predict_trajectory(inputs)
        return {"trajectory": pred.squeeze(0)}

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        self.eval()
        features: Dict[str, torch.Tensor] = {}
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        features = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in features.items()}
        if self._vlm_cache_ds is not None and self._vlm_cache_scene_token is not None:
            idx = self._vlm_cache_token_to_idx.get(self._vlm_cache_scene_token)
            if idx is not None:
                sample = self._vlm_cache_ds[idx]
                for key, val in sample.items():
                    if key.startswith("vlm_"):
                        # Mirror InlineNavsimEvaluator: keep original dtype for
                        # index tensors (int64) — do NOT force bfloat16.
                        if isinstance(val, torch.Tensor):
                            features[key] = val.to(self._device).unsqueeze(0)
                        else:
                            features[key] = val
            else:
                raise FileNotFoundError(
                    f"[WorldModelNavsimAgent] VLM feature cache miss: "
                    f"token={self._vlm_cache_scene_token} not in val split.\n"
                    f"Regenerate val cache or set model.use_cached_features=false."
                )

        from torch.amp import autocast
        device_type = "cuda" if str(self._device).startswith("cuda") else "cpu"
        with torch.no_grad(), autocast(device_type, dtype=torch.bfloat16):
            predictions = self.forward(features)
            poses = predictions["trajectory"].squeeze(0).cpu().numpy()
        return Trajectory(poses.astype(np.float32))
