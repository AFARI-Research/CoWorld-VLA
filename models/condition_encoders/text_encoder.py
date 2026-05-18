"""Text condition encoder using frozen UMT5 (from Wan2.2-TI2V-5B)."""

from __future__ import annotations

import html
import os
import re
from typing import Optional

import torch
from transformers import AutoTokenizer, UMT5EncoderModel

from models.registry import CONDITION_REGISTRY
from utils.utils import cfg_get, cfg_model_torch_dtype
from .base import ConditionEncoder

try:
    import ftfy
except ImportError:
    ftfy = None

_FIXED_TEXT_MAX_SEQ_LEN = 512


def _prompt_clean(text: str) -> str:
    if ftfy is not None:
        text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Natural-language strings for the 4 driving-command classes (0=left/1=straight/2=right/3=unknown).
# Matches NuPlan's geometric ``infer_navigation_command`` output and NAVSIM's ``driving_command`` one-hot.
_NAV_CMD_TEXT = ["turn left", "go straight", "turn right", "unknown"]


def _speed_bucket(speed_mps: float) -> str:
    """Coarse speed phrase, phrased so it fits after ``"The ego vehicle is "``.

    Each return value is a valid completion of that subject (e.g.
    ``"The ego vehicle is driving at moderate speed."``).
    """
    if speed_mps < 1.0:
        return "nearly stopped"
    if speed_mps < 5.0:
        return "driving slowly"
    if speed_mps < 10.0:
        return "driving at moderate speed"
    return "driving at high speed"


def compose_scene_and_trajectory_prompt(
    scene_description: str,
    future_traj_f3: torch.Tensor,
    step_seconds: float,
    navigation_command: Optional[int] = None,
    ego_speed: Optional[float] = None,
) -> str:
    """Compose a UMT5-friendly prompt with scene + speed phrase + command + future trajectory.

    Structure::

        <scene>. The ego vehicle is <speed_phrase>. Command: <cmd>. Starting at (+0.00, +0.00), future trajectory: [(x,y), ...].

    Design rationale:

    - **Speed phrase only (no motion/direction phrase):** the old ``_infer_motion_description``
      used hard thresholds on the *future* trajectory (e.g. ``total_heading > 0.3 rad → "turning left"``)
      which (a) duplicates the ``Command:`` signal in steady-state turns and (b) can contradict
      ``Command:`` during the *intention* phase (driver commanded to turn but the next 8 frames
      haven't started turning yet). ``Command`` is kept as the sole qualitative direction signal.
      ``_speed_bucket`` is kept because speed is orthogonal to direction and helps UMT5 activate
      matching caption priors.
    - **``Command:`` is the only direction phrase**, driven by ``navigation_command`` (the strategic intent).
    - **Future-only ``Starting at (0,0), future trajectory: [...]``**: local ego frame always has
      the current ego at origin, so explicitly telling the model "you start at (0, 0)" anchors
      the coordinate system; history points are omitted (they'd only repeat what the input image shows).
    - **``(x, y)`` with 2 decimals, no heading**: heading's information content is already carried
      by the polyline shape + ``Command``; UMT5 tokenises small floats poorly anyway.

    Args:
        scene_description: Static scene caption (e.g. from config ``text_prompt``).
        future_traj_f3: ``[F, 3]`` future poses in local ego frame.
        step_seconds: Inter-sample interval, used as fallback when ``ego_speed`` is ``None``.
        navigation_command: Integer 0/1/2/3 (left/straight/right/unknown). ``None`` skips the ``Command:`` clause.
        ego_speed: Instantaneous ego speed in m/s (rear-axle body-frame ``‖(vx, vy)‖``). ``None``
            falls back to ``‖future[-1]‖ / (step_seconds * F)``.
    """
    t = future_traj_f3.detach().float().cpu()
    if ego_speed is None:
        total_dist = float(torch.norm(t[-1, :2]))
        n_steps = t.shape[0]
        ego_speed = total_dist / max(step_seconds * n_steps, 1e-6)

    scene_base = (scene_description or "").strip() or "Front-view dashcam driving video"
    scene = f"{scene_base}. The ego vehicle is {_speed_bucket(float(ego_speed))}."

    if navigation_command is not None:
        idx = max(0, min(int(navigation_command), len(_NAV_CMD_TEXT) - 1))
        scene = f"{scene} Command: {_NAV_CMD_TEXT[idx]}."

    fut_parts = [
        f"({float(t[i, 0]):+.2f}, {float(t[i, 1]):+.2f})" for i in range(t.shape[0])
    ]
    traj_str = ", ".join(fut_parts)

    return f"{scene} Starting at (+0.00, +0.00), future trajectory: [{traj_str}]."


@CONDITION_REGISTRY.register("text")
class TextConditionEncoder(ConditionEncoder):
    """Frozen UMT5 text encoder extracted from Wan2.2-TI2V-5B.

    Supports optional trajectory-to-text: when the input dict contains
    ``future_trajectory [B,F,3]`` the trajectory is serialized into the
    prompt before encoding.
    """

    def __init__(self, cfg, **kwargs):
        super().__init__()
        # ``cfg`` is the ``model.condition`` subtree (flat keys: name, text_prompt, ...).
        pretrained = kwargs.get(
            "pretrained_model_path",
            cfg_get(cfg, "pretrained_model_path", "Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
        )
        nested = cfg_get(cfg, "condition", None)
        prompt_src = nested if isinstance(nested, dict) else cfg
        self._scene_prompt = str(cfg_get(prompt_src, "text_prompt", "") or "").strip()

        torch_dtype = kwargs.get("torch_dtype")
        if torch_dtype is None:
            torch_dtype = cfg_model_torch_dtype(cfg)

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, subfolder="tokenizer")
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            pretrained, subfolder="text_encoder", torch_dtype=torch_dtype,
        )
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)

        self._text_dim: int = self.text_encoder.config.d_model

    @property
    def output_dim(self) -> int:
        return self._text_dim

    def train(self, mode: bool = True):
        # Frozen UMT5: ignore ``mode`` so WanWorldModel.train(True) never marks this submodule training.
        super().train(False)
        self.text_encoder.eval()
        return self

    def build_text_prompts(self, inputs: dict, batch_size: int) -> list[str]:
        """Same string list as used inside :meth:`forward` / ``encode_prompt``.

        Plumbs per-sample ``navigation_command`` and instantaneous ``ego_speed`` into
        :func:`compose_scene_and_trajectory_prompt`. Falls back gracefully when any
        field is absent (e.g. pure-caption inference).
        """
        ft = inputs.get("future_trajectory")
        has_ft = (
            isinstance(ft, torch.Tensor)
            and ft.dim() == 3
            and ft.shape[0] == batch_size
        )
        if not has_ft:
            return [self._scene_prompt] * batch_size

        dt_t = inputs.get("trajectory_interval_s")
        if isinstance(dt_t, torch.Tensor) and dt_t.numel() > 0:
            dt = float(dt_t.reshape(-1)[0].item())
        else:
            dt = 0.5

        nav = inputs.get("navigation_command")
        spd = inputs.get("ego_speed")
        es = inputs.get("ego_status")

        def _pick_nav(b: int) -> Optional[int]:
            if nav is None:
                return None
            v = nav[b] if isinstance(nav, torch.Tensor) and nav.dim() > 0 else nav
            return int(v.item()) if isinstance(v, torch.Tensor) else int(v)

        def _pick_speed(b: int) -> Optional[float]:
            if spd is not None:
                v = spd[b] if isinstance(spd, torch.Tensor) and spd.dim() > 0 else spd
                return float(v.item()) if isinstance(v, torch.Tensor) else float(v)
            if isinstance(es, torch.Tensor) and es.dim() == 2 and es.shape[-1] >= 6:
                vx = float(es[b, 4].item())
                vy = float(es[b, 5].item())
                return (vx * vx + vy * vy) ** 0.5
            return None

        return [
            compose_scene_and_trajectory_prompt(
                self._scene_prompt,
                ft[b],
                dt,
                navigation_command=_pick_nav(b),
                ego_speed=_pick_speed(b),
            )
            for b in range(batch_size)
        ]

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: str | list[str],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        text_dim = self._text_dim
        max_len = _FIXED_TEXT_MAX_SEQ_LEN
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        prompts = [_prompt_clean(p) for p in prompts]
        _dbg = os.environ.get("debug")
        if _dbg not in (None, "", "0", "false", "False"):
            for i, p in enumerate(prompts):
                print(f"[debug] world model text condition [sample {i}]:\n{p}", flush=True)
        if device is None:
            device = next(self.text_encoder.parameters()).device
        te = self.text_encoder.to(device)
        batch = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=max_len,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        out = te(input_ids, attention_mask).last_hidden_state
        seq_lens = attention_mask.gt(0).sum(dim=1).long()
        outs: list[torch.Tensor] = []
        for i, sl in enumerate(seq_lens.tolist()):
            e = out[i, :sl]
            pad = torch.zeros(max_len - sl, text_dim, device=device, dtype=e.dtype)
            outs.append(torch.cat([e, pad], dim=0))
        return torch.stack(outs, dim=0)

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        prompts = self.build_text_prompts(inputs, batch_size)
        return self.encode_prompt(prompts, device=device)
