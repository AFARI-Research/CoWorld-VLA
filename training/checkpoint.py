"""Checkpoint management: save / load / top-K best tracking."""

from __future__ import annotations

import json
import os
import re
import shutil
from typing import Any, Callable

import torch


MODEL_CHECKPOINT = "model_state_dict.pt"
STATE_FILE = "state.pt"


def _log_state_dict_diff(
    log_fn: Callable[[str], None], miss: list[str], unex: list[str]
) -> None:
    if not miss and not unex:
        log_fn("  Checkpoint 与模型完美适配（无 missing / unexpected keys）。")
        return
    if miss:
        n = len(miss)
        preview = f"{miss[:8]}{'...' if n > 8 else ''}"
        log_fn(f"  Missing keys: {n} — {preview}")
    if unex:
        n = len(unex)
        preview = f"{unex[:8]}{'...' if n > 8 else ''}"
        log_fn(f"  Unexpected keys: {n} — {preview}")


def _write_key_diff_reports(
    ckpt_path: str,
    miss: list[str],
    unex: list[str],
    log_fn: Callable[[str], None],
    *,
    report_dir: str | None = None,
    report_stem: str | None = None,
) -> None:
    """Write full key lists for missing / unexpected ``load_state_dict`` keys.

    Default (``report_dir`` unset): next to the checkpoint — ``missing_keys.txt`` /
    ``unexpected_keys.txt`` inside a checkpoint **directory**, or
    ``<stem>_missing_keys.txt`` / ``<stem>_unexpected_keys.txt`` beside a single
    ``.pt`` file.

    If ``report_dir`` is set, files are written under that directory as
    ``{stem}_missing_keys.txt`` / ``{stem}_unexpected_keys.txt`` where ``stem`` is
    ``report_stem`` if given, else the checkpoint directory basename or ``.pt`` stem.
    """
    p = os.path.normpath(os.path.expanduser(str(ckpt_path)))
    if report_dir:
        d = os.path.normpath(os.path.expanduser(str(report_dir)))
        if report_stem:
            stem = report_stem
        elif os.path.isdir(p):
            stem = os.path.basename(p.rstrip(os.sep)) or "checkpoint"
        else:
            stem = os.path.splitext(os.path.basename(p))[0] or "checkpoint"
        os.makedirs(d, exist_ok=True)
        miss_path = os.path.join(d, f"{stem}_missing_keys.txt")
        unex_path = os.path.join(d, f"{stem}_unexpected_keys.txt")
    elif os.path.isdir(p):
        miss_path = os.path.join(p, "missing_keys.txt")
        unex_path = os.path.join(p, "unexpected_keys.txt")
    else:
        d = os.path.dirname(p)
        stem = os.path.splitext(os.path.basename(p))[0]
        miss_path = os.path.join(d, f"{stem}_missing_keys.txt")
        unex_path = os.path.join(d, f"{stem}_unexpected_keys.txt")
    if miss:
        with open(miss_path, "w", encoding="utf-8") as f:
            f.write("\n".join(miss))
        log_fn(f"  Saved missing keys ({len(miss)}): {miss_path}")
    if unex:
        with open(unex_path, "w", encoding="utf-8") as f:
            f.write("\n".join(unex))
        log_fn(f"  Saved unexpected keys ({len(unex)}): {unex_path}")


def _report_state_dict_diff(
    ckpt_path: str,
    log_fn: Callable[[str], None],
    miss: list[str],
    unex: list[str],
    *,
    write_key_reports: bool,
    key_report_dir: str | None = None,
    key_report_stem: str | None = None,
) -> None:
    _log_state_dict_diff(log_fn, miss, unex)
    if write_key_reports and (miss or unex):
        _write_key_diff_reports(
            ckpt_path,
            miss,
            unex,
            log_fn,
            report_dir=key_report_dir,
            report_stem=key_report_stem,
        )


def _unwrap(obj):
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict, got {type(obj).__name__}")
    for key in ("state_dict", "model_state_dict"):
        if key in obj and isinstance(obj[key], dict):
            return obj[key]
    return obj


def _load_sd_and_state_from_ckpt_path(
    ckpt_path: str,
    map_location: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load ``model_state_dict.pt`` (+ optional ``state.pt``) from a checkpoint dir or a single ``.pt`` file."""
    p = os.path.normpath(os.path.expanduser(str(ckpt_path)))
    state: dict[str, Any] = {}
    if os.path.isdir(p):
        mp = os.path.join(p, MODEL_CHECKPOINT)
        if not os.path.isfile(mp):
            raise FileNotFoundError(f"Missing {MODEL_CHECKPOINT} in {p}")
        sd = _unwrap(torch.load(mp, map_location=map_location, weights_only=False))
        sp = os.path.join(p, STATE_FILE)
        if os.path.isfile(sp):
            raw = torch.load(sp, map_location=map_location, weights_only=False)
            state = raw if isinstance(raw, dict) else {}
    elif os.path.isfile(p):
        sd = _unwrap(torch.load(p, map_location=map_location, weights_only=False))
    else:
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return sd, state


def _should_write_checkpoint(accelerator, save_only_rank0: bool) -> bool:
    return accelerator is None or not save_only_rank0 or bool(accelerator.is_main_process)


def save_checkpoint(
    model,
    scheduler,
    config,
    save_dir: str,
    step: int,
    epoch: int | None = None,
    accelerator=None,
    save_only_rank0: bool = True,
    checkpoint_name: str | None = None,
) -> str:
    """Save model weights + training state into a sub-directory.

    Args:
        checkpoint_name: if set, use this subfolder under ``save_dir`` instead of
            ``epoch_*_step_*`` / ``step_*``. Typical value ``last`` for a rolling
            resume checkpoint (overwrite).

    Returns the checkpoint directory path.
    """
    step_int = int(step)
    if checkpoint_name:
        ckpt_name = checkpoint_name
    elif epoch is not None:
        ckpt_name = f"epoch_{int(epoch):04d}_step_{step_int}"
    else:
        ckpt_name = f"step_{step_int}"
    ckpt_dir = os.path.join(save_dir, ckpt_name)

    if accelerator is not None:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped = getattr(unwrapped, "_orig_mod", unwrapped)
        if getattr(accelerator.state, "deepspeed_plugin", None):
            state_dict = accelerator.get_state_dict(model)
        else:
            state_dict = unwrapped.state_dict()
    else:
        unwrapped = getattr(model, "module", getattr(model, "_orig_mod", model))
        state_dict = unwrapped.state_dict()

    state_payload = {
        "step": step_int,
        "epoch": int(epoch) if epoch is not None else None,
        "config": config,
    }
    if scheduler is not None:
        state_payload["scheduler_state_dict"] = scheduler.state_dict()

    if _should_write_checkpoint(accelerator, save_only_rank0):
        if checkpoint_name == "last" and os.path.isdir(ckpt_dir):
            shutil.rmtree(ckpt_dir, ignore_errors=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(state_dict, os.path.join(ckpt_dir, MODEL_CHECKPOINT))
        torch.save(state_payload, os.path.join(ckpt_dir, STATE_FILE))

    return ckpt_dir


def load_checkpoint(
    ckpt_path: str,
    model,
    map_location: Any = "cpu",
    log_fn: Callable[[str], None] = print,
    write_key_reports: bool = True,
    key_report_dir: str | None = None,
    key_report_stem: str | None = None,
) -> tuple[dict[str, Any], int]:
    """Load model weights and return ``(training_state, start_step)``.

    Compatible with both new directory layout and legacy single-file ``.pt``.
    If ``write_key_reports``, non-empty missing / unexpected key lists are written
    next to the checkpoint, or under ``key_report_dir`` when provided (optional
    ``key_report_stem`` chooses the filename prefix; default is derived from the
    checkpoint path).
    """
    p = os.path.normpath(os.path.expanduser(str(ckpt_path)))
    log_fn(f"Loading checkpoint: {p}")

    if os.path.isdir(p):
        sd, state = _load_sd_and_state_from_ckpt_path(p, map_location)
    elif os.path.isfile(p) and p.endswith(".pt"):
        raw = torch.load(p, map_location=map_location, weights_only=False)
        sd = _unwrap(raw) if isinstance(raw, dict) else {}
        state = {}
    else:
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    miss, unex = model.load_state_dict(sd, strict=False)
    _report_state_dict_diff(
        p,
        log_fn,
        miss,
        unex,
        write_key_reports=write_key_reports,
        key_report_dir=key_report_dir,
        key_report_stem=key_report_stem,
    )

    if state.get("step") is not None:
        start_step = int(state["step"]) + 1
    else:
        m = re.search(r"step_(\d+)", os.path.basename(p.rstrip(os.sep)))
        start_step = int(m.group(1)) + 1 if m else 0

    log_fn(f"  Resuming from step {start_step}")
    if state.get("epoch") is not None:
        log_fn(f"  Checkpoint epoch (completed passes)={int(state['epoch'])}")
    return state, start_step


def _partial_copy_mismatched_shapes(
    model, state_dict: dict, log_fn: Callable[[str], None] = print
):
    """For keys where checkpoint and model shapes differ (e.g. resized embeddings),
    copy the overlapping region in-place and remove the key from state_dict
    so that load_state_dict won't silently skip it.
    """
    model_sd = model.state_dict()
    handled = []
    for k, ckpt_v in list(state_dict.items()):
        if k not in model_sd:
            continue
        target_v = model_sd[k]
        if ckpt_v.shape == target_v.shape:
            continue
        if ckpt_v.dim() != target_v.dim():
            continue
        min_sizes = [min(s1, s2) for s1, s2 in zip(ckpt_v.shape, target_v.shape)]
        slices = tuple(slice(0, s) for s in min_sizes)
        with torch.no_grad():
            model_sd[k][slices] = ckpt_v[slices]
        state_dict.pop(k)
        handled.append(f"{k}: ckpt {list(ckpt_v.shape)} → model {list(target_v.shape)}")
    if handled:
        log_fn(f"  Partial-copy (shape mismatch): {len(handled)} keys")
        for desc in handled:
            log_fn(f"    {desc}")


def load_stage1_into_vlm_worldmodel(
    ckpt_path: str,
    model,
    map_location="cpu",
    log_fn: Callable[[str], None] = print,
    write_key_reports: bool = True,
    key_report_dir: str | None = None,
    key_report_stem: str | None = None,
) -> int:
    """Load a stage-1 WanWorldModel checkpoint into a WmVlmJoint.

    Maps ``transformer.*`` → ``world_model.transformer.*`` etc., and skips
    the old TextConditionEncoder weights (replaced by LatentConditionEncoder).

    Returns the starting step.
    """
    ckpt_norm = os.path.normpath(os.path.expanduser(str(ckpt_path)))
    sd, state = _load_sd_and_state_from_ckpt_path(ckpt_path, map_location)

    remapped = {}
    n_skip = 0
    for k, v in sd.items():
        if k.startswith("condition_encoder."):
            n_skip += 1
            continue
        remapped[f"world_model.{k}"] = v

    log_fn(
        f"[stage1→WmVlmJoint] Remapped {len(remapped)} keys, skipped {n_skip} (condition_encoder.*)"
    )

    _partial_copy_mismatched_shapes(model, remapped, log_fn)
    miss, unex = model.load_state_dict(remapped, strict=False)
    _report_state_dict_diff(
        ckpt_norm,
        log_fn,
        miss,
        unex,
        write_key_reports=write_key_reports,
        key_report_dir=key_report_dir,
        key_report_stem=key_report_stem,
    )

    start_step = int(state.get("step", -1)) + 1
    return start_step


def load_stage2_vlm_into_trajectory_model(
    ckpt_path: str,
    model,
    map_location="cpu",
    log_fn: Callable[[str], None] = print,
    write_key_reports: bool = True,
    key_report_dir: str | None = None,
    key_report_stem: str | None = None,
) -> int:
    """Load weights from a stage-2 WmVlmJoint checkpoint into a stage-3
    trajectory model (``AeActtokenVla``).

    Always maps ``vlm.*`` from the checkpoint onto the target model.

    If the target model has a ``world_model`` submodule (stage-3 with
    ``use_wan: true``) **and** the checkpoint contains ``world_model.*``
    keys, those are loaded as well; otherwise WAN weights are skipped.

    Handles embedding size mismatch: copies the overlapping rows so that
    trained wan_action token embeddings are preserved, and only truly new
    tokens stay random.
    """
    ckpt_norm = os.path.normpath(os.path.expanduser(str(ckpt_path)))
    sd, state = _load_sd_and_state_from_ckpt_path(ckpt_path, map_location)

    vlm_keys = {k: v for k, v in sd.items() if k.startswith("vlm.")}
    load_sd: dict[str, Any] = dict(vlm_keys)

    load_wm = getattr(model, "world_model", None) is not None
    wm_keys: dict[str, Any] = {}
    if load_wm:
        wm_keys = {k: v for k, v in sd.items() if k.startswith("world_model.")}
        load_sd.update(wm_keys)

    n_other = len(sd) - len(load_sd)
    log_fn(
        f"[stage2→stage3] vlm.*={len(vlm_keys)}"
        + (f", world_model.*={len(wm_keys)}" if load_wm else "")
        + f", skipped_other={n_other}"
    )

    _partial_copy_mismatched_shapes(model, load_sd, log_fn)

    miss, unex = model.load_state_dict(load_sd, strict=False)
    _report_state_dict_diff(
        ckpt_norm,
        log_fn,
        miss,
        unex,
        write_key_reports=write_key_reports,
        key_report_dir=key_report_dir,
        key_report_stem=key_report_stem,
    )

    return int(state.get("step", -1)) + 1


class BestCheckpointTracker:
    """Keeps track of the top-K checkpoints by validation metric."""

    def __init__(self, save_dir: str, k: int = 5, mode: str = "min"):
        self.save_dir = save_dir
        self.k = k
        self.mode = mode
        self._higher_is_better = mode == "max"
        self._meta_path = os.path.join(save_dir, "best_checkpoints.json")
        self._entries: list[dict] = []
        if os.path.exists(self._meta_path):
            try:
                with open(self._meta_path) as f:
                    self._entries = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._entries = []
        if self._entries:
            self._sort_entries_inplace()

    def _sort_entries_inplace(self) -> None:
        self._entries.sort(key=lambda e: float(e["metric"]), reverse=self._higher_is_better)

    def _is_better(self, new_val: float, old_val: float) -> bool:
        return new_val < old_val if self.mode == "min" else new_val > old_val

    def would_keep(self, metric_value: float) -> bool:
        """True if a checkpoint with this metric would enter (or stay in) the top-K."""
        if len(self._entries) < self.k:
            return True
        self._sort_entries_inplace()
        worst_metric = float(self._entries[-1]["metric"])
        return self._is_better(float(metric_value), worst_metric)

    def ranked_entries(self) -> list[dict[str, Any]]:
        """Return ``{"path": basename, "metric": float}`` sorted best → worst for the configured mode."""
        if not self._entries:
            return []
        return sorted(
            (dict(e) for e in self._entries),
            key=lambda e: float(e["metric"]),
            reverse=self._higher_is_better,
        )

    def update(
        self,
        ckpt_dir: str,
        metric_value: float,
    ) -> bool:
        """Register a checkpoint. Returns True if it entered the top-K."""
        entry = {"path": os.path.basename(ckpt_dir), "metric": float(metric_value)}

        if len(self._entries) < self.k:
            self._entries.append(entry)
            self._sort_entries_inplace()
            self._persist()
            return True

        self._sort_entries_inplace()
        worst = self._entries[-1]
        if not self._is_better(metric_value, worst["metric"]):
            return False

        evicted = self._entries.pop()
        self._entries.append(entry)
        self._sort_entries_inplace()
        self._persist()

        old_dir = os.path.join(self.save_dir, evicted["path"])
        if os.path.isdir(old_dir):
            shutil.rmtree(old_dir, ignore_errors=True)

        return True

    def _persist(self):
        os.makedirs(self.save_dir, exist_ok=True)
        with open(self._meta_path, "w") as f:
            json.dump(self._entries, f, indent=2)
