"""Small data helpers used by the inference-only CoWorld branch."""

from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch


def nuplan_collate_fn(batch: list) -> dict:
    """Stack samples while preserving variable-length cached VLM hidden states."""
    import torch.nn.utils.rnn as rnn_utils

    elem = batch[0]
    result: dict[str, Any] = {}
    for key in elem:
        vals = [d[key] for d in batch]
        if key == "vlm_last_hidden" and isinstance(vals[0], torch.Tensor):
            padded = rnn_utils.pad_sequence(vals, batch_first=True, padding_value=0.0)
            result[key] = padded
            lengths = torch.tensor([v.shape[0] for v in vals], dtype=torch.long)
            max_len = padded.shape[1]
            result["vlm_last_hidden_mask"] = (
                torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
            )
        elif isinstance(vals[0], torch.Tensor):
            try:
                result[key] = torch.stack(vals, dim=0)
            except RuntimeError:
                result[key] = vals
        elif isinstance(vals[0], (int, float)):
            result[key] = torch.tensor(vals)
        elif isinstance(vals[0], str):
            result[key] = vals
        else:
            result[key] = vals
    return result


def visualize_reconstruction(
    input_frames: torch.Tensor,
    pred_frames: torch.Tensor,
    save_path: Optional[str] = None,
    num_samples: int = 4,
    resolution: Optional[Tuple[int, int]] = None,
    sample_ids: Optional[object] = None,
    text_labels: Optional[List[str]] = None,
    future_label: str = "pred",
    gt_frames: Optional[torch.Tensor] = None,
    compare_label: str = "gt",
) -> None:
    """Render input and predicted frames for optional debugging/visualization."""
    inp = input_frames.detach()
    prd = pred_frames.detach()
    if inp.dim() == 4:
        inp = inp.unsqueeze(1)
    if prd.dim() == 4:
        prd = prd.unsqueeze(1)

    batch_size, num_input_frames, _, image_h, image_w = inp.shape
    if num_input_frames < 1:
        raise ValueError(f"input_frames needs T>=1, got T={num_input_frames}")
    if prd.shape[0] != batch_size:
        raise ValueError(f"pred_frames batch {prd.shape[0]} != input_frames batch {batch_size}")
    num_pred_frames = prd.shape[1]
    if num_pred_frames < 1:
        raise ValueError("pred_frames must have at least 1 frame")

    gt: Optional[torch.Tensor] = None
    if gt_frames is not None:
        gt = gt_frames.detach()
        if gt.dim() == 4:
            gt = gt.unsqueeze(1)
        if gt.shape[0] != batch_size:
            raise ValueError(f"gt_frames batch {gt.shape[0]} != input_frames batch {batch_size}")
        if gt.shape[1] != num_pred_frames:
            raise ValueError(f"gt_frames T={gt.shape[1]} must match pred_frames T={num_pred_frames}")
    compare_mode = gt is not None

    n_export = min(int(num_samples), batch_size)
    if n_export <= 0:
        return

    input_frames = inp[:n_export].to("cpu", dtype=torch.float32).mul(0.5).add(0.5).clamp(0.0, 1.0)
    pred_frames = prd[:n_export].to("cpu", dtype=torch.float32).mul(0.5).add(0.5).clamp(0.0, 1.0)
    gt_t: Optional[torch.Tensor] = None
    if gt is not None:
        gt_t = gt[:n_export].to("cpu", dtype=torch.float32).mul(0.5).add(0.5).clamp(0.0, 1.0)

    ids_list: Optional[list] = None
    if sample_ids is not None:
        if isinstance(sample_ids, torch.Tensor):
            ids_list = sample_ids.detach()[:n_export].flatten().cpu().tolist()
        elif isinstance(sample_ids, (list, tuple)):
            ids_list = list(sample_ids[:n_export])

    num_panels = num_input_frames + num_pred_frames
    dpi = 100
    text_height_in = 1.2 if text_labels else 0.0

    if resolution is not None:
        width_px, height_px = int(resolution[0]), int(resolution[1])
        fig_w_in = max(width_px / dpi, 0.1)
        if compare_mode:
            fig_w_in = max(fig_w_in * 1.9, 3.2)
        fig_h_in = (height_px / dpi) * max(num_panels / 3, 0.45) + text_height_in
    else:
        row_img_in = max(int(image_h) / dpi, 0.75)
        fig_w_in = max(int(image_w) / dpi + 0.2, 2.0)
        if compare_mode:
            fig_w_in = max(fig_w_in * 1.95, 3.2)
        fig_h_in = (row_img_in + 0.36) * num_panels + 0.45 + text_height_in
        max_side_in = 40.0
        if fig_w_in > max_side_in or fig_h_in > max_side_in:
            scale = min(max_side_in / fig_w_in, max_side_in / fig_h_in)
            fig_w_in *= scale
            fig_h_in *= scale

    input_titles = [f"frame {idx + 1}" for idx in range(num_input_frames)]

    def _frame_title(label: str, idx: int) -> str:
        return label if num_pred_frames == 1 else f"{label} {idx + 1}"

    for sample_idx in range(n_export):
        fig = plt.figure(figsize=(fig_w_in, fig_h_in), layout="constrained")
        grid = fig.add_gridspec(num_panels, 2, hspace=0.22, wspace=0.06)

        for panel_idx in range(num_panels):
            if panel_idx < num_input_frames:
                ax = fig.add_subplot(grid[panel_idx, :])
                image = input_frames[sample_idx, panel_idx]
                title = input_titles[panel_idx]
            elif gt_t is not None:
                pred_idx = panel_idx - num_input_frames
                ax_l = fig.add_subplot(grid[panel_idx, 0])
                ax_r = fig.add_subplot(grid[panel_idx, 1])
                ax_l.imshow(pred_frames[sample_idx, pred_idx].permute(1, 2, 0).contiguous().numpy(), aspect="auto")
                ax_l.axis("off")
                ax_l.set_title(_frame_title(future_label, pred_idx), loc="left", fontsize=11, fontweight="medium", pad=8)
                ax_r.imshow(gt_t[sample_idx, pred_idx].permute(1, 2, 0).contiguous().numpy(), aspect="auto")
                ax_r.axis("off")
                ax_r.set_title(_frame_title(compare_label, pred_idx), loc="left", fontsize=11, fontweight="medium", pad=8)
                continue
            else:
                pred_idx = panel_idx - num_input_frames
                ax = fig.add_subplot(grid[panel_idx, :])
                image = pred_frames[sample_idx, pred_idx]
                title = _frame_title(future_label, pred_idx)

            ax.imshow(image.permute(1, 2, 0).contiguous().numpy(), aspect="auto")
            ax.axis("off")
            ax.set_title(title, loc="left", fontsize=11, fontweight="medium", pad=8)

        id_part = ids_list[sample_idx] if ids_list is not None and sample_idx < len(ids_list) else None
        if id_part is not None:
            fig.suptitle(f"dataset id = {id_part}", fontsize=12, fontweight="bold")

        if text_labels is not None and sample_idx < len(text_labels) and text_labels[sample_idx]:
            label = text_labels[sample_idx]
            if len(label) > 300:
                label = label[:297] + "..."
            fig.text(
                0.02,
                0.005,
                label,
                fontsize=6,
                family="monospace",
                verticalalignment="bottom",
                wrap=True,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray", alpha=0.85),
            )

        try:
            if save_path:
                root, ext = os.path.splitext(save_path)
                if not ext:
                    root, ext = save_path, ".png"
                out_dir = os.path.dirname(root)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                suffix = f"id{id_part}" if id_part is not None else f"sample{sample_idx}"
                plt.savefig(f"{root}_{suffix}{ext}", dpi=dpi, bbox_inches="tight", pad_inches=0.2)
            else:
                plt.show()
        finally:
            plt.close(fig)
