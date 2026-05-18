"""
``WanWorldModel`` — Wan2.2-TI2V-5B with pluggable condition encoders (stage 1).

Data flow:
  history_images [B,T_h,3,H,W] + future_images [B,T_f,3,H,W]
    → pack time into batch → ``encode_frames`` (VAE + z-score) / ``decode_latents`` (un-z-score + VAE) → unpack
    → concat on time axis → per-token timestep masking
    → WanTransformer3DModel(hidden_states, timestep, encoder_hidden_states)
    → velocity MSE loss on future segment

The condition encoder is selected by ``model.condition.name`` in the config:
  - "text"   : frozen UMT5 (default)
  - "latent" : external embeddings via projection MLP
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchmetrics import MeanMetric

from diffusers import WanTransformer3DModel, AutoencoderKLWan
from diffusers.schedulers import UniPCMultistepScheduler

from datasets.data_utils import visualize_reconstruction
from models.registry import MODEL_REGISTRY, CONDITION_REGISTRY
from training.metrics import psnr
from utils.utils import cfg_get, cfg_model_torch_dtype

import models.condition_encoders  # trigger registration


@MODEL_REGISTRY.register("WanWorldModel")
class WanWorldModel(nn.Module):
    """Wan+VAE world model with pluggable condition encoder.

    Trainable parameters are mainly in ``transformer`` and ``condition_encoder``
    (VAE is always frozen).
    """

    def __init__(self, cfg, **kwargs):
        super().__init__()

        model_cfg = cfg_get(cfg, "model", cfg)
        torch_dtype = cfg_model_torch_dtype(model_cfg)

        pretrained = cfg_get(model_cfg, "pretrained_model_path", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
        load_tr = cfg_get(model_cfg, "load_transformer_pretrained", True)
        self.num_history_frames = int(cfg_get(model_cfg, "num_history_frames", 1))
        self.num_future_frames = int(cfg_get(model_cfg, "num_future_frames", 8))
        self.num_train_timesteps = int(cfg_get(model_cfg, "num_train_timesteps", 1000))
        self.pretrained_model_path = pretrained

        # --- VAE (frozen, eval) — dtype from ``model.dtype`` ---
        self.vae = AutoencoderKLWan.from_pretrained(
            pretrained, subfolder="vae", torch_dtype=torch_dtype,
        )
        self.vae.requires_grad_(False)
        self.vae.eval()

        self.register_buffer(
            "latents_mean",
            rearrange(torch.tensor(self.vae.config.latents_mean), "z -> 1 z 1 1 1"),
        )
        self.register_buffer(
            "latents_std",
            rearrange(torch.tensor(self.vae.config.latents_std), "z -> 1 z 1 1 1"),
        )

        # --- Transformer ---
        if load_tr:
            self.transformer = WanTransformer3DModel.from_pretrained(
                pretrained, subfolder="transformer", torch_dtype=torch_dtype,
            )
        else:
            config = WanTransformer3DModel.load_config(pretrained, subfolder="transformer")
            self.transformer = WanTransformer3DModel.from_config(config).to(torch_dtype)

        self.transformer_patch_size = tuple(self.transformer.config.patch_size)
        if cfg_get(model_cfg, "gradient_checkpointing", True):
            self.transformer.enable_gradient_checkpointing()

        # --- Scheduler (inference only) ---
        self.scheduler = UniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            prediction_type="flow_prediction",
            flow_shift=float(cfg_get(model_cfg, "flow_shift", 5.0)),
            use_flow_sigmas=True,
            time_shift_type="exponential",
        )

        # --- Condition Encoder (pluggable) ---
        cond_cfg = dict(cfg_get(model_cfg, "condition", {"name": "text"}))
        if cfg_get(cond_cfg, "name", None) is None:
            cond_cfg["name"] = "text"

        self.condition_encoder = CONDITION_REGISTRY.build(
            cond_cfg,
            pretrained_model_path=pretrained,
            torch_dtype=torch_dtype,
        )

        self.val_gen_psnr = MeanMetric(sync_on_compute=True, dist_sync_on_step=False)
        self._val_vis_done = False
        self._val_wm_num_inference_steps = int(cfg_get(model_cfg, "validation_num_inference_steps", 20))

    # ------------------------------------------------------------------
    # Ensure frozen parts stay eval
    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()
        self.condition_encoder.train(mode)
        return self

    # ------------------------------------------------------------------
    # VAE encode / decode (Wan + ``latents_mean`` / ``latents_std``; reshape outside)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_frames(self, sample: torch.Tensor) -> torch.Tensor:
        """``AutoencoderKLWan.encode`` → latent mode → z-score (transformer latent space)."""
        vae_dtype = next(self.vae.parameters()).dtype
        sample = sample.to(vae_dtype)
        with torch.autocast(device_type="cuda", enabled=False):
            z = self.vae.encode(sample, return_dict=True).latent_dist.mode()
        return (z - self.latents_mean) / self.latents_std

    @torch.no_grad()
    def decode_latents(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse z-score then ``AutoencoderKLWan.decode``. ``z`` is in normalized latent space."""
        z = z * self.latents_std + self.latents_mean
        vae_dtype = next(self.vae.parameters()).dtype
        z = z.to(vae_dtype)
        with torch.autocast(device_type="cuda", enabled=False):
            return self.vae.decode(z, return_dict=False)[0]

    # ------------------------------------------------------------------
    # Timestep masking
    # ------------------------------------------------------------------
    def _build_per_token_timestep(
        self, timesteps: torch.Tensor, n_frames: int, lat_h: int, lat_w: int,
    ) -> torch.Tensor:
        _, p_h, p_w = self.transformer_patch_size
        ph, pw = lat_h // p_h, lat_w // p_w
        dt = timesteps.dtype
        dev = timesteps.device
        mask = torch.zeros(n_frames, ph, pw, device=dev, dtype=dt)
        mask[self.num_history_frames:].fill_(1)
        flat_mask = mask.flatten()
        return rearrange(timesteps, "b -> b 1") * rearrange(flat_mask, "p -> 1 p")

    # ------------------------------------------------------------------
    # forward (training) — returns dict of named losses
    # ------------------------------------------------------------------
    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, Any]:
        pixel = torch.cat(
            [
                rearrange(inputs["history_images"], "b t c h w -> b c t h w"),
                rearrange(inputs["future_images"], "b t c h w -> b c t h w"),
            ],
            dim=2,
        )
        B = pixel.shape[0]
        device = pixel.device
        n_hist = self.num_history_frames
        n_fut = self.num_future_frames

        # 1. VAE encode (time → batch outside ``encode_frames``; z-score inside)
        n_frames = n_hist + n_fut
        x_pack = rearrange(pixel, "b c t h w -> (b t) c 1 h w")
        all_z = rearrange(
            self.encode_frames(x_pack),
            "(b t) z 1 h w -> b z t h w",
            b=B,
            t=n_frames,
        )
        his_z = all_z[:, :, :n_hist]
        target_z = all_z[:, :, n_hist:]
        _, _, _, lat_h, lat_w = all_z.shape

        # 2. Noise & timestep sampling
        noise = torch.randn_like(target_z)
        t_unit = torch.sigmoid(torch.randn(B, device=device, dtype=target_z.dtype)).clamp_(1e-4, 1.0 - 1e-4)
        sigma = rearrange(t_unit, "b -> b 1 1 1 1")
        t = t_unit * float(self.num_train_timesteps)

        # 3. Flow matching
        noisy = (1.0 - sigma) * target_z + sigma * noise
        model_in = torch.cat([his_z, noisy], dim=2)

        # 4. Per-token timestep
        per_tok_t = self._build_per_token_timestep(t, n_hist + n_fut, lat_h, lat_w)

        # 5. Condition encoding (pluggable)
        tr_dtype = next(self.transformer.parameters()).dtype
        enc_hidden = self.condition_encoder(inputs, B, device).to(dtype=tr_dtype)

        # 6. Transformer forward
        pred = self.transformer(
            hidden_states=model_in.to(tr_dtype),
            timestep=per_tok_t,
            encoder_hidden_states=enc_hidden,
            return_dict=False,
        )[0]

        # 7. Loss: velocity target on future segment only
        pred_gen = pred[:, :, n_hist:]
        target_v = noise - target_z
        flow_loss = F.mse_loss(pred_gen, target_v)

        # 8. Per-sigma bucket stats (detached) → other_log (not in total loss unless listed under loss)
        loss_out: dict[str, torch.Tensor] = {"wan_flow_matching": flow_loss}
        other_log: dict[str, torch.Tensor] = {}
        sigma_flat = rearrange(sigma, "b 1 1 1 1 -> b")
        per_sample_mse = (pred_gen - target_v).pow(2).mean(dim=list(range(1, pred_gen.dim())))
        for lo, hi in [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]:
            mask_b = (sigma_flat >= lo) & (sigma_flat < hi)
            key = f"mse_sigma_{lo:.2f}_{hi:.2f}"
            other_log[key] = per_sample_mse[mask_b].mean().detach() if mask_b.any() else torch.tensor(0.0, device=device)

        return {"loss": loss_out, "other_log": other_log}

    # ------------------------------------------------------------------
    # generate (inference)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        inputs: dict[str, torch.Tensor],
        num_inference_steps: int = 50,
    ) -> torch.Tensor:
        """Generate future frames from history frames. Returns ``[B, n_fut, 3, H, W]``."""
        pixel = rearrange(inputs["history"], "b t c h w -> b c t h w")
        B = pixel.shape[0]
        device = pixel.device
        n_hist = self.num_history_frames
        n_fut = self.num_future_frames

        x_pack = rearrange(pixel, "b c t h w -> (b t) c 1 h w")
        his_z = rearrange(
            self.encode_frames(x_pack),
            "(b t) z 1 h w -> b z t h w",
            b=B,
            t=n_hist,
        )
        _, z_dim, _, lat_h, lat_w = his_z.shape
        tr_dtype = next(self.transformer.parameters()).dtype
        latents = torch.randn(B, z_dim, n_fut, lat_h, lat_w, device=device, dtype=tr_dtype)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        enc_hidden = self.condition_encoder(inputs, B, device).to(dtype=tr_dtype)

        for t in self.scheduler.timesteps:
            full_in = torch.cat([his_z, latents], dim=2)
            per_tok_t = self._build_per_token_timestep(t.expand(B), n_hist + n_fut, lat_h, lat_w)
            out = self.transformer(
                hidden_states=full_in.to(tr_dtype),
                timestep=per_tok_t,
                encoder_hidden_states=enc_hidden,
                return_dict=False,
            )[0]
            noise_pred = out[:, :, n_hist:].to(dtype=latents.dtype)
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        z_pack = rearrange(latents, "b z t h w -> (b t) z 1 h w")
        pixels = rearrange(
            self.decode_latents(z_pack),
            "(b t) c 1 h w -> b c t h w",
            b=B,
            t=n_fut,
        )
        return rearrange(pixels, "b c t h w -> b t c h w")

    # ------------------------------------------------------------------
    # Validation (Trainer.validate): TorchMetrics + ``validation_step``
    # ------------------------------------------------------------------
    def reset_validation_metrics(self) -> None:
        self.val_gen_psnr.reset()
        self._val_vis_done = False

    @torch.no_grad()
    def validation_step(
        self,
        batch: dict,
        batch_idx: int,
        *,
        accelerator,
        vis_save_dir: str | None = None,
        global_step: int = 0,
        vis_num_samples: int = 3,
        vis_every_batch: bool = False,
    ) -> None:
        bs = batch["history_images"].shape[0]

        device = next(self.transformer.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        gen_input = {"history": inputs["history_images"]}
        for key in ("future_trajectory", "trajectory_interval_s", "history_trajectory"):
            if key in inputs:
                gen_input[key] = inputs[key]

        pred_frames = self.generate(
            gen_input,
            num_inference_steps=self._val_wm_num_inference_steps,
        )
        gt_frames = inputs["future_images"]
        dev = pred_frames.device
        for b in range(bs):
            p = psnr(pred_frames[b], gt_frames[b], data_range=2.0)
            self.val_gen_psnr.update(p.detach().to(device=dev, dtype=torch.float32))

        if vis_save_dir and (vis_every_batch or not self._val_vis_done):
            step_dir = os.path.join(vis_save_dir, f"step_{global_step}")
            os.makedirs(step_dir, exist_ok=True)
            stem = os.path.join(
                step_dir, f"b{batch_idx}_rank{accelerator.process_index}"
            )
            n_vis = min(vis_num_samples, bs)
            vis_prompts = self.condition_encoder.build_text_prompts(gen_input, bs)
            text_labels = (
                vis_prompts[:n_vis] if vis_prompts is not None else None
            )
            visualize_reconstruction(
                input_frames=inputs["history_images"][:n_vis],
                pred_frames=pred_frames[:n_vis],
                save_path=f"{stem}_compare.png",
                num_samples=n_vis,
                text_labels=text_labels,
                future_label="pred",
                gt_frames=gt_frames[:n_vis],
                compare_label="gt",
            )
            if not vis_every_batch:
                self._val_vis_done = True

    def compute_validation_metrics(self) -> dict[str, float]:
        v = self.val_gen_psnr.compute()
        self.val_gen_psnr.reset()
        self._val_vis_done = False
        return {"gen_psnr": float(v.detach().cpu())}

