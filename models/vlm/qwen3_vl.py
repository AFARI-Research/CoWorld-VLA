"""Qwen3-VL wrapper for driving action-token extraction.

**Token layout in sequence**::

    [non_action_tokens]  <|start_action|>
    [jepa_action_0...N]  (if num_jepa_action_tokens > 0)
    [vggt_action_0...N]  (if num_vggt_action_tokens > 0)
    [wan_action_0...N]   (if num_wan_action_tokens > 0)
    [traj_action_0...N]  (if num_traj_action_tokens > 0)
    <|end_action|>
    [CE text / assistant response]

``<|start_action|>`` and ``<|end_action|>`` are boundary markers.
CE loss starts from the token immediately after ``<|end_action|>``.

**Token extraction modes** (``vlm_conditioning`` / :meth:`extract_tokens`):

Fixed-size (return ``Tensor``, no mask when used alone):
    ``"jepa_action_tokens"`` → ``[B, num_jepa_action_tokens, H]`` (empty when 0)
    ``"vggt_action_tokens"`` → ``[B, num_vggt_action_tokens, H]``
    ``"wan_action_tokens"``  → ``[B, num_wan_action_tokens,  H]``
    ``"traj_action_tokens"`` → ``[B, num_traj_action_tokens, H]``

Variable-size (``forward_extract`` sets ``<mode>`` and ``<mode>_mask``):
    ``"non_action_tokens"``  → tokens before ``<|start_action|>``

:meth:`forward_extract` / :meth:`generate_text` 的 ``vlm_conditioning`` 为**模式名 ``list``**（与
Stage-3 从 YAML 展成 ``[str, ...]`` 同形，见 :class:`AeActtokenVla`），按该列表顺序取各槽。

Shared index computation via :meth:`_compute_content_indices` is used by both the
online forward path and offline cache generation (:meth:`forward_full_hidden`).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PILImage
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.generation.utils import GenerateDecoderOnlyOutput

from omegaconf import ListConfig

from models.vlm.driving_prompt import (
    ACTION_PLACEHOLDER,
    END_ACTION_TOKEN,
    START_ACTION_TOKEN,
    TRAJ_ACTION_TOKEN,
    VLM_DRIVING_SYSTEM_MESSAGE,
    build_action_token_string,
    get_all_jepa_action_token_strings,
    get_all_vggt_action_token_strings,
    get_all_vggt_cam_action_token_strings,
    get_all_wan_action_token_strings,
)

IGNORE_INDEX = -100

# 定长四槽的序列内顺序，与 *Token layout* 一致。Stage-2/3 的 ``vlm_conditioning`` 可整表传入；元素集 = ``_FIXED_SIZE_MODES``。
VLM_FIXED_SIZE_MODES_ORDER: tuple[str, ...] = (
    "jepa_action_tokens",
    "vggt_action_tokens",
    "wan_action_tokens",
    "traj_action_tokens",
)
_FIXED_SIZE_MODES = frozenset(VLM_FIXED_SIZE_MODES_ORDER)
_VARIABLE_SIZE_MODES = frozenset({"non_action_tokens", "visual_tokens"})
# 单键、列表、YAML bool dict 的合法名（定长 + non_action）；dict 展开为列表时保留键写序。
VLM_MODE_KEYS: frozenset[str] = _FIXED_SIZE_MODES | _VARIABLE_SIZE_MODES


class Qwen3VLWrapper(nn.Module):
    """Lightweight wrapper around Qwen3-VL for action-token extraction.

    Training mode keeps ``text_config.use_cache=False``; eval mode enables it.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-VL-2B-Instruct",
        num_jepa_action_tokens: int = 0,
        num_jepa_token_steps: int = 8,
        num_vggt_action_tokens: int = 0,
        num_vggt_token_steps: int = 8,
        num_vggt_cam_action_tokens: int = 0,
        num_wan_action_tokens: int = 24,
        num_wan_token_steps: int = 8,
        num_traj_action_tokens: int = 0,
        max_action_token_slots: int = 64,
        torch_dtype=torch.bfloat16,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            dtype=torch_dtype,
        )
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.tokenizer.padding_side = "left"

        self.hidden_size = self.model.config.text_config.hidden_size
        # Validate divisibility before registering tokens.
        for _name, _total, _steps in (
            ("jepa", num_jepa_action_tokens, num_jepa_token_steps),
            ("vggt", num_vggt_action_tokens, num_vggt_token_steps),
            ("wan",  num_wan_action_tokens,  num_wan_token_steps),
        ):
            if _total > 0 and _total % _steps != 0:
                raise ValueError(
                    f"num_{_name}_action_tokens ({_total}) must be divisible by "
                    f"num_{_name}_token_steps ({_steps})."
                )

        self.num_jepa_action_tokens = num_jepa_action_tokens
        self.num_jepa_token_steps   = num_jepa_token_steps
        self.num_vggt_action_tokens = num_vggt_action_tokens
        self.num_vggt_token_steps   = num_vggt_token_steps
        self.num_vggt_cam_action_tokens = num_vggt_cam_action_tokens
        self.num_wan_action_tokens  = num_wan_action_tokens
        self.num_wan_token_steps    = num_wan_token_steps
        self.num_traj_action_tokens = num_traj_action_tokens

        # Register special tokens:
        #   - Boundary markers: <|start_action|>, <|end_action|>
        #   - Step-indexed types (jepa, vggt, wan): unique token per step
        #   - Traj type: single token <|traj_action|> repeated K times (no index)
        jepa_strs = get_all_jepa_action_token_strings(max_action_token_slots) if num_jepa_action_tokens > 0 else []
        vggt_strs = get_all_vggt_action_token_strings(max_action_token_slots) if num_vggt_action_tokens > 0 else []
        vggt_cam_strs = get_all_vggt_cam_action_token_strings(max_action_token_slots) if num_vggt_cam_action_tokens > 0 else []
        wan_strs  = get_all_wan_action_token_strings(max_action_token_slots) if num_wan_action_tokens > 0 else []

        extra_tokens = (
            [START_ACTION_TOKEN, END_ACTION_TOKEN, TRAJ_ACTION_TOKEN]
            + jepa_strs + vggt_strs + vggt_cam_strs + wan_strs
        )
        self.processor.tokenizer.add_tokens(extra_tokens, special_tokens=True)
        self.model.resize_token_embeddings(len(self.processor.tokenizer))

        def _ids(strings: List[str]) -> List[int]:
            return [self.processor.tokenizer.convert_tokens_to_ids(s) for s in strings]

        self.start_action_token_id: int  = _ids([START_ACTION_TOKEN])[0]
        self.end_action_token_id: int    = _ids([END_ACTION_TOKEN])[0]
        self.traj_action_token_id: int   = _ids([TRAJ_ACTION_TOKEN])[0]   # single id, no list
        self.jepa_action_token_ids: List[int] = _ids(jepa_strs)
        self.vggt_action_token_ids: List[int] = _ids(vggt_strs)
        self.vggt_cam_action_token_ids: List[int] = _ids(vggt_cam_strs)
        self.wan_action_token_ids: List[int]  = _ids(wan_strs)

        self.action_replace_string = build_action_token_string(
            num_jepa_action_tokens=num_jepa_action_tokens,
            num_jepa_token_steps=num_jepa_token_steps,
            num_vggt_action_tokens=num_vggt_action_tokens,
            num_vggt_token_steps=num_vggt_token_steps,
            num_vggt_cam_action_tokens=num_vggt_cam_action_tokens,
            num_wan_action_tokens=num_wan_action_tokens,
            num_wan_token_steps=num_wan_token_steps,
            num_traj_action_tokens=num_traj_action_tokens,
        )

        self._sync_use_cache()

    def _sync_use_cache(self) -> None:
        tc = getattr(self.model.config, "text_config", None)
        if tc is None:
            raise RuntimeError("text_config not found in model config")
        tc.use_cache = not self.training

    def train(self, mode: bool = True) -> "Qwen3VLWrapper":
        super().train(mode)
        self._sync_use_cache()
        return self

    def build_inputs(
        self,
        images: List[PILImage.Image],
        user_prompts: List[str],
        assistant_answers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Build tokenized inputs from images + prompt strings.

        The ``{actions}`` placeholder is replaced with :attr:`action_replace_string`
        which contains START + all enabled action tokens + END.
        No separate traj suffix is appended — everything is inside the placeholder.
        """
        messages_batch = []
        for idx, (img, prompt) in enumerate(zip(images, user_prompts)):
            user_text = prompt.replace(ACTION_PLACEHOLDER, self.action_replace_string)
            content = [
                {"type": "image", "image": img},
                {"type": "text", "text": user_text},
            ]
            msg = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": VLM_DRIVING_SYSTEM_MESSAGE}],
                },
                {"role": "user", "content": content},
            ]
            if assistant_answers is not None:
                msg.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_answers[idx]}],
                })
            messages_batch.append(msg)

        add_gen = assistant_answers is None
        _dbg = os.environ.get("debug")
        if _dbg not in (None, "", "0", "false", "False"):
            for i, msg in enumerate(messages_batch):
                rendered = self.processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=add_gen
                )
                print(f"[debug] vlm training text [sample {i}]:\n{rendered}", flush=True)

        return self.processor.apply_chat_template(
            messages_batch,
            tokenize=True,
            add_generation_prompt=add_gen,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )

    def _compute_content_indices(
        self, content_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute all token boundary indices for a single content sequence (no padding).

        Used by both offline cache generation (:meth:`forward_full_hidden`) and online
        inference, ensuring index semantics are identical in both paths.

        Args:
            content_ids: ``[L_content]`` int64, real tokens only (no padding).

        Returns:
            Dict with:
                ``start_action_idx``:   scalar int64 — position of ``<|start_action|>``.
                ``end_action_idx``:     scalar int64 — position of ``<|end_action|>``.
                ``jepa_action_indices``: ``[Mj]`` int64 (empty when not used).
                ``vggt_action_indices``: ``[Mv]`` int64 (empty when not used).
                ``wan_action_indices``:  ``[N]``  int64 (empty when not used).
                ``traj_action_indices``: ``[K]``  int64 (empty when not used).

        Cache semantics (what each index set covers):
            ``content[:start_action_idx]``       → non_action_tokens
            ``content[jepa_action_indices]``     → jepa_action_tokens
            ``content[vggt_action_indices]``     → vggt_action_tokens
            ``content[wan_action_indices]``      → wan_action_tokens
            ``content[traj_action_indices]``     → traj_action_tokens
            ``content[end_action_idx]``          → end_action (boundary, not in any group)
        """
        dev = content_ids.device

        def _find_unique(token_id: int, name: str) -> torch.Tensor:
            hits = (content_ids == token_id).nonzero(as_tuple=True)[0]
            if hits.numel() != 1:
                raise ValueError(
                    f"_compute_content_indices: expected exactly 1 {name} token, "
                    f"found {hits.numel()}"
                )
            return hits[0]

        def _find_indexed(token_ids: List[int], name: str, expected: int) -> torch.Tensor:
            """Find all positions of step-indexed tokens (jepa/vggt/wan)."""
            if expected == 0 or not token_ids:
                return torch.empty(0, dtype=torch.long, device=dev)
            id_t = torch.tensor(token_ids, device=dev)
            indices = torch.isin(content_ids, id_t).nonzero(as_tuple=True)[0]
            if indices.numel() != expected:
                raise ValueError(
                    f"_compute_content_indices: expected {expected} {name} tokens, "
                    f"found {indices.numel()}"
                )
            return indices

        start_action_idx = _find_unique(self.start_action_token_id, "<|start_action|>")
        end_action_idx   = _find_unique(self.end_action_token_id,   "<|end_action|>")

        # Traj: single repeated token — all K positions share one token id.
        if self.num_traj_action_tokens > 0:
            traj_indices = (content_ids == self.traj_action_token_id).nonzero(as_tuple=True)[0]
            if traj_indices.numel() != self.num_traj_action_tokens:
                raise ValueError(
                    f"_compute_content_indices: expected {self.num_traj_action_tokens} "
                    f"traj_action tokens, found {traj_indices.numel()}"
                )
        else:
            traj_indices = torch.empty(0, dtype=torch.long, device=dev)

        return {
            "start_action_idx":    start_action_idx,
            "end_action_idx":      end_action_idx,
            "jepa_action_indices": _find_indexed(self.jepa_action_token_ids, "jepa_action", self.num_jepa_action_tokens),
            "vggt_action_indices": _find_indexed(self.vggt_action_token_ids, "vggt_action", self.num_vggt_action_tokens),
            "vggt_cam_action_indices": _find_indexed(self.vggt_cam_action_token_ids, "vggt_cam_action", self.num_vggt_cam_action_tokens),
            "wan_action_indices":  _find_indexed(self.wan_action_token_ids,  "wan_action",  self.num_wan_action_tokens),
            "traj_action_indices": traj_indices,
        }

    def extract_tokens(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        """Fixed-size token extraction from a padded batch.

        Args:
            hidden_states: ``[B, L, H]``
            input_ids:     ``[B, L]``
            mode: one of ``_FIXED_SIZE_MODES``.

        Returns:
            ``[B, M, H]`` where M is the token count for the given type (0 if disabled).
        """
        if mode not in _FIXED_SIZE_MODES:
            raise ValueError(
                f"extract_tokens: mode={mode!r} is not a fixed-size mode. "
                f"Variable-length modes must be used via forward_extract(). "
                f"Fixed modes: {sorted(_FIXED_SIZE_MODES)}."
            )

        B, _, H = hidden_states.shape
        device = input_ids.device

        if mode == "traj_action_tokens":
            expected = self.num_traj_action_tokens
            if expected == 0:
                return torch.empty(B, 0, H, device=device, dtype=hidden_states.dtype)
            # Single repeated token id.
            mask = input_ids == self.traj_action_token_id
            counts = mask.sum(dim=1)
            if not (counts == expected).all():
                raise ValueError(
                    f"[extract_tokens:traj_action_tokens] expected {expected} "
                    f"positions per sequence; got {counts.tolist()}"
                )
            indices = mask.nonzero(as_tuple=True)
            return hidden_states[indices[0], indices[1], :].view(B, expected, H)

        # Step-indexed token types (jepa, vggt, wan): each step has a unique token id.
        _cfg = {
            "jepa_action_tokens": (self.jepa_action_token_ids, self.num_jepa_action_tokens),
            "vggt_action_tokens": (self.vggt_action_token_ids, self.num_vggt_action_tokens),
            "vggt_cam_action_tokens": (self.vggt_cam_action_token_ids, self.num_vggt_cam_action_tokens),
            "wan_action_tokens":  (self.wan_action_token_ids,  self.num_wan_action_tokens),
        }
        token_ids, expected = _cfg[mode]

        if expected == 0 or not token_ids:
            return torch.empty(B, 0, H, device=device, dtype=hidden_states.dtype)

        id_set = torch.tensor(token_ids, device=device)
        mask = torch.isin(input_ids, id_set)
        counts = mask.sum(dim=1)
        if not (counts == expected).all():
            raise ValueError(
                f"[extract_tokens:{mode}] each sequence must have exactly {expected} "
                f"positions; got {counts.tolist()}"
            )
        indices = mask.nonzero(as_tuple=True)
        return hidden_states[indices[0], indices[1], :].view(B, expected, H)

    def _extract_variable_tokens(
        self,
        last_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Variable-length token extraction returning ``(features, mask)``.

        ``True`` in mask = real token, ``False`` = padding.
        """
        B, _, H = last_hidden.shape
        device = input_ids.device

        # First non-pad column; must match :meth:`forward_full_hidden` (first ``mask==1``),
        # not ``(mask==0).sum()`` which also counts trailing pad zeros if any.
        first_real = attention_mask.long().argmax(dim=1)
        start_action_pos = (input_ids == self.start_action_token_id).long().argmax(dim=1)
        non_action_lens = start_action_pos - first_real  # [B]

        max_non = int(non_action_lens.max().item()) if non_action_lens.numel() > 0 else 0
        non_act = torch.zeros(B, max_non, H, device=device, dtype=last_hidden.dtype)
        non_act_mask = torch.zeros(B, max_non, dtype=torch.bool, device=device)
        for b in range(B):
            fr = int(first_real[b].item())
            sp = int(start_action_pos[b].item())
            L_b = sp - fr
            if L_b > 0:
                non_act[b, :L_b] = last_hidden[b, fr:sp]
                non_act_mask[b, :L_b] = True

        if mode == "non_action_tokens":
            return non_act, non_act_mask
            
        if mode == "visual_tokens":
            image_pad_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            vision_pad_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_pad|>")
            if image_pad_id is None: image_pad_id = -100
            if vision_pad_id is None: vision_pad_id = -100
            
            mask = (input_ids == image_pad_id) | (input_ids == vision_pad_id)
            counts = mask.sum(dim=1)
            max_count = int(counts.max().item()) if counts.numel() > 0 else 0

            features = torch.zeros(B, max_count, H, device=device, dtype=last_hidden.dtype)
            features_mask = torch.zeros(B, max_count, dtype=torch.bool, device=device)
            for b in range(B):
                c = int(counts[b].item())
                if c > 0:
                    features[b, :c] = last_hidden[b][mask[b]]
                    features_mask[b, :c] = True
            return features, features_mask

        raise ValueError(f"_extract_variable_tokens: unhandled mode={mode!r}")

    def compute_ce_loss(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy on tokens **after** ``<|end_action|>`` in each sequence.

        ``<|end_action|>`` serves as the CE probe token — CE loss begins
        at the token immediately following it, covering the assistant response.
        If ``<|end_action|>`` is absent in a row, that row contributes no CE targets.
        """
        labels = input_ids.clone()
        B, seq_len = labels.shape[0], labels.shape[1]

        col_idx = torch.arange(seq_len, device=labels.device).unsqueeze(0).expand(B, -1)
        past_end = seq_len + 1
        idx_for_first_valid = torch.where(attention_mask.bool(), col_idx, past_end)
        first_non_pad = idx_for_first_valid.min(dim=1).values
        has_valid = attention_mask.any(dim=1)

        end_id = int(self.end_action_token_id)
        device = input_ids.device
        content_start = torch.zeros(B, dtype=torch.long, device=device)
        for b in range(B):
            if not bool(has_valid[b].item()):
                continue
            fn = int(first_non_pad[b].item())
            row = input_ids[b, fn:]
            valid = attention_mask[b, fn:].bool()
            hits = ((row == end_id) & valid).nonzero(as_tuple=True)[0]
            if hits.numel() == 0:
                content_start[b] = seq_len
            else:
                content_start[b] = fn + int(hits[0].item()) + 1  # unique token → hits[0]

        bad = has_valid & (content_start > seq_len)
        if bad.any():
            idx_bad = bad.nonzero(as_tuple=True)[0]
            raise RuntimeError(
                f"[compute_ce_loss] end_action overflows sequence "
                f"(batch indices {idx_bad.tolist()}, seq_len={seq_len})."
            )

        _dbg = os.environ.get("debug")
        if _dbg not in (None, "", "0", "false", "False"):
            tok = self.processor.tokenizer
            for b in range(B):
                if not bool(has_valid[b].item()):
                    continue
                cs = int(content_start[b].item())
                seg = input_ids[b, cs:][attention_mask[b, cs:].bool()]
                ids_cpu = seg.detach().cpu().tolist()
                text = tok.decode(ids_cpu, skip_special_tokens=False)
                print(
                    f"[VLM CE debug] sample {b}: content_start={cs} len={len(ids_cpu)}\n"
                    f"{text!r}\n",
                    flush=True,
                )

        for b in range(B):
            if not bool(has_valid[b].item()):
                labels[b, :] = IGNORE_INDEX
                continue
            cs = int(content_start[b].item())
            labels[b, :cs] = IGNORE_INDEX

        labels[attention_mask == 0] = IGNORE_INDEX

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        if not (shift_labels != IGNORE_INDEX).any():
            return (logits * 0).sum()

        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
        )

    @torch.no_grad()
    def generate_text(
        self,
        images: List[PILImage.Image],
        user_prompts: List[str],
        max_new_tokens: int = 512,
        *,
        assistant_answers: Optional[List[str]] = None,
        vlm_conditioning: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Greedy assistant text via ``model.generate`` (``do_sample=False``).

        必含 ``"texts"``。``vlm_conditioning`` 为模式名 list 时，对 **定长** action 槽在 prefill hidden 上抽与 :meth:`forward_extract` 同形张量（**不含** non_action；该槽仅 ``forward`` 路径）。"""
        n = len(images)
        if len(user_prompts) != n:
            raise ValueError(f"len(user_prompts)={len(user_prompts)} != len(images)={n}")

        model_inputs = self.build_inputs(images, user_prompts, assistant_answers=assistant_answers)
        device = next(self.model.parameters()).device
        model_inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in model_inputs.items()
        }
        prompt_len = model_inputs["input_ids"].shape[1]
        tokenizer = self.processor.tokenizer

        def _decode_assistant(sequences: torch.Tensor) -> List[str]:
            return [
                tokenizer.decode(sequences[i, prompt_len:], skip_special_tokens=True)
                for i in range(sequences.shape[0])
            ]

        gen_common = dict(
            max_new_tokens=max_new_tokens,
            min_new_tokens=50,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=50,
        )

        if vlm_conditioning is not None and not isinstance(vlm_conditioning, (list, tuple, ListConfig)):
            raise TypeError(
                f"generate_text: vlm_conditioning 须为 list 或 None，得 {type(vlm_conditioning).__name__}。"
            )
        modes = [str(x) for x in vlm_conditioning] if vlm_conditioning is not None else []
        names = [k for k in modes if k in _FIXED_SIZE_MODES]
        if not names:
            sequences = self.model.generate(**model_inputs, **gen_common)  # type: ignore[arg-type]
            if not isinstance(sequences, torch.Tensor):
                raise TypeError("expected tensor ids from generate()")
            return {"texts": _decode_assistant(sequences)}

        gen_out = self.model.generate(
            **model_inputs,  # type: ignore[arg-type]
            **gen_common,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
        if not isinstance(gen_out, GenerateDecoderOnlyOutput):
            raise TypeError("expected GenerateDecoderOnlyOutput from generate()")
        hs = gen_out.hidden_states
        if not hs:
            raise RuntimeError("generate returned no hidden_states.")
        last_hidden = hs[0][-1]
        in_ids = model_inputs["input_ids"]
        out: Dict[str, Any] = {"texts": _decode_assistant(gen_out.sequences)}
        for key in names:
            out[key] = self.extract_tokens(last_hidden, in_ids, key)
        return out

    def forward_extract(
        self,
        images: List[PILImage.Image],
        user_prompts: List[str],
        assistant_answers: Optional[List[str]] = None,
        *,
        compute_ce_loss: bool = True,
        vlm_conditioning: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Build inputs → VLM forward → extract features.

        ``vlm_conditioning`` 为模式名 list（同 ``AeActtokenVla`` 的 ``_vlm_conditioning`` 展开）；
        ``None`` 时仅可配合 ``ce_loss``。多键不事先并一条张量（由 :class:`~models.action_head.recog.recog_action_head.RecogActionHead` 内拼）。
        """
        if vlm_conditioning is not None and not isinstance(vlm_conditioning, (list, tuple, ListConfig)):
            raise TypeError(
                f"forward_extract: vlm_conditioning 须为 list 或 None，得 {type(vlm_conditioning).__name__}。"
            )
        modes: list[str] | None = (
            [str(x) for x in vlm_conditioning] if vlm_conditioning is not None else None
        )
        has_answers = assistant_answers is not None
        want_ce = has_answers and compute_ce_loss
        if not modes and not want_ce:
            raise ValueError(
                "forward_extract: 须指定非空 vlm_conditioning 列表，或提供 assistant_answers 且 compute_ce_loss=True 以只算 ce_loss。"
            )

        batch_inputs = self.build_inputs(images, user_prompts, assistant_answers)
        device = next(self.model.parameters()).device
        batch_inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch_inputs.items()
        }

        outputs = self.model(
            **batch_inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]  # [B, L, H]
        in_ids = batch_inputs["input_ids"]

        result: dict[str, Any] = {}
        if modes:
            att = batch_inputs["attention_mask"]
            for name in modes:
                if name in _VARIABLE_SIZE_MODES:
                    f, fm = self._extract_variable_tokens(
                        last_hidden, in_ids, att, name,
                    )
                    result[name] = f
                    result[f"{name}_mask"] = fm
                else:
                    result[name] = self.extract_tokens(last_hidden, in_ids, name)

        if has_answers and compute_ce_loss:
            assert assistant_answers is not None
            result["ce_loss"] = self.compute_ce_loss(
                outputs.logits,
                batch_inputs["input_ids"],
                batch_inputs["attention_mask"],
            )

        return result

    @torch.no_grad()
    def forward_full_hidden(
        self,
        images: List[PILImage.Image],
        user_prompts: List[str],
    ) -> List[Dict[str, torch.Tensor]]:
        """Full VLM forward — returns per-sample last_hidden and all token indices for caching.

        Uses :meth:`_compute_content_indices` (shared with online path) to ensure
        index semantics are identical between cached and non-cached training paths.

        Returns:
            Per-sample list of dicts, each containing:

            ``last_hidden``:          ``[L_content, H]`` bfloat16, no padding.
            ``start_action_idx``:     scalar int64.
            ``end_action_idx``:       scalar int64.
            ``jepa_action_indices``:  ``[Mj]`` int64 (empty ``[0]`` when unused).
            ``vggt_action_indices``:  ``[Mv]`` int64.
            ``wan_action_indices``:   ``[N]``  int64.
            ``traj_action_indices``:  ``[K]``  int64.
        """
        batch_inputs = self.build_inputs(images, user_prompts, assistant_answers=None)
        device = next(self.model.parameters()).device
        batch_inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch_inputs.items()
        }

        outputs = self.model(
            **batch_inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden_batch = outputs.hidden_states[-1]
        input_ids_batch   = batch_inputs["input_ids"]
        attention_mask_batch = batch_inputs["attention_mask"]

        if self.processor.tokenizer.padding_side != "left":
            raise RuntimeError(
                f"forward_full_hidden requires padding_side='left', "
                f"got '{self.processor.tokenizer.padding_side}'."
            )

        results: List[Dict[str, torch.Tensor]] = []
        for b in range(last_hidden_batch.shape[0]):
            real_cols = attention_mask_batch[b].bool()
            first_real = int(real_cols.nonzero(as_tuple=True)[0][0].item())
            content_hidden = last_hidden_batch[b, first_real:].cpu().to(torch.bfloat16)
            content_ids = input_ids_batch[b, first_real:].cpu()

            idx = self._compute_content_indices(content_ids)
            results.append({
                "last_hidden":          content_hidden,
                "start_action_idx":     idx["start_action_idx"],
                "end_action_idx":       idx["end_action_idx"],
                "jepa_action_indices":  idx["jepa_action_indices"],
                "vggt_action_indices":  idx["vggt_action_indices"],
                "wan_action_indices":   idx["wan_action_indices"],
                "traj_action_indices":  idx["traj_action_indices"],
            })

        return results
