"""Unified driving prompt construction for VLM (stage 2) training and inference.

Templates are designed for Qwen3-VL chat format.

**Token layout in VLM sequence**::

    [non_action_tokens]
    <|start_action|>
    [jepa_action tokens]    (if num_jepa_action_tokens > 0)
    [vggt_action tokens]    (if num_vggt_action_tokens > 0)
    [wan_action tokens]     (if num_wan_action_tokens > 0)
    [traj_action tokens]    (if num_traj_action_tokens > 0)
    <|end_action|>
    [CE text / assistant response]

``<|start_action|>`` marks the start of all action tokens.
``<|end_action|>`` marks the end — CE loss starts from the token immediately after it.
All action token types follow the same step-based template pattern as wan_action:
each step token is repeated ``num_action_tokens // num_token_steps`` times.
"""

from __future__ import annotations

import numpy as np

NAVIGATION_COMMANDS = ["TURN LEFT", "GO STRAIGHT", "TURN RIGHT", "UNKNOWN"]

# ── Special boundary tokens ────────────────────────────────────────────────
START_ACTION_TOKEN = "<|start_action|>"
END_ACTION_TOKEN = "<|end_action|>"

# ── Action token templates (step-indexed) ─────────────────────────────────
# jepa / vggt / wan: num_xxx_token_steps unique token ids, each repeated
# (num_xxx_action_tokens / num_xxx_token_steps) times in the sequence.
JEPA_ACTION_TOKEN_TEMPLATE = "<|jepa_action_{i}|>"
VGGT_ACTION_TOKEN_TEMPLATE = "<|vggt_action_{i}|>"
VGGT_CAM_ACTION_TOKEN_TEMPLATE = "<|vggt_cam_action_{i}|>"
WAN_ACTION_TOKEN_TEMPLATE  = "<|wan_action_{i}|>"

# ── Traj action token (single token, no step index) ────────────────────────
# All K traj positions use the same token ID, repeated num_traj_action_tokens times.
# This lets the model attend to K independent contextual positions while keeping
# the embedding initialisation shared.
TRAJ_ACTION_TOKEN = "<|traj_action|>"

ACTION_PLACEHOLDER = "{actions}"

# Optional extra user text after the predict-dynamics / action-token line.
POST_ACTION_USER_HINT = ""

VLM_DRIVING_SYSTEM_MESSAGE = """
You are a vehicle trajectory prediction model for autonomous driving. Your task is to predict the ego vehicle's 4-second trajectory based on the following inputs: multi-view images from 8 cameras, ego vehicle states (position), and discrete navigation commands. The input provides a 2-second history, and your output should ensure a safe trajectory for the next 4 seconds. Your predictions must adhere to the following metrics:
1. **No at-fault Collisions (NC)**: Avoid collisions with other objects/vehicles.
2. **Drivable Area Compliance (DAC)**: Stay within the drivable area.
3. **Time to Collision (TTC)**: Maintain a safe distance from other vehicles.
4. **Ego Progress (EP)**: Ensure the ego vehicle moves forward without being stuck.
5. **Comfort (C)**: Avoid sharp turns and sudden decelerations.
6. **Driving Direction Compliance (DDC)**: Align with the intended driving direction.
For evaluation, use the **PDM Score**, which combines these metrics: **PDM Score** = NC * DAC * (5*TTC + 5*EP + 2*C + 0*DDC) / 12.
Your predictions will be evaluated through a non-reactive 4-second simulation with an LQR controller and background actors following their recorded trajectories. The better your predictions, the higher your score.
"""


def format_trajectory_number(n: float | np.floating, decimal_places: int = 2) -> str:
    """Format one scalar like ReCogDrive ``format_number``: 2 decimals, tiny values as ``0.0``."""
    x = float(n)
    if abs(round(x, decimal_places)) <= 1e-2:
        return "0.0"
    return f"{x:+.{decimal_places}f}"


def _build_indexed_block(template: str, num_action_tokens: int, num_token_steps: int) -> str:
    """Build the token string for one step-indexed action-token type.

    ``num_token_steps`` unique token strings are each repeated
    ``num_action_tokens // num_token_steps`` times, for a total of
    ``num_action_tokens`` token positions.

    Args:
        template:          Token template, e.g. ``"<|wan_action_{i}|>"``.
        num_action_tokens: Total token positions (0 → empty string).
        num_token_steps:   Number of unique step token ids.
                           Must divide ``num_action_tokens`` evenly.

    Returns:
        Concatenated token string (empty when ``num_action_tokens == 0``).
    """
    if num_action_tokens <= 0 or num_token_steps <= 0:
        return ""
    if num_action_tokens % num_token_steps != 0:
        raise ValueError(
            f"num_action_tokens ({num_action_tokens}) must be divisible by "
            f"num_token_steps ({num_token_steps})."
        )
    reps = num_action_tokens // num_token_steps   # repetitions per step token
    return "".join(template.format(i=step) * reps for step in range(num_token_steps))


def build_action_token_string(
    num_jepa_action_tokens: int = 0,
    num_jepa_token_steps: int = 8,
    num_vggt_action_tokens: int = 0,
    num_vggt_token_steps: int = 8,
    num_vggt_cam_action_tokens: int = 0,
    num_wan_action_tokens: int = 24,
    num_wan_token_steps: int = 8,
    num_traj_action_tokens: int = 0,
) -> str:
    """Build the replacement string for the ``{actions}`` placeholder.

    Layout::

        <|start_action|>
        [jepa_action tokens] (if num_jepa_action_tokens > 0)
        [vggt_action tokens] (if num_vggt_action_tokens > 0)
        [wan_action tokens]  (if num_wan_action_tokens  > 0)
        [traj_action tokens] (if num_traj_action_tokens > 0)
        <|end_action|>

    For step-indexed types (jepa / vggt / wan):
        ``num_xxx_token_steps`` unique step token ids, each repeated
        ``num_xxx_action_tokens // num_xxx_token_steps`` times.

    For traj: single token ``<|traj_action|>`` repeated ``num_traj_action_tokens`` times.

    Any token type with count 0 contributes nothing to the string.
    """
    traj_block = TRAJ_ACTION_TOKEN * num_traj_action_tokens if num_traj_action_tokens > 0 else ""
    vggt_block = ""
    if num_vggt_action_tokens > 0:
        if num_vggt_cam_action_tokens > 0:
            # Interleave <vggt_cam_0>*C <vggt_geo_0>*K <vggt_cam_1>*C <vggt_geo_1>*K ...
            steps = num_vggt_token_steps
            tokens_per_step = num_vggt_action_tokens // steps
            cam_tokens_per_step = num_vggt_cam_action_tokens // steps
            for i in range(steps):
                if cam_tokens_per_step > 0:
                    vggt_block += VGGT_CAM_ACTION_TOKEN_TEMPLATE.format(i=i) * cam_tokens_per_step
                vggt_block += VGGT_ACTION_TOKEN_TEMPLATE.format(i=i) * tokens_per_step
        else:
            vggt_block = _build_indexed_block(VGGT_ACTION_TOKEN_TEMPLATE, num_vggt_action_tokens, num_vggt_token_steps)

    return (
        START_ACTION_TOKEN
        + _build_indexed_block(JEPA_ACTION_TOKEN_TEMPLATE, num_jepa_action_tokens, num_jepa_token_steps)
        + vggt_block
        + _build_indexed_block(WAN_ACTION_TOKEN_TEMPLATE,  num_wan_action_tokens,  num_wan_token_steps)
        + traj_block
        + END_ACTION_TOKEN
    )


def build_driving_prompt(
    history_trajectory: np.ndarray,
    navigation_command: int,
    ego_velocity: np.ndarray,
    ego_acceleration: np.ndarray,
    interval_s: float = 0.5,
    action_placeholder: str = ACTION_PLACEHOLDER,
    post_action_user_hint: str = "",
) -> str:
    """Build the **user message** for VLM driving prompt (ReCogDrive ``vel_and_acc`` style).

    Args:
        history_trajectory: ``[Hn, 3]`` array of (x, y, heading) in local ego frame.
        navigation_command: 0 = left, 1 = straight, 2 = right, 3 = unknown.
        ego_velocity: length-2 array ``(vx, vy)`` in rear-axle body frame (m/s).
        ego_acceleration: length-2 array ``(ax, ay)`` in rear-axle body frame (m/s²).
        interval_s: time interval between trajectory samples.
        action_placeholder: replaced with the action token string later.
        post_action_user_hint: extra user text appended after the predict-dynamics line.

    Returns:
        User message string (without ``<image>`` prefix — caller adds that).
    """
    n_hist = len(history_trajectory)
    hist_lines = "\n".join(
        (
            f"   - t-{n_hist - 1 - i}: "
            f"({format_trajectory_number(t[0])}, {format_trajectory_number(t[1])}, "
            f"{format_trajectory_number(t[2])})"
        )
        for i, t in enumerate(history_trajectory)
    )
    cmd = NAVIGATION_COMMANDS[min(int(navigation_command), len(NAVIGATION_COMMANDS) - 1)]

    vel = np.asarray(ego_velocity, dtype=np.float64).ravel()
    acc = np.asarray(ego_acceleration, dtype=np.float64).ravel()
    if vel.size != 2 or acc.size != 2:
        raise ValueError(
            f"ego_velocity / ego_acceleration must be length-2; got {vel.size} / {acc.size}"
        )
    vel_str = f"({format_trajectory_number(vel[0])}, {format_trajectory_number(vel[1])})"
    acc_str = f"({format_trajectory_number(acc[0])}, {format_trajectory_number(acc[1])})"

    output_requirements = (
        "\nOutput requirements:\n"
        "- Predict 8 future trajectory points\n"
        "- Each point format: (x:float, y:float, heading:float)\n"
        "- Use [PT, ...] to encapsulate the trajectory\n"
        "- Maintain numerical precision to 2 decimal places"
    )

    # Order: inputs → output contract → generation cue (action tokens last).
    body = (
        "As an autonomous driving system, predict the vehicle's trajectory based on:\n"
        "1. Visual perception from the front camera (image)\n"
        f"2. Historical motion context (last {n_hist} timesteps at {interval_s:.2f}s spacing):\n"
        f"{hist_lines}\n"
        f"3. Active navigation command: [{cmd}]\n"
        f"4. Current velocity: {vel_str}\n"
        f"5. Current acceleration: {acc_str}"
        f"{output_requirements}\n\n"
        f"Predict future dynamics (latent action tokens, then trajectory text): {action_placeholder}"
    )
    hint = (post_action_user_hint or "").strip()
    if hint:
        body = f"{body}\n\n{hint}"
    return body


def build_trajectory_answer(
    future_trajectory: np.ndarray,
    num_points: int = 8,
) -> str:
    """Build the **assistant message** (GT for CE loss).

    Matches ReCogDrive SFT: prefix + ``[PT, (x, y, h), ...]``.
    """
    pts = ", ".join(
        (
            f"({format_trajectory_number(t[0])}, {format_trajectory_number(t[1])}, "
            f"{format_trajectory_number(t[2])})"
        )
        for t in future_trajectory[:num_points]
    )
    return f"Here is the planning trajectory [PT, {pts}]."


def get_all_jepa_action_token_strings(max_slots: int = 64) -> list[str]:
    """Return all ``<|jepa_action_i|>`` strings for tokenizer expansion."""
    return [JEPA_ACTION_TOKEN_TEMPLATE.format(i=i) for i in range(max_slots)]


def get_all_vggt_cam_action_token_strings(max_slots: int = 64) -> list[str]:
    """Return all ``<|vggt_cam_action_i|>`` strings for tokenizer expansion."""
    return [VGGT_CAM_ACTION_TOKEN_TEMPLATE.format(i=i) for i in range(max_slots)]

def get_all_vggt_action_token_strings(max_slots: int = 64) -> list[str]:
    """Return all ``<|vggt_action_i|>`` strings for tokenizer expansion."""
    return [VGGT_ACTION_TOKEN_TEMPLATE.format(i=i) for i in range(max_slots)]


def get_all_wan_action_token_strings(max_slots: int = 64) -> list[str]:
    """Return all ``<|wan_action_i|>`` strings for tokenizer expansion."""
    return [WAN_ACTION_TOKEN_TEMPLATE.format(i=i) for i in range(max_slots)]
