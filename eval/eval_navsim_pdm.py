"""Evaluate a worldmodel trajectory model on NAVSIM using PDM score.

Each rank evaluates one token chunk using navsim's ``SceneLoader`` +
``MetricCacheLoader`` + ``pdm_score()`` and dumps a JSON. When launched under
``torch.distributed.run`` (``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK`` set), the
chunk id / GPU are derived from those env vars; otherwise it runs single-rank
on ``cuda:0``. Use ``scripts/evaluation/run_eval_navsim_pdm_multigpu.sh`` to
launch + merge.
"""

from __future__ import annotations

import argparse
import json
import lzma
import os
import pickle
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _resolve_scene_filter_yaml(explicit: Optional[str]) -> Path:
    """Default: $NAVSIM_DEVKIT_ROOT/.../scene_filter/navtest.yaml (official navtest split)."""
    if explicit:
        p = Path(os.path.expanduser(explicit))
        if not p.is_file():
            raise SystemExit(f"Scene filter YAML not found: {p}")
        return p
    root = os.environ.get("NAVSIM_DEVKIT_ROOT")
    if not root:
        raise SystemExit(
            "Set NAVSIM_DEVKIT_ROOT (recommended) or pass --scene-filter-yaml "
            "pointing to navsim's scene_filter/navtest.yaml"
        )
    p = Path(root) / (
        "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
    )
    if not p.is_file():
        raise SystemExit(f"Scene filter YAML not found: {p}")
    return p


def _load_scene_filter_from_yaml(yaml_path: Path):
    """Load navsim ``SceneFilter`` from devkit YAML (same files as Hydra / recogdrive PDM eval)."""
    from omegaconf import OmegaConf
    from navsim.common.dataclasses import SceneFilter

    cfg = OmegaConf.load(yaml_path)
    d = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(d, dict)
    d.pop("_target_", None)
    d.pop("_convert_", None)
    return SceneFilter(**d)


def main():
    parser = argparse.ArgumentParser(description="NAVSIM PDM Score evaluation")
    parser.add_argument("--config", required=True, help="Model YAML config")
    parser.add_argument("--checkpoint", default=None, help="Stage-3 model checkpoint")
    parser.add_argument("--vlm-feature-cache-dir", default=None,
                        help="Path to VLM features cache (overrides vlm_feature_cache_dir in yaml)")
    parser.add_argument("--navsim-log-path", required=True)
    parser.add_argument("--sensor-blobs-path", required=True)
    parser.add_argument("--metric-cache-path", required=True)
    parser.add_argument(
        "--scene-filter-yaml",
        default=None,
        help="Hydra SceneFilter YAML (default: navtest under $NAVSIM_DEVKIT_ROOT)",
    )
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--output-dir", default=None, help="Save per-token results as JSON")
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    print(
        f"[eval_navsim_pdm] rank={rank}/{world_size} device={device}",
        flush=True,
    )

    from navsim.common.dataloader import SceneLoader, MetricCacheLoader
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    from eval.navsim_agent import WorldModelNavsimAgent

    # Resolve vlm_feature_cache_dir: CLI override takes precedence, then yaml.
    from omegaconf import OmegaConf
    _cfg = OmegaConf.load(args.config)
    _cfg_dict = OmegaConf.to_container(_cfg, resolve=True)
    _cache_dir = args.vlm_feature_cache_dir or _cfg_dict.get("vlm_feature_cache_dir")

    print("Initializing agent...")
    agent = WorldModelNavsimAgent(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=device,
        vlm_feature_cache_dir=_cache_dir,
    )
    agent.initialize()

    proposal_sampling = TrajectorySampling(time_horizon=4, interval_length=0.1)
    simulator = PDMSimulator(proposal_sampling)
    scorer = PDMScorer(proposal_sampling)

    sf_path = _resolve_scene_filter_yaml(args.scene_filter_yaml)
    scene_filter = _load_scene_filter_from_yaml(sf_path)
    print(f"Scene filter: {sf_path} "
          f"(frame_interval={scene_filter.frame_interval}, "
          f"{len(scene_filter.log_names or [])} logs, "
          f"{len(scene_filter.tokens or [])} tokens)")
    if args.max_scenes is not None:
        scene_filter.max_scenes = args.max_scenes

    sensor_config = agent.get_sensor_config()
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(args.sensor_blobs_path),
        data_path=Path(args.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=sensor_config,
    )

    metric_cache_loader = MetricCacheLoader(Path(args.metric_cache_path))
    all_tokens = sorted(set(scene_loader.tokens) & set(metric_cache_loader.tokens))

    if world_size > 1:
        tokens = list(np.array_split(all_tokens, world_size)[rank])
        print(f"Rank {rank}/{world_size}: {len(tokens)} scenes (total {len(all_tokens)})")
    else:
        tokens = all_tokens
        print(f"Evaluating {len(tokens)} scenes...")

    results = []
    for token in tqdm(tokens, desc=f"PDM[{rank}]"):
        score_row = {"token": token, "valid": True}
        try:
            metric_path = metric_cache_loader.metric_cache_paths[token]
            with lzma.open(metric_path, "rb") as f:
                metric_cache = pickle.load(f)

            if hasattr(agent, "set_vlm_cache_scene_token"):
                agent.set_vlm_cache_scene_token(token)
            agent_input = scene_loader.get_agent_input_from_token(token)
            trajectory = agent.compute_trajectory(agent_input)

            pdm_result = pdm_score(
                metric_cache, trajectory,
                simulator.proposal_sampling, simulator, scorer,
            )
            score_row.update(asdict(pdm_result))
        except Exception as e:
            print(f"  Error on {token}: {e}")
            score_row["valid"] = False
            score_row["score"] = 0.0
        results.append(score_row)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"pdm_results_chunk{rank}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out_path}")

    valid = [r for r in results if r["valid"]]
    scores = [r["score"] for r in valid]
    if scores:
        print(f"\n{'='*50}")
        print(f"  Rank {rank}: {len(valid)} / {len(results)} valid")
        print(f"  Mean PDM Score: {np.mean(scores):.4f}")
        print(f"  Median:         {np.median(scores):.4f}")
        print(f"  Min / Max:      {np.min(scores):.4f} / {np.max(scores):.4f}")
        print(f"{'='*50}")
    else:
        print("No valid results.")


if __name__ == "__main__":
    main()
