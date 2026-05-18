#!/usr/bin/env python3
"""合并 ``pdm_results_chunk*.json`` 并写出聚合分数（历史实验补档用）。

默认在目录下写入 ``pdm_merged_aggregate.json``（各指标在 valid 样本上的均值 + 计数信息）。
评测脚本 ``run_eval_navsim_pdm_multigpu.sh`` 在 torchrun 结束后也会调用本脚本。

示例::

    python scripts/evaluation/merge_pdm_eval_chunks.py /path/to/run/2026.04.20.20.52.20
"""

from __future__ import annotations

import argparse
import json
import glob
import os
import sys

import numpy as np

# 与 multigpu 脚本里打印的列一致；缺键的样本在 mean 时跳过
_METRIC_KEYS = (
    "score",
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
)


def _mean_for_key(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if key in r and r[key] is not None]
    return float(np.mean(vals)) if vals else float("nan")


def merge_chunks(out_dir: str) -> dict:
    files = sorted(glob.glob(os.path.join(out_dir, "pdm_results_chunk*.json")))
    if not files:
        raise FileNotFoundError(f"未找到 pdm_results_chunk*.json: {out_dir}")

    rows: list[dict] = []
    for f in files:
        with open(f, encoding="utf-8") as fp:
            rows.extend(json.load(fp))

    valid = [r for r in rows if r.get("valid")]
    if not valid:
        raise ValueError("无有效结果（valid=true 的行为空）")

    mean_metrics = {k: _mean_for_key(valid, k) for k in _METRIC_KEYS}
    return {
        "output_dir": os.path.abspath(out_dir),
        "chunk_files": [os.path.basename(f) for f in files],
        "num_chunks": len(files),
        "total_rows": len(rows),
        "valid_rows": len(valid),
        "mean_metrics": mean_metrics,
    }


def _print_summary(agg: dict) -> None:
    mm = agg["mean_metrics"]
    print(
        f"Chunks: {agg['num_chunks']} files, total rows: {agg['total_rows']}, valid: {agg['valid_rows']}"
    )
    print(f"  score                              : {mm['score']:.6f}")
    for k in _METRIC_KEYS[1:]:
        print(f"  {k:35s}: {mm[k]:.6f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "output_dir",
        help="含 pdm_results_chunk*.json 的目录（通常为 run_root/<timestamp>/）",
    )
    p.add_argument(
        "-o",
        "--aggregate-out",
        default=None,
        help="聚合 JSON 输出路径（默认：<output_dir>/pdm_merged_aggregate.json）",
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="不写 stdout，只写文件",
    )
    args = p.parse_args()
    out_dir = os.path.expanduser(args.output_dir)
    if not os.path.isdir(out_dir):
        print(f"不是目录: {out_dir}", file=sys.stderr)
        return 1

    try:
        agg = merge_chunks(out_dir)
    except (FileNotFoundError, ValueError) as e:
        print(str(e), file=sys.stderr)
        return 1

    out_path = args.aggregate_out or os.path.join(out_dir, "pdm_merged_aggregate.json")
    with open(out_path, "w", encoding="utf-8") as wf:
        json.dump(agg, wf, indent=2)
    if not args.quiet:
        print(f"已写入聚合分数: {out_path}\n")
        _print_summary(agg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
