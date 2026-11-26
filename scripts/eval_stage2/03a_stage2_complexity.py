#!/usr/bin/env python3
"""Stage 2 Step 03a: Static Complexity (Params & FLOPs decomposition) for Stage 2 UAVReIDNet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common_stage2 import (  # noqa: E402
    load_stage2_config,
    measure_stage2_complexity,
    reset_gpu_memory,
    resolve_ckpt_path,
    resolve_output_dir,
    write_csv,
    write_json,
)
from stage2_models import load_stage2_checkpoint_strict  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 Step 03a: Static Complexity")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_stage2_config(args.config)
    out_root = resolve_output_dir(cfg, args.output_dir)
    out_dir = out_root / "03_efficiency_and_latency_orin"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eff_cfg = cfg.get("efficiency", {})
    comp_lens = [int(x) for x in eff_cfg.get("complexity_seq_lens", [8, 12])]
    ckpts = cfg.get("checkpoints") or {}

    specs = [
        ("convnext_rga_mamba", "uavreid_convnext_rga_mamba", False),
        ("convnext_bcgg_unfolded_mamba", "uavreid_convnext_bcgg_mamba", False),
        ("convnext_bcgg_folded_mamba", "uavreid_convnext_bcgg_mamba", True),
        ("convnext_bcgg_folded_none", "uavreid_convnext_bcgg_mean_head", True),
        ("convnext_bcgg_folded_attn", "uavreid_convnext_bcgg_attn", True),
        ("convnext_bcgg_folded_mamba_mean", "uavreid_convnext_bcgg_mamba_meanpool", True),
    ]

    csv_rows: List[Dict[str, Any]] = []
    json_rows: List[Dict[str, Any]] = []

    for cfg_name, ckpt_key, deploy in specs:
        if ckpt_key not in ckpts:
            continue
        ckpt_path = resolve_ckpt_path(cfg, ckpt_key)
        if not ckpt_path.is_file():
            continue
        reset_gpu_memory(device)
        model, audit = load_stage2_checkpoint_strict(ckpt_key, ckpt_path, deploy_bcgg=deploy, device=device)
        for L in comp_lens:
            stats = measure_stage2_complexity(model, seq_len=L, device=device)
            csv_rows.append(
                {
                    "config": cfg_name,
                    "seq_len": L,
                    "backbone_params_M": f"{stats['backbone_params_M']:.4f}",
                    "temporal_head_params_M": f"{stats['temporal_head_params_M']:.4f}",
                    "total_params_M": f"{stats['total_params_M']:.4f}",
                    "backbone_GFLOPs": f"{stats['backbone_GFLOPs']:.4f}",
                    "temporal_head_GFLOPs": f"{stats['temporal_head_GFLOPs']:.4f}",
                    "total_GFLOPs": f"{stats['total_GFLOPs']:.4f}",
                }
            )
            json_rows.append(
                {
                    "config": cfg_name,
                    "checkpoint_key": ckpt_key,
                    "checkpoint_sha256": audit["sha256"],
                    "bcgg_folded": deploy,
                    "seq_len": L,
                    **stats,
                }
            )
            print(
                f"[03a | L={L:2d}] {cfg_name:<32} | params={stats['total_params_M']:.2f}M "
                f"(bb={stats['backbone_params_M']:.2f}M, th={stats['temporal_head_params_M']:.2f}M) | "
                f"GFLOPs={stats['total_GFLOPs']:.2f} (bb={stats['backbone_GFLOPs']:.2f}, th={stats['temporal_head_GFLOPs']:.4f})"
            )
        del model
        reset_gpu_memory(device)

    write_csv(csv_rows, out_dir / "stage2_static_complexity.csv")
    write_json(json_rows, out_dir / "stage2_static_complexity.json")


if __name__ == "__main__":
    main()
