#!/usr/bin/env python3
"""Stage 2 Step 03b: Benchmark PyTorch Clip Latency (FP32/FP16) on Jetson AGX Orin."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common_stage2 import (  # noqa: E402
    load_stage2_config,
    maybe_lock_jetson_clocks,
    read_jetson_thermals,
    reset_gpu_memory,
    resolve_ckpt_path,
    resolve_output_dir,
    summarize_latencies,
    write_csv,
    write_json,
)
from stage2_models import Stage2UAVReIDNet, load_stage2_checkpoint_strict  # noqa: E402


@torch.no_grad()
def _bench_pytorch_clip(
    model: Stage2UAVReIDNet,
    seq_len: int,
    precision: str,
    warmup: int,
    runs: int,
    device: torch.device,
) -> Dict[str, Any]:
    reset_gpu_memory(device)
    model.eval().to(device)
    dtype = torch.float16 if precision == "fp16" else torch.float32
    use_amp = precision == "fp16"
    clip = torch.randn(1, seq_len, 3, 224, 224, device=device, dtype=torch.float32)

    def _run_once():
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            seq_feats = model.extract_clip_visual(clip)
            v_mean = seq_feats.mean(dim=1)
            if model.temporal_encoder is None:
                _ = model.head(v_mean, None)
            else:
                t_tok, _ = model.temporal_encoder(seq_feats)
                _ = model.head(v_mean, t_tok)

    for _ in range(warmup):
        _run_once()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    bb_times: List[float] = []
    th_times: List[float] = []
    tot_times: List[float] = []
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e2 = torch.cuda.Event(enable_timing=True)

    for _ in range(runs):
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            e0.record()
            seq_feats = model.extract_clip_visual(clip)
            e1.record()
            v_mean = seq_feats.mean(dim=1)
            if model.temporal_encoder is None:
                _ = model.head(v_mean, None)
            else:
                t_tok, _ = model.temporal_encoder(seq_feats)
                _ = model.head(v_mean, t_tok)
            e2.record()
        torch.cuda.synchronize(device)
        bb_times.append(float(e0.elapsed_time(e1)))
        th_times.append(float(e1.elapsed_time(e2)))
        tot_times.append(float(e0.elapsed_time(e2)))

    peak_vram = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    bb_st = summarize_latencies(bb_times, batch_size=1)
    th_st = summarize_latencies(th_times, batch_size=1)
    tot_st = summarize_latencies(tot_times, batch_size=1)
    return {
        "backbone_clip_p50_ms": bb_st["p50_ms"],
        "temporal_head_p50_ms": th_st["p50_ms"],
        "total_clip_p5_ms": tot_st["p5_ms"],
        "total_clip_p50_ms": tot_st["p50_ms"],
        "total_clip_p95_ms": tot_st["p95_ms"],
        "total_clip_mean_ms": tot_st["mean_ms"],
        "total_clip_std_ms": tot_st["std_ms"],
        "clips_per_sec": tot_st["throughput"],
        "peak_vram_MiB": peak_vram,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 Step 03b: Benchmark PyTorch Latency")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--lock-clocks", action="store_true")
    parser.add_argument(
        "--merge-missing",
        action="store_true",
        help="Reuse existing stage2_pytorch_latency.{csv,json} rows and benchmark only missing configs",
    )
    args = parser.parse_args()

    cfg = load_stage2_config(args.config)
    out_root = resolve_output_dir(cfg, args.output_dir)
    out_dir = out_root / "03_efficiency_and_latency_orin"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clock_meta = maybe_lock_jetson_clocks(args.lock_clocks)

    eff_cfg = cfg.get("efficiency", {})
    seq_lens = [int(x) for x in eff_cfg.get("seq_lens", [4, 8, 12, 16])]
    precs = [str(x).lower() for x in eff_cfg.get("precisions", ["fp32", "fp16"])]
    warmup = int(eff_cfg.get("warmup_runs", 50))
    runs = int(eff_cfg.get("measure_runs", 200))
    ckpts = cfg.get("checkpoints") or {}

    specs = [
        ("convnext_rga_mamba", "uavreid_convnext_rga_mamba", False),
        ("convnext_bcgg_unfolded_mamba", "uavreid_convnext_bcgg_mamba", False),
        ("convnext_bcgg_folded_mamba", "uavreid_convnext_bcgg_mamba", True),
        ("convnext_bcgg_folded_none", "uavreid_convnext_bcgg_mean_head", True),
        ("convnext_bcgg_folded_attn", "uavreid_convnext_bcgg_attn", True),
        ("convnext_bcgg_folded_mamba_mean", "uavreid_convnext_bcgg_mamba_meanpool", True),
    ]

    csv_path = out_dir / "stage2_pytorch_latency.csv"
    json_path = out_dir / "stage2_pytorch_latency.json"
    existing_csv: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    existing_json: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    if args.merge_missing and csv_path.is_file() and json_path.is_file():
        with csv_path.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                existing_csv[(r["config"], r["precision"].upper(), int(r["seq_len"]))] = r
        jdata = json.loads(json_path.read_text(encoding="utf-8"))
        for r in jdata.get("records", []):
            existing_json[(r["config"], r["precision"].upper(), int(r["seq_len"]))] = r
        print(f"[OK] --merge-missing loaded {len(existing_csv)} existing PyTorch latency entries.")

    csv_rows: List[Dict[str, Any]] = []
    json_rows: List[Dict[str, Any]] = []

    for cfg_name, ckpt_key, deploy in specs:
        if ckpt_key not in ckpts:
            continue
        ckpt_path = resolve_ckpt_path(cfg, ckpt_key)
        if not ckpt_path.is_file():
            continue

        needed_pairs = [
            (prec, L)
            for prec in precs
            for L in seq_lens
            if (cfg_name, prec.upper(), L) not in existing_csv or (cfg_name, prec.upper(), L) not in existing_json
        ]
        if not needed_pairs:
            for prec in precs:
                for L in seq_lens:
                    csv_rows.append(existing_csv[(cfg_name, prec.upper(), L)])
                    json_rows.append(existing_json[(cfg_name, prec.upper(), L)])
            print(f"[OK] Reusing cached PyTorch latency for {cfg_name}")
            continue

        reset_gpu_memory(device)
        model, audit = load_stage2_checkpoint_strict(ckpt_key, ckpt_path, deploy_bcgg=deploy, device=device)
        for prec in precs:
            for L in seq_lens:
                k = (cfg_name, prec.upper(), L)
                if k in existing_csv and k in existing_json:
                    csv_rows.append(existing_csv[k])
                    json_rows.append(existing_json[k])
                    continue
                st = _bench_pytorch_clip(model, L, prec, warmup=warmup, runs=runs, device=device)
                c_row = {
                    "config": cfg_name,
                    "precision": prec.upper(),
                    "batch_size": 1,
                    "seq_len": L,
                    "backbone_clip_p50_ms": f"{st['backbone_clip_p50_ms']:.3f}",
                    "temporal_head_p50_ms": f"{st['temporal_head_p50_ms']:.3f}",
                    "total_clip_p5_ms": f"{st['total_clip_p5_ms']:.3f}",
                    "total_clip_p50_ms": f"{st['total_clip_p50_ms']:.3f}",
                    "total_clip_p95_ms": f"{st['total_clip_p95_ms']:.3f}",
                    "total_clip_mean_ms": f"{st['total_clip_mean_ms']:.3f}",
                    "clips_per_sec": f"{st['clips_per_sec']:.2f}",
                    "peak_vram_MiB": f"{st['peak_vram_MiB']:.2f}",
                }
                j_row = {
                    "config": cfg_name,
                    "checkpoint_sha256": audit["sha256"],
                    "precision": prec.upper(),
                    "batch_size": 1,
                    "seq_len": L,
                    **st,
                }
                csv_rows.append(c_row)
                json_rows.append(j_row)
                print(
                    f"[03b | {prec.upper():4s} | L={L:2d}] {cfg_name:<30} | "
                    f"bb={st['backbone_clip_p50_ms']:.2f}ms + th={st['temporal_head_p50_ms']:.2f}ms "
                    f"=> total_p50={st['total_clip_p50_ms']:.2f}ms (p95={st['total_clip_p95_ms']:.2f}ms, vram={st['peak_vram_MiB']:.1f}MiB)"
                )
        del model
        reset_gpu_memory(device)

    write_csv(csv_rows, csv_path)
    write_json(
        {
            "hardware": clock_meta,
            "thermals_after_c": read_jetson_thermals(),
            "warmup_runs": warmup,
            "measure_runs": runs,
            "records": json_rows,
        },
        json_path,
    )


if __name__ == "__main__":
    main()
