#!/usr/bin/env python3
"""Stage 2 Step 04: Precision Consistency (PyTorch FP32 vs FP16 vs TensorRT) for Model A."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common_stage2 import (  # noqa: E402
    TensorRT10Runner,
    UAVClipPairDataset,
    build_or_load_stage2_engines,
    build_stage2_transform,
    evaluate_uav_retrieval,
    extract_stage2_embeddings,
    load_processed_pairs,
    load_stage2_config,
    make_clip_loader,
    reset_gpu_memory,
    resolve_ckpt_path,
    resolve_output_dir,
    resolve_path,
    write_csv,
    write_json,
)
from stage2_models import load_stage2_checkpoint_strict  # noqa: E402


def _extract_trt_embeddings(
    bb_runner: TensorRT10Runner,
    tp_runner: TensorRT10Runner,
    loader,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    q_list, g_list = [], []
    for b_clip, a_clip, _, _, _ in loader:
        B, N, C, H, W = b_clip.shape
        # Backbone TRT executes in chunks <= 16 images (max_bs_backbone=16)
        b_flat = b_clip.view(B * N, C, H, W).to(device, non_blocking=True)
        a_flat = a_clip.view(B * N, C, H, W).to(device, non_blocking=True)

        def _run_bb_chunks(flat_t: torch.Tensor) -> torch.Tensor:
            outs = []
            for start in range(0, flat_t.size(0), 16):
                chunk = flat_t[start : start + 16]
                outs.append(bb_runner.infer(chunk).float().clone())
            return torch.cat(outs, dim=0).view(B, N, -1)

        g_seq = _run_bb_chunks(b_flat)
        q_seq = _run_bb_chunks(a_flat)
        g_emb = tp_runner.infer(g_seq).float().clone().cpu()
        q_emb = tp_runner.infer(q_seq).float().clone().cpu()
        g_list.append(g_emb)
        q_list.append(q_emb)
    return {"qf": torch.cat(q_list, dim=0), "gf": torch.cat(g_list, dim=0)}


def _compare_embeddings(emb_ref: torch.Tensor, emb_cmp: torch.Tensor) -> Dict[str, float]:
    n_ref = F.normalize(emb_ref.float(), p=2, dim=1)
    n_cmp = F.normalize(emb_cmp.float(), p=2, dim=1)
    diff = (n_ref - n_cmp).abs()
    cos = F.cosine_similarity(n_ref, n_cmp, dim=1)
    return {
        "min_cosine_sim": float(cos.min().item()),
        "mean_cosine_sim": float(cos.mean().item()),
        "p5_cosine_sim": float(np.percentile(cos.numpy(), 5)),
        "mean_abs_diff": float(diff.mean().item()),
        "max_abs_diff": float(diff.max().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 Step 04: Precision Consistency")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--temporal-fp32", action="store_true", help="Evaluate TRT with FP32 temporal engine")
    args = parser.parse_args()

    cfg = load_stage2_config(args.config)
    out_root = resolve_output_dir(cfg, args.output_dir)
    out_dir = out_root / "04_precision_consistency"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_cfg = cfg.get("eval", {})
    proc_dir = resolve_path(cfg.get("processed_dir", "data/processed"))
    L = int(eval_cfg.get("primary_num_frames", 8))
    bs = int(eval_cfg.get("batch_size", 16))
    nw = int(eval_cfg.get("num_workers", 4))
    tfm = build_stage2_transform(cfg)

    clean_pairs = load_processed_pairs(proc_dir, split="test", clean_uav_only=True)
    clean_pids = np.array([p["identity_id"] for p in clean_pairs], dtype=np.int64)
    clean_seqs = [p["sequence_id"] for p in clean_pairs]
    loader = make_clip_loader(UAVClipPairDataset(proc_dir / "test", clean_pairs, tfm, num_frames=L), bs, nw)

    ckpt_a = resolve_ckpt_path(cfg, "uavreid_convnext_bcgg_mamba")
    reset_gpu_memory(device)
    model_a, _ = load_stage2_checkpoint_strict("uavreid_convnext_bcgg_mamba", ckpt_a, deploy_bcgg=True, device=device)

    emb_fp32 = extract_stage2_embeddings(model_a, loader, device=device, precision="fp32")["feats"]["fused"]
    emb_fp16 = extract_stage2_embeddings(model_a, loader, device=device, precision="fp16")["feats"]["fused"]
    del model_a
    reset_gpu_memory(device)

    engines_fp16 = build_or_load_stage2_engines(cfg, out_root, seq_lens=[L], temporal_fp16=True)
    bb_runner = TensorRT10Runner(engines_fp16["backbone_engine"], device=device)
    tp_runner_fp16 = TensorRT10Runner(engines_fp16["temporal_engines"][L], device=device)
    try:
        emb_trt_fp16 = _extract_trt_embeddings(bb_runner, tp_runner_fp16, loader, device=device)
    finally:
        tp_runner_fp16.close()

    res_fp32 = evaluate_uav_retrieval(emb_fp32["qf"], emb_fp32["gf"], clean_pids, clean_seqs, device=device)
    res_fp16 = evaluate_uav_retrieval(emb_fp16["qf"], emb_fp16["gf"], clean_pids, clean_seqs, device=device)
    res_trt_fp16 = evaluate_uav_retrieval(emb_trt_fp16["qf"], emb_trt_fp16["gf"], clean_pids, clean_seqs, device=device)

    diff_fp16_vs_fp32 = {
        "query": _compare_embeddings(emb_fp32["qf"], emb_fp16["qf"]),
        "gallery": _compare_embeddings(emb_fp32["gf"], emb_fp16["gf"]),
    }
    diff_trt_fp16_vs_fp32 = {
        "query": _compare_embeddings(emb_fp32["qf"], emb_trt_fp16["qf"]),
        "gallery": _compare_embeddings(emb_fp32["gf"], emb_trt_fp16["gf"]),
    }

    records = [
        ("pytorch_fp32", emb_fp32, res_fp32, None),
        ("pytorch_fp16", emb_fp16, res_fp16, diff_fp16_vs_fp32),
        ("tensorrt_fp16", emb_trt_fp16, res_trt_fp16, diff_trt_fp16_vs_fp32),
    ]

    # Check whether TRT FP16 shows any temporal exp() numerical degradation and also test mixed TRT (FP16 bb + FP32 temporal) if requested or if |ΔR1| > 0.5%
    d_r1_trt = res_trt_fp16["per_query_R1"] - res_fp32["per_query_R1"]
    diff_trt_mixed = None
    if args.temporal_fp32 or abs(d_r1_trt) > 0.5:
        print(f"[INFO] Measuring mixed TRT (FP16 backbone + FP32 temporal) since ΔR1={d_r1_trt:+.2f}% or --temporal-fp32 passed...")
        engines_tp32 = build_or_load_stage2_engines(cfg, out_root, seq_lens=[L], temporal_fp16=False)
        tp_runner_fp32 = TensorRT10Runner(engines_tp32["temporal_engines"][L], device=device)
        try:
            emb_trt_mixed = _extract_trt_embeddings(bb_runner, tp_runner_fp32, loader, device=device)
        finally:
            tp_runner_fp32.close()
        res_trt_mixed = evaluate_uav_retrieval(emb_trt_mixed["qf"], emb_trt_mixed["gf"], clean_pids, clean_seqs, device=device)
        diff_trt_mixed = {
            "query": _compare_embeddings(emb_fp32["qf"], emb_trt_mixed["qf"]),
            "gallery": _compare_embeddings(emb_fp32["gf"], emb_trt_mixed["gf"]),
        }
        records.append(("tensorrt_fp16bb_fp32tp", emb_trt_mixed, res_trt_mixed, diff_trt_mixed))

    bb_runner.close()
    reset_gpu_memory(device)

    csv_rows: List[Dict[str, Any]] = []
    json_rows: List[Dict[str, Any]] = []
    for rt_name, _, res, diff_info in records:
        d_map = res["per_query_mAP"] - res_fp32["per_query_mAP"]
        d_r1 = res["per_query_R1"] - res_fp32["per_query_R1"]
        csv_rows.append(
            {
                "runtime": rt_name,
                "eval_mode": "clean_uav_anti_uav_2565",
                "num_frames": L,
                "mAP": f"{res['per_query_mAP']:.2f}",
                "R1": f"{res['per_query_R1']:.2f}",
                "R5": f"{res['per_query_R5']:.2f}",
                "R10": f"{res['per_query_R10']:.2f}",
                "mINP": f"{res['per_query_mINP']:.2f}",
                "delta_mAP_vs_fp32": f"{d_map:+.2f}",
                "delta_R1_vs_fp32": f"{d_r1:+.2f}",
                "min_cosine_sim_vs_fp32": f"{diff_info['query']['min_cosine_sim']:.6f}" if diff_info else "1.000000",
                "mean_abs_diff_vs_fp32": f"{diff_info['query']['mean_abs_diff']:.6e}" if diff_info else "0.000000e+00",
            }
        )
        json_rows.append(
            {
                "runtime": rt_name,
                "eval_mode": "clean_uav_anti_uav_2565",
                "num_frames": L,
                "mAP": float(res["per_query_mAP"]),
                "R1": float(res["per_query_R1"]),
                "R5": float(res["per_query_R5"]),
                "R10": float(res["per_query_R10"]),
                "mINP": float(res["per_query_mINP"]),
                "per_video_R1": float(res["per_video_R1"]),
                "per_video_mAP": float(res["per_video_mAP"]),
                "delta_mAP_vs_fp32": float(d_map),
                "delta_R1_vs_fp32": float(d_r1),
                "embedding_diff_vs_fp32": diff_info,
            }
        )
        print(
            f"[04 | {rt_name:<22}] mAP={res['per_query_mAP']:.2f}% ({d_map:+.2f}) | "
            f"R1={res['per_query_R1']:.2f}% ({d_r1:+.2f}) | R5={res['per_query_R5']:.2f}% | mINP={res['per_query_mINP']:.2f}%"
        )

    write_csv(csv_rows, out_dir / "stage2_precision_comparison.csv")
    write_json(json_rows, out_dir / "stage2_precision_comparison.json")
    write_json(
        {
            "pytorch_fp16_vs_fp32": diff_fp16_vs_fp32,
            "tensorrt_fp16_vs_fp32": diff_trt_fp16_vs_fp32,
            "tensorrt_mixed_fp32tp_vs_fp32": diff_trt_mixed,
        },
        out_dir / "trt_vs_pytorch_embedding_diff.json",
    )


if __name__ == "__main__":
    main()
