#!/usr/bin/env python3
"""Stage 2 Step 03c: Export ONNX + Benchmark TensorRT (kClip & kStream) + Re-Acquisition Time Budget."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common_stage2 import (  # noqa: E402
    TensorRT10Runner,
    build_or_load_stage2_engines,
    load_stage2_config,
    maybe_lock_jetson_clocks,
    read_jetson_thermals,
    reset_gpu_memory,
    resolve_output_dir,
    summarize_latencies,
    write_csv,
    write_json,
)


def _preprocess_bgr_256_crop(bgr_crop: np.ndarray, mean_arr: np.ndarray, inv_std_arr: np.ndarray) -> np.ndarray:
    """Match Resize(256)+CenterCrop(224)+Normalize preprocess on CPU OpenCV BGR image."""
    resized = cv2.resize(bgr_crop, (256, 256), interpolation=cv2.INTER_LINEAR)
    cropped = resized[16:240, 16:240]
    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normed = (rgb - mean_arr) * inv_std_arr
    return np.ascontiguousarray(normed.transpose(2, 0, 1)[None, ...], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 Step 03c: Export ONNX & Benchmark TRT (kClip & kStream)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--lock-clocks", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--temporal-fp32", action="store_true", help="Build temporal Mamba engine in FP32 instead of FP16")
    args = parser.parse_args()

    cfg = load_stage2_config(args.config)
    out_root = resolve_output_dir(cfg, args.output_dir)
    out_dir = out_root / "03_efficiency_and_latency_orin"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clock_meta = maybe_lock_jetson_clocks(args.lock_clocks)

    eff_cfg = cfg.get("efficiency", {})
    trt_cfg = eff_cfg.get("trt", {})
    seq_lens = [int(x) for x in eff_cfg.get("seq_lens", [4, 8, 12, 16])]
    warmup = int(trt_cfg.get("warmup_runs", 50))
    runs = int(trt_cfg.get("measure_runs", 200))
    gallery_sz = int(eff_cfg.get("kstream_gallery_size", 2630))
    stride = int(cfg.get("eval", {}).get("frame_stride", 4))

    engines = build_or_load_stage2_engines(
        cfg, out_root, seq_lens=seq_lens, temporal_fp16=(not args.temporal_fp32), force_rebuild=args.force_rebuild
    )
    bb_runner = TensorRT10Runner(engines["backbone_engine"], device=device)
    bb_size_mb = engines["backbone_engine"].stat().st_size / 1e6

    mean_arr = np.array(cfg.get("preprocess", {}).get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    inv_std_arr = 1.0 / np.array(cfg.get("preprocess", {}).get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
    gallery_bank = F.normalize(torch.randn(gallery_sz, 1472, device=device, dtype=torch.float32), p=2, dim=1)
    dummy_bgr = np.random.randint(0, 256, (180, 180, 3), dtype=np.uint8)

    lat_csv: List[Dict[str, Any]] = []
    lat_json: List[Dict[str, Any]] = []

    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e2 = torch.cuda.Event(enable_timing=True)

    # Benchmark per-frame preprocessing + H2D + backbone BS=1 once for kStream
    t_prep_list, t_bb1_list, t_frame_tot_list = [], [], []
    inp_cpu_pin = torch.empty((1, 3, 224, 224), dtype=torch.float32, pin_memory=True)
    import time

    for _ in range(warmup):
        arr = _preprocess_bgr_256_crop(dummy_bgr, mean_arr, inv_std_arr)
        inp_cpu_pin.copy_(torch.from_numpy(arr))
        gpu_in = inp_cpu_pin.to(device, non_blocking=False)
        _ = bb_runner.infer(gpu_in)
        bb_runner.synchronize()

    for _ in range(runs):
        t0 = time.perf_counter()
        arr = _preprocess_bgr_256_crop(dummy_bgr, mean_arr, inv_std_arr)
        inp_cpu_pin.copy_(torch.from_numpy(arr))
        gpu_in = inp_cpu_pin.to(device, non_blocking=False)
        t1 = time.perf_counter()
        _ = bb_runner.infer(gpu_in)
        bb_runner.synchronize()
        t2 = time.perf_counter()
        t_prep_list.append((t1 - t0) * 1000.0)
        t_bb1_list.append((t2 - t1) * 1000.0)
        t_frame_tot_list.append((t2 - t0) * 1000.0)

    kstream_prep_st = summarize_latencies(t_prep_list)
    kstream_bb1_st = summarize_latencies(t_bb1_list)
    kstream_frame_st = summarize_latencies(t_frame_tot_list)
    print(
        f"[03c | kStream Per-Frame] prep+H2D={kstream_prep_st['p50_ms']:.3f}ms | "
        f"backbone_BS1={kstream_bb1_st['p50_ms']:.3f}ms => frame_total_p50={kstream_frame_st['p50_ms']:.3f}ms"
    )

    kstream_reacq_by_L: Dict[int, Dict[str, float]] = {}
    kclip_by_L: Dict[int, Dict[str, float]] = {}

    for L in seq_lens:
        tp_path = engines["temporal_engines"][L]
        tp_runner = TensorRT10Runner(tp_path, device=device)
        tp_size_mb = tp_path.stat().st_size / 1e6

        try:
            # --- Scenario 1: kClip (Offline / Batch Clip: BS=L backbone + BS=1 temporal) ---
            clip_inp = torch.randn(L, 3, 224, 224, device=device, dtype=torch.float32)
            for _ in range(warmup):
                sf = bb_runner.infer(clip_inp).float().unsqueeze(0)
                _ = tp_runner.infer(sf)
            torch.cuda.synchronize(device)

            bb_ms, tp_ms, tot_ms = [], [], []
            for _ in range(runs):
                e0.record()
                sf = bb_runner.infer(clip_inp).float().unsqueeze(0)
                e1.record()
                _ = tp_runner.infer(sf)
                e2.record()
                torch.cuda.synchronize(device)
                bb_ms.append(float(e0.elapsed_time(e1)))
                tp_ms.append(float(e1.elapsed_time(e2)))
                tot_ms.append(float(e0.elapsed_time(e2)))

            bb_st = summarize_latencies(bb_ms)
            tp_st = summarize_latencies(tp_ms)
            tot_st = summarize_latencies(tot_ms)
            kclip_by_L[L] = {
                "backbone_p50_ms": bb_st["p50_ms"],
                "temporal_p50_ms": tp_st["p50_ms"],
                "total_p50_ms": tot_st["p50_ms"],
                "total_p95_ms": tot_st["p95_ms"],
            }
            lat_csv.append(
                {
                    "mode": "kClip",
                    "seq_len": L,
                    "backbone_precision": "FP16",
                    "temporal_precision": engines["temporal_precision"].upper(),
                    "per_frame_backbone_p50_ms": f"{bb_st['p50_ms'] / L:.3f}",
                    "backbone_clip_p50_ms": f"{bb_st['p50_ms']:.3f}",
                    "temporal_and_match_p50_ms": f"{tp_st['p50_ms']:.3f}",
                    "total_p5_ms": f"{tot_st['p5_ms']:.3f}",
                    "total_p50_ms": f"{tot_st['p50_ms']:.3f}",
                    "total_p95_ms": f"{tot_st['p95_ms']:.3f}",
                    "clips_per_sec": f"{tot_st['throughput']:.2f}",
                    "engine_size_MB": f"{bb_size_mb + tp_size_mb:.2f}",
                }
            )
            lat_json.append(
                {
                    "mode": "kClip",
                    "seq_len": L,
                    "backbone_precision": "FP16",
                    "temporal_precision": engines["temporal_precision"].upper(),
                    "backbone_p50_ms": bb_st["p50_ms"],
                    "temporal_p50_ms": tp_st["p50_ms"],
                    **tot_st,
                    "clips_per_sec": tot_st["throughput"],
                    "backbone_engine_MB": round(bb_size_mb, 3),
                    "temporal_engine_MB": round(tp_size_mb, 3),
                    "engine_size_MB": round(bb_size_mb + tp_size_mb, 3),
                }
            )
            print(
                f"[03c | kClip   | L={L:2d}] bb({L})={bb_st['p50_ms']:.2f}ms + temporal={tp_st['p50_ms']:.2f}ms "
                f"=> p50={tot_st['p50_ms']:.2f}ms (p95={tot_st['p95_ms']:.2f}ms, {tot_st['throughput']:.1f} clips/s)"
            )

            # --- Scenario 2: kStream (Online L-th frame: prep+H2D+backbone BS=1 -> ring buffer -> temporal + match(2630)) ---
            ring_buf = torch.randn(1, L, 960, device=device, dtype=torch.float32)
            for _ in range(warmup):
                arr = _preprocess_bgr_256_crop(dummy_bgr, mean_arr, inv_std_arr)
                inp_cpu_pin.copy_(torch.from_numpy(arr))
                gpu_in = inp_cpu_pin.to(device, non_blocking=False)
                feat1 = bb_runner.infer(gpu_in).float()
                ring_buf[:, -1, :].copy_(feat1)
                emb = F.normalize(tp_runner.infer(ring_buf).float(), p=2, dim=1)
                sims = emb @ gallery_bank.t()
                _, _ = torch.topk(sims, k=10, dim=1)
            torch.cuda.synchronize(device)

            tm_list, ks_tot_list = [], []
            for _ in range(runs):
                t_a = time.perf_counter()
                arr = _preprocess_bgr_256_crop(dummy_bgr, mean_arr, inv_std_arr)
                inp_cpu_pin.copy_(torch.from_numpy(arr))
                gpu_in = inp_cpu_pin.to(device, non_blocking=False)
                feat1 = bb_runner.infer(gpu_in).float()
                ring_buf[:, -1, :].copy_(feat1)
                torch.cuda.synchronize(device)
                t_b = time.perf_counter()
                emb = F.normalize(tp_runner.infer(ring_buf).float(), p=2, dim=1)
                sims = emb @ gallery_bank.t()
                vals, idxs = torch.topk(sims, k=10, dim=1)
                _ = idxs.cpu()
                torch.cuda.synchronize(device)
                t_c = time.perf_counter()
                tm_list.append((t_c - t_b) * 1000.0)
                ks_tot_list.append((t_c - t_a) * 1000.0)

            tm_st = summarize_latencies(tm_list)
            ks_st = summarize_latencies(ks_tot_list)
            kstream_reacq_by_L[L] = {
                "per_frame_prep_bb_p50_ms": kstream_frame_st["p50_ms"],
                "per_frame_backbone_p50_ms": kstream_bb1_st["p50_ms"],
                "temporal_and_match_p50_ms": tm_st["p50_ms"],
                "final_step_p50_ms": ks_st["p50_ms"],
                "final_step_p95_ms": ks_st["p95_ms"],
            }
            lat_csv.append(
                {
                    "mode": "kStream",
                    "seq_len": L,
                    "backbone_precision": "FP16",
                    "temporal_precision": engines["temporal_precision"].upper(),
                    "per_frame_backbone_p50_ms": f"{kstream_bb1_st['p50_ms']:.3f}",
                    "backbone_clip_p50_ms": f"{kstream_frame_st['p50_ms']:.3f}",
                    "temporal_and_match_p50_ms": f"{tm_st['p50_ms']:.3f}",
                    "total_p5_ms": f"{ks_st['p5_ms']:.3f}",
                    "total_p50_ms": f"{ks_st['p50_ms']:.3f}",
                    "total_p95_ms": f"{ks_st['p95_ms']:.3f}",
                    "clips_per_sec": f"{1000.0 / ks_st['p50_ms']:.2f}",
                    "engine_size_MB": f"{bb_size_mb + tp_size_mb:.2f}",
                }
            )
            lat_json.append(
                {
                    "mode": "kStream",
                    "seq_len": L,
                    "backbone_precision": "FP16",
                    "temporal_precision": engines["temporal_precision"].upper(),
                    "gallery_size": gallery_sz,
                    **kstream_reacq_by_L[L],
                    "total_p5_ms": ks_st["p5_ms"],
                    "total_p50_ms": ks_st["p50_ms"],
                    "total_p95_ms": ks_st["p95_ms"],
                    "engine_size_MB": round(bb_size_mb + tp_size_mb, 3),
                }
            )
            print(
                f"[03c | kStream | L={L:2d}] frame_bb={kstream_bb1_st['p50_ms']:.2f}ms | "
                f"temporal+match({gallery_sz})={tm_st['p50_ms']:.2f}ms => final_step_p50={ks_st['p50_ms']:.2f}ms"
            )
        finally:
            tp_runner.close()

    bb_runner.close()
    reset_gpu_memory(device)

    write_csv(lat_csv, out_dir / "stage2_tensorrt_latency.csv")
    write_json(
        {
            "hardware": clock_meta,
            "thermals_after_c": read_jetson_thermals(),
            "warmup_runs": warmup,
            "measure_runs": runs,
            "kstream_per_frame_stats": {
                "prep_h2d_p50_ms": kstream_prep_st["p50_ms"],
                "backbone_bs1_p50_ms": kstream_bb1_st["p50_ms"],
                "frame_total_p50_ms": kstream_frame_st["p50_ms"],
            },
            "records": lat_json,
        },
        out_dir / "stage2_tensorrt_latency.json",
    )

    # Re-Acquisition Time Budget using real FPS from 00_env_and_manifest/dataset_audit.json
    audit_path = out_root / "00_env_and_manifest" / "dataset_audit.json"
    median_fps = 30.0
    mean_fps = 30.0
    if audit_path.is_file():
        da = json.loads(audit_path.read_text(encoding="utf-8"))
        fps_st = da.get("clean_test_fps_stats", {})
        if fps_st.get("median"):
            median_fps = float(fps_st["median"])
        if fps_st.get("mean"):
            mean_fps = float(fps_st["mean"])

    budget_csv: List[Dict[str, Any]] = []
    budget_json: List[Dict[str, Any]] = []
    for L in seq_lens:
        cam_med_ms = ((L - 1) * stride / median_fps) * 1000.0
        cam_mean_ms = ((L - 1) * stride / mean_fps) * 1000.0
        ks = kstream_reacq_by_L[L]
        kc = kclip_by_L[L]
        tot_med_ms = cam_med_ms + ks["final_step_p50_ms"]
        tot_mean_ms = cam_mean_ms + ks["final_step_p50_ms"]
        budget_csv.append(
            {
                "seq_len": L,
                "stride": stride,
                "median_fps": f"{median_fps:.2f}",
                "camera_wait_median_fps_ms": f"{cam_med_ms:.2f}",
                "mean_fps": f"{mean_fps:.2f}",
                "camera_wait_mean_fps_ms": f"{cam_mean_ms:.2f}",
                "orin_kstream_per_frame_ms": f"{ks['per_frame_prep_bb_p50_ms']:.3f}",
                "orin_kstream_final_step_ms": f"{ks['final_step_p50_ms']:.3f}",
                "orin_kclip_total_ms": f"{kc['total_p50_ms']:.3f}",
                "total_reacq_kstream_median_fps_ms": f"{tot_med_ms:.2f}",
                "total_reacq_kstream_mean_fps_ms": f"{tot_mean_ms:.2f}",
                "compute_pct_of_budget": f"{100.0 * ks['final_step_p50_ms'] / max(1e-6, tot_med_ms):.2f}",
            }
        )
        budget_json.append(
            {
                "seq_len": L,
                "stride": stride,
                "median_fps": median_fps,
                "mean_fps": mean_fps,
                "camera_wait_median_fps_ms": cam_med_ms,
                "camera_wait_mean_fps_ms": cam_mean_ms,
                "orin_kstream": ks,
                "orin_kclip": kc,
                "total_reacq_kstream_median_fps_ms": tot_med_ms,
                "total_reacq_kstream_mean_fps_ms": tot_mean_ms,
            }
        )
    write_csv(budget_csv, out_dir / "stage2_reacquisition_time_budget.csv")
    write_json(budget_json, out_dir / "stage2_reacquisition_time_budget.json")


if __name__ == "__main__":
    main()
