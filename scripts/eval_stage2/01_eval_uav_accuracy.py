#!/usr/bin/env python3
"""Stage 2 Step 01: Evaluate UAV-Anti-UAV Retrieval Accuracy, Win/Loss & Video-Level Bootstrap CI."""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common_stage2 import (  # noqa: E402
    UAVClipPairDataset,
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


def _paired_win_loss(r1_a: np.ndarray, r1_b: np.ndarray) -> Dict[str, Any]:
    a_ok = r1_a > 50.0
    b_ok = r1_b > 50.0
    both_ok = int(np.sum(a_ok & b_ok))
    a_wins = int(np.sum(a_ok & (~b_ok)))
    a_losses = int(np.sum((~a_ok) & b_ok))
    both_fail = int(np.sum((~a_ok) & (~b_ok)))
    discordant = a_wins + a_losses
    chi2 = ((abs(a_wins - a_losses) - 1.0) ** 2 / discordant) if discordant > 0 else 0.0
    p_approx = math.erfc(math.sqrt(chi2 / 2.0)) if discordant > 0 else 1.0
    return {
        "n_queries": int(len(r1_a)),
        "both_correct": both_ok,
        "model_a_wins": a_wins,
        "model_a_losses": a_losses,
        "both_wrong": both_fail,
        "net_gain": int(a_wins - a_losses),
        "mcnemar_chi2_cc": round(float(chi2), 4),
        "mcnemar_p_approx": float(p_approx),
    }


def _video_bootstrap_ci(
    seq_ids: Sequence[str],
    metrics_a: Dict[str, np.ndarray],
    metrics_ref: Dict[str, np.ndarray],
    n_boot: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    vid_groups: Dict[str, List[int]] = defaultdict(list)
    for idx, sid in enumerate(seq_ids):
        vid_groups[str(sid)].append(idx)
    vids = sorted(vid_groups.keys())
    V = len(vids)

    n_q_v = np.array([len(vid_groups[v]) for v in vids], dtype=np.float64)
    dr1_v = np.array([np.mean(metrics_a["r1"][vid_groups[v]] - metrics_ref["r1"][vid_groups[v]]) for v in vids])
    dmap_v = np.array([np.mean(metrics_a["ap"][vid_groups[v]] - metrics_ref["ap"][vid_groups[v]]) for v in vids])
    dminp_v = np.array([np.mean(metrics_a["inp"][vid_groups[v]] - metrics_ref["inp"][vid_groups[v]]) for v in vids])
    sum_dr1_v = dr1_v * n_q_v
    sum_dmap_v = dmap_v * n_q_v
    sum_dminp_v = dminp_v * n_q_v

    rng = np.random.default_rng(seed)
    chunk = 2000
    pv_r1, pv_map, pv_minp = [], [], []
    pq_r1, pq_map, pq_minp = [], [], []
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        idx = rng.integers(0, V, size=(m, V))
        pv_r1.append(dr1_v[idx].mean(axis=1))
        pv_map.append(dmap_v[idx].mean(axis=1))
        pv_minp.append(dminp_v[idx].mean(axis=1))

        denom = np.maximum(1.0, n_q_v[idx].sum(axis=1))
        pq_r1.append(sum_dr1_v[idx].sum(axis=1) / denom)
        pq_map.append(sum_dmap_v[idx].sum(axis=1) / denom)
        pq_minp.append(sum_dminp_v[idx].sum(axis=1) / denom)
        done += m

    def _ci(arr_list: List[np.ndarray], obs: float) -> Dict[str, float]:
        arr = np.concatenate(arr_list)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return {
            "observed_diff": float(obs),
            "boot_mean": float(np.mean(arr)),
            "ci95_low": float(lo),
            "ci95_high": float(hi),
            "p_boot_le_zero": float(np.mean(arr <= 0.0)),
        }

    return {
        "n_videos": V,
        "bootstrap_iters": n_boot,
        "seed": seed,
        "per_query_delta_R1": _ci(pq_r1, float(np.mean(metrics_a["r1"] - metrics_ref["r1"]))),
        "per_query_delta_mAP": _ci(pq_map, float(np.mean(metrics_a["ap"] - metrics_ref["ap"]))),
        "per_query_delta_mINP": _ci(pq_minp, float(np.mean(metrics_a["inp"] - metrics_ref["inp"]))),
        "per_video_delta_R1": _ci(pv_r1, float(np.mean(dr1_v))),
        "per_video_delta_mAP": _ci(pv_map, float(np.mean(dmap_v))),
        "per_video_delta_mINP": _ci(pv_minp, float(np.mean(dminp_v))),
    }


def _summary_row(model_name: str, eval_mode: str, num_frames: int, res: Dict[str, Any], round_vals: bool = True) -> Dict[str, Any]:
    def _fmt(v: float) -> Any:
        return f"{v:.2f}" if round_vals else float(v)

    return {
        "model": model_name,
        "eval_mode": eval_mode,
        "num_frames": int(num_frames),
        "num_queries": int(res["num_queries"]),
        "num_videos": int(res["num_videos"]),
        "per_query_mAP": _fmt(res["per_query_mAP"]),
        "per_query_R1": _fmt(res["per_query_R1"]),
        "per_query_R5": _fmt(res["per_query_R5"]),
        "per_query_R10": _fmt(res["per_query_R10"]),
        "per_query_mINP": _fmt(res["per_query_mINP"]),
        "per_video_mAP": _fmt(res["per_video_mAP"]),
        "per_video_R1": _fmt(res["per_video_R1"]),
        "per_video_R5": _fmt(res["per_video_R5"]),
        "per_video_R10": _fmt(res["per_video_R10"]),
        "per_video_mINP": _fmt(res["per_video_mINP"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 Step 01: UAV-Anti-UAV Retrieval Accuracy & Bootstrap CI")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--force-extract", action="store_true", help="Ignore existing cache and re-extract all models")
    args = parser.parse_args()

    cfg = load_stage2_config(args.config)
    out_root = resolve_output_dir(cfg, args.output_dir)
    out_dir = out_root / "01_accuracy_benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_root / "03_efficiency_and_latency_orin" / "engines" / "stage2_eval_cache.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_cfg = cfg.get("eval", {})
    proc_dir = resolve_path(cfg.get("processed_dir", "data/processed"))
    test_img_dir = proc_dir / "test"
    L = int(eval_cfg.get("primary_num_frames", 8))
    bs = int(eval_cfg.get("batch_size", 16))
    nw = int(eval_cfg.get("num_workers", 4))
    n_boot = int(eval_cfg.get("bootstrap_iters", 10000))
    seed = int(eval_cfg.get("seed", 42))
    tfm = build_stage2_transform(cfg)

    all_pairs = load_processed_pairs(proc_dir, split="test", clean_uav_only=False)
    clean_mask = np.array([p["origin"] == "uav_anti_uav" for p in all_pairs], dtype=bool)
    clean_indices = np.flatnonzero(clean_mask)
    clean_pairs = [all_pairs[i] for i in clean_indices]
    all_pids = np.array([p["identity_id"] for p in all_pairs], dtype=np.int64)
    all_seqs = [p["sequence_id"] for p in all_pairs]
    all_evs = np.array([p["event_index"] for p in all_pairs], dtype=np.int64)
    clean_pids = all_pids[clean_indices]
    clean_seqs = [all_seqs[i] for i in clean_indices]
    clean_evs = all_evs[clean_indices]

    print(
        f"[INFO] 01_eval_uav_accuracy: full_pool={len(all_pairs)} pairs ({len(set(all_seqs))} vids) | "
        f"clean_uav={len(clean_pairs)} pairs ({len(set(clean_seqs))} vids) | L={L}"
    )

    loader_multi = make_clip_loader(UAVClipPairDataset(test_img_dir, all_pairs, tfm, num_frames=L), bs, nw)
    loader_1f = make_clip_loader(UAVClipPairDataset(test_img_dir, all_pairs, tfm, num_frames=1, single_frame=True), bs, nw)

    extracted: Dict[str, Dict[str, Any]] = {}
    if not args.force_extract and cache_path.is_file():
        blob = torch.load(cache_path, map_location="cpu", weights_only=False)
        if int(blob.get("all_pairs_count", -1)) == len(all_pairs) and len(blob.get("clean_indices", [])) == len(clean_indices):
            extracted = dict(blob.get("embeddings", {}))
            print(f"[OK] Loaded existing embedding cache ({len(extracted)} spaces): {cache_path}")

    cache_dirty = False

    # 1. Model A (ConvNeXt + BCGG + Bi-Mamba + AttnPool) + GASNet 1-frame
    if "model_a::fused" not in extracted or "gasnet_1frame" not in extracted:
        ckpt_a = resolve_ckpt_path(cfg, "uavreid_convnext_bcgg_mamba")
        reset_gpu_memory(device)
        model_a, _ = load_stage2_checkpoint_strict("uavreid_convnext_bcgg_mamba", ckpt_a, deploy_bcgg=True, device=device)
        extracted["gasnet_1frame"] = extract_stage2_embeddings(model_a, loader_1f, device=device, single_frame=True)["feats"]["gasnet_1frame"]
        ext_a = extract_stage2_embeddings(model_a, loader_multi, device=device, single_frame=False)["feats"]
        for sp, tens in ext_a.items():
            extracted[f"model_a::{sp}"] = tens
        del model_a
        reset_gpu_memory(device)
        cache_dirty = True

    # 2. Model B (ConvNeXt + BCGG + Visual Mean + BNNeck, temporal=none)
    if "model_b::fused" not in extracted:
        ckpt_b = resolve_ckpt_path(cfg, "uavreid_convnext_bcgg_mean_head")
        reset_gpu_memory(device)
        model_b, _ = load_stage2_checkpoint_strict("uavreid_convnext_bcgg_mean_head", ckpt_b, deploy_bcgg=True, device=device)
        extracted["model_b::fused"] = extract_stage2_embeddings(model_b, loader_multi, device=device)["feats"]["fused"]
        del model_b
        reset_gpu_memory(device)
        cache_dirty = True

    # 3. Model C (ConvNeXt + RGA + Bi-Mamba)
    if "model_c::fused" not in extracted:
        ckpt_c = resolve_ckpt_path(cfg, "uavreid_convnext_rga_mamba")
        reset_gpu_memory(device)
        model_c, _ = load_stage2_checkpoint_strict("uavreid_convnext_rga_mamba", ckpt_c, deploy_bcgg=False, device=device)
        extracted["model_c::fused"] = extract_stage2_embeddings(model_c, loader_multi, device=device)["feats"]["fused"]
        del model_c
        reset_gpu_memory(device)
        cache_dirty = True

    # 4. Extended 1: ConvNeXt + BCGG + TemporalAttentionEncoder + AttnPool
    ckpts_map = cfg.get("checkpoints") or {}
    if "uavreid_convnext_bcgg_attn" in ckpts_map and "model_attn::fused" not in extracted:
        ckpt_attn = resolve_ckpt_path(cfg, "uavreid_convnext_bcgg_attn")
        if ckpt_attn.is_file():
            reset_gpu_memory(device)
            model_attn, _ = load_stage2_checkpoint_strict("uavreid_convnext_bcgg_attn", ckpt_attn, deploy_bcgg=True, device=device)
            extracted["model_attn::fused"] = extract_stage2_embeddings(model_attn, loader_multi, device=device)["feats"]["fused"]
            del model_attn
            reset_gpu_memory(device)
            cache_dirty = True

    # 5. Extended 2: ConvNeXt + BCGG + Bi-Mamba + MeanPool
    if "uavreid_convnext_bcgg_mamba_meanpool" in ckpts_map and "model_mamba_mean::fused" not in extracted:
        ckpt_mm = resolve_ckpt_path(cfg, "uavreid_convnext_bcgg_mamba_meanpool")
        if ckpt_mm.is_file():
            reset_gpu_memory(device)
            model_mm, _ = load_stage2_checkpoint_strict("uavreid_convnext_bcgg_mamba_meanpool", ckpt_mm, deploy_bcgg=True, device=device)
            extracted["model_mamba_mean::fused"] = extract_stage2_embeddings(model_mm, loader_multi, device=device)["feats"]["fused"]
            del model_mm
            reset_gpu_memory(device)
            cache_dirty = True

    if cache_dirty:
        torch.save(
            {"all_pairs_count": len(all_pairs), "clean_indices": clean_indices, "embeddings": extracted},
            cache_path,
        )
        print(f"[OK] Saved/updated Stage 2 FP32 embedding cache: {cache_path}")

    eval_specs_clean = [
        ("gasnet_1frame", "gasnet_1frame", 1),
        ("gasnet_visual_mean_L8_no_bn", "model_a::visual_mean", L),
        ("uavreid_convnext_bcgg_mean_head", "model_b::fused", L),
        ("uavreid_convnext_bcgg_mamba", "model_a::fused", L),
        ("uavreid_convnext_rga_mamba", "model_c::fused", L),
    ]
    eval_specs_full = [
        ("gasnet_1frame", "gasnet_1frame", 1),
        ("uavreid_convnext_bcgg_mean_head", "model_b::fused", L),
        ("uavreid_convnext_bcgg_mamba", "model_a::fused", L),
        ("uavreid_convnext_rga_mamba", "model_c::fused", L),
    ]
    if "model_attn::fused" in extracted:
        eval_specs_clean.append(("uavreid_convnext_bcgg_attn", "model_attn::fused", L))
        eval_specs_full.append(("uavreid_convnext_bcgg_attn", "model_attn::fused", L))
    if "model_mamba_mean::fused" in extracted:
        eval_specs_clean.append(("uavreid_convnext_bcgg_mamba_meanpool", "model_mamba_mean::fused", L))
        eval_specs_full.append(("uavreid_convnext_bcgg_mamba_meanpool", "model_mamba_mean::fused", L))
    eval_specs_clean.extend(
        [
            ("model_a_temporal_token_512", "model_a::temporal_token", L),
            ("model_a_pre_bn_1472", "model_a::pre_bn", L),
        ]
    )

    summary_csv: List[Dict[str, Any]] = []
    summary_json: List[Dict[str, Any]] = []
    per_query_rows: List[Dict[str, Any]] = []
    clean_results: Dict[str, Dict[str, Any]] = {}

    clean_t = torch.from_numpy(clean_indices).long()
    for disp_name, ek, nf in eval_specs_clean:
        qf = extracted[ek]["qf"][clean_t]
        gf = extracted[ek]["gf"][clean_t]
        res = evaluate_uav_retrieval(qf, gf, clean_pids, clean_seqs, device=device)
        clean_results[disp_name] = res
        summary_csv.append(_summary_row(disp_name, "clean_uav_anti_uav_2565", nf, res, round_vals=True))
        summary_json.append(_summary_row(disp_name, "clean_uav_anti_uav_2565", nf, res, round_vals=False))
        print(
            f"[CLEAN 2565] {disp_name:<38} | PQ mAP={res['per_query_mAP']:.2f}% R1={res['per_query_R1']:.2f}% "
            f"R5={res['per_query_R5']:.2f}% mINP={res['per_query_mINP']:.2f}% | "
            f"PV R1={res['per_video_R1']:.2f}% PV mAP={res['per_video_mAP']:.2f}%"
        )
        pq = res["per_query"]
        for qi in range(len(clean_pairs)):
            per_query_rows.append(
                {
                    "eval_mode": "clean_uav_anti_uav_2565",
                    "model": disp_name,
                    "sequence_id": clean_seqs[qi],
                    "event_index": int(clean_evs[qi]),
                    "identity_id": int(clean_pids[qi]),
                    "rank1_pos": int(pq["rank1_pos"][qi]),
                    "r1": int(round(pq["r1"][qi] / 100.0)),
                    "ap": f"{pq['ap'][qi]:.4f}",
                    "inp": f"{pq['inp'][qi]:.4f}",
                }
            )

    for disp_name, ek, nf in eval_specs_full:
        res = evaluate_uav_retrieval(extracted[ek]["qf"], extracted[ek]["gf"], all_pids, all_seqs, device=device)
        summary_csv.append(_summary_row(disp_name, "full_processed_2630", nf, res, round_vals=True))
        summary_json.append(_summary_row(disp_name, "full_processed_2630", nf, res, round_vals=False))
        print(
            f"[FULL  2630] {disp_name:<38} | PQ mAP={res['per_query_mAP']:.2f}% R1={res['per_query_R1']:.2f}% "
            f"R5={res['per_query_R5']:.2f}% mINP={res['per_query_mINP']:.2f}%"
        )

    write_csv(summary_csv, out_dir / "uav_accuracy_summary.csv")
    write_json(summary_json, out_dir / "uav_accuracy_summary.json")
    write_csv(per_query_rows, out_dir / "per_query_ranks.csv")

    pq_a = clean_results["uavreid_convnext_bcgg_mamba"]["per_query"]
    pq_b = clean_results["uavreid_convnext_bcgg_mean_head"]["per_query"]
    pq_g1 = clean_results["gasnet_1frame"]["per_query"]
    pq_c = clean_results["uavreid_convnext_rga_mamba"]["per_query"]

    comparisons: Dict[str, Any] = {
        "eval_mode": "clean_uav_anti_uav_2565",
        "model_a_vs_model_b_visual_mean_bnneck": {
            "win_loss": _paired_win_loss(pq_a["r1"], pq_b["r1"]),
            "bootstrap_ci": _video_bootstrap_ci(clean_seqs, pq_a, pq_b, n_boot=n_boot, seed=seed),
        },
        "model_a_vs_gasnet_1frame": {
            "win_loss": _paired_win_loss(pq_a["r1"], pq_g1["r1"]),
            "bootstrap_ci": _video_bootstrap_ci(clean_seqs, pq_a, pq_g1, n_boot=n_boot, seed=seed),
        },
        "model_a_bcgg_vs_model_c_rga": {
            "win_loss": _paired_win_loss(pq_a["r1"], pq_c["r1"]),
            "bootstrap_ci": _video_bootstrap_ci(clean_seqs, pq_a, pq_c, n_boot=n_boot, seed=seed),
        },
    }
    if "uavreid_convnext_bcgg_attn" in clean_results:
        pq_attn = clean_results["uavreid_convnext_bcgg_attn"]["per_query"]
        comparisons["model_a_mamba_attn_vs_transformer_attn"] = {
            "win_loss": _paired_win_loss(pq_a["r1"], pq_attn["r1"]),
            "bootstrap_ci": _video_bootstrap_ci(clean_seqs, pq_a, pq_attn, n_boot=n_boot, seed=seed),
        }
    if "uavreid_convnext_bcgg_mamba_meanpool" in clean_results:
        pq_mm = clean_results["uavreid_convnext_bcgg_mamba_meanpool"]["per_query"]
        comparisons["model_a_mamba_attn_vs_mamba_meanpool"] = {
            "win_loss": _paired_win_loss(pq_a["r1"], pq_mm["r1"]),
            "bootstrap_ci": _video_bootstrap_ci(clean_seqs, pq_a, pq_mm, n_boot=n_boot, seed=seed),
        }

    write_json(comparisons, out_dir / "bootstrap_ci_and_winloss.json")

    boot_rows = []
    for name, block in comparisons.items():
        if not isinstance(block, dict):
            continue
        wl = block["win_loss"]
        ci = block["bootstrap_ci"]
        boot_rows.append(
            {
                "comparison": name,
                "n_queries": wl["n_queries"],
                "model_a_wins": wl["model_a_wins"],
                "model_a_losses": wl["model_a_losses"],
                "net_gain": wl["net_gain"],
                "mcnemar_p": f"{wl['mcnemar_p_approx']:.6g}",
                "pq_delta_R1": f"{ci['per_query_delta_R1']['observed_diff']:+.2f}",
                "pq_delta_R1_ci95": f"[{ci['per_query_delta_R1']['ci95_low']:+.2f}, {ci['per_query_delta_R1']['ci95_high']:+.2f}]",
                "pq_delta_mAP": f"{ci['per_query_delta_mAP']['observed_diff']:+.2f}",
                "pq_delta_mAP_ci95": f"[{ci['per_query_delta_mAP']['ci95_low']:+.2f}, {ci['per_query_delta_mAP']['ci95_high']:+.2f}]",
                "pv_delta_R1": f"{ci['per_video_delta_R1']['observed_diff']:+.2f}",
                "pv_delta_R1_ci95": f"[{ci['per_video_delta_R1']['ci95_low']:+.2f}, {ci['per_video_delta_R1']['ci95_high']:+.2f}]",
            }
        )
    write_csv(boot_rows, out_dir / "bootstrap_ci_and_winloss.csv")


if __name__ == "__main__":
    main()
