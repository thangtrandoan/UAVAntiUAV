#!/usr/bin/env python3
"""Stage 2 Step 02: Occlusion/OV, Attribute, Gap Duration, Padding & Length/Order Ablations."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common_stage2 import (  # noqa: E402
    ATTR_NAME_TO_IDX,
    ATTR_NAMES,
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


def _subset_metrics(res: Dict[str, Any], mask: np.ndarray) -> Dict[str, Any]:
    idxs = np.flatnonzero(mask)
    if len(idxs) == 0:
        return {"query_count": 0, "mAP": "", "R1": "", "R5": "", "R10": "", "mINP": ""}
    pq = res["per_query"]
    return {
        "query_count": int(len(idxs)),
        "mAP": float(np.mean(pq["ap"][idxs])),
        "R1": float(np.mean(pq["r1"][idxs])),
        "R5": float(np.mean(pq["r5"][idxs])),
        "R10": float(np.mean(pq["r10"][idxs])),
        "mINP": float(np.mean(pq["inp"][idxs])),
    }


def _fmt_sub_row(group_name: str, model_key: str, sub: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    row: Dict[str, Any] = {"group": group_name, "model": model_key, "query_count": sub["query_count"]}
    if extra:
        row.update(extra)
    for k in ("mAP", "R1", "R5", "R10", "mINP"):
        v = sub[k]
        row[k] = f"{v:.2f}" if isinstance(v, float) else v
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 Step 02: Occlusion, OOV, Gap, Padding & SeqLen Ablation")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--from-cache-only",
        action="store_true",
        help="Recompute only attribute/gap/padding tables from embedding cache without re-running L/order ablation",
    )
    args = parser.parse_args()

    cfg = load_stage2_config(args.config)
    out_root = resolve_output_dir(cfg, args.output_dir)
    out_dir = out_root / "02_occlusion_and_oov_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_root / "03_efficiency_and_latency_orin" / "engines" / "stage2_eval_cache.pt"
    events_csv = out_root / "00_env_and_manifest" / "dataset_events_table.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_cfg = cfg.get("eval", {})
    proc_dir = resolve_path(cfg.get("processed_dir", "data/processed"))
    test_img_dir = proc_dir / "test"
    L0 = int(eval_cfg.get("primary_num_frames", 8))
    seq_lens = [int(x) for x in eval_cfg.get("seq_lens", [4, 8, 12, 16])]
    bs = int(eval_cfg.get("batch_size", 16))
    nw = int(eval_cfg.get("num_workers", 4))
    seed = int(eval_cfg.get("seed", 42))
    tfm = build_stage2_transform(cfg)

    clean_pairs = load_processed_pairs(proc_dir, split="test", clean_uav_only=True)
    clean_pids = np.array([p["identity_id"] for p in clean_pairs], dtype=np.int64)
    clean_seqs = [p["sequence_id"] for p in clean_pairs]

    fps_lookup: Dict[Tuple[str, int], Tuple[float | None, float | None]] = {}
    if events_csv.is_file():
        with events_csv.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("split") == "test":
                    fps_v = float(r["fps"]) if r.get("fps") else None
                    gs_v = float(r["gap_seconds"]) if r.get("gap_seconds") else None
                    fps_lookup[(r["sequence_id"], int(r["event_index"]))] = (fps_v, gs_v)

    gap_frames = np.array([p["gap_frames"] for p in clean_pairs], dtype=np.int64)
    gap_secs = np.array(
        [
            fps_lookup.get((p["sequence_id"], p["event_index"]), (None, None))[1] or np.nan
            for p in clean_pairs
        ],
        dtype=np.float64,
    )
    valid_b = np.array([p["valid_before_frames"] for p in clean_pairs], dtype=np.int64)
    valid_a = np.array([p["valid_after_frames"] for p in clean_pairs], dtype=np.int64)
    attrs_mat = np.array([p["attributes"] for p in clean_pairs], dtype=np.int64)

    emb_l8: Dict[str, Dict[str, torch.Tensor]] = {}
    if cache_path.is_file():
        print(f"[OK] Reusing cached embeddings from {cache_path}")
        blob = torch.load(cache_path, map_location="cpu", weights_only=False)
        clean_t = torch.from_numpy(np.asarray(blob["clean_indices"])).long()
        embs = blob["embeddings"]
        key_map = [
            ("gasnet_1frame", "gasnet_1frame"),
            ("uavreid_convnext_bcgg_mean_head", "model_b::fused"),
            ("uavreid_convnext_bcgg_mamba", "model_a::fused"),
            ("uavreid_convnext_rga_mamba", "model_c::fused"),
            ("uavreid_convnext_bcgg_attn", "model_attn::fused"),
            ("uavreid_convnext_bcgg_mamba_meanpool", "model_mamba_mean::fused"),
        ]
        for mk, ck in key_map:
            if ck in embs:
                emb_l8[mk] = {k: v[clean_t] for k, v in embs[ck].items()}
    else:
        print("[INFO] Cache not found; extracting L=8 embeddings...")
        loader_multi = make_clip_loader(UAVClipPairDataset(test_img_dir, clean_pairs, tfm, num_frames=L0), bs, nw)
        loader_1f = make_clip_loader(UAVClipPairDataset(test_img_dir, clean_pairs, tfm, num_frames=1, single_frame=True), bs, nw)
        ckpt_a = resolve_ckpt_path(cfg, "uavreid_convnext_bcgg_mamba")
        ckpt_b = resolve_ckpt_path(cfg, "uavreid_convnext_bcgg_mean_head")
        reset_gpu_memory(device)
        m_a, _ = load_stage2_checkpoint_strict("uavreid_convnext_bcgg_mamba", ckpt_a, deploy_bcgg=True, device=device)
        emb_l8["gasnet_1frame"] = extract_stage2_embeddings(m_a, loader_1f, device=device, single_frame=True)["feats"]["gasnet_1frame"]
        emb_l8["uavreid_convnext_bcgg_mamba"] = extract_stage2_embeddings(m_a, loader_multi, device=device)["feats"]["fused"]
        del m_a
        reset_gpu_memory(device)
        m_b, _ = load_stage2_checkpoint_strict("uavreid_convnext_bcgg_mean_head", ckpt_b, deploy_bcgg=True, device=device)
        emb_l8["uavreid_convnext_bcgg_mean_head"] = extract_stage2_embeddings(m_b, loader_multi, device=device)["feats"]["fused"]
        del m_b
        reset_gpu_memory(device)

    ordered_candidates = (
        "gasnet_1frame",
        "uavreid_convnext_bcgg_mean_head",
        "uavreid_convnext_bcgg_mamba",
        "uavreid_convnext_rga_mamba",
        "uavreid_convnext_bcgg_attn",
        "uavreid_convnext_bcgg_mamba_meanpool",
    )
    eval_models = [mk for mk in ordered_candidates if mk in emb_l8]
    results_l8 = {
        mk: evaluate_uav_retrieval(emb_l8[mk]["qf"], emb_l8[mk]["gf"], clean_pids, clean_seqs, device=device)
        for mk in eval_models
    }

    # 1. Disjoint OC/OV groups (idx 2 = Out-of-View, idx 7 = Occlusion)
    ov = attrs_mat[:, 2] == 1
    oc = attrs_mat[:, 7] == 1
    oc_ov_groups = [
        ("OV_only", ov & (~oc)),
        ("OC_only", (~ov) & oc),
        ("Both_OV_OC", ov & oc),
        ("Neither", (~ov) & (~oc)),
    ]
    rows_oc_ov, json_oc_ov = [], []
    for gname, mask in oc_ov_groups:
        for mk in eval_models:
            sub = _subset_metrics(results_l8[mk], mask)
            rows_oc_ov.append(_fmt_sub_row(gname, mk, sub))
            json_oc_ov.append({"group": gname, "model": mk, **sub})
    write_csv(rows_oc_ov, out_dir / "by_disjoint_oc_ov.csv")
    write_json(json_oc_ov, out_dir / "by_disjoint_oc_ov.json")

    # 2. All Attributes breakdown
    priority_attrs = [
        "Out-of-View",
        "Occlusion",
        "Fast Motion",
        "Scale Variation",
        "Motion Blur",
        "Low Resolution",
        "Fake UAV",
        "Background Clutter",
    ]
    other_attrs = [ATTR_NAMES[i] for i in sorted(ATTR_NAMES) if ATTR_NAMES[i] not in priority_attrs]
    rows_attrs, json_attrs = [], []
    for aname in priority_attrs + other_attrs:
        aidx = ATTR_NAME_TO_IDX[aname]
        mask = attrs_mat[:, aidx] == 1
        for mk in eval_models:
            sub = _subset_metrics(results_l8[mk], mask)
            rows_attrs.append(_fmt_sub_row(aname, mk, sub))
            json_attrs.append({"attribute": aname, "attr_idx": aidx, "model": mk, **sub})
    write_csv(rows_attrs, out_dir / "by_all_attributes.csv")
    write_json(json_attrs, out_dir / "by_all_attributes.json")

    # 3. Gap Duration Buckets
    gap_buckets = [
        ("1-4 frames", (gap_frames >= 1) & (gap_frames <= 4)),
        ("5-16 frames", (gap_frames >= 5) & (gap_frames <= 16)),
        ("17-64 frames", (gap_frames >= 17) & (gap_frames <= 64)),
        (">=65 frames", gap_frames >= 65),
    ]
    rows_gap, json_gap = [], []
    for bname, mask in gap_buckets:
        gs_sub = gap_secs[mask]
        gs_valid = gs_sub[np.isfinite(gs_sub)]
        sec_mean = float(np.mean(gs_valid)) if len(gs_valid) else None
        sec_med = float(np.median(gs_valid)) if len(gs_valid) else None
        for mk in eval_models:
            sub = _subset_metrics(results_l8[mk], mask)
            extra = {
                "gap_seconds_mean": f"{sec_mean:.3f}" if sec_mean is not None else "",
                "gap_seconds_median": f"{sec_med:.3f}" if sec_med is not None else "",
            }
            rows_gap.append(_fmt_sub_row(bname, mk, sub, extra=extra))
            json_gap.append(
                {
                    "gap_bucket": bname,
                    "model": mk,
                    "gap_seconds_mean": sec_mean,
                    "gap_seconds_median": sec_med,
                    **sub,
                }
            )
    write_csv(rows_gap, out_dir / "by_gap_duration.csv")
    write_json(json_gap, out_dir / "by_gap_duration.json")

    # 4. Valid Frames & Padding (m = L vs m < L)
    pad_groups = [
        (f"Full_both_ge_{L0}", (valid_b >= L0) & (valid_a >= L0)),
        (f"Padded_either_lt_{L0}", (valid_b < L0) | (valid_a < L0)),
        (f"Query_padded_lt_{L0}", valid_a < L0),
        (f"Gallery_padded_lt_{L0}", valid_b < L0),
    ]
    rows_pad, json_pad = [], []
    for pname, mask in pad_groups:
        for mk in eval_models:
            sub = _subset_metrics(results_l8[mk], mask)
            rows_pad.append(_fmt_sub_row(pname, mk, sub))
            json_pad.append({"padding_group": pname, "model": mk, **sub})
    write_csv(rows_pad, out_dir / "by_valid_frames_padding.csv")
    write_json(json_pad, out_dir / "by_valid_frames_padding.json")

    if args.from_cache_only:
        print("[OK] --from-cache-only complete; existing seq_len & order ablation files preserved.")
        return

    # 5. Sequence Length (L in {4, 8, 12, 16}) & Order Ablation (shuffle, reverse) on Model A + Model B L scaling
    ckpt_a = resolve_ckpt_path(cfg, "uavreid_convnext_bcgg_mamba")
    ckpt_b = resolve_ckpt_path(cfg, "uavreid_convnext_bcgg_mean_head")
    rows_abl: List[Dict[str, Any]] = []
    json_abl: List[Dict[str, Any]] = []

    reset_gpu_memory(device)
    model_a, _ = load_stage2_checkpoint_strict("uavreid_convnext_bcgg_mamba", ckpt_a, deploy_bcgg=True, device=device)

    l_runs_a: List[Dict[str, Any]] = []
    for L in seq_lens:
        if L == L0:
            res = results_l8["uavreid_convnext_bcgg_mamba"]
        else:
            loader = make_clip_loader(UAVClipPairDataset(test_img_dir, clean_pairs, tfm, num_frames=L, order_mode="normal"), bs, nw)
            ext = extract_stage2_embeddings(model_a, loader, device=device)["feats"]["fused"]
            res = evaluate_uav_retrieval(ext["qf"], ext["gf"], clean_pids, clean_seqs, device=device)
        rows_abl.append(
            {
                "model": "uavreid_convnext_bcgg_mamba",
                "ablation": f"seq_len_{L}",
                "num_frames": L,
                "order_mode": "normal",
                "per_query_mAP": f"{res['per_query_mAP']:.2f}",
                "per_query_R1": f"{res['per_query_R1']:.2f}",
                "per_query_R5": f"{res['per_query_R5']:.2f}",
                "per_query_R10": f"{res['per_query_R10']:.2f}",
                "per_query_mINP": f"{res['per_query_mINP']:.2f}",
                "per_video_mAP": f"{res['per_video_mAP']:.2f}",
                "per_video_R1": f"{res['per_video_R1']:.2f}",
            }
        )
        jrec = {k: v for k, v in res.items() if k != "per_query"}
        jrec.update({"model": "uavreid_convnext_bcgg_mamba", "ablation": f"seq_len_{L}", "num_frames": L, "order_mode": "normal"})
        json_abl.append(jrec)
        l_runs_a.append(jrec)
        print(f"[ABLATION Model A | L={L} | normal] mAP={res['per_query_mAP']:.2f}% R1={res['per_query_R1']:.2f}% mINP={res['per_query_mINP']:.2f}%")

    for ord_mode in ("shuffle", "reverse"):
        loader = make_clip_loader(
            UAVClipPairDataset(test_img_dir, clean_pairs, tfm, num_frames=L0, order_mode=ord_mode, shuffle_seed=seed), bs, nw
        )
        ext = extract_stage2_embeddings(model_a, loader, device=device)["feats"]["fused"]
        res = evaluate_uav_retrieval(ext["qf"], ext["gf"], clean_pids, clean_seqs, device=device)
        rows_abl.append(
            {
                "model": "uavreid_convnext_bcgg_mamba",
                "ablation": f"order_{ord_mode}",
                "num_frames": L0,
                "order_mode": ord_mode,
                "per_query_mAP": f"{res['per_query_mAP']:.2f}",
                "per_query_R1": f"{res['per_query_R1']:.2f}",
                "per_query_R5": f"{res['per_query_R5']:.2f}",
                "per_query_R10": f"{res['per_query_R10']:.2f}",
                "per_query_mINP": f"{res['per_query_mINP']:.2f}",
                "per_video_mAP": f"{res['per_video_mAP']:.2f}",
                "per_video_R1": f"{res['per_video_R1']:.2f}",
            }
        )
        jrec = {k: v for k, v in res.items() if k != "per_query"}
        jrec.update({"model": "uavreid_convnext_bcgg_mamba", "ablation": f"order_{ord_mode}", "num_frames": L0, "order_mode": ord_mode})
        json_abl.append(jrec)

    del model_a
    reset_gpu_memory(device)

    model_b, _ = load_stage2_checkpoint_strict("uavreid_convnext_bcgg_mean_head", ckpt_b, deploy_bcgg=True, device=device)
    for L in seq_lens:
        if L == L0:
            res = results_l8["uavreid_convnext_bcgg_mean_head"]
        else:
            loader = make_clip_loader(UAVClipPairDataset(test_img_dir, clean_pairs, tfm, num_frames=L, order_mode="normal"), bs, nw)
            ext = extract_stage2_embeddings(model_b, loader, device=device)["feats"]["fused"]
            res = evaluate_uav_retrieval(ext["qf"], ext["gf"], clean_pids, clean_seqs, device=device)
        rows_abl.append(
            {
                "model": "uavreid_convnext_bcgg_mean_head",
                "ablation": f"seq_len_{L}",
                "num_frames": L,
                "order_mode": "normal",
                "per_query_mAP": f"{res['per_query_mAP']:.2f}",
                "per_query_R1": f"{res['per_query_R1']:.2f}",
                "per_query_R5": f"{res['per_query_R5']:.2f}",
                "per_query_R10": f"{res['per_query_R10']:.2f}",
                "per_query_mINP": f"{res['per_query_mINP']:.2f}",
                "per_video_mAP": f"{res['per_video_mAP']:.2f}",
                "per_video_R1": f"{res['per_video_R1']:.2f}",
            }
        )
        jrec = {k: v for k, v in res.items() if k != "per_query"}
        jrec.update({"model": "uavreid_convnext_bcgg_mean_head", "ablation": f"seq_len_{L}", "num_frames": L, "order_mode": "normal"})
        json_abl.append(jrec)
    del model_b
    reset_gpu_memory(device)

    spread_a = {
        "mAP_spread": float(max(r["per_query_mAP"] for r in l_runs_a) - min(r["per_query_mAP"] for r in l_runs_a)),
        "R1_spread": float(max(r["per_query_R1"] for r in l_runs_a) - min(r["per_query_R1"] for r in l_runs_a)),
        "mINP_spread": float(max(r["per_query_mINP"] for r in l_runs_a) - min(r["per_query_mINP"] for r in l_runs_a)),
    }
    write_csv(rows_abl, out_dir / "by_seq_len_and_order_ablation.csv")
    write_json({"runs": json_abl, "model_a_seq_len_spread": spread_a}, out_dir / "by_seq_len_and_order_ablation.json")


if __name__ == "__main__":
    main()
