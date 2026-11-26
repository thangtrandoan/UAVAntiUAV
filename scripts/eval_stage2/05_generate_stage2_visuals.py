#!/usr/bin/env python3
"""Stage 2 Step 05: Generate Chapter 4/6 Visualizations (300 DPI PNG + vector PDF)."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

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


def _save_fig_dual(fig: plt.Figure, stem_path: Path, dpi: int = 300) -> Tuple[Path, Path]:
    stem_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = stem_path.with_suffix(".png")
    pdf_path = stem_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved Stage 2 visual: {png_path} & {pdf_path}")
    return png_path, pdf_path


def _resize_pil(im: Image.Image, size: Tuple[int, int]) -> Image.Image:
    resamp = getattr(Image, "Resampling", Image).BICUBIC
    return im.convert("RGB").resize(size, resamp)


def _draw_uav_retrieval_figure(
    q_pair: Dict[str, Any],
    q_dir_root: Path,
    top5: List[Tuple[Dict[str, Any], float, bool]],
    title: str,
    subtitle: str,
    stem_path: Path,
    dpi: int,
    fps_val: Optional[float],
) -> Tuple[Path, Path]:
    fig = plt.figure(figsize=(16.5, 4.4))
    gs = fig.add_gridspec(1, 7, width_ratios=[1.8, 0.15, 1, 1, 1, 1, 1], wspace=0.14)
    fig.suptitle(f"{title}\n{subtitle}", fontsize=11, fontweight="bold", y=1.02)

    # Left: 3 representative Query frames from T_after
    ax_q = fig.add_subplot(gs[0, 0])
    q_frames = q_pair["query_frames"][:8]
    if len(q_frames) >= 3:
        rep_names = [q_frames[0], q_frames[len(q_frames) // 2], q_frames[-1]]
    elif len(q_frames) == 2:
        rep_names = [q_frames[0], q_frames[1], q_frames[1]]
    else:
        rep_names = [q_frames[0], q_frames[0], q_frames[0]]
    strip = Image.new("RGB", (3 * 112 + 8, 112), (245, 245, 245))
    for k, fn in enumerate(rep_names):
        im_p = q_dir_root / q_pair["query_dir"] / fn
        with Image.open(im_p) as raw:
            thumb = _resize_pil(raw, (112, 112))
        strip.paste(thumb, (k * 116, 0))
    ax_q.imshow(ImageOps.expand(strip, border=5, fill="#1E90FF"))
    gap_s_txt = f" ({q_pair['gap_frames']/fps_val:.2f}s)" if fps_val else ""
    ax_q.set_title(
        f"Query T_after (3 khung đại diện)\n{q_pair['sequence_id']}\n"
        f"ID:{q_pair['identity_id']} | Gap:{q_pair['gap_frames']}f{gap_s_txt} | m={q_pair['valid_after_frames']}",
        fontsize=8.5,
        fontweight="bold",
        color="#0B5394",
    )
    ax_q.axis("off")

    ax_sep = fig.add_subplot(gs[0, 1])
    ax_sep.axis("off")

    # Right: Top-5 Gallery candidates (T_before)
    for idx, (gp, sim, is_match) in enumerate(top5):
        ax = fig.add_subplot(gs[0, idx + 2])
        rep_g = gp["gallery_frames"][-1]
        gp_path = q_dir_root / gp["gallery_dir"] / rep_g
        with Image.open(gp_path) as raw:
            gim = _resize_pil(raw, (150, 150))
        color = "#28A745" if is_match else "#DC3545"
        status = "ĐÚNG" if is_match else "SAI"
        ax.imshow(ImageOps.expand(gim, border=6, fill=color))
        ax.set_title(
            f"Rank-{idx + 1}: {status}\nID:{gp['identity_id']} | cos={sim:.3f}\nGap:{gp['gap_frames']}f | {gp['sequence_id'][-6:]}",
            fontsize=8.5,
            fontweight="bold",
            color=color,
        )
        ax.axis("off")

    return _save_fig_dual(fig, stem_path, dpi=dpi)


def generate_uav_retrieval_examples(
    cfg: Dict[str, Any],
    out_dir: Path,
    cache_path: Path,
    clean_pairs: List[Dict[str, Any]],
    fps_by_seq: Dict[str, float],
    dpi: int,
) -> List[Dict[str, Any]]:
    ex_dir = out_dir / "uav_retrieval_examples"
    ex_dir.mkdir(parents=True, exist_ok=True)
    proc_dir = resolve_path(cfg.get("processed_dir", "data/processed"))
    test_img_dir = proc_dir / "test"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cache_path.is_file():
        blob = torch.load(cache_path, map_location="cpu", weights_only=False)
        clean_t = torch.from_numpy(np.asarray(blob["clean_indices"])).long()
        qa, ga = blob["embeddings"]["model_a::fused"]["qf"][clean_t], blob["embeddings"]["model_a::fused"]["gf"][clean_t]
        qb, gb = blob["embeddings"]["model_b::fused"]["qf"][clean_t], blob["embeddings"]["model_b::fused"]["gf"][clean_t]
        qg, gg = blob["embeddings"]["gasnet_1frame"]["qf"][clean_t], blob["embeddings"]["gasnet_1frame"]["gf"][clean_t]
    else:
        tfm = build_stage2_transform(cfg)
        L = int(cfg.get("eval", {}).get("primary_num_frames", 8))
        bs = int(cfg.get("eval", {}).get("batch_size", 16))
        nw = int(cfg.get("eval", {}).get("num_workers", 4))
        lm = make_clip_loader(UAVClipPairDataset(test_img_dir, clean_pairs, tfm, num_frames=L), bs, nw)
        l1 = make_clip_loader(UAVClipPairDataset(test_img_dir, clean_pairs, tfm, num_frames=1, single_frame=True), bs, nw)
        ma, _ = load_stage2_checkpoint_strict("uavreid_convnext_bcgg_mamba", resolve_ckpt_path(cfg, "uavreid_convnext_bcgg_mamba"), True, device)
        eg = extract_stage2_embeddings(ma, l1, device, single_frame=True)["feats"]["gasnet_1frame"]
        ea = extract_stage2_embeddings(ma, lm, device)["feats"]["fused"]
        del ma
        reset_gpu_memory(device)
        mb, _ = load_stage2_checkpoint_strict("uavreid_convnext_bcgg_mean_head", resolve_ckpt_path(cfg, "uavreid_convnext_bcgg_mean_head"), True, device)
        eb = extract_stage2_embeddings(mb, lm, device)["feats"]["fused"]
        del mb
        reset_gpu_memory(device)
        qa, ga, qb, gb, qg, gg = ea["qf"], ea["gf"], eb["qf"], eb["gf"], eg["qf"], eg["gf"]

    pids = np.array([p["identity_id"] for p in clean_pairs], dtype=np.int64)
    seqs = [p["sequence_id"] for p in clean_pairs]
    sim_a = (F.normalize(qa.float(), p=2, dim=1) @ F.normalize(ga.float(), p=2, dim=1).t()).numpy()
    res_a = evaluate_uav_retrieval(qa, ga, pids, seqs, device=device)
    res_b = evaluate_uav_retrieval(qb, gb, pids, seqs, device=device)
    res_g = evaluate_uav_retrieval(qg, gg, pids, seqs, device=device)

    r1_a = res_a["per_query"]["r1"] > 50.0
    r1_b = res_b["per_query"]["r1"] > 50.0
    r1_g = res_g["per_query"]["r1"] > 50.0

    # Pick Case 1: Out-of-View (prefer large gap) where Model A succeeds and (GASNet 1f or Model B) fails
    ov_cands = [
        i
        for i, p in enumerate(clean_pairs)
        if p["attributes"][2] == 1 and r1_a[i] and ((not r1_b[i]) or (not r1_g[i]))
    ]
    ov_cands.sort(key=lambda i: ((not r1_b[i]) and (not r1_g[i]), clean_pairs[i]["gap_frames"]), reverse=True)
    idx_ov = ov_cands[0]

    # Pick Case 2: Occlusion (distinct seq from idx_ov)
    oc_cands = [
        i
        for i, p in enumerate(clean_pairs)
        if p["sequence_id"] != clean_pairs[idx_ov]["sequence_id"]
        and p["attributes"][7] == 1
        and r1_a[i]
        and ((not r1_b[i]) or (not r1_g[i]))
    ]
    oc_cands.sort(key=lambda i: ((not r1_b[i]) and (not r1_g[i]), clean_pairs[i]["gap_frames"]), reverse=True)
    idx_oc = oc_cands[0] if oc_cands else ov_cands[1]

    # Pick Case 3: Failure case where Model A (and baselines) miss R1, preferring Low Resolution (idx 9) or Fake UAV (idx 13)
    fail_cands = [
        i
        for i, p in enumerate(clean_pairs)
        if (not r1_a[i]) and (not r1_b[i]) and (not r1_g[i]) and p["sequence_id"] not in {clean_pairs[idx_ov]["sequence_id"], clean_pairs[idx_oc]["sequence_id"]}
    ]
    fail_cands.sort(
        key=lambda i: (clean_pairs[i]["attributes"][9], clean_pairs[i]["attributes"][13], clean_pairs[i]["gap_frames"]),
        reverse=True,
    )
    idx_fail = fail_cands[0]

    chosen = [
        ("success_oov_long_gap", idx_ov, "Ca thành công 1 — Out-of-View (Mất dấu dài)"),
        ("success_occlusion", idx_oc, "Ca thành công 2 — Occlusion (Che khuất)"),
        ("failure_case_hard", idx_fail, "Ca thất bại — Độ phân giải thấp / Fake UAV (Cả hai cùng nhầm)"),
    ]

    meta_rows: List[Dict[str, Any]] = []
    for tag, qi, headline in chosen:
        qp = clean_pairs[qi]
        order_row = np.argsort(-sim_a[qi])[:5]
        top5 = [(clean_pairs[int(gi)], float(sim_a[qi, int(gi)]), clean_pairs[int(gi)]["identity_id"] == qp["identity_id"]) for gi in order_row]
        fps_v = fps_by_seq.get(qp["sequence_id"])
        sub = (
            f"UAVReIDNet (Model A) Rank-1={'ĐÚNG' if r1_a[qi] else 'SAI'} | "
            f"Visual Mean+BN (Model B) Rank-1={'ĐÚNG' if r1_b[qi] else 'SAI'} | "
            f"GASNet 1-khung Rank-1={'ĐÚNG' if r1_g[qi] else 'SAI'}"
        )
        stem = ex_dir / f"uav_retrieval_{tag}_{qp['sequence_id']}_ev{qp['event_index']}"
        png_p, pdf_p = _draw_uav_retrieval_figure(qp, test_img_dir, top5, headline, sub, stem, dpi, fps_v)
        meta_rows.append(
            {
                "case_type": tag,
                "sequence_id": qp["sequence_id"],
                "event_index": qp["event_index"],
                "identity_id": qp["identity_id"],
                "gap_frames": qp["gap_frames"],
                "model_a_r1": bool(r1_a[qi]),
                "model_b_r1": bool(r1_b[qi]),
                "gasnet_1f_r1": bool(r1_g[qi]),
                "top1_score": float(top5[0][1]),
                "png_path": str(png_p),
                "pdf_path": str(pdf_p),
            }
        )

    write_csv(meta_rows, ex_dir / "retrieval_cases.csv")
    write_json(meta_rows, ex_dir / "retrieval_cases.json")
    return meta_rows


@torch.no_grad()
def generate_temporal_weight_visuals(
    cfg: Dict[str, Any],
    out_dir: Path,
    clean_pairs: List[Dict[str, Any]],
    dpi: int,
) -> List[Dict[str, Any]]:
    tw_dir = out_dir / "temporal_weights"
    tw_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc_dir = resolve_path(cfg.get("processed_dir", "data/processed"))
    test_img_dir = proc_dir / "test"
    L = int(cfg.get("visualizations", {}).get("num_frames_visual", 8))
    tfm = build_stage2_transform(cfg)

    # Pick 3 distinct clips: (1) Padded short clip (3<=m<8), (2) Full Motion Blur/Occlusion, (3) Full Out-of-View
    c_padded = sorted(
        [i for i, p in enumerate(clean_pairs) if 2 <= p["valid_after_frames"] < L],
        key=lambda i: clean_pairs[i]["valid_after_frames"],
    )[len([p for p in clean_pairs if 2 <= p["valid_after_frames"] < L]) // 2]
    c_blur = next(
        i
        for i, p in enumerate(clean_pairs)
        if p["valid_after_frames"] >= L and p["attributes"][8] == 1 and p["attributes"][7] == 1 and p["sequence_id"] != clean_pairs[c_padded]["sequence_id"]
    )
    c_oov = next(
        i
        for i, p in enumerate(clean_pairs)
        if p["valid_after_frames"] >= L and p["attributes"][2] == 1 and p["sequence_id"] not in {clean_pairs[c_padded]["sequence_id"], clean_pairs[c_blur]["sequence_id"]}
    )

    chosen_clips = [
        ("padded_short_clip", c_padded, "Clip ngắn bị đệm lặp (AttentionPooling tự giảm trọng số khung đệm)"),
        ("full_blur_occlusion", c_blur, "Clip đủ khung có Motion Blur + Occlusion"),
        ("full_out_of_view", c_oov, "Clip đủ khung sau sự kiện Out-of-View"),
    ]

    ckpt_a = resolve_ckpt_path(cfg, "uavreid_convnext_bcgg_mamba")
    reset_gpu_memory(device)
    model_a, audit = load_stage2_checkpoint_strict("uavreid_convnext_bcgg_mamba", ckpt_a, deploy_bcgg=True, device=device)
    model_a.set_temporal_attention_capture(True)

    stats_rows: List[Dict[str, Any]] = []
    try:
        ds = UAVClipPairDataset(test_img_dir, [clean_pairs[idx] for _, idx, _ in chosen_clips], tfm, num_frames=L)
        for item_i, (tag, qi, desc) in enumerate(chosen_clips):
            qp = clean_pairs[qi]
            _, a_clip, _, _, _ = ds[item_i]
            _ = model_a(a_clip.unsqueeze(0).to(device))
            alpha_t = model_a.temporal_encoder.attn_pool.last_weights
            assert alpha_t is not None, "[FAIL] AttentionPooling.last_weights returned None! No fake weights allowed."
            alpha = alpha_t.squeeze(0).float().cpu().numpy()
            assert np.isfinite(alpha).all() and abs(float(alpha.sum()) - 1.0) < 1e-4

            valid_m = min(L, qp["valid_after_frames"])
            raw_frames = list(qp["query_frames"][:L])
            padded_flags = [False] * len(raw_frames)
            while len(raw_frames) < L:
                raw_frames.append(raw_frames[-1])
                padded_flags.append(True)

            fig = plt.figure(figsize=(14.5, 5.2))
            gs = fig.add_gridspec(2, L, height_ratios=[1.1, 1.3], hspace=0.28, wspace=0.12)
            fig.suptitle(
                f"{desc}\n{qp['sequence_id']} (event {qp['event_index']}, valid={valid_m}/{L} khung)",
                fontsize=11,
                fontweight="bold",
                y=0.99,
            )

            for t in range(L):
                ax_im = fig.add_subplot(gs[0, t])
                with Image.open(test_img_dir / qp["query_dir"] / raw_frames[t]) as raw:
                    im = _resize_pil(raw, (128, 128))
                border_col = "#DC3545" if padded_flags[t] else "#2B6CB0"
                ax_im.imshow(ImageOps.expand(im, border=4, fill=border_col))
                lbl = f"t={t+1} [PAD]" if padded_flags[t] else f"t={t+1}"
                ax_im.set_title(lbl, fontsize=9, fontweight="bold", color=border_col)
                ax_im.axis("off")

            ax_bar = fig.add_subplot(gs[1, :])
            xs = np.arange(L)
            bar_cols = ["#CBD5E0" if pf else "#2F855A" for pf in padded_flags]
            bars = ax_bar.bar(xs, alpha * 100.0, width=0.55, color=bar_cols, edgecolor="#1A202C", linewidth=0.8)
            ax_bar.axhline(100.0 / L, color="#E53E3E", linestyle="--", linewidth=1.2, label=f"Mean đều (1/{L} = {100.0/L:.1f}%)")
            ax_bar.set_xticks(xs)
            ax_bar.set_xticklabels([f"Khung {t+1}\n({'Đệm' if padded_flags[t] else 'Thật'})" for t in range(L)], fontsize=9)
            ax_bar.set_ylabel("Trọng số AttentionPooling α_t (%)", fontsize=10, fontweight="bold")
            ax_bar.set_ylim(0, max(25.0, float(np.max(alpha) * 125.0)))
            ax_bar.grid(axis="y", linestyle="--", alpha=0.4)
            ax_bar.legend(fontsize=9, loc="upper right")
            for b in bars:
                val = b.get_height()
                ax_bar.annotate(f"{val:.1f}%", (b.get_x() + b.get_width() / 2, val), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")

            stem = tw_dir / f"temporal_weights_{tag}_{qp['sequence_id']}_ev{qp['event_index']}"
            _save_fig_dual(fig, stem, dpi=dpi)

            valid_mean = float(np.mean(alpha[:valid_m]))
            pad_mean = float(np.mean(alpha[valid_m:])) if valid_m < L else None
            stats_rows.append(
                {
                    "clip_tag": tag,
                    "sequence_id": qp["sequence_id"],
                    "event_index": qp["event_index"],
                    "valid_frames": valid_m,
                    "seq_len": L,
                    "weights_sum": float(alpha.sum()),
                    "valid_frame_mean_weight": valid_mean,
                    "padded_frame_mean_weight": pad_mean,
                    "alpha_min": float(alpha.min()),
                    "alpha_max": float(alpha.max()),
                    "alpha_weights": [round(float(v), 6) for v in alpha],
                    "hook_verified": True,
                }
            )
    finally:
        model_a.set_temporal_attention_capture(False)
        del model_a
        reset_gpu_memory(device)

    write_csv(
        [{k: (str(v) if isinstance(v, list) else v) for k, v in r.items()} for r in stats_rows],
        tw_dir / "temporal_weights_stats.csv",
    )
    write_json(
        {"hook_verified": True, "checkpoint_sha256": audit["sha256"], "clips": stats_rows},
        tw_dir / "temporal_weights_stats.json",
    )
    return stats_rows


def generate_stage2_charts(out_root: Path, out_dir: Path, dpi: int) -> None:
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    gap_csv = out_root / "02_occlusion_and_oov_analysis" / "by_gap_duration.csv"
    ocov_csv = out_root / "02_occlusion_and_oov_analysis" / "by_disjoint_oc_ov.csv"
    abl_csv = out_root / "02_occlusion_and_oov_analysis" / "by_seq_len_and_order_ablation.csv"
    bud_csv = out_root / "03_efficiency_and_latency_orin" / "stage2_reacquisition_time_budget.csv"
    pt_csv = out_root / "03_efficiency_and_latency_orin" / "stage2_pytorch_latency.csv"
    trt_csv = out_root / "03_efficiency_and_latency_orin" / "stage2_tensorrt_latency.csv"

    model_specs = [
        ("gasnet_1frame", "GASNet 1 khung", "#718096", "o--"),
        ("uavreid_convnext_bcgg_mean_head", "Visual Mean + BNNeck", "#3182CE", "s-."),
        ("uavreid_convnext_bcgg_mamba", "UAVReIDNet (BCGG + Mamba)", "#2F855A", "^-"),
    ]

    # Chart 1: rank1_by_gap_duration
    if gap_csv.is_file():
        df_g = pd.read_csv(gap_csv)
        buckets = ["1-4 frames", "5-16 frames", "17-64 frames", ">=65 frames"]
        labels = []
        for b in buckets:
            sub = df_g[df_g["group"] == b]
            med = sub["gap_seconds_median"].iloc[0] if not sub.empty and pd.notna(sub["gap_seconds_median"].iloc[0]) else ""
            labels.append(f"{b}\n(med ~{float(med):.2f}s)" if med != "" else b)

        x = np.arange(len(buckets))
        w = 0.25
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        for idx, (mk, lbl, col, _) in enumerate(model_specs):
            vals = [float(df_g[(df_g["group"] == b) & (df_g["model"] == mk)]["R1"].iloc[0]) for b in buckets]
            bars = ax.bar(x + (idx - 1) * w, vals, w, label=lbl, color=col, alpha=0.92)
            for bar in bars:
                ax.annotate(f"{bar.get_height():.1f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9.5)
        ax.set_ylabel("Rank-1 (%)", fontsize=11, fontweight="bold")
        ax.set_title("Độ chính xác Rank-1 theo độ dài khoảng vắng mặt (UAV-Anti-UAV Clean)", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=9)
        plt.tight_layout()
        _save_fig_dual(fig, charts_dir / "rank1_by_gap_duration", dpi=dpi)

    # Chart 2: rank1_by_oc_ov_attributes
    if ocov_csv.is_file():
        df_o = pd.read_csv(ocov_csv)
        groups = [
            ("OV_only", "Chỉ Out-of-View"),
            ("OC_only", "Chỉ Occlusion"),
            ("Both_OV_OC", "Cả OV + OC"),
            ("Neither", "Không có OV/OC"),
        ]
        labels = []
        for gk, gdisp in groups:
            sub = df_o[df_o["group"] == gk]
            qc = int(sub["query_count"].iloc[0]) if not sub.empty else 0
            labels.append(f"{gdisp}\n(n={qc})")

        x = np.arange(len(groups))
        w = 0.25
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        for idx, (mk, lbl, col, _) in enumerate(model_specs):
            vals = [float(df_o[(df_o["group"] == gk) & (df_o["model"] == mk)]["R1"].iloc[0]) for gk, _ in groups]
            bars = ax.bar(x + (idx - 1) * w, vals, w, label=lbl, color=col, alpha=0.92)
            for bar in bars:
                ax.annotate(f"{bar.get_height():.1f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9.5)
        ax.set_ylabel("Rank-1 (%)", fontsize=11, fontweight="bold")
        ax.set_title("Độ chính xác Rank-1 theo 4 nhóm thuộc tính Che khuất & Mất tầm nhìn", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=9)
        plt.tight_layout()
        _save_fig_dual(fig, charts_dir / "rank1_by_oc_ov_attributes", dpi=dpi)

    # Chart 3: seq_len_stability
    if abl_csv.is_file():
        df_a = pd.read_csv(abl_csv)
        df_norm = df_a[df_a["order_mode"] == "normal"].copy()
        df_b = pd.read_csv(bud_csv) if bud_csv.is_file() else pd.DataFrame()
        Ls = sorted(df_norm["num_frames"].unique().tolist())

        fig, ax1 = plt.subplots(figsize=(8.8, 5.0))
        sub_ma = df_norm[df_norm["model"] == "uavreid_convnext_bcgg_mamba"].sort_values("num_frames")
        sub_mb = df_norm[df_norm["model"] == "uavreid_convnext_bcgg_mean_head"].sort_values("num_frames")
        ax1.plot(sub_ma["num_frames"], sub_ma["per_query_R1"].astype(float), "o-", color="#2F855A", linewidth=2.2, label="UAVReIDNet (Mamba) Rank-1")
        ax1.plot(sub_ma["num_frames"], sub_ma["per_query_mAP"].astype(float), "s--", color="#276749", linewidth=1.8, label="UAVReIDNet (Mamba) mAP")
        if not sub_mb.empty:
            ax1.plot(sub_mb["num_frames"], sub_mb["per_query_R1"].astype(float), "^:", color="#3182CE", linewidth=1.8, label="Visual Mean + BNNeck Rank-1")
        for _, r in sub_ma.iterrows():
            ax1.annotate(f"{float(r['per_query_R1']):.2f}%", (r["num_frames"], float(r["per_query_R1"])), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold", color="#2F855A")

        ax1.set_xticks(Ls)
        ax1.set_xlabel("Độ dài cửa sổ L (khung)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Độ chính xác (%)", fontsize=11, fontweight="bold")
        ax1.set_ylim(25, 90)
        ax1.grid(True, linestyle="--", alpha=0.4)

        if not df_b.empty:
            ax2 = ax1.twinx()
            df_b_s = df_b.sort_values("seq_len")
            ax2.bar(df_b_s["seq_len"], df_b_s["camera_wait_median_fps_ms"].astype(float), width=1.1, alpha=0.22, color="#DD6B20", label="Thời gian chờ camera thu L khung (ms)")
            ax2.set_ylabel("Thời gian chờ camera (ms)", fontsize=10, color="#C05621", fontweight="bold")
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8.5)
        else:
            ax1.legend(loc="lower right", fontsize=9)

        ax1.set_title("Độ ổn định Rank-1/mAP theo độ dài chuỗi L & Thời gian chờ thu khung", fontsize=11.5, fontweight="bold")
        plt.tight_layout()
        _save_fig_dual(fig, charts_dir / "seq_len_stability", dpi=dpi)

    # Chart 4: stage2_latency_breakdown (at L=8)
    if pt_csv.is_file() and trt_csv.is_file():
        df_pt = pd.read_csv(pt_csv)
        df_trt = pd.read_csv(trt_csv)
        pt_fp32 = df_pt[(df_pt["config"] == "convnext_bcgg_folded_mamba") & (df_pt["precision"] == "FP32") & (df_pt["seq_len"] == 8)]
        pt_fp16 = df_pt[(df_pt["config"] == "convnext_bcgg_folded_mamba") & (df_pt["precision"] == "FP16") & (df_pt["seq_len"] == 8)]
        trt_clip = df_trt[(df_trt["mode"] == "kClip") & (df_trt["seq_len"] == 8)]
        trt_stream = df_trt[(df_trt["mode"] == "kStream") & (df_trt["seq_len"] == 8)]

        bars_data = []
        if not pt_fp32.empty:
            r = pt_fp32.iloc[0]
            bars_data.append(("PyTorch FP32\n(Clip L=8)", float(r["backbone_clip_p50_ms"]), float(r["temporal_head_p50_ms"])))
        if not pt_fp16.empty:
            r = pt_fp16.iloc[0]
            bars_data.append(("PyTorch FP16\n(Clip L=8)", float(r["backbone_clip_p50_ms"]), float(r["temporal_head_p50_ms"])))
        if not trt_clip.empty:
            r = trt_clip.iloc[0]
            bars_data.append(("TensorRT FP16\n(kClip L=8)", float(r["backbone_clip_p50_ms"]), float(r["temporal_and_match_p50_ms"])))
        if not trt_stream.empty:
            r = trt_stream.iloc[0]
            bars_data.append(("TensorRT FP16\n(kStream bước cuối)", float(r["backbone_clip_p50_ms"]), float(r["temporal_and_match_p50_ms"])))

        if bars_data:
            labels = [b[0] for b in bars_data]
            bb_vals = np.array([b[1] for b in bars_data])
            th_vals = np.array([b[2] for b in bars_data])
            x = np.arange(len(labels))
            fig, ax = plt.subplots(figsize=(9.0, 4.9))
            ax.bar(x, bb_vals, width=0.48, label="Backbone (ConvNeXt + BCGG folded)", color="#3182CE")
            ax.bar(x, th_vals, width=0.48, bottom=bb_vals, label="Temporal (Bi-Mamba + Head [+Match])", color="#D69E2E")
            ax.axhline(33.33, color="#E53E3E", linestyle="--", linewidth=1.2, label="Ngưỡng khung 30 FPS (33.33 ms)")
            for i in range(len(labels)):
                tot = bb_vals[i] + th_vals[i]
                ax.annotate(f"Tổng: {tot:.2f} ms\n(bb {bb_vals[i]:.1f} + th {th_vals[i]:.1f})", (x[i], tot), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9.5, fontweight="bold")
            ax.set_ylabel("Độ trễ p50 trên Jetson AGX Orin (ms)", fontsize=10.5, fontweight="bold")
            ax.set_ylim(0, max(bb_vals + th_vals) * 1.22)
            ax.set_title("Phân rã độ trễ xử lý UAVReIDNet trên Jetson AGX Orin (L=8)", fontsize=12, fontweight="bold")
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.legend(fontsize=9)
            plt.tight_layout()
            _save_fig_dual(fig, charts_dir / "stage2_latency_breakdown", dpi=dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 Step 05: Generate Stage 2 Visualizations")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--charts-only", action="store_true")
    args = parser.parse_args()

    cfg = load_stage2_config(args.config)
    out_root = resolve_output_dir(cfg, args.output_dir)
    out_dir = out_root / "05_visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(cfg.get("visualizations", {}).get("dpi", 300))

    if not args.charts_only:
        proc_dir = resolve_path(cfg.get("processed_dir", "data/processed"))
        clean_pairs = load_processed_pairs(proc_dir, split="test", clean_uav_only=True)
        cache_path = out_root / "03_efficiency_and_latency_orin" / "engines" / "stage2_eval_cache.pt"
        events_csv = out_root / "00_env_and_manifest" / "dataset_events_table.csv"
        fps_by_seq: Dict[str, float] = {}
        if events_csv.is_file():
            with events_csv.open("r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r.get("fps"):
                        fps_by_seq[r["sequence_id"]] = float(r["fps"])
        generate_uav_retrieval_examples(cfg, out_dir, cache_path, clean_pairs, fps_by_seq, dpi)
        generate_temporal_weight_visuals(cfg, out_dir, clean_pairs, dpi)

    generate_stage2_charts(out_root, out_dir, dpi)


if __name__ == "__main__":
    main()
