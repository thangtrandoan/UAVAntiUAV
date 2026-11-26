#!/usr/bin/env python3
"""Stage 2 Step 00: Audit Jetson Orin env, UAV-Anti-UAV dataset (origin + real FPS), and checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common_stage2 import (  # noqa: E402
    MODEL_KEYS,
    classify_sequence_origin,
    collect_env_info,
    load_processed_pairs,
    load_stage2_config,
    resolve_ckpt_path,
    resolve_output_dir,
    resolve_path,
    write_csv,
    write_json,
)
from stage2_models import load_stage2_checkpoint_strict  # noqa: E402


def _parse_fps_str(rate_str: str) -> Optional[float]:
    if not rate_str or rate_str == "0/0":
        return None
    if "/" in rate_str:
        num, den = rate_str.split("/", 1)
        d = float(den)
        return (float(num) / d) if d > 0 else None
    val = float(rate_str)
    return val if val > 0 else None


def _probe_video_fps(seq_id: str, video_path: Path) -> Tuple[str, Dict[str, Any]]:
    if not video_path.is_file():
        return seq_id, {"fps": None, "method": "missing", "nb_frames": None}
    import subprocess

    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate,avg_frame_rate,nb_frames,width,height",
                "-of",
                "json",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            streams = json.loads(proc.stdout).get("streams", [])
            if streams:
                st = streams[0]
                fps = _parse_fps_str(st.get("avg_frame_rate", "")) or _parse_fps_str(st.get("r_frame_rate", ""))
                if fps is not None and fps > 0:
                    return seq_id, {
                        "fps": round(float(fps), 4),
                        "avg_frame_rate": st.get("avg_frame_rate"),
                        "r_frame_rate": st.get("r_frame_rate"),
                        "nb_frames": int(st["nb_frames"]) if str(st.get("nb_frames", "")).isdigit() else None,
                        "width": st.get("width"),
                        "height": st.get("height"),
                        "method": "ffprobe",
                    }
    except Exception:
        pass

    cap = cv2.VideoCapture(str(video_path))
    fps_cv = float(cap.get(cv2.CAP_PROP_FPS))
    n_cv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return seq_id, {
        "fps": round(fps_cv, 4) if fps_cv > 0 else None,
        "nb_frames": n_cv if n_cv > 0 else None,
        "method": "cv2" if fps_cv > 0 else "failed",
    }


def _find_video_for_sequence(raw_uav_dir: Path, split: str, seq_id: str) -> Path:
    sub = "Train" if split == "train" else "Test"
    cand = raw_uav_dir / sub / seq_id / f"{seq_id}.mp4"
    if cand.is_file():
        return cand
    other = "Test" if sub == "Train" else "Train"
    return raw_uav_dir / other / seq_id / f"{seq_id}.mp4"


def _dist_stats(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"count": 0}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "count": int(len(arr)),
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "min": round(float(np.min(arr)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "max": round(float(np.max(arr)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 Step 00: Audit Env, Dataset & Checkpoints")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_stage2_config(args.config)
    out_root = resolve_output_dir(cfg, args.output_dir)
    out_dir = out_root / "00_env_and_manifest"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("[INFO] 00_audit_stage2: Collecting Orin env, auditing UAV-Anti-UAV & checkpoints")
    print("=" * 80)

    env_info = collect_env_info()
    write_json(env_info, out_dir / "env_info.json")

    # 1. Dataset & Real FPS Audit
    proc_dir = resolve_path(cfg.get("processed_dir", "data/processed"))
    raw_uav_dir = resolve_path(cfg.get("raw_uav_dir", "data/UAV-Anti-UAV"))
    seq_lens = [int(x) for x in cfg.get("eval", {}).get("seq_lens", [4, 8, 12, 16])]

    all_train = load_processed_pairs(proc_dir, split="train", clean_uav_only=False)
    all_test = load_processed_pairs(proc_dir, split="test", clean_uav_only=False)
    clean_train = [p for p in all_train if p["origin"] == "uav_anti_uav"]
    clean_test = [p for p in all_test if p["origin"] == "uav_anti_uav"]

    # Check train/test disjointness
    train_seqs = {p["sequence_id"] for p in all_train}
    test_seqs = {p["sequence_id"] for p in all_test}
    train_ids = {p["identity_id"] for p in all_train}
    test_ids = {p["identity_id"] for p in all_test}
    seq_overlap = sorted(train_seqs & test_seqs)
    id_overlap = sorted(train_ids & test_ids)
    assert len(seq_overlap) == 0 and len(id_overlap) == 0, (
        f"[FAIL] Train/Test leakage detected: seqs={seq_overlap[:5]}, ids={id_overlap[:5]}"
    )

    # Probe real FPS for all UAV-Anti-UAV sequences used in train & test
    uav_seq_tasks: List[Tuple[str, Path]] = []
    seen_seqs = set()
    for p in clean_train + clean_test:
        sid = p["sequence_id"]
        if sid not in seen_seqs:
            seen_seqs.add(sid)
            uav_seq_tasks.append((sid, _find_video_for_sequence(raw_uav_dir, p["split"], sid)))

    print(f"[INFO] Probing real FPS via ffprobe on {len(uav_seq_tasks)} UAV-Anti-UAV videos...")
    fps_map: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for sid, info in ex.map(lambda it: _probe_video_fps(it[0], it[1]), uav_seq_tasks):
            fps_map[sid] = info

    events_rows: List[Dict[str, Any]] = []
    test_fps_vals: List[float] = []
    all_fps_vals: List[float] = []
    gap_frames_clean_test: List[float] = []
    gap_secs_clean_test: List[float] = []

    for p in clean_train + clean_test:
        finfo = fps_map.get(p["sequence_id"], {})
        fps = finfo.get("fps")
        gap_f = int(p["gap_frames"])
        gap_s = (gap_f / float(fps)) if (fps is not None and fps > 0) else None
        if fps is not None:
            all_fps_vals.append(float(fps))
        if p["split"] == "test":
            gap_frames_clean_test.append(float(gap_f))
            if fps is not None:
                test_fps_vals.append(float(fps))
            if gap_s is not None:
                gap_secs_clean_test.append(float(gap_s))
        events_rows.append(
            {
                "split": p["split"],
                "origin": p["origin"],
                "sequence_id": p["sequence_id"],
                "event_index": p["event_index"],
                "identity_id": p["identity_id"],
                "fps": f"{fps:.4f}" if fps is not None else "",
                "valid_before_frames": p["valid_before_frames"],
                "valid_after_frames": p["valid_after_frames"],
                "gap_frames": gap_f,
                "gap_seconds": f"{gap_s:.4f}" if gap_s is not None else "",
            }
        )

    write_csv(events_rows, out_dir / "dataset_events_table.csv")

    def _origin_summary(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        c_pairs = Counter(p["origin"] for p in pairs)
        seqs_by_origin: Dict[str, set] = {}
        for p in pairs:
            seqs_by_origin.setdefault(p["origin"], set()).add(p["sequence_id"])
        return {
            "total_pairs": len(pairs),
            "total_sequences": len({p["sequence_id"] for p in pairs}),
            "pairs_by_origin": dict(sorted(c_pairs.items())),
            "sequences_by_origin": {k: len(v) for k, v in sorted(seqs_by_origin.items())},
            "non_uav_sequences_examples": {
                k: sorted(v)[:5] for k, v in sorted(seqs_by_origin.items()) if k != "uav_anti_uav"
            },
        }

    padding_stats = {}
    for L in seq_lens:
        q_short = sum(1 for p in clean_test if p["valid_after_frames"] < L)
        g_short = sum(1 for p in clean_test if p["valid_before_frames"] < L)
        either_short = sum(1 for p in clean_test if p["valid_after_frames"] < L or p["valid_before_frames"] < L)
        padding_stats[f"L={L}"] = {
            "query_padded_count": q_short,
            "query_padded_pct": round(100.0 * q_short / max(1, len(clean_test)), 2),
            "gallery_padded_count": g_short,
            "gallery_padded_pct": round(100.0 * g_short / max(1, len(clean_test)), 2),
            "either_padded_count": either_short,
            "either_padded_pct": round(100.0 * either_short / max(1, len(clean_test)), 2),
        }

    fps_hist_all = Counter(round(v, 2) for v in all_fps_vals)
    fps_hist_test_videos = Counter(
        round(fps_map[sid]["fps"], 2)
        for sid in sorted({p["sequence_id"] for p in clean_test})
        if fps_map.get(sid, {}).get("fps") is not None
    )

    train_orig = _origin_summary(all_train)
    test_orig = _origin_summary(all_test)
    total_non_uav_seqs = (
        sum(v for k, v in train_orig["sequences_by_origin"].items() if k != "uav_anti_uav")
        + sum(v for k, v in test_orig["sequences_by_origin"].items() if k != "uav_anti_uav")
    )

    dataset_audit = {
        "processed_dir": str(proc_dir),
        "raw_uav_dir": str(raw_uav_dir),
        "clean_protocol_note": (
            f"Identified {total_non_uav_seqs} non-UAV-Anti-UAV sequences across train+test "
            "(Anti-UAV-RGBT YYYYMMDD_* + UAV123 named sequences). Primary clean benchmark excludes them "
            "for strict UAV-Anti-UAV single-source evaluation, while 01_eval_uav_accuracy also reports the full 2630 pool for reproduction."
        ),
        "train_test_sequence_overlap": len(seq_overlap),
        "train_test_identity_overlap": len(id_overlap),
        "train_split": train_orig,
        "test_split": test_orig,
        "clean_test_gap_frames_stats": _dist_stats(gap_frames_clean_test),
        "clean_test_gap_seconds_stats": _dist_stats(gap_secs_clean_test),
        "clean_test_fps_stats": _dist_stats(test_fps_vals),
        "clean_test_video_fps_distribution_hz": {str(k): int(v) for k, v in sorted(fps_hist_test_videos.items())},
        "all_event_fps_distribution_hz": {str(k): int(v) for k, v in sorted(fps_hist_all.items())},
        "clean_test_short_clip_padding": padding_stats,
    }
    write_json(dataset_audit, out_dir / "dataset_audit.json")

    # 2. Strict Checkpoint Manifest Audit
    ckpt_audits: List[Dict[str, Any]] = []
    ckpt_csv_rows: List[Dict[str, Any]] = []
    for model_key in MODEL_KEYS:
        ckpt_path = resolve_ckpt_path(cfg, model_key)
        model, audit = load_stage2_checkpoint_strict(model_key, ckpt_path, deploy_bcgg=False, device=torch.device("cpu"))
        del model
        ckpt_audits.append(audit)
        ckpt_csv_rows.append(
            {
                "model_key": audit["model_key"],
                "path": audit["path"],
                "sha256": audit["sha256"],
                "size_MB": audit["size_MB"],
                "backbone": audit["backbone"],
                "gate": audit["gate"],
                "temporal_type": audit["temporal_type"],
                "temporal_pool": audit["temporal_pool"],
                "temporal_pe": audit["temporal_pe"],
                "temporal_impl": audit["temporal_impl"],
                "num_frames_ckpt": audit["num_frames_ckpt"],
                "stride_ckpt": audit["stride_ckpt"],
                "num_identities": audit["num_identities"],
                "best_val_rank1": f"{audit['best_val_rank1']*100:.4f}" if audit["best_val_rank1"] is not None else "",
                "matched_feature_keys": audit["matched_feature_keys"],
                "total_matched_keys": audit["total_matched_keys"],
                "ignored_classifier_keys": ";".join(audit["ignored_classifier_keys"]),
                "strict_feature_match": audit["strict_feature_match"],
            }
        )
        print(
            f"[OK] {model_key}: gate={audit['gate']} temporal={audit['temporal_type']}({audit['temporal_pool']}) "
            f"feature_keys={audit['matched_feature_keys']} total={audit['total_matched_keys']} "
            f"sha256={audit['sha256'][:16]}"
        )

    write_csv(ckpt_csv_rows, out_dir / "checkpoints_manifest.csv")
    write_json(ckpt_audits, out_dir / "checkpoints_manifest.json")


if __name__ == "__main__":
    main()
