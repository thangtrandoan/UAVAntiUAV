#!/usr/bin/env python3
"""Master orchestrator for Stage 2 (UAVReIDNet) Jetson AGX Orin evaluation & verification."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common_stage2 import load_stage2_config, resolve_output_dir  # noqa: E402


def _run_step(script_name: str, extra_args: List[str]) -> None:
    cmd = [sys.executable, "-u", str(SCRIPT_DIR / script_name), *extra_args]
    print(f"\n>>> RUNNING: {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"[FAIL] Step {script_name} exited with code {proc.returncode}")


def verify_stage2_checklist(out_root: Path) -> bool:
    print("\n" + "=" * 80)
    print("STAGE 2 CHECKLIST VERIFICATION")
    print("=" * 80)
    ok_all = True

    # 1. dataset_audit.json (clean + real FPS)
    da_path = out_root / "00_env_and_manifest" / "dataset_audit.json"
    c1 = da_path.is_file()
    c1_msg = "missing"
    if c1:
        da = json.loads(da_path.read_text(encoding="utf-8"))
        c1 = (
            da.get("train_test_sequence_overlap") == 0
            and da.get("train_test_identity_overlap") == 0
            and da.get("clean_test_fps_stats", {}).get("count", 0) > 0
        )
        c1_msg = (
            f"clean_test_pairs={da.get('test_split', {}).get('pairs_by_origin', {}).get('uav_anti_uav')}, "
            f"fps_probed={da.get('clean_test_fps_stats', {}).get('count')}, median_fps={da.get('clean_test_fps_stats', {}).get('median')}"
        )
    print(f"[{'OK' if c1 else 'FAIL'}] 1. Dataset audit: {da_path} -> {c1_msg}")
    ok_all &= c1

    # 2. checkpoints_manifest.csv (100% strict feature key match)
    cm_path = out_root / "00_env_and_manifest" / "checkpoints_manifest.csv"
    c2 = cm_path.is_file()
    c2_msg = "missing"
    if c2:
        with cm_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        c2 = len(rows) == 5 and all(r.get("strict_feature_match") == "True" for r in rows)
        c2_msg = f"{len(rows)} checkpoints verified ({[r['matched_feature_keys'] for r in rows]})"
    print(f"[{'OK' if c2 else 'FAIL'}] 2. Checkpoints manifest: {cm_path} -> {c2_msg}")
    ok_all &= c2

    # 3. uav_accuracy_summary.csv
    acc_path = out_root / "01_accuracy_benchmarks" / "uav_accuracy_summary.csv"
    c3 = acc_path.is_file()
    c3_msg = "missing"
    if c3:
        with acc_path.open("r", encoding="utf-8") as f:
            arows = list(csv.DictReader(f))
        models = {r["model"] for r in arows}
        req = {"gasnet_1frame", "uavreid_convnext_bcgg_mean_head", "uavreid_convnext_bcgg_mamba", "uavreid_convnext_rga_mamba", "uavreid_convnext_bcgg_attn", "uavreid_convnext_bcgg_mamba_meanpool"}
        c3 = req.issubset(models)
        c3_msg = f"{len(arows)} rows, models={sorted(models)}"
    print(f"[{'OK' if c3 else 'FAIL'}] 3. UAV accuracy summary: {acc_path} -> {c3_msg}")
    ok_all &= c3

    # 4. stage2_tensorrt_latency.csv (both kClip and kStream)
    trt_path = out_root / "03_efficiency_and_latency_orin" / "stage2_tensorrt_latency.csv"
    c4 = trt_path.is_file()
    c4_msg = "missing"
    if c4:
        with trt_path.open("r", encoding="utf-8") as f:
            trows = list(csv.DictReader(f))
        modes = {r["mode"] for r in trows}
        c4 = {"kClip", "kStream"}.issubset(modes) and len(trows) >= 8
        c4_msg = f"{len(trows)} rows, modes={sorted(modes)}"
    print(f"[{'OK' if c4 else 'FAIL'}] 4. Orin TRT kClip + kStream: {trt_path} -> {c4_msg}")
    ok_all &= c4

    # 5. Visualizations (retrieval examples + attention pooling weights)
    rc_path = out_root / "05_visualizations" / "uav_retrieval_examples" / "retrieval_cases.json"
    tw_path = out_root / "05_visualizations" / "temporal_weights" / "temporal_weights_stats.json"
    c5 = rc_path.is_file() and tw_path.is_file()
    c5_msg = "missing"
    if c5:
        rc = json.loads(rc_path.read_text(encoding="utf-8"))
        tw = json.loads(tw_path.read_text(encoding="utf-8"))
        has_succ = any("success" in x.get("case_type", "") for x in rc)
        has_fail = any("failure" in x.get("case_type", "") for x in rc)
        c5 = has_succ and has_fail and bool(tw.get("hook_verified"))
        c5_msg = f"retrieval_cases={len(rc)} (succ={has_succ}, fail={has_fail}), temporal_hook={tw.get('hook_verified')}"
    print(f"[{'OK' if c5 else 'FAIL'}] 5. Visuals & real temporal weights: -> {c5_msg}")
    ok_all &= c5
    return ok_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 2 (UAVReIDNet) evaluation suite")
    parser.add_argument("--config", type=str, default=str(SCRIPT_DIR / "eval_stage2_config.yaml"))
    parser.add_argument("--output-dir", type=str, default="report1")
    parser.add_argument("--lock-clocks", action="store_true")
    parser.add_argument("--temporal-fp32", action="store_true")
    parser.add_argument(
        "--stages",
        nargs="*",
        default=None,
        choices=["00", "01", "02", "03a", "03b", "03c", "04", "05"],
    )
    args = parser.parse_args()

    cfg = load_stage2_config(args.config)
    out_root = resolve_output_dir(cfg, args.output_dir)
    common = ["--config", str(args.config), "--output-dir", str(out_root)]
    clock_args = [*common, "--lock-clocks"] if args.lock_clocks else list(common)
    trt_args = [*clock_args, "--temporal-fp32"] if args.temporal_fp32 else list(clock_args)
    v_args = [*common, "--temporal-fp32"] if args.temporal_fp32 else list(common)

    all_stages = ["00", "01", "02", "03a", "03b", "03c", "04", "05"]
    selected = set(args.stages) if args.stages else set(all_stages)

    for st in all_stages:
        if st not in selected:
            continue
        if st == "00":
            _run_step("00_audit_stage2.py", common)
        elif st == "01":
            _run_step("01_eval_uav_accuracy.py", common)
        elif st == "02":
            _run_step("02_eval_attributes_and_robustness.py", common)
        elif st == "03a":
            _run_step("03a_stage2_complexity.py", common)
        elif st == "03b":
            _run_step("03b_bench_stage2_pytorch.py", clock_args)
        elif st == "03c":
            _run_step("03c_export_and_bench_stage2_trt.py", trt_args)
        elif st == "04":
            _run_step("04_verify_stage2_trt_accuracy.py", v_args)
        elif st == "05":
            _run_step("05_generate_stage2_visuals.py", common)

    if not args.stages:
        verify_stage2_checklist(out_root)


if __name__ == "__main__":
    main()
