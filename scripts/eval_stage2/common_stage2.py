"""Shared helpers for Stage 2 (UAVReIDNet) Jetson AGX Orin evaluation."""

from __future__ import annotations

import csv
import datetime as dt
import gc
import json
import os
import platform
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from stage2_models import (
    RGAChannel,
    RGASpatial,
    SimpleS6Block,
    Stage2BackboneWrapper,
    Stage2TemporalWrapper,
    Stage2UAVReIDNet,
    load_stage2_checkpoint_strict,
    sha256_file,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "eval_stage2_config.yaml"

MODEL_KEYS = (
    "uavreid_convnext_bcgg_mamba",
    "uavreid_convnext_bcgg_mean_head",
    "uavreid_convnext_rga_mamba",
    "uavreid_convnext_bcgg_attn",
    "uavreid_convnext_bcgg_mamba_meanpool",
)

ATTR_NAMES: Dict[int, str] = {
    0: "Fast Motion",
    1: "Background Clutter",
    2: "Out-of-View",
    3: "Illumination Variation",
    4: "Viewpoint Change",
    5: "Scale Variation",
    6: "Deformation",
    7: "Occlusion",
    8: "Motion Blur",
    9: "Low Resolution",
    10: "Camera Motion",
    11: "Thermal Crossover",
    12: "Night-time",
    13: "Fake UAV",
    14: "Zoom",
}
ATTR_NAME_TO_IDX: Dict[str, int] = {v: k for k, v in ATTR_NAMES.items()}


def resolve_path(path_like: str | Path) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (ROOT_DIR / p).resolve()


def load_stage2_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    path = resolve_path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"[FAIL] Stage 2 config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_output_dir(cfg: Dict[str, Any], cli_output_dir: Optional[str | Path] = None) -> Path:
    raw = cli_output_dir if cli_output_dir is not None else cfg.get("output_dir", "report1")
    out = resolve_path(raw)
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_ckpt_path(cfg: Dict[str, Any], model_key: str) -> Path:
    ckpts = cfg.get("checkpoints") or {}
    if model_key not in ckpts:
        raise KeyError(f"[FAIL] Checkpoint key {model_key!r} not in config")
    return resolve_path(ckpts[model_key])


def write_csv(rows: Iterable[Dict[str, Any]], path: Path, fieldnames: Optional[List[str]] = None) -> Path:
    rows_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows_list[0].keys()) if rows_list else []
    with path.open("w", encoding="utf-8", newline="") as f:
        if fieldnames:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_list)
    print(f"[OK] Wrote CSV ({len(rows_list)} rows): {path}")
    return path


def write_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"[OK] Wrote JSON: {path}")
    return path


def reset_gpu_memory(device: Optional[torch.device] = None) -> None:
    gc.collect()
    if torch.cuda.is_available():
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        else:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


def _run_cmd(cmd: List[str], timeout: int = 10) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as exc:
        return -1, "", str(exc)


def read_nvpmodel_status() -> Dict[str, Any]:
    rc, out, err = _run_cmd(["nvpmodel", "-q"])
    mode_line = ""
    for line in (out + "\n" + err).splitlines():
        if "Power Mode" in line:
            mode_line = line.strip()
            break
    return {
        "returncode": rc,
        "power_mode": mode_line or (out.splitlines()[0] if out else "unknown"),
        "raw_output": out,
        "raw_error": err,
    }


def read_jetson_thermals() -> Dict[str, float]:
    temps: Dict[str, float] = {}
    base = Path("/sys/devices/virtual/thermal")
    if not base.is_dir():
        return temps
    for zone in sorted(base.glob("thermal_zone*")):
        tf, tp = zone / "type", zone / "temp"
        if tf.is_file() and tp.is_file():
            try:
                val = tp.read_text(encoding="utf-8").strip()
                if val:
                    temps[tf.read_text(encoding="utf-8").strip()] = round(float(val) / 1000.0, 2)
            except Exception:
                pass
    return temps


def maybe_lock_jetson_clocks(lock_clocks: bool) -> Dict[str, Any]:
    if lock_clocks:
        print("[INFO] Locking Jetson Orin clocks: sudo nvpmodel -m 0 && sudo jetson_clocks")
        rc1, out1, err1 = _run_cmd(["sudo", "-n", "nvpmodel", "-m", "0"], timeout=15)
        if rc1 != 0:
            print(f"[WARNING] `sudo -n nvpmodel -m 0` failed (rc={rc1}): {err1 or out1}")
        else:
            print("[OK] nvpmodel set to mode 0 (MAXN)")
        rc2, out2, err2 = _run_cmd(["sudo", "-n", "jetson_clocks"], timeout=15)
        if rc2 != 0:
            print(f"[WARNING] `sudo -n jetson_clocks` failed (rc={rc2}): {err2 or out2}")
        else:
            print("[OK] jetson_clocks enabled")
    nvp = read_nvpmodel_status()
    thermals = read_jetson_thermals()
    print(f"[INFO] Power mode: {nvp.get('power_mode')} | Thermals (C): {thermals}")
    return {"lock_requested": bool(lock_clocks), "nvpmodel": nvp, "thermals_c": thermals}


def collect_env_info() -> Dict[str, Any]:
    device_model = "unknown"
    dt_model = Path("/proc/device-tree/model")
    if dt_model.is_file():
        device_model = dt_model.read_bytes().replace(b"\x00", b"").decode("utf-8", errors="ignore").strip()
    tegra_release = Path("/etc/nv_tegra_release").read_text(encoding="utf-8", errors="ignore").strip() if Path("/etc/nv_tegra_release").is_file() else ""
    try:
        import tensorrt as trt
        trt_ver = str(trt.__version__)
    except Exception as exc:
        trt_ver = f"unavailable ({exc})"

    _, git_sha, _ = _run_cmd(["git", "-C", str(ROOT_DIR), "rev-parse", "HEAD"])
    _, git_branch, _ = _run_cmd(["git", "-C", str(ROOT_DIR), "rev-parse", "--abbrev-ref", "HEAD"])
    gpu_info: Dict[str, Any] = {"available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_info.update(
            {
                "name": torch.cuda.get_device_name(0),
                "total_vram_MiB": round(props.total_memory / (1024.0 * 1024.0), 2),
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": int(props.multi_processor_count),
            }
        )
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "device_tree_model": device_model,
        "l4t_tegra_release": tegra_release,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "tensorrt_version": trt_ver,
        "trtexec_exists": Path("/usr/src/tensorrt/bin/trtexec").is_file(),
        "gpu": gpu_info,
        "nvpmodel": read_nvpmodel_status(),
        "thermals_c": read_jetson_thermals(),
        "git": {"branch": git_branch, "commit_sha": git_sha},
    }


def summarize_latencies(latencies_ms: List[float], batch_size: int = 1) -> Dict[str, float]:
    arr = np.asarray(latencies_ms, dtype=np.float64)
    p5 = float(np.percentile(arr, 5))
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return {
        "p5_ms": p5,
        "p50_ms": p50,
        "p95_ms": p95,
        "mean_ms": mean,
        "std_ms": std,
        "throughput": (1000.0 * batch_size / p50) if p50 > 0 else 0.0,
        "pct_under_33_33ms": float(np.mean(arr <= 33.33) * 100.0),
    }


# --------------------------------------------------------------------------- #
# Dataset & Origin Classification
# --------------------------------------------------------------------------- #
def classify_sequence_origin(sequence_id: str) -> str:
    if sequence_id.startswith("UAV-Anti-UAV_"):
        return "uav_anti_uav"
    if len(sequence_id) >= 8 and sequence_id[:8].isdigit():
        return "anti_uav_rgbt"
    return "uav123"


def load_processed_pairs(
    processed_dir: Path | str,
    split: str = "test",
    clean_uav_only: bool = True,
) -> List[Dict[str, Any]]:
    pdir = resolve_path(processed_dir)
    q_path = pdir / f"query_{split}.json"
    g_path = pdir / f"gallery_{split}.json"
    if not q_path.is_file() or not g_path.is_file():
        raise FileNotFoundError(f"[FAIL] Missing query/gallery JSON under {pdir} for split={split!r}")

    queries = json.loads(q_path.read_text(encoding="utf-8"))
    galleries = json.loads(g_path.read_text(encoding="utf-8"))
    g_map = {(g["sequence_id"], int(g["event_index"])): g for g in galleries}

    pairs: List[Dict[str, Any]] = []
    for q in queries:
        seq_id = str(q["sequence_id"])
        origin = classify_sequence_origin(seq_id)
        if clean_uav_only and origin != "uav_anti_uav":
            continue
        key = (seq_id, int(q["event_index"]))
        g = g_map.get(key)
        if g is None or q.get("identity_id") is None:
            continue
        pairs.append(
            {
                "split": split,
                "origin": origin,
                "sequence_id": seq_id,
                "event_index": int(q["event_index"]),
                "identity_id": int(q["identity_id"]),
                "gallery_frames": list(g.get("frames", [])),
                "gallery_dir": str(g["frame_dir"]),
                "query_frames": list(q.get("frames", [])),
                "query_dir": str(q["frame_dir"]),
                "valid_before_frames": len(g.get("frames", [])),
                "valid_after_frames": len(q.get("frames", [])),
                "gap_frames": int(q.get("disappearance_duration_frames", 0)),
                "attributes": list(q.get("attributes", [])),
                "language_description": str(q.get("language_description", "")),
            }
        )
    return pairs


def build_stage2_transform(cfg: Dict[str, Any]) -> transforms.Compose:
    pc = cfg.get("preprocess", {})
    resize_sz = int(pc.get("resize_size", 256))
    img_sz = int(pc.get("img_size", 224))
    mean = pc.get("mean", [0.485, 0.456, 0.406])
    std = pc.get("std", [0.229, 0.224, 0.225])
    return transforms.Compose(
        [
            transforms.Resize((resize_sz, resize_sz)),
            transforms.CenterCrop((img_sz, img_sz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


class UAVClipPairDataset(Dataset):
    """Dataset returning `(before_clip, after_clip, identity_id, sequence_id, event_index)`."""

    def __init__(
        self,
        split_img_dir: Path | str,
        pairs: Sequence[Dict[str, Any]],
        transform,
        num_frames: int = 8,
        single_frame: bool = False,
        order_mode: str = "normal",
        shuffle_seed: int = 42,
    ):
        self.split_img_dir = resolve_path(split_img_dir)
        self.pairs = list(pairs)
        self.transform = transform
        self.num_frames = int(num_frames)
        self.single_frame = bool(single_frame)
        self.order_mode = str(order_mode)
        self.shuffle_seed = int(shuffle_seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def _select_frames(self, frames: List[str], take_last: bool, item_idx: int, role_offset: int) -> List[str]:
        frames = list(frames)
        if not frames:
            return []
        if self.single_frame:
            return [frames[-1]] if take_last else [frames[0]]
        if len(frames) > self.num_frames:
            frames = frames[-self.num_frames :] if take_last else frames[: self.num_frames]
        elif len(frames) < self.num_frames:
            while len(frames) < self.num_frames:
                frames.append(frames[-1])
        if self.order_mode == "reverse":
            frames = frames[::-1]
        elif self.order_mode == "shuffle":
            rng = random.Random(self.shuffle_seed + item_idx * 2 + role_offset)
            idxs = list(range(len(frames)))
            rng.shuffle(idxs)
            frames = [frames[i] for i in idxs]
        return frames

    def _load_clip_tensor(self, rel_dir: str, frame_names: List[str]) -> torch.Tensor:
        if not frame_names:
            n = 1 if self.single_frame else self.num_frames
            return torch.zeros((n, 3, 224, 224), dtype=torch.float32)
        imgs = []
        base = self.split_img_dir / rel_dir
        for fn in frame_names:
            fpath = base / fn
            with Image.open(fpath) as im:
                rgb = im.convert("RGB")
                imgs.append(self.transform(rgb) if self.transform is not None else transforms.ToTensor()(rgb))
        return torch.stack(imgs, dim=0)

    def __getitem__(self, idx: int):
        p = self.pairs[idx]
        b_names = self._select_frames(p["gallery_frames"], take_last=True, item_idx=idx, role_offset=0)
        a_names = self._select_frames(p["query_frames"], take_last=False, item_idx=idx, role_offset=1)
        b_clip = self._load_clip_tensor(p["gallery_dir"], b_names)
        a_clip = self._load_clip_tensor(p["query_dir"], a_names)
        return b_clip, a_clip, int(p["identity_id"]), p["sequence_id"], int(p["event_index"])


def make_clip_loader(ds: UAVClipPairDataset, batch_size: int = 16, num_workers: int = 4) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def extract_stage2_embeddings(
    model: Stage2UAVReIDNet,
    loader: DataLoader,
    device: torch.device,
    precision: str = "fp32",
    single_frame: bool = False,
) -> Dict[str, Any]:
    model.eval()
    model.to(device)
    use_amp = (device.type == "cuda") and (precision == "fp16")

    if single_frame:
        spaces = ["gasnet_1frame"]
    elif model.temporal_encoder is None:
        spaces = ["visual_mean", "fused"]
    else:
        spaces = ["visual_mean", "temporal_token", "pre_bn", "fused"]

    store: Dict[str, Dict[str, List[torch.Tensor]]] = {s: {"qf": [], "gf": []} for s in spaces}
    pids: List[int] = []
    seq_ids: List[str] = []
    event_idxs: List[int] = []

    for b_clip, a_clip, batch_pids, batch_seqs, batch_evs in loader:
        b_clip = b_clip.to(device, non_blocking=True)
        a_clip = a_clip.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            if single_frame:
                gf = model.extract_clip_visual(b_clip).squeeze(1)
                qf = model.extract_clip_visual(a_clip).squeeze(1)
                store["gasnet_1frame"]["gf"].append(gf.float().cpu())
                store["gasnet_1frame"]["qf"].append(qf.float().cpu())
            else:
                g_out = model.forward_all_spaces(b_clip)
                q_out = model.forward_all_spaces(a_clip)
                for sp in spaces:
                    store[sp]["gf"].append(g_out[sp].float().cpu())
                    store[sp]["qf"].append(q_out[sp].float().cpu())
        pids.extend(int(x) for x in batch_pids)
        seq_ids.extend(str(s) for s in batch_seqs)
        event_idxs.extend(int(e) for e in batch_evs)

    feats = {
        sp: {"qf": torch.cat(store[sp]["qf"], dim=0), "gf": torch.cat(store[sp]["gf"], dim=0)}
        for sp in spaces
    }
    return {
        "feats": feats,
        "pids": np.asarray(pids, dtype=np.int64),
        "seq_ids": seq_ids,
        "event_indices": np.asarray(event_idxs, dtype=np.int64),
    }


def compute_metrics_from_sim(
    sim_cpu: torch.Tensor,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_seq_ids: Sequence[str],
    top_k: int = 5,
) -> Dict[str, Any]:
    """Compute exact ReID mAP, R1, R5, R10, mINP on BOTH Per-Query and Per-Video views."""
    sim_np = sim_cpu.numpy()
    order = np.argsort(-sim_np, axis=1, kind="mergesort")
    matches = g_pids[order] == q_pids[:, None]

    num_q = sim_np.shape[0]
    r1_arr = np.zeros(num_q, dtype=np.float64)
    r5_arr = np.zeros(num_q, dtype=np.float64)
    r10_arr = np.zeros(num_q, dtype=np.float64)
    ap_arr = np.zeros(num_q, dtype=np.float64)
    inp_arr = np.zeros(num_q, dtype=np.float64)
    rank1_pos_arr = np.zeros(num_q, dtype=np.int64)

    for i in range(num_q):
        rel = matches[i].astype(np.float64)
        pos_idxs = np.flatnonzero(rel > 0)
        if len(pos_idxs) == 0:
            rank1_pos_arr[i] = -1
            continue
        first_pos = int(pos_idxs[0])
        last_pos = int(pos_idxs[-1])
        rank1_pos_arr[i] = first_pos + 1
        r1_arr[i] = 100.0 if first_pos < 1 else 0.0
        r5_arr[i] = 100.0 if first_pos < 5 else 0.0
        r10_arr[i] = 100.0 if first_pos < 10 else 0.0

        cumsum = np.cumsum(rel)
        ranks = np.arange(1, len(rel) + 1, dtype=np.float64)
        precisions = cumsum / ranks
        ap_arr[i] = float((precisions * rel).sum() / len(pos_idxs) * 100.0)
        inp_arr[i] = float(len(pos_idxs) / (last_pos + 1.0) * 100.0)

    # Per-video uniform average
    vid_groups: Dict[str, List[int]] = defaultdict(list)
    for i, sid in enumerate(q_seq_ids):
        vid_groups[str(sid)].append(i)

    vid_r1, vid_r5, vid_r10, vid_map, vid_minp = [], [], [], [], []
    for idxs in vid_groups.values():
        vid_r1.append(float(np.mean(r1_arr[idxs])))
        vid_r5.append(float(np.mean(r5_arr[idxs])))
        vid_r10.append(float(np.mean(r10_arr[idxs])))
        vid_map.append(float(np.mean(ap_arr[idxs])))
        vid_minp.append(float(np.mean(inp_arr[idxs])))

    top_idx = order[:, :top_k]
    top_sim = np.take_along_axis(sim_np, top_idx, axis=1)

    return {
        "num_queries": int(num_q),
        "gallery_size": int(sim_np.shape[1]),
        "num_videos": int(len(vid_groups)),
        "per_query_mAP": float(np.mean(ap_arr)),
        "per_query_R1": float(np.mean(r1_arr)),
        "per_query_R5": float(np.mean(r5_arr)),
        "per_query_R10": float(np.mean(r10_arr)),
        "per_query_mINP": float(np.mean(inp_arr)),
        "per_video_mAP": float(np.mean(vid_map)),
        "per_video_R1": float(np.mean(vid_r1)),
        "per_video_R5": float(np.mean(vid_r5)),
        "per_video_R10": float(np.mean(vid_r10)),
        "per_video_mINP": float(np.mean(vid_minp)),
        "per_query": {
            "r1": r1_arr,
            "r5": r5_arr,
            "r10": r10_arr,
            "ap": ap_arr,
            "inp": inp_arr,
            "rank1_pos": rank1_pos_arr,
            "top_indices": top_idx,
            "top_sims": top_sim,
        },
    }


def evaluate_uav_retrieval(
    qf: torch.Tensor,
    gf: torch.Tensor,
    q_pids: np.ndarray,
    q_seq_ids: Sequence[str],
    g_pids: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    if g_pids is None:
        g_pids = q_pids
    dev = device if (device is not None and device.type == "cuda" and torch.cuda.is_available()) else torch.device("cpu")
    qn = F.normalize(qf.float().to(dev), p=2, dim=1)
    gn = F.normalize(gf.float().to(dev), p=2, dim=1)
    sim = (qn @ gn.t()).cpu()
    return compute_metrics_from_sim(sim, np.asarray(q_pids), np.asarray(g_pids), q_seq_ids, top_k=top_k)


# --------------------------------------------------------------------------- #
# Static Complexity (Exact MACs & Params)
# --------------------------------------------------------------------------- #
class _MACAccumulator:
    def __init__(self) -> None:
        self.macs = 0
        self.handles: List[Any] = []

    def _conv2d(self, mod: nn.Conv2d, inp, out) -> None:
        x = inp[0]
        b = int(x.shape[0])
        oh, ow = int(out.shape[2]), int(out.shape[3])
        kops = mod.kernel_size[0] * mod.kernel_size[1] * (mod.in_channels // mod.groups)
        self.macs += b * kops * mod.out_channels * oh * ow

    def _conv1d(self, mod: nn.Conv1d, inp, out) -> None:
        x = inp[0]
        b = int(x.shape[0])
        ol = int(out.shape[2])
        kops = mod.kernel_size[0] * (mod.in_channels // mod.groups)
        self.macs += b * kops * mod.out_channels * ol

    def _linear(self, mod: nn.Linear, inp, out) -> None:
        x = inp[0]
        inst = int(np.prod(x.shape[:-1])) if x.dim() > 1 else 1
        self.macs += inst * mod.in_features * mod.out_features

    def _rga_spatial(self, mod: RGASpatial, inp, out) -> None:
        b = int(inp[0].shape[0])
        inter_ch = int(mod.theta[0].out_channels)
        self.macs += b * mod.num_nodes * mod.num_nodes * inter_ch

    def _rga_channel(self, mod: RGAChannel, inp, out) -> None:
        b, c = int(inp[0].shape[0]), int(inp[0].shape[1])
        inter_sp = int(mod.theta[0].out_channels)
        self.macs += b * c * c * inter_sp

    def _s6_scan(self, mod: SimpleS6Block, inp, out) -> None:
        b, l, _ = inp[0].shape
        # Explicit selective scan elementwise + state recurrence MACs:
        # W=dt*A (1), dB=dt*B (1), V=dB*x (1), a*h+V (1), y=sum(h*C) (1) -> 5 ops per (b,l,d_inner,d_state) + gating 2*(b,l,d_inner)
        self.macs += int(b) * int(l) * mod.d_inner * (5 * mod.d_state + 2)

    def _mha(self, mod: nn.MultiheadAttention, inp, out) -> None:
        q = inp[0]
        if getattr(mod, "batch_first", False):
            b, l, d = int(q.shape[0]), int(q.shape[1]), int(q.shape[2])
        else:
            l, b, d = int(q.shape[0]), int(q.shape[1]), int(q.shape[2])
        # PyTorch MultiheadAttention calls F.multi_head_attention_forward bypassing out_proj forward hook:
        # count Q,K,V,O projections (4 * B * L * d^2) + Q@K^T & Attn@V matmuls (2 * B * L^2 * d)
        self.macs += (4 * b * l * d * d) + (2 * b * l * l * d)

    def register(self, module: nn.Module) -> None:
        self.macs = 0
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                self.handles.append(m.register_forward_hook(self._conv2d))
            elif isinstance(m, nn.Conv1d):
                self.handles.append(m.register_forward_hook(self._conv1d))
            elif isinstance(m, nn.Linear):
                self.handles.append(m.register_forward_hook(self._linear))
            elif isinstance(m, RGASpatial):
                self.handles.append(m.register_forward_hook(self._rga_spatial))
            elif isinstance(m, RGAChannel):
                self.handles.append(m.register_forward_hook(self._rga_channel))
            elif isinstance(m, SimpleS6Block):
                self.handles.append(m.register_forward_hook(self._s6_scan))
            elif isinstance(m, nn.MultiheadAttention):
                self.handles.append(m.register_forward_hook(self._mha))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


@torch.no_grad()
def measure_stage2_complexity(model: Stage2UAVReIDNet, seq_len: int, device: torch.device) -> Dict[str, float]:
    model.eval().to(device)
    bb_acc = _MACAccumulator()
    th_acc = _MACAccumulator()
    bb_acc.register(model.backbone)
    if model.temporal_encoder is not None:
        th_acc.register(model.temporal_encoder)
    th_acc.register(model.head)
    try:
        dummy = torch.randn(1, seq_len, 3, 224, 224, device=device)
        _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        bb_macs = int(bb_acc.macs)
        th_macs = int(th_acc.macs)
    finally:
        bb_acc.remove()
        th_acc.remove()

    bb_params = sum(p.numel() for p in model.backbone.parameters())
    th_params = (
        sum(p.numel() for p in model.temporal_encoder.parameters()) if model.temporal_encoder is not None else 0
    ) + sum(p.numel() for p in model.head.parameters())
    tot_params = sum(p.numel() for p in model.parameters())
    return {
        "backbone_params_M": bb_params / 1e6,
        "temporal_head_params_M": th_params / 1e6,
        "total_params_M": tot_params / 1e6,
        "backbone_GFLOPs": (2.0 * bb_macs) / 1e9,
        "temporal_head_GFLOPs": (2.0 * th_macs) / 1e9,
        "total_GFLOPs": (2.0 * (bb_macs + th_macs)) / 1e9,
    }


# --------------------------------------------------------------------------- #
# TensorRT 10.3 Build & Inference Helpers
# --------------------------------------------------------------------------- #
def build_trt_engine_generic(
    onnx_path: Path,
    engine_path: Path,
    input_name: str,
    min_shape: str,
    opt_shape: str,
    max_shape: str,
    trtexec_bin: str = "/usr/src/tensorrt/bin/trtexec",
    timing_cache_file: Optional[Path] = None,
    builder_opt_level: int = 2,
    avg_timing: int = 2,
    fp16: bool = True,
) -> Path:
    if not Path(trtexec_bin).is_file():
        raise FileNotFoundError(f"[FAIL] trtexec not found: {trtexec_bin}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        trtexec_bin,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--minShapes={input_name}:{min_shape}",
        f"--optShapes={input_name}:{opt_shape}",
        f"--maxShapes={input_name}:{max_shape}",
        f"--builderOptimizationLevel={int(builder_opt_level)}",
        f"--avgTiming={int(avg_timing)}",
        "--skipInference",
    ]
    if fp16:
        cmd.append("--fp16")
    if timing_cache_file is not None:
        timing_cache_file.parent.mkdir(parents=True, exist_ok=True)
        cmd.append(f"--timingCacheFile={timing_cache_file}")
    print(f"[INFO] Building TensorRT engine ({'FP16' if fp16 else 'FP32'}): {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not engine_path.is_file():
        raise RuntimeError(
            f"[FAIL] trtexec failed (rc={proc.returncode}):\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}"
        )
    print(f"[OK] Built engine: {engine_path} ({engine_path.stat().st_size / 1e6:.2f} MB)")
    return engine_path


class TensorRT10Runner:
    def __init__(self, engine_path: Path, device: Optional[torch.device] = None):
        import tensorrt as trt

        self.engine_path = Path(engine_path)
        self.device = device or torch.device("cuda:0")
        self.stream = torch.cuda.Stream(device=self.device)
        logger = trt.Logger(trt.Logger.WARNING)
        with self.engine_path.open("rb") as f:
            raw = f.read()
        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(raw)
        if self.engine is None:
            raise RuntimeError(f"[FAIL] Cannot deserialize engine: {engine_path}")
        self.context = self.engine.create_execution_context()

        self.input_name = None
        self.output_name = None
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            elif self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                self.output_name = name
        self.input_dtype = self._to_torch_dtype(self.engine.get_tensor_dtype(self.input_name))
        self.output_dtype = self._to_torch_dtype(self.engine.get_tensor_dtype(self.output_name))
        self._out_cache: Dict[Tuple[int, ...], torch.Tensor] = {}

    @staticmethod
    def _to_torch_dtype(trt_dtype) -> torch.dtype:
        import tensorrt as trt

        return {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT8: torch.int8,
            trt.DataType.BOOL: torch.bool,
        }.get(trt_dtype, torch.float32)

    def synchronize(self) -> None:
        self.stream.synchronize()

    def infer(self, inp: torch.Tensor) -> torch.Tensor:
        if inp.device != self.device:
            inp = inp.to(self.device)
        if inp.dtype != self.input_dtype:
            inp = inp.to(self.input_dtype)
        inp = inp.contiguous()
        cur = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(cur)
        with torch.cuda.stream(self.stream):
            self.context.set_input_shape(self.input_name, tuple(inp.shape))
            out_shape = tuple(self.context.get_tensor_shape(self.output_name))
            out = self._out_cache.get(out_shape)
            if out is None:
                out = torch.empty(out_shape, dtype=self.output_dtype, device=self.device)
                self._out_cache[out_shape] = out
            self.context.set_tensor_address(self.input_name, int(inp.data_ptr()))
            self.context.set_tensor_address(self.output_name, int(out.data_ptr()))
            if not self.context.execute_async_v3(int(self.stream.cuda_stream)):
                raise RuntimeError(f"[FAIL] execute_async_v3 failed on {self.engine_path.name}")
        cur.wait_stream(self.stream)
        return out

    def close(self) -> None:
        self.synchronize()
        self._out_cache.clear()
        del self.context
        del self.engine
        del self.runtime


def build_or_load_stage2_engines(
    cfg: Dict[str, Any],
    out_root: Path,
    seq_lens: Sequence[int],
    temporal_fp16: bool = True,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    import onnx

    eff_cfg = cfg.get("efficiency", {})
    trt_cfg = eff_cfg.get("trt", {})
    engines_dir = out_root / "03_efficiency_and_latency_orin" / "engines"
    engines_dir.mkdir(parents=True, exist_ok=True)
    timing_cache = engines_dir / "trt_stage2_timing.cache"
    ckpt_path = resolve_ckpt_path(cfg, "uavreid_convnext_bcgg_mamba")
    ckpt_sha = sha256_file(ckpt_path)

    trtexec_bin = str(trt_cfg.get("trtexec_bin", "/usr/src/tensorrt/bin/trtexec"))
    opset = int(trt_cfg.get("opset", 17))
    opt_lvl = int(trt_cfg.get("builder_opt_level", 2))
    avg_tim = int(trt_cfg.get("avg_timing", 2))
    min_bs = int(trt_cfg.get("min_bs", 1))
    opt_bs = int(trt_cfg.get("opt_bs", 1))
    max_bs_bb = int(trt_cfg.get("max_bs_backbone", 16))
    max_bs_tp = int(trt_cfg.get("max_bs_temporal", 32))
    img_sz = int(cfg.get("preprocess", {}).get("img_size", 224))

    model_cpu: Optional[Stage2UAVReIDNet] = None

    def _get_model() -> Stage2UAVReIDNet:
        nonlocal model_cpu
        if model_cpu is None:
            model_cpu, _ = load_stage2_checkpoint_strict(
                "uavreid_convnext_bcgg_mamba", ckpt_path, deploy_bcgg=True, device=torch.device("cpu")
            )
        return model_cpu

    # 1. Backbone engine
    bb_onnx = engines_dir / "uav_backbone.onnx"
    bb_engine = engines_dir / "uav_backbone_fp16.engine"
    bb_meta_p = Path(f"{bb_engine}.meta.json")
    bb_sig = {"ckpt_sha256": ckpt_sha, "kind": "backbone", "fp16": True, "max_bs": max_bs_bb, "img_size": img_sz}
    bb_reuse = False
    if not force_rebuild and bb_engine.is_file() and bb_onnx.is_file() and bb_meta_p.is_file():
        try:
            if json.loads(bb_meta_p.read_text(encoding="utf-8")) == bb_sig:
                bb_reuse = True
                print(f"[OK] Reusing backbone TRT engine: {bb_engine}")
        except Exception:
            pass
    if not bb_reuse:
        bb_wrap = Stage2BackboneWrapper(_get_model()).eval()
        dummy_img = torch.randn(2, 3, img_sz, img_sz)
        torch.onnx.export(
            bb_wrap,
            dummy_img,
            str(bb_onnx),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input_img"],
            output_names=["visual_feat"],
            dynamic_axes={"input_img": {0: "batch_size"}, "visual_feat": {0: "batch_size"}},
        )
        onnx.checker.check_model(onnx.load(str(bb_onnx)))
        build_trt_engine_generic(
            bb_onnx,
            bb_engine,
            "input_img",
            f"{min_bs}x3x{img_sz}x{img_sz}",
            f"{opt_bs}x3x{img_sz}x{img_sz}",
            f"{max_bs_bb}x3x{img_sz}x{img_sz}",
            trtexec_bin=trtexec_bin,
            timing_cache_file=timing_cache,
            builder_opt_level=opt_lvl,
            avg_timing=avg_tim,
            fp16=True,
        )
        bb_meta_p.write_text(json.dumps(bb_sig, indent=2), encoding="utf-8")

    # 2. Temporal engines (for each requested L)
    prec_tag = "fp16" if temporal_fp16 else "fp32"
    temp_engines: Dict[int, Path] = {}
    temp_onnxs: Dict[int, Path] = {}
    for L in seq_lens:
        L = int(L)
        tp_onnx = engines_dir / f"uav_temporal_L{L}.onnx"
        tp_engine = engines_dir / f"uav_temporal_L{L}_{prec_tag}.engine"
        tp_meta_p = Path(f"{tp_engine}.meta.json")
        tp_sig = {"ckpt_sha256": ckpt_sha, "kind": "temporal", "seq_len": L, "fp16": bool(temporal_fp16), "max_bs": max_bs_tp}
        tp_reuse = False
        if not force_rebuild and tp_engine.is_file() and tp_onnx.is_file() and tp_meta_p.is_file():
            try:
                if json.loads(tp_meta_p.read_text(encoding="utf-8")) == tp_sig:
                    tp_reuse = True
                    print(f"[OK] Reusing temporal TRT engine (L={L}, {prec_tag}): {tp_engine}")
            except Exception:
                pass
        if not tp_reuse:
            tp_wrap = Stage2TemporalWrapper(_get_model()).eval()
            dummy_seq = torch.randn(2, L, 960)
            torch.onnx.export(
                tp_wrap,
                dummy_seq,
                str(tp_onnx),
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=["seq_feats"],
                output_names=["reid_embed"],
                dynamic_axes={"seq_feats": {0: "batch_size"}, "reid_embed": {0: "batch_size"}},
            )
            onnx.checker.check_model(onnx.load(str(tp_onnx)))
            build_trt_engine_generic(
                tp_onnx,
                tp_engine,
                "seq_feats",
                f"{min_bs}x{L}x960",
                f"{opt_bs}x{L}x960",
                f"{max_bs_tp}x{L}x960",
                trtexec_bin=trtexec_bin,
                timing_cache_file=timing_cache,
                builder_opt_level=opt_lvl,
                avg_timing=avg_tim,
                fp16=bool(temporal_fp16),
            )
            tp_meta_p.write_text(json.dumps(tp_sig, indent=2), encoding="utf-8")
        temp_onnxs[L] = tp_onnx
        temp_engines[L] = tp_engine

    if model_cpu is not None:
        del model_cpu
    return {
        "backbone_onnx": bb_onnx,
        "backbone_engine": bb_engine,
        "temporal_onnxs": temp_onnxs,
        "temporal_engines": temp_engines,
        "temporal_precision": prec_tag,
    }
