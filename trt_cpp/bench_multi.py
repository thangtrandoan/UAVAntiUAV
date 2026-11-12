#!/usr/bin/env python3
"""Do toc do NHIEU LAN tren NHIEU sequence, cho ca ban C++ va ban Python.

Vi sao can nhieu lan + nhieu input:
  - Mot lan chay chiu anh huong nhiet do, xung nhip, page cache -> sai so 10-20%.
  - Mot sequence chi co vai tram frame chay model -> trung binh khong on dinh.
  - FPS ca video phu thuoc TI LE frame vang cua sequence, nen so cua mot
    sequence KHONG dai dien duoc cho sequence khac.

Cach lam:
  1. Chon K sequence dai (>= --min-frames) trai deu theo co bbox.
  2. Moi sequence chay R lan, bo lan dau (warmup: page cache + TRT lazy init).
  3. Bao cao TRUNG VI (khong phai trung binh) + do lech + min/max.
  4. Kiem troi nhiet: so lan chay dau voi lan cuoi, canh bao neu lech > 5%.

Dung:
    python trt_cpp/bench_multi.py --config configs/config_resnet50_ibn.yaml \\
        --backbone v1_backbone.engine --temporal v1_n8_temporal.engine \\
        --impl both --nseq 4 --reps 3
"""
import argparse
import glob
import os
import re
import statistics as st
import subprocess
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_ROOT = os.path.join(PROJ, "data", "UAV-Anti-UAV", "Test")
CPP_BIN = os.path.join(PROJ, "trt_cpp", "build", "uav_reid_infer")
PY_WRAP = os.path.join(PROJ, "trt_cpp", "infer_trt.py")


# ---------------------------------------------------------------- chon input
def seq_stats(seq_dir):
    """(so frame, median dien tich bbox, ti le vang). None neu thieu du lieu."""
    name = os.path.basename(seq_dir)
    gt = os.path.join(seq_dir, "groundtruth_rect.txt")
    if not os.path.exists(os.path.join(seq_dir, name + ".mp4")) or not os.path.exists(gt):
        return None
    areas = []
    for line in open(gt, "r", encoding="utf-8", errors="replace"):
        p = line.replace(",", " ").split()
        if len(p) >= 4:
            try:
                areas.append(float(p[2]) * float(p[3]))
            except ValueError:
                pass
    if not areas:
        return None
    ab = os.path.join(seq_dir, "absent.txt")
    absent = 0
    if os.path.exists(ab):
        absent = sum(1 for l in open(ab) if l.strip() == "1")
    return len(areas), st.median(areas), absent / len(areas)


def seq_resolution(seq_dir):
    import cv2
    name = os.path.basename(seq_dir)
    cap = cv2.VideoCapture(os.path.join(seq_dir, name + ".mp4"))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return w, h, fps


def pick_sequences(nseq, min_frames, names, min_bbox=100):
    if names:
        return [os.path.join(TEST_ROOT, n) for n in names]
    rows = []
    for d in sorted(glob.glob(os.path.join(TEST_ROOT, "*"))):
        if not os.path.isdir(d):
            continue
        s = seq_stats(d)
        if s and s[0] >= min_frames and s[1] >= min_bbox:
            rows.append((d, s))
    if not rows:
        print(f"  Khong tim thay sequence nao >= {min_frames} frame trong {TEST_ROOT}")
        sys.exit(2)
    rows.sort(key=lambda r: r[1][1])          # theo dien tich bbox
    if nseq >= len(rows):
        return [r[0] for r in rows]
    # Lay trai deu tu bbox nho nhat den lon nhat.
    idx = [round(i * (len(rows) - 1) / (nseq - 1)) for i in range(nseq)]
    return [rows[i][0] for i in sorted(set(idx))]


# ---------------------------------------------------------------- chay + doc
def run(cmd, log_path):
    with open(log_path, "w", encoding="utf-8", errors="replace") as f:
        p = subprocess.run(cmd, cwd=PROJ, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode


def grab(path, pat, cast=float):
    if not os.path.exists(path):
        return None
    text = open(path, "r", encoding="utf-8", errors="replace").read()
    m = re.search(pat, text)
    if not m:
        return None
    try:
        return cast(m.group(1))
    except (TypeError, ValueError):
        return None


def parse_cpp(out_dir, seq_frames=None):
    s = os.path.join(out_dir, "summary.txt")
    return {
        "frames": grab(s, r"frame xu ly\s*:\s*(\d+)", int),
        "backbone_ms": grab(s, r"backbone \(prep\+TRT\)\s*:\s*([\d.]+)"),
        "n_bb": grab(s, r"backbone \(prep\+TRT\)\s*:[^(]*\(n=(\d+)\)", int),
        "temporal_ms": grab(s, r"temporal \(H2D\+TRT\)\s*:\s*([\d.]+)"),
        "n_tp": grab(s, r"temporal \(H2D\+TRT\)\s*:[^(]*\(n=(\d+)\)", int),
        "fps": grab(s, r"throughput \(model\+UI\)\s*:\s*([\d.]+)"),
        "fps_write": grab(s, r"throughput \(\+ghi video\):\s*([\d.]+)"),
        "hard": grab(s, r"HARD LOCK\s*:\s*(\d+)", int),
        "false": grab(s, r"False alarm \(fine fail\):\s*(\d+)", int),
        "lat": grab(s, r"Re-acquisition latency : mean ([\d.]+)"),
        "verified": cpp_verified(out_dir),
        "seq_frames": seq_frames,
    }


def find_metrics(out_dir):
    """infer.py coi --out-dir la thu muc CHA roi them ten sequence vao duoi."""
    direct = os.path.join(out_dir, "metrics.txt")
    if os.path.exists(direct):
        return direct
    hits = sorted(glob.glob(os.path.join(out_dir, "*", "metrics.txt")))
    return hits[0] if hits else None


def parse_py(out_dir, seq_frames=None):
    m = find_metrics(out_dir) or os.path.join(out_dir, "metrics.txt")
    # Python do tien xu ly rieng, nen backbone that = CNN + preprocess.
    cnn = grab(m, r"Avg CNN Feature Extraction\s*:\s*([\d.]+)")
    pre = grab(m, r"Avg Preprocess \(cvt\+resize\+H2D\)\s*:\s*([\d.]+)")
    return {
        "frames": None,   # metrics.txt khong ghi tong so frame
        "backbone_ms": (cnn + pre) if (cnn is not None and pre is not None) else cnn,
        "cnn_ms": cnn,
        "pre_ms": pre,
        "n_bb": grab(m, r"Avg CNN Feature Extraction\s*:[^(]*\(n=(\d+)\)", int),
        "temporal_ms": grab(m, r"Avg Mamba \+ Head Time\s*:\s*([\d.]+)"),
        "n_tp": grab(m, r"Avg Mamba \+ Head Time\s*:[^(]*\(n=(\d+)", int),
        "fps": grab(m, r"Avg Throughput \(model\+UI\)\s*:\s*([\d.]+)"),
        "fps_write": grab(m, r"Avg Throughput \(\+ghi video\):\s*([\d.]+)"),
        "hard": (open(m, encoding="utf-8", errors="replace").read().count("HARD LOCK!")
                        if os.path.exists(m) else None),
        "false": grab(m, r"False Alarms \(Fine Fails\)\s*:\s*(\d+)", int),
        "lat": grab(m, r"Re-acquisition Latency\s*:\s*([\d.]+)"),
        "verified": py_verified(m, seq_frames or 0),
        "seq_frames": seq_frames,
    }


def py_verified(metrics_path, fallback_end):
    """So frame o trang thai T3_VERIFIED, dung lai tu log Python.

    Python khong ghi per-frame nen phai suy ra: moi lan HARD LOCK mo mot doan
    verified, doan ket thuc o lan mat ke tiep.
    """
    hl, lost = [], []
    if not os.path.exists(metrics_path):
        return None
    for line in open(metrics_path, "r", encoding="utf-8", errors="replace"):
        m = re.match(r"^\[(\d+)\]\s+(.*)$", line.strip())
        if not m:
            continue
        f, msg = int(m.group(1)), m.group(2)
        if "HARD LOCK!" in msg:
            hl.append(f)
        if "Target LOST" in msg or "HIJACK DETECTED" in msg:
            lost.append(f)
    total = 0
    for h in hl:
        nxt = [f for f in lost if f >= h]
        total += (nxt[0] if nxt else fallback_end) - h
    return total


def cpp_verified(out_dir):
    """Dem truc tiep cot `state` trong frames.csv — khong phai suy ra."""
    path = os.path.join(out_dir, "frames.csv")
    if not os.path.exists(path):
        return None
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        next(f, None)
        for line in f:
            if ",T3_VERIFIED," in line:
                n += 1
    return n


def check_clocks():
    """Canh bao neu xung nhip khong bi khoa — sai so 20-30% la binh thuong."""
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") as f:
            cur = int(f.read().strip())
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq") as f:
            mx = int(f.read().strip())
        if cur < mx * 0.95:
            print(f"  CANH BAO: CPU dang {cur/1e6:.2f} GHz < max {mx/1e6:.2f} GHz.")
            print("            Chua khoa xung -> so do se thap va dao dong.")
            print("            Chay: sudo nvpmodel -m 0 && sudo jetson_clocks")
        else:
            print(f"  Xung CPU: {cur/1e6:.2f} / {mx/1e6:.2f} GHz (da o muc cao)")
    except OSError:
        print("  Khong doc duoc xung CPU (bo qua kiem tra)")


# ---------------------------------------------------------------- bao cao
def agg(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None
    return {
        "median": st.median(v),
        "mean": st.fmean(v),
        "std": st.pstdev(v) if len(v) > 1 else 0.0,
        "min": min(v),
        "max": max(v),
        "n": len(v),
    }


def sf(v, nd=2):
    """Format mot so co the None -> '-'."""
    return "-" if v is None else f"{v:.{nd}f}"


def fmt(a, nd=2):
    if not a:
        return "-"
    return f"{a['median']:.{nd}f} ± {a['std']:.{nd}f}  [{a['min']:.{nd}f}-{a['max']:.{nd}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_resnet50_ibn.yaml")
    ap.add_argument("--backbone", default="v1_backbone.engine")
    ap.add_argument("--temporal", default="v1_n8_temporal.engine")
    ap.add_argument("--impl", choices=["cpp", "py", "both"], default="both")
    ap.add_argument("--seqs", default="auto", help="auto | danh sach ten cach nhau dau phay")
    ap.add_argument("--nseq", type=int, default=4, help="so sequence khi --seqs auto")
    ap.add_argument("--min-frames", type=int, default=400,
                    help="chi chon sequence dai hon muc nay (do dai -> on dinh)")
    ap.add_argument("--min-bbox", type=int, default=100,
                    help="bo sequence co median bbox nho hon muc nay (bbox suy bien)")
    ap.add_argument("--reps", type=int, default=3, help="so lan chay moi sequence")
    ap.add_argument("--warmup-runs", type=int, default=1,
                    help="so lan chay dau bi bo (page cache + TRT lazy init)")
    ap.add_argument("--warmup-frames", type=int, default=30,
                    help="so frame dau bo khoi thong ke (chi ban C++)")
    ap.add_argument("--limit", type=int, default=0, help="gioi han so frame (0 = tat ca)")
    ap.add_argument("--no-video-out", action="store_true",
                    help="ban C++ khong ghi video (do rieng model)")
    ap.add_argument("--out-root", default="bench_multi_out")
    ap.add_argument("--csv", default="bench_multi.csv")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reuse", action="store_true",
                    help="khong chay lai, chi doc lai thu muc <out-root>/<impl>/<seq>/r<rep> da co")
    args = ap.parse_args()

    names = None
    if args.seqs and args.seqs != "auto":
        names = [s.strip() for s in args.seqs.split(",") if s.strip()]
    seqs = pick_sequences(args.nseq, args.min_frames, names, args.min_bbox)
    impls = ["cpp", "py"] if args.impl == "both" else [args.impl]

    print("=" * 78)
    print(f"  DO TOC DO: {len(impls)} ban x {len(seqs)} sequence x "
          f"{args.reps} lan (+{args.warmup_runs} warmup)")
    print("=" * 78)
    check_clocks()
    for s in seqs:
        meta = seq_stats(s)
        try:
            w, h, fv = seq_resolution(s)
            res = f"{w}x{h}@{fv:.0f}"
        except Exception:
            res = "?"
        print(f"  {os.path.basename(s):<32} {meta[0]:>6} frame  {res:>12}  "
              f"bbox~{meta[1]:>7.0f}px  vang {100*meta[2]:>5.1f}%")
    print()

    if args.dry_run:
        print("  (dry-run: khong chay gi)")
        return 0

    for exe in ([CPP_BIN] if "cpp" in impls else []) + ([PY_WRAP] if "py" in impls else []):
        if not os.path.exists(exe):
            print(f"  THIEU: {exe}")
            return 2
    os.makedirs(os.path.join(PROJ, args.out_root), exist_ok=True)

    rows = []
    total = len(seqs) * (args.reps + args.warmup_runs) * len(impls)
    done = 0
    for seq in seqs:
        name = os.path.basename(seq)
        seq_frames, seq_bbox, _ = seq_stats(seq)
        for rep in range(args.warmup_runs + args.reps):
            for impl in impls:
                out_dir = os.path.join(PROJ, args.out_root, impl, name, f"r{rep}")
                os.makedirs(out_dir, exist_ok=True)
                log = os.path.join(out_dir, "run.log")
                if impl == "cpp":
                    cmd = [CPP_BIN, "--config", args.config, "--seq-dir", seq,
                           "--backbone", args.backbone, "--temporal", args.temporal,
                           "--out-dir", out_dir, "--warmup", str(args.warmup_frames),
                           "--quiet"]
                    if args.limit:
                        cmd += ["--limit", str(args.limit)]
                    if args.no_video_out:
                        cmd.append("--no-video-out")
                    rec = parse_cpp
                else:
                    cmd = [sys.executable, PY_WRAP, "--config", args.config,
                           "--engine-backbone", args.backbone,
                           "--engine-temporal", args.temporal,
                           "--seq-dir", seq, "--out-dir", out_dir]
                    rec = parse_py
                if args.reuse:
                    if impl == "cpp":
                        ok = os.path.exists(os.path.join(out_dir, "summary.txt"))
                    else:
                        ok = find_metrics(out_dir) is not None
                    rc = 0 if ok else 1
                else:
                    rc = run(cmd, log)
                done += 1
                tag = "warmup" if rep < args.warmup_runs else f"rep{rep - args.warmup_runs + 1}"
                if rc != 0:
                    print(f"  [{done}/{total}] {impl:>3} {name} {tag}: LOI (xem {log})")
                    continue
                r = rec(out_dir, seq_frames)
                r.update({"impl": impl, "seq": name, "rep": rep,
                          "warmup": rep < args.warmup_runs,
                          "bbox": seq_stats(seq)[1]})
                print(f"  [{done}/{total}] {impl:>3} {name} {tag}: "
                      f"backbone {sf(r['backbone_ms'])} ms | temporal "
                      f"{sf(r['temporal_ms'])} ms | {sf(r['fps'], 1)} FPS | "
                      f"n_bb={r['n_bb']} n_tp={r['n_tp']}")
                if r["backbone_ms"] is None:
                    print(f"      CANH BAO: khong doc duoc so tu {out_dir} — xem run.log")
                if not r["warmup"]:
                    rows.append(r)

    if not rows:
        print("  Khong co ket qua hop le")
        return 2

    # ---------------- bao cao ----------------
    def medv(vals):
        v = [x for x in vals if x is not None]
        return st.median(v) if v else None

    def med(impl, key, seq=None):
        v = [r[key] for r in rows
             if r["impl"] == impl and (seq is None or r["seq"] == seq) and r[key] is not None]
        return st.median(v) if v else None

    def fmtc(a, nd=1):
        """Gon: trung vi±do lech, khong in [min-max] — dung cho bang nhieu cot."""
        if not a:
            return "-"
        return f"{a['median']:.{nd}f}±{a['std']:.{nd}f}"

    def diff_cell(c, p, nd=2, unit=""):
        """In ra: cpp / py / chenh lech tuyet doi / chenh lech tuong doi."""
        if c is None or p is None:
            return "-"
        d = c - p
        pct = (100.0 * d / p) if p else 0.0
        return f"{c:.{nd}f}{unit} / {p:.{nd}f}{unit} / {d:+.{nd}f} ({pct:+.1f}%)"

    print("\n" + "=" * 92)
    print("  TOC DO  —  dinh dang: C++ / Python / chenh lech (C++ so voi Python)")
    print("=" * 92)
    print(f"  {'':<8}{'backbone (ms)':<44}{'temporal (ms)':<32}")
    for impl in impls:
        sub = [r for r in rows if r["impl"] == impl]
        if sub:
            print(f"  {impl:<8}"
                  f"{fmt(agg([r['backbone_ms'] for r in sub])):<44}"
                  f"{fmt(agg([r['temporal_ms'] for r in sub])):<32}")
    if len(impls) == 2:
        print(f"\n  {'':<8}{'throughput (FPS)':<44}{'+ghi video (FPS)':<32}")
        for impl in impls:
            sub = [r for r in rows if r["impl"] == impl]
            if sub:
                print(f"  {impl:<8}"
                      f"{fmt(agg([r['fps'] for r in sub]), 1):<44}"
                      f"{fmt(agg([r['fps_write'] for r in sub]), 1):<32}")
        for key, name, nd, unit in (("backbone_ms", "backbone", 2, " ms"),
                                    ("temporal_ms", "temporal", 2, " ms"),
                                    ("fps", "throughput", 1, " FPS"),
                                    ("fps_write", "throughput+ghi video", 1, " FPS")):
            c, p_ = med("cpp", key), med("py", key)
            if c is None or p_ is None:
                continue
            gain = (c / p_) if (key.startswith("fps") and p_) else None
            extra = f"   = {gain:.2f}x" if gain else ""
            print(f"  => {name:<20} C++ {sf(c, nd)}{unit}  vs  Python {sf(p_, nd)}{unit}"
                  f"   ({c - p_:+.{nd}f}{unit}, {100.0*(c-p_)/p_:+.1f}%){extra}")

    print("\n" + "=" * 92)
    print("  CHINH XAC  —  cung dinh dang: C++ / Python / chenh lech")
    print("=" * 92)
    print(f"  {'':<8}{'HARD LOCK':<26}{'latency (frame)':<28}{'false alarm':<18}{'%verified':<14}")
    for impl in impls:
        sub = [r for r in rows if r["impl"] == impl]
        if not sub:
            continue
        ver = [100.0 * r["verified"] / r["seq_frames"] for r in sub
               if r.get("verified") is not None and r.get("seq_frames")]
        print(f"  {impl:<8}"
              f"{fmtc(agg([float(r['hard']) for r in sub if r['hard'] is not None])):<26}"
              f"{fmtc(agg([r['lat'] for r in sub])):<28}"
              f"{fmtc(agg([float(r['false']) for r in sub if r['false'] is not None])):<18}"
              f"{fmtc(agg(ver)):<14}")
    if len(impls) == 2:
        def close(key, tol):
            c, p_ = med("cpp", key), med("py", key)
            if c is None or p_ is None:
                return True          # mot ben khong co so -> khong ket luan lech
            return abs(c - p_) <= tol
        same = close("hard", 0) and close("false", 0) and close("lat", 0.5)
        verc = [100.0 * r["verified"] / r["seq_frames"] for r in rows
                if r["impl"] == "cpp" and r.get("verified") is not None and r.get("seq_frames")]
        verp = [100.0 * r["verified"] / r["seq_frames"] for r in rows
                if r["impl"] == "py" and r.get("verified") is not None and r.get("seq_frames")]
        if verc and verp:
            same = same and abs(st.median(verc) - st.median(verp)) <= 0.5
        pair = (("HARD LOCK", "hard", 1), ("latency", "lat", 1),
                ("false alarm", "false", 1))
        parts = []
        for name, key, nd in pair:
            c, p_ = med("cpp", key), med("py", key)
            if c is not None and p_ is not None:
                parts.append(f"{name} {c - p_:+.{nd}f}")
        if verc and verp:
            parts.append(f"%verified {st.median(verc) - st.median(verp):+.1f} diem")
        print(f"\n  => chenh lech (C++ - Python): {', '.join(parts) if parts else '-'}")
        # Dem theo tung sequence — ro nghia hon viec gop trung vi cua nhieu input khac nhau.
        n_ok = n_tot = 0
        for seq in seqs:
            nm = os.path.basename(seq)
            c = [r for r in rows if r["impl"] == "cpp" and r["seq"] == nm]
            p_ = [r for r in rows if r["impl"] == "py" and r["seq"] == nm]
            if not c or not p_:
                continue
            n_tot += 1
            c, p_ = c[0], p_[0]
            ok = True
            for key, tol in (("hard", 0), ("false", 0), ("lat", 0.5), ("verified", 0)):
                a, b = c.get(key), p_.get(key)
                if a is None or b is None:
                    continue
                if abs(a - b) > tol:
                    ok = False
            n_ok += 1 if ok else 0
        if n_tot:
            print(f"  => khop {n_ok}/{n_tot} sequence (lan chay dau moi sequence; "
                  f"{args.reps} lan/sequence de kiem tinh tat dinh)")
        print(f"  => {'KHOP HOAN TOAN' if same else 'CO LECH — xem bang theo sequence duoi'}")

    # Bang theo sequence: nhin thay input nao lech.
    if len(impls) == 2:
        print(f"\n  {'sequence':<24}{'px':>7} | {'bb cpp/py (ms)':>24} | "
              f"{'FPS cpp/py':>18} | {'verif cpp/py':>14}")
        print("  " + "-" * 92)
        for seq in seqs:
            name = os.path.basename(seq)
            c = [r for r in rows if r["impl"] == "cpp" and r["seq"] == name]
            p = [r for r in rows if r["impl"] == "py" and r["seq"] == name]
            if not c or not p:
                continue
            cb = medv([r["backbone_ms"] for r in c])
            pb = medv([r["backbone_ms"] for r in p])
            cf = medv([r["fps"] for r in c])
            pf = medv([r["fps"] for r in p])
            cv = medv([r["verified"] for r in c])
            pv = medv([r["verified"] for r in p])
            if cb is None and pb is None:
                continue
            pxs = [r["bbox"] for r in c if r["bbox"] is not None]
            print(f"  {name:<24}{(st.median(pxs) if pxs else 0):>7.0f} | "
                  f"{sf(cb):>10}/{sf(pb):<10} | {sf(cf, 1):>8}/{sf(pf, 1):<8} | "
                  f"{sf(cv, 0):>6}/{sf(pv, 0):<6}")

    # Tinh lap lai: cung input + cung ban phai ra cung quyet dinh.
    for impl in impls:
        seen = {}
        for r in rows:
            if r["impl"] != impl:
                continue
            key = r["seq"]
            sig = (r["verified"], r["hard"], r["false"])
            if key in seen and seen[key] != sig:
                print(f"\n  CANH BAO: {impl} khong tat dinh tren {key}: "
                      f"{seen[key]} vs {sig} giua cac lan chay.")
            seen[key] = sig

    # Troi nhiet: lan chay cuoi cham hon lan dau bao nhieu.
    for impl in impls:
        sub = [r for r in rows if r["impl"] == impl]
        if len(sub) < 2:
            continue
        first, last = sub[0]["backbone_ms"], sub[-1]["backbone_ms"]
        if first and last and last / first > 1.05:
            print(f"\n  CANH BAO troi nhiet ({impl}): backbone "
                  f"{first:.2f} -> {last:.2f} ms (+{100*(last/first-1):.1f}%).")

    n_cpp = {r["n_bb"] for r in rows if r["impl"] == "cpp"}
    n_py = {r["n_bb"] for r in rows if r["impl"] == "py"}
    if n_cpp and n_py and n_cpp != n_py:
        print(f"\n  CANH BAO: so lan chay backbone khac nhau (cpp={n_cpp} py={n_py}).")
        print("            Hai ben khong lam cung khoi luong -> so sanh khong cong bang.")

    with open(os.path.join(PROJ, args.csv), "w", encoding="utf-8") as f:
        cols = ["impl", "seq", "rep", "bbox", "seq_frames", "frames", "backbone_ms",
                "n_bb", "temporal_ms", "n_tp", "fps", "fps_write", "hard", "false",
                "lat", "verified"]
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n")
    print(f"\n  ghi ra: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
