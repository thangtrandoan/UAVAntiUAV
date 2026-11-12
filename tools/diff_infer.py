#!/usr/bin/env python3
"""So QUYET DINH giua ban Python (infer.py / infer_trt.py) va ban C++ (uav_reid_infer).

Khong so vector dac trung — da biet hai ben lech ~0.97 o nhanh resize. O day chi so
nhung gi quyet dinh ket qua: frame nao LOST, frame nao HARD LOCK, latency bao nhieu,
bao nhieu false alarm.

Doc:
    <py-dir>/metrics.txt        (infer.py in ra, echo moi dong log)
    <py-dir>/summary_metrics.txt (tuy chon, de doi chieu so tong hop)
    <cpp-dir>/events.txt        (uav_reid_infer)
    <cpp-dir>/summary.txt       (uav_reid_infer)

Dung:
    python tools/diff_infer.py --py-dir infer_output/resnet50_ibn_20thg10 \\
                               --cpp-dir infer_cpp/UAV-Anti-UAV_Test_000059
"""
import argparse
import os
import re
import sys

# Thu tu QUAN TRONG:
#  - "Anti-Hijack check" co chu Hijack viet hoa khac (chu i thuong) nen khong bi luat
#    HIJACK bat nham, nhung van phai dat sau cho chac.
#  - dong chuyen trang thai cua C++ la "<cu> -> T1_LOST (ly do)", cua Python la
#    "Target LOST! -> T1_LOST" — ca hai deu chua "-> T1_LOST".
#  - luat T2_LOST phai dat TRUOC luat T1_LOST, vi dong "T2_SEARCH -> T1_LOST"
#    chua ca hai mau.
RULES = [
    ("HIJACK",       r"HIJACK"),
    ("HIJACK_CHECK", r"Anti-Hijack check"),
    ("HARD_LOCK",    r"HARD LOCK!"),
    ("HARD_COLLECT", r"HARD LOCK collecting"),
    ("FINE_FAIL",    r"Fine FAILED"),
    ("SOFT_PASS",    r"SOFT LOCK PASS"),
    ("SOFT_FAIL",    r"SOFT LOCK FAIL"),
    ("COARSE_FAIL",  r"Coarse FAILED"),
    ("SOFT_COLLECT", r"Soft Lock collecting"),
    ("GAP_SKIP",     r"T2_SEARCH: vang"),
    ("T2_LOST",      r"UAV lost during T2_SEARCH|T2_SEARCH -> T1_LOST"),
    ("REAPPEAR",     r"reappeared from GT|-> T2_SEARCH"),
    ("T1_LOST",      r"Target LOST|-> T1_LOST"),
    ("BANK_MOMENT",  r"Last-moment Memory Bank"),
    ("BANK_UPDATE",  r"bank update|Anchor updated|Recent updated"),
    ("DEBUG",        r"DEBUG "),
]

LABEL = {
    "T1_LOST": "T1_LOST (mat muc tieu)",
    "REAPPEAR": "T2_SEARCH (tai xuat)",
    "GAP_SKIP": "bo qua frame vang",
    "T2_LOST": "T2_SEARCH -> LOST",
    "SOFT_COLLECT": "soft dang thu",
    "SOFT_PASS": "SOFT LOCK PASS",
    "SOFT_FAIL": "SOFT LOCK FAIL",
    "COARSE_FAIL": "coarse truot",
    "HARD_COLLECT": "hard dang thu",
    "HARD_LOCK": "HARD LOCK",
    "FINE_FAIL": "fine truot (false alarm)",
    "HIJACK_CHECK": "kiem anti-hijack",
    "HIJACK": "phat hien bi chiem",
    "BANK_MOMENT": "cap nhat bank truoc LOST",
    "BANK_UPDATE": "cap nhat bank",
    "DEBUG": "dong chan doan",
}


def classify(msg: str):
    for kind, pat in RULES:
        if re.search(pat, msg):
            return kind
    return None


def parse_log(path: str):
    """Tra ve [(frame, kind, msg, vals)] — hai dinh dang cua hai ben.

    Python: '[158] Target LOST! -> T1_LOST'
    C++   : '[f 158] T1_LOST -> T1_LOST (bbox vang)'
    """
    out, unknown = [], []
    if not os.path.exists(path):
        return out, unknown
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.match(r"^\[(?:f )?(\d+)\]\s+(.*)$", line.strip())
            if not m:
                continue
            frame, msg = int(m.group(1)), m.group(2)
            kind = classify(msg)
            if kind is None:
                unknown.append((frame, msg))
                continue
            vals = {}
            for key, pat in (("fine", r"fine=([0-9.]+)"),
                             ("soft", r"soft=([0-9.]+)"),
                             ("coarse", r"coarse=([0-9.]+)"),
                             ("lat", r"latency=(\d+)|Latency: (\d+)")):
                mm = re.search(pat, msg)
                if mm:
                    g = next(g for g in mm.groups() if g is not None)
                    vals[key] = float(g) if key != "lat" else int(g)
            out.append((frame, kind, msg, vals))
    return out, unknown


def parse_numbers(path: str, patterns):
    """Lay cac so tong hop theo cap (nhan, regex). Tra ve dict nhan -> so."""
    got = {}
    if not os.path.exists(path):
        return got
    text = open(path, "r", encoding="utf-8", errors="replace").read()
    for name, pat in patterns:
        m = re.search(pat, text)
        if m:
            got[name] = float(m.group(1))
    return got


def dedupe_loss(events):
    """Mot lan mat co the bi in 2 dong (nguyen nhan + 'Target LOST'). Chi giu dong
    co NGUYEN NHAN (T2_LOST), bo dong T1_LOST trung frame do."""
    lost_cause = {f for f, k, _, _ in events if k == "T2_LOST"}
    return [(f, k, m, v) for f, k, m, v in events
            if not (k == "T1_LOST" and f in lost_cause)]


def frames_by_kind(events):
    d = {}
    for frame, kind, _, _ in events:
        d.setdefault(kind, []).append(frame)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py-dir", required=True, help="thu muc infer.py ghi ra")
    ap.add_argument("--cpp-dir", required=True, help="thu muc uav_reid_infer ghi ra")
    ap.add_argument("--py-log", default=None, help="duong dan metrics.txt (neu khac)")
    ap.add_argument("--max-detail", type=int, default=8,
                    help="in toi da bao nhieu dong lech chi tiet cho moi loai")
    args = ap.parse_args()

    py_log = args.py_log or os.path.join(args.py_dir, "metrics.txt")
    cpp_events = os.path.join(args.cpp_dir, "events.txt")
    py_sum = os.path.join(args.py_dir, "summary_metrics.txt")
    if not os.path.exists(py_sum):
        py_sum = py_log  # infer.py ghi ca summary vao metrics.txt
    cpp_sum = os.path.join(args.cpp_dir, "summary.txt")

    py_ev, py_unk = parse_log(py_log)
    cpp_ev, cpp_unk = parse_log(cpp_events)
    py_ev, cpp_ev = dedupe_loss(py_ev), dedupe_loss(cpp_ev)
    if not py_ev:
        print(f"  KHONG doc duoc su kien Python: {py_log}")
    if not cpp_ev:
        print(f"  KHONG doc duoc su kien C++: {cpp_events}")
    if not py_ev or not cpp_ev:
        return 2

    py_k = frames_by_kind(py_ev)
    cpp_k = frames_by_kind(cpp_ev)

    print(f"\n  Python: {len(py_ev)} su kien  ({py_log})")
    print(f"  C++   : {len(cpp_ev)} su kien  ({cpp_events})\n")

    print("  %-28s %6s %6s  %s" % ("loai su kien", "py", "cpp", "frame lech"))
    print("  " + "-" * 74)
    kinds = sorted(set(py_k) | set(cpp_k),
                   key=lambda k: [r[0] for r in RULES].index(k))
    bad = 0
    for k in kinds:
        a, b = py_k.get(k, []), cpp_k.get(k, [])
        only_py = [f for f in a if f not in b]
        only_cpp = [f for f in b if f not in a]
        # DEBUG la dong chan doan, khong phai quyet dinh: ban C++ khong in duoc phan
        # PRE-BN (engine khong xuat raw_feat), nen thieu khong tinh la lech.
        if k == "DEBUG":
            print("  %-28s %6d %6d  (chi de tham khao, khong tinh lech)"
                  % (LABEL[k], len(a), len(b)))
            continue
        mark = "" if not (only_py or only_cpp) else "  <-- LECH"
        print("  %-28s %6d %6d  %s%s"
              % (LABEL[k], len(a), len(b),
                 (f"py:{only_py[:args.max_detail]} cpp:{only_cpp[:args.max_detail]}"
                  if (only_py or only_cpp) else ""), mark))
        bad += len(only_py) + len(only_cpp)

    # --- HARD LOCK: so ca diem va latency, khong chi so frame ---
    py_h = [(f, v) for f, k, _, v in py_ev if k == "HARD_LOCK"]
    cpp_h = [(f, v) for f, k, _, v in cpp_ev if k == "HARD_LOCK"]
    if py_h or cpp_h:
        print("\n  HARD LOCK chi tiet:")
        print("  %-8s %-22s %-22s" % ("frame", "py (fine/lat)", "cpp (fine/lat)"))
        for i in range(max(len(py_h), len(cpp_h))):
            a = py_h[i] if i < len(py_h) else None
            b = cpp_h[i] if i < len(cpp_h) else None
            fmt = lambda t: (f"f{t[0]:<5} fine={t[1].get('fine','-')}"
                             f" lat={t[1].get('lat','-')}") if t else "-"
            same = (a and b and a[0] == b[0]
                    and abs(a[1].get("fine", 0) - b[1].get("fine", 0)) < 1e-3)
            print("  %-8s %-22s %-22s %s" % (a[0] if a else "-", fmt(a), fmt(b),
                                             "" if same else "<-- khac"))

    # --- so tong hop ---
    P = [("false", r"False Alarms \(Fine Fails\)\s*:\s*(\d+)"),
         ("tn", r"Avg Mamba \+ Head Time\s*:\s*[\d.]+ ms\s*\(n=(\d+)"),
         ("lat", r"Re-acquisition Latency\s*:\s*([\d.]+) frames")]
    C = [("false", r"False alarm \(fine fail\):\s*(\d+)"),
         ("tn", r"temporal \(H2D\+TRT\)\s*:\s*[\d.]+ ms\s+\(n=(\d+)\)"),
         ("lat", r"Re-acquisition latency\s*:\s*mean ([\d.]+)")]
    ps, cs = parse_numbers(py_sum, P), parse_numbers(cpp_sum, C)
    ps["hard"] = float(len(py_k.get("HARD_LOCK", [])))
    cs["hard"] = float(len(cpp_k.get("HARD_LOCK", [])))
    print("\n  %-32s %10s %10s" % ("tong hop", "py", "cpp"))
    print("  " + "-" * 54)
    for key, name in (("hard", "so lan HARD LOCK"),
                      ("false", "false alarm"),
                      ("tn", "so lan chay temporal"),
                      ("lat", "latency trung binh (frame)")):
        pv, cv = ps.get(key), cs.get(key)
        flag = "" if (pv is not None and cv is not None
                      and abs(pv - cv) < 1e-6) else "  <-- khac"
        print("  %-32s %10s %10s%s"
              % (name, "-" if pv is None else f"{pv:g}",
                 "-" if cv is None else f"{cv:g}", flag))

    # Dong khong phan loai duoc: thuong la cau chu khac nhau giua hai ban.
    # In ra de biet co can them luat khong, thay vi am tham bo qua.
    for name, unk in (("Python", py_unk), ("C++", cpp_unk)):
        if unk:
            print(f"\n  {name}: {len(unk)} dong khong phan loai duoc (bo qua):")
            for frame, msg in unk[:args.max_detail]:
                print(f"    [{frame}] {msg[:100]}")
            if len(unk) > args.max_detail:
                print(f"    ... con {len(unk) - args.max_detail} dong")

    print()
    if bad == 0:
        print("  KET LUAN: khop su kien. Vector lech 0.97 khong lam doi quyet dinh.")
        return 0
    print(f"  KET LUAN: {bad} su kien lech. Xem bang tren de biet loai nao.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
