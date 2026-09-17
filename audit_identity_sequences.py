#!/usr/bin/env python3
"""
=============================================================================
 AUDIT: quan hệ identity_id <-> sequence_id trong query/gallery JSON
=============================================================================
Vì sao cần: khối `=== CHAN DOAN NHIEU NHAN / TRUNG DANH TINH ===` trong
`calibrate_threshold.py` cho biết CẶP PID NÀO bị nhầm, nhưng KHÔNG cho biết
liệu điều đó là do **cấu trúc dataset** (một drone thật bị tách thành 2 nhãn,
hoặc nhãn là cục bộ theo từng chuỗi) hay do **embedding thật sự dễ nhầm**.

Script này chỉ đọc JSON — KHÔNG cần model, KHÔNG cần GPU, chạy vài giây.

Cách dùng:
    python audit_identity_sequences.py \
        --query   processed/query_test.json \
        --gallery processed/gallery_test.json

Hoặc lấy đường dẫn từ config:
    python audit_identity_sequences.py --config configs/config_colab.yaml
=============================================================================
"""

import argparse
import json
import os
from collections import Counter, defaultdict


def load_cfg(path):
    import yaml
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    ec = cfg.get('eval', {})
    return ec.get('query_json'), ec.get('gallery_json'), cfg.get('paths', {}).get('data_dir')


def _get(rec, *names):
    """Lấy giá trị đầu tiên tồn tại trong các tên key có thể có."""
    for n in names:
        if n in rec and rec[n] is not None:
            return rec[n]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--query', default=None)
    ap.add_argument('--gallery', default=None)
    ap.add_argument('--config', default=None)
    ap.add_argument('--top', type=int, default=25)
    args = ap.parse_args()

    q_path, g_path = args.query, args.gallery
    if args.config:
        cq, cg, _ = load_cfg(args.config)
        q_path = q_path or cq
        g_path = g_path or cg
    if not q_path or not g_path:
        ap.error("Cần --query/--gallery hoặc --config")

    print("=" * 74)
    print("  AUDIT: identity_id <-> sequence_id")
    print("=" * 74)
    print(f"  Query  : {q_path}")
    print(f"  Gallery: {g_path}")

    with open(q_path) as f:
        queries = json.load(f)
    with open(g_path) as f:
        galleries = json.load(f)
    if isinstance(queries, dict):
        queries = list(queries.values())
    if isinstance(galleries, dict):
        galleries = list(galleries.values())

    print(f"\n[1] Kích thước & key có sẵn")
    print(f"  queries  : {len(queries)}   | galleries: {len(galleries)}")
    if queries:
        print(f"  keys(query)  : {sorted(queries[0].keys())}")
    if galleries:
        print(f"  keys(gallery): {sorted(galleries[0].keys())}")

    # ---- Thu thập ----
    id2seqs = defaultdict(set)        # identity -> {sequence_id}
    id_events = defaultdict(int)      # identity -> số event
    seq2ids = defaultdict(set)        # sequence -> {identity}
    pid2frames = defaultdict(int)     # identity -> tổng số frame
    for rec in list(queries) + list(galleries):
        pid = _get(rec, 'identity_id', 'pid')
        seq = _get(rec, 'sequence_id', 'seq_id', 'sequence')
        if pid is None or seq is None:
            continue
        id2seqs[pid].add(seq)
        seq2ids[seq].add(pid)
        id_events[pid] += 1
        pid2frames[pid] += len(_get(rec, 'frames') or [])

    if not id2seqs:
        print("\n❌ Không đọc được identity_id/sequence_id. Kiểm tra tên key ở mục [1].")
        return

    n_ids, n_seqs = len(id2seqs), len(seq2ids)
    print(f"\n[2] Tổng quan")
    print(f"  Số identity_id khác nhau : {n_ids}")
    print(f"  Số sequence_id khác nhau : {n_seqs}")
    print(f"  Số event                 : {sum(id_events.values())}")

    # ---- [3] Mỗi identity xuất hiện ở bao nhiêu sequence? ----
    nseq_hist = Counter(len(s) for s in id2seqs.values())
    print(f"\n[3] Mỗi identity_id xuất hiện ở BAO NHIÊU sequence?")
    print(f"  {'#seq/identity':>14} {'#identities':>12}  {'tỉ lệ':>8}")
    for k in sorted(nseq_hist):
        print(f"  {k:>14} {nseq_hist[k]:>12}  {nseq_hist[k]/n_ids*100:>7.1f}%")
    share_one = nseq_hist.get(1, 0) / n_ids
    if share_one > 0.9:
        print(f"  ⚠️ {share_one*100:.1f}% identity CHỈ ở 1 sequence ⇒ `identity_id` là CỤC BỘ theo chuỗi.")
        print(f"     ⇒ Cùng một drone thật ở 2 chuỗi sẽ mang 2 identity_id KHÁC NHAU ⇒ mọi cặp")
        print(f"       xuyên-chuỗi của nó bị tính là IMPOSTOR ⇒ chính là 'đuôi nặng' làm sập TAR@FAR.")
    else:
        print(f"  ⇒ {100-share_one*100:.1f}% identity trải trên ≥2 sequence ⇒ `identity_id` mang tính TOÀN CỤC.")

    # ---- [4] Một sequence có bao nhiêu identity cùng lúc? ----
    per_seq = Counter(len(v) for v in seq2ids.values())
    print(f"\n[4] Mỗi sequence chứa BAO NHIÊU identity_id?")
    print(f"  {'#identity/seq':>13} {'#sequences':>11}  {'tỉ lệ':>8}")
    for k in sorted(per_seq):
        print(f"  {k:>13} {per_seq[k]:>11}  {per_seq[k]/n_seqs*100:>7.1f}%")
    print(f"  -> Nếu phần lớn sequence có NHIỀU identity ⇒ có nhiều drone cùng lúc ⇒ hard negative")
    print(f"     CÙNG CHUỖI là chuyện bình thường (không phải lỗi nhãn).")

    # ---- [5] Cặp identity hay đi cùng nhau nhất ----
    co = Counter()
    for seq, ids in seq2ids.items():
        ids = sorted(ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                co[(ids[i], ids[j])] += 1
    print(f"\n[5] Cặp identity_id XUẤT HIỆN CÙNG NHAU ở nhiều sequence nhất:")
    print(f"  {'pid_a':>8} {'pid_b':>8} {'#seq chung':>11} {'#seq(a)':>8} {'#seq(b)':>8}  "
          f"{'Jaccard':>8}  {'nghi trung nhan?':>16}")
    for (a, b), c in co.most_common(args.top):
        sa, sb = id2seqs[a], id2seqs[b]
        jac = c / len(sa | sb)
        # ⚠️ Nếu MỌI identity chỉ ở 1 sequence thì Jaccard=1.0 là TẤT YẾU (không có thông tin).
        # Dấu hiệu thật chỉ có khi cặp đó dùng chung ≥2 sequence, hoặc ≥1 bên trải nhiều sequence.
        informative = (c >= 2) or (len(sa) > 1) or (len(sb) > 1)
        if not informative:
            flag = "- (Jaccard vo nghia)"
        elif jac >= 0.8:
            flag = "NGHI CAO"
        elif jac >= 0.5:
            flag = "nghi vua"
        else:
            flag = "khong"
        print(f"  {a:>8} {b:>8} {c:>11} {len(sa):>8} {len(sb):>8}  {jac:>8.3f}  {flag:>16}")
    if n_seq_hist_multi := sum(v for k, v in nseq_hist.items() if k >= 2):
        print(f"  -> Có {n_seq_hist_multi} identity trải trên ≥2 sequence ⇒ cột Jaccard MỚI có ý nghĩa.")
        print(f"     Jaccard ≈ 1.0 (và #seq chung ≥2) ⇒ hai nhãn gần như TRÙNG tập sequence ⇒")
        print(f"     nghi mạnh một drone bị tách thành 2 nhãn (đối chiếu mục [B] của chẩn đoán).")
    else:
        print(f"  -> ⚠️ MỌI identity chỉ ở 1 sequence ⇒ cột Jaccard KHÔNG có ý nghĩa (luôn = 1.0 cho")
        print(f"     hai identity bất kỳ trong cùng chuỗi). Chỉ dùng cột '#seq chung' và mục [3].")
        print(f"     Tín hiệu trùng nhãn phải đến từ ẢNH (mục [A] của chẩn đoán), không từ Jaccard.")

    # ---- [6] Ước lượng tỉ lệ cặp impostor XUYÊN CHUỖI (không cần model) ----
    ev2id = {}
    for rec in list(queries) + list(galleries):
        pid = _get(rec, 'identity_id', 'pid')
        seq = _get(rec, 'sequence_id', 'seq_id', 'sequence')
        ev = _get(rec, 'event_index')
        if pid is not None and seq is not None:
            ev2id[(seq, ev)] = pid
    entries = list(ev2id.items())
    same_seq_imp = cross_seq_imp = gen = 0
    for (s1, e1), p1 in entries:
        for (s2, e2), p2 in entries:
            if (s1, e1) == (s2, e2):
                continue
            if p1 == p2:
                gen += 1
            elif s1 == s2:
                same_seq_imp += 1
            else:
                cross_seq_imp += 1
    tot_imp = same_seq_imp + cross_seq_imp
    if tot_imp:
        print(f"\n[6] Cấu trúc cặp impostor (trên toàn bộ event, không cần model)")
        print(f"  genuine (cùng pid)          : {gen:>10,}")
        print(f"  impostor CÙNG chuỗi         : {same_seq_imp:>10,}  ({same_seq_imp/tot_imp*100:.1f}%)")
        print(f"  impostor KHÁC chuỗi         : {cross_seq_imp:>10,}  ({cross_seq_imp/tot_imp*100:.1f}%)")
        print(f"  -> Nếu impostor KHÁC chuỗi chiếm ưu thế ⇒ ngưỡng FAR toàn cục chủ yếu bị quyết định")
        print(f"     bởi các cặp XUYÊN CHUỖI, mà đó không phải thứ pipeline phải chống (§ KL7).")
        print(f"     Nếu impostor CÙNG chuỗi chiếm ưu thế ⇒ ngưỡng phản ánh đúng bài toán pipeline.")

    print("\n" + "=" * 74)
    print("  Gửi lại TOÀN BỘ output này (mục [2]–[6]) để kết luận.")
    print("=" * 74)


if __name__ == '__main__':
    main()
