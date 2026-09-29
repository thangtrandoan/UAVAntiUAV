"""
Khóa pipeline: một nguồn sự thật cho train / eval / infer / calibrate.

Mục đích: chỉ sửa model. Mọi tham số pipeline lấy từ block `train` và
`data_pipeline`; các script khác bị ép theo và cảnh báo nếu config ghi lệch.

Lý do cần: trước đây mỗi script đọc tham số từ block của riêng nó
(`train_reid.py` -> `train.num_frames`, `infer.py` -> `infer.num_frames`), và
`infer.stride` chỉ được đồng bộ nếu config có `data_pipeline.frame_stride`.
Thiếu key đó thì `infer.py` chạy bước 2 trong khi train bước 4, không báo gì
(xem md/29thg9.md §2.4, §2.5).

Dùng:
    from pipeline_lock import resolve_pipeline, provenance

    inf_cfg = resolve_pipeline(cfg, 'infer', script='infer.py')
"""

import json
import os

# key trong section -> (block nguồn, key nguồn)
SOURCES = {
    'num_frames':    ('train',         'num_frames'),
    'stride':        ('data_pipeline', 'frame_stride'),
    'backbone':      ('train',         'backbone'),
    'bbox_padding':  ('data_pipeline', 'bbox_padding'),
    # Nhận dạng model: phải khớp giữa lúc train và mọi khâu nạp checkpoint.
    'temporal_type': ('train', 'temporal_type'),
    'temporal_pool': ('train', 'temporal_pool'),
    'temporal_pe':   ('train', 'temporal_pe'),
}

# Nhãn nguồn để in ra cho dễ đối chiếu.
_SRC_LABEL = {k: f'{blk}.{sk}' for k, (blk, sk) in SOURCES.items()}


def resolve_pipeline(cfg, section, script='', verbose=True):
    """Ép các tham số pipeline trong `cfg[section]` theo nguồn sự thật.

    Sửa dict tại chỗ và trả về chính nó. Section không tồn tại thì tạo rỗng.

    Không raise: config thiếu key chỉ sinh cảnh báo, để không chặn một lượt
    eval/infer đang chạy.
    """
    sec = cfg.get(section)
    if not isinstance(sec, dict):
        sec = {}
        cfg[section] = sec

    warnings = []
    resolved = {}   # key -> (giá trị nguồn, giá trị config ghi)

    for key, (blk, src_key) in SOURCES.items():
        source = (cfg.get(blk) or {}).get(src_key)
        own = sec.get(key)

        if source is None:
            if own is None:
                warnings.append(f"thiếu nguồn {blk}.{src_key} và {section}.{key} "
                                f"cũng không có -> không xác định được giá trị")
            else:
                warnings.append(f"thiếu nguồn {blk}.{src_key}; giữ nguyên "
                                f"{section}.{key} = {own!r} (không đồng bộ được)")
            resolved[key] = (own, own)
            continue

        if own is not None and own != source:
            warnings.append(f"lệch {section}.{key} = {own!r} vs {blk}.{src_key} = "
                            f"{source!r} -> ép dùng {source!r}")
        sec[key] = source
        resolved[key] = (source, own)

    if verbose:
        _print_banner(section, script, resolved, warnings)

    check_data_meta(cfg, verbose=verbose)
    return sec


# Các key mà `data_pipeline.py` dùng để sinh dữ liệu. Đổi chúng mà không sinh lại
# thì dữ liệu cũ và config mới lệch nhau.
_DATA_KEYS = ('frame_stride', 'num_before_frames', 'num_after_frames',
              'bbox_padding', 'crop_size')
_meta_seen = set()


def check_data_meta(cfg, verbose=True):
    """So `data_pipeline` trong config với `pipeline_meta.json` của dữ liệu đã sinh.

    Bốn file JSON (`query_test.json`, ...) chỉ chứa TÊN FILE, không cho biết bước
    thời gian đã dùng. Nếu config đổi `frame_stride` mà chưa sinh lại dữ liệu thì
    train (JSON cũ) và infer (đọc config mới) lệch nhau trong im lặng.

    Trả về list cảnh báo. Chỉ in một lần cho mỗi `data_dir`.
    """
    dp = cfg.get('data_pipeline') or {}
    data_dir = (cfg.get('paths') or {}).get('data_dir')
    if not data_dir:
        return []

    meta_path = os.path.join(data_dir, 'pipeline_meta.json')
    if not os.path.isfile(meta_path) or meta_path in _meta_seen:
        return []
    _meta_seen.add(meta_path)

    try:
        with open(meta_path, encoding='utf-8') as f:
            meta = json.load(f)
    except Exception as e:
        msg = f"không đọc được {meta_path}: {e}"
        if verbose:
            print(f"  [!] {msg}")
        return [msg]

    diffs = [f"{k}: dữ liệu sinh ở {meta[k]!r}, config đang là {dp[k]!r}"
             for k in _DATA_KEYS if k in meta and k in dp and meta[k] != dp[k]]
    if diffs:
        msg = (f"config `data_pipeline` KHÁC `{meta_path}` — dữ liệu CHƯA được sinh "
               f"lại, train và infer sẽ lệch nhau:")
        if verbose:
            print('-' * 62)
            print(f"  [!] {msg}")
            for d in diffs:
                print(f"      - {d}")
            print(f"      Sinh lại dữ liệu: python data_pipeline.py --config <config>")
            print('-' * 62)
        return [msg] + diffs
    return []


def _print_banner(section, script, resolved, warnings):
    line = '=' * 62
    print(line)
    print(f"  PIPELINE  script={script or '?'}  section={section}")
    print(line)
    for key, (val, own) in resolved.items():
        flag = f"   (config ghi {own!r})" if (own is not None and own != val) else ''
        print(f"  {key:<14} {val!r:<22} <- {_SRC_LABEL[key]}{flag}")
    print('-' * 62)
    if warnings:
        for w in warnings:
            print(f"  [!] {w}")
    else:
        print("  không có lệch; mọi script dùng cùng một pipeline")
    print(f"  biến tự do duy nhất: train.temporal_type")
    print(line + "\n")


def provenance(cfg):
    """Metadata ghi kèm kết quả đo, để biết số liệu thuộc protocol nào.

    Được ghi vào `calibrated_threshold.json` (xem calibrate_threshold.py).
    """
    tr = cfg.get('train') or {}
    dp = cfg.get('data_pipeline') or {}
    return {
        'num_frames':     tr.get('num_frames'),
        'frame_stride':   dp.get('frame_stride'),
        'temporal_type':  tr.get('temporal_type'),
        'temporal_pool':  tr.get('temporal_pool'),
        'temporal_pe':    tr.get('temporal_pe'),
        'backbone':       tr.get('backbone'),
        'n_frames_train': tr.get('n_frames_choices', 'fixed'),   # 'fixed' = N cố định
        'pipeline_lock':  'md/pipeline.md',
    }


def assert_frozen(cfg):
    """Kiểm tra các hằng số đã chốt. Trả về list cảnh báo (rỗng = đạt).

    Dùng ở đầu `train_reid.py` để phát hiện sớm việc vô tình mở lại biến đã khóa.
    """
    tr = cfg.get('train') or {}
    dp = cfg.get('data_pipeline') or {}
    loss = tr.get('loss') or {}
    out = []

    if tr.get('n_frames_choices'):
        out.append("train.n_frames_choices đang bật nhưng pipeline chốt N cố định; "
                   "mở lại biến này phá so sánh giữa các model")
    if not tr.get('temporal_type'):
        out.append("thiếu train.temporal_type; code sẽ im lặng rơi về 'mamba'. "
                   "Ghi rõ 'mamba' hoặc 'attention'")
    if tr.get('num_frames') != 12:
        out.append(f"train.num_frames = {tr.get('num_frames')} != 12 (hằng số đã chốt)")
    if tr.get('batch_size') != 12:
        out.append(f"train.batch_size = {tr.get('batch_size')} != 12; khác là bình thường "
                   f"khi chạy trên phần cứng khác, nhưng kết quả không so trực tiếp được")
    if dp.get('frame_stride') is None:
        out.append("thiếu data_pipeline.frame_stride -> mất đồng bộ toàn pipeline")

    # Cần đủ frame trong JSON cho N lớn nhất. `_load_clip` pad bằng cách LẶP frame
    # cuối, nên thiếu frame vẫn chạy nhưng im lặng tạo ra clip sai.
    n_max = max(list(tr.get('val_n_list') or []) + [tr.get('num_frames') or 0] or [0])
    for side in ('num_before_frames', 'num_after_frames'):
        have = dp.get(side)
        if isinstance(have, int) and n_max > have:
            out.append(f"{side} = {have} < N lớn nhất cần dùng ({n_max}); "
                       f"clip sẽ bị pad bằng cách lặp frame cuối")
    if tr.get('temporal_pool') == 'mean':
        out.append("train.temporal_pool = 'mean'. mean chỉ bất biến theo N khi frame iid; "
                   "frame UAV tương quan cao nên không. Pipeline chốt 'attn'")
    if loss.get('lam2') not in (0.0, 0):
        out.append(f"train.loss.lam2 = {loss.get('lam2')} != 0.0; TemporalConsistencyLoss "
                   f"hiện không đóng góp gradient (bản cũ độc hại, bản mới trơ)")
    return out
