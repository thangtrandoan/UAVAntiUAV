"""Nạp checkpoint kèm báo cáo missing / unexpected / lệch shape."""

import torch

from .temporal_mamba import HAS_MAMBA


def _is_classifier_key(name):
    """
    Key của các đầu CLASSIFIER (không tham gia trích feature khi eval).

    Vì sao cần phân biệt: checkpoint train với `num_identities` khác model khởi tạo
    (vd: 502 danh tính lúc train vs 1000 default) sẽ khiến các key này lệch shape.
    Đó là chuyện BÌNH THƯỜNG, không phải lỗi:
      - `backbone.classifier_global` / `classifier_fs`: đầu phân loại pretrain của DINOv3/GASNet
        (chỉ dùng cho pretraining objective)
      - `head.classifier`: đầu ID classifier trong ReIDHead — `ReIDHead.forward` chỉ gọi nó khi
        `self.training == True`; eval trả về `bn_feat` TRƯỚC đó (xem ReIDHead.forward).
    Nên thiếu/lệch các key này KHÔNG ảnh hưởng feature dùng để matching.
    """
    n = name.lower()
    return ('classifier' in n) or n.endswith('.fc.weight') or n.endswith('.fc.bias')


def load_checkpoint_verbose(model, checkpoint_path, tag="checkpoint", log=print):
    """
    Load checkpoint vào model, ĐỒNG THỜI báo cáo đầy đủ:
      - missing        : key có trong model nhưng checkpoint không có → giữ init hiện tại
      - unexpected     : key có trong checkpoint nhưng model không có → bị bỏ
      - shape mismatch : key cùng tên nhưng khác shape → bị bỏ (trước đây lặng im)

    Lý do: mọi chỗ load đều dùng `strict=False`, nghĩa là checkpoint thiếu key
    (ví dụ TOÀN BỘ `backbone.*`) vẫn in ra "Loaded" như thành công. Nếu backbone
    không được nạp, visual feature là feature pretrain chung chứ không phải feature
    đã train → mọi kết luận eval/infer đều nhiễu.

    🛠️ (14/9) FIX: các key CLASSIFIER (xem `_is_classifier_key`) được tách riêng và KHÔNG
    kích hoạt cảnh báo nghiêm trọng — lệch `num_identities` giữa checkpoint và model khởi tạo
    là chuyện bình thường và chúng không nằm trên đường trích feature. Cảnh báo nghiêm trọng
    chỉ bật khi thiếu key `backbone.*` KHÔNG phải classifier.

    Trả về dict summary để caller log/lưu.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model_state = model.state_dict()

    new_state_dict = {}
    skipped_shape = []
    for k, v in state_dict.items():
        new_k = k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k
        if new_k in model_state and tuple(v.shape) != tuple(model_state[new_k].shape):
            skipped_shape.append((new_k, tuple(v.shape), tuple(model_state[new_k].shape)))
            continue
        new_state_dict[new_k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

    n_model = len(model_state)
    n_loaded = sum(1 for k in model_state if k in new_state_dict)
    # Chỉ tính các key ẢNH HƯỞNG ĐẶC TRƯNG là "nghiêm trọng"
    backbone_missing = [k for k in missing
                        if k.startswith('backbone.') and not _is_classifier_key(k)]
    head_feature_missing = [k for k in missing
                            if k.startswith('head.') and not _is_classifier_key(k)
                            and 'bnneck' not in k]
    # (24/9) `temporal_encoder.*` CŨNG là trọng số ĐẶC TRƯNG. Thiếu key ở đây nghĩa là
    # kiến trúc lúc DỰNG MODEL khác lúc TRAIN (`temporal_type` hoặc `temporal_pool` truyền sai)
    # -> encoder chạy random init. Trước đây nhóm này chỉ nằm trong  MISSING chung nên RẤT DỄ
    # bỏ sót (đã xảy ra thật: eval dựng `pool='attn'` cho checkpoint train `pool='mean'`).
    # Loại `pos_embed` ra vì nó là thành phần TÙY CHỌN đã có khối ℹ riêng giải thích bên dưới.
    temporal_missing = [k for k in missing
                        if k.startswith('temporal_encoder.') and 'pos_embed' not in k]
    # (24/9) Lỗ hổng ĐỐI XỨNG với `temporal_missing`: nếu config nói `pool='mean'` nhưng
    # checkpoint train bằng `pool='attn'` thì `attn_pool.*` là UNEXPECTED (KHÔNG phải MISSING)
    # -> trọng số attention pooling bị VỨT ĐI âm thầm và pooling rơi về `mean*sqrt(N)`.
    temporal_unexpected = [k for k in unexpected
                           if k.startswith('temporal_encoder.') and 'pos_embed' not in k]
    classifier_missing = [k for k in missing if _is_classifier_key(k)]
    classifier_mismatch = [s for s in skipped_shape if _is_classifier_key(s[0])]

    log(f"  [{tag}] {os.path.basename(str(checkpoint_path))}: "
        f"{n_loaded}/{n_model} keys của model được nạp "
        f"({len(state_dict)} keys trong file)")

    def _group(keys):
        groups = {}
        for k in keys:
            groups.setdefault(k.split('.')[0], []).append(k)
        return groups

    # (24/9) TÁCH key CLASSIFIER ra khỏi khối  . `_is_classifier_key` đã có sẵn và khối ℹ
    # bên dưới giải thích chúng là BÌNH THƯỜNG, NHƯNG trước đây chúng vẫn bị liệt kê dưới  và
    # in RA TRƯỚC khối ℹ đó -> người đọc thấy "  MISSING 3 keys" rồi tưởng hỏng, dù 3 key đó
    # chỉ là đầu classifier (lệch `num_identities`), KHÔNG nằm trên đường trích feature.
    missing_feature = [k for k in missing if not _is_classifier_key(k)]
    skipped_shape_feature = [s for s in skipped_shape if not _is_classifier_key(s[0])]

    if missing_feature:
        log(f"  [{tag}] ⚠️ MISSING {len(missing_feature)} keys (giữ init hiện tại):")
        for prefix, keys in sorted(_group(missing_feature).items(), key=lambda kv: -len(kv[1])):
            log(f"      - {prefix}.* : {len(keys)} keys (vd: {keys[0]})")
    # (22/9) `pos_embed` bị BỎ khỏi kiến trúc (§16) nên key này thành "unexpected".
    # Đây là thay đổi CHỦ Ý, không phải lỗi -> báo riêng để không gây hoang mang.
    _pe = [k for k in unexpected if 'pos_embed' in k]
    _te_set = set(temporal_unexpected)
    _other_unexp = [k for k in unexpected if 'pos_embed' not in k and k not in _te_set]
    if _pe:
        log(f"  [{tag}] ℹ️ {len(_pe)} key `pos_embed` bị bỏ — CHỦ Ý, không phải lỗi "
            f"(md/22thg9.md §16): {_pe}")
    if _other_unexp:
        log(f"  [{tag}] ⚠️ UNEXPECTED {len(_other_unexp)} keys trong checkpoint (bị bỏ):")
        for prefix, keys in sorted(_group(_other_unexp).items(), key=lambda kv: -len(kv[1])):
            log(f"      - {prefix}.* : {len(keys)} keys (vd: {keys[0]})")
    if skipped_shape_feature:
        log(f"  [{tag}] ⚠️ SHAPE MISMATCH {len(skipped_shape_feature)} keys (bị bỏ):")
        for name, ck_shape, md_shape in skipped_shape_feature[:10]:
            log(f"      - {name}: checkpoint{ck_shape} vs model{md_shape}")

    # --- Chẩn đoán implementation của TEMPORAL ENCODER (Mamba thật vs SimpleS6Block fallback) ---
    # Vì sao quan trọng: `TemporalMambaEncoder.__init__` chọn module theo `HAS_MAMBA`. Nếu lúc TRAIN
    # có `mamba_ssm` mà lúc EVAL không (hoặc ngược lại) thì trọng số temporal KHÔNG khớp và nhánh
    # temporal chạy random — một confound bậc một khi đánh giá "temporal có ý nghĩa không".
    # Phân biệt bằng shape (hai implementation KHÁC shape):
    # SimpleS6Block : x_proj = Linear(d_inner, d_state*2 + 1) = (33, d_inner), dt_proj = (d_inner, 1)
    # mamba_ssm     : x_proj = Linear(d_inner, dt_rank + 2*d_state) = (64, d_inner) với d_model=512,
    # dt_rank = ceil(512/16) = 32, dt_proj = (d_inner, dt_rank)
    def _find_shape(sd, target):
        for k, v in sd.items():
            nk = k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k
            if nk == target:
                return tuple(v.shape)
        return None

    _xk = 'temporal_encoder.layers.0.x_proj.weight'
    _dk = 'temporal_encoder.layers.0.dt_proj.weight'
    ck_x = _find_shape(state_dict, _xk)
    ck_d = _find_shape(state_dict, _dk)
    ms_x = _find_shape(model_state, _xk)
    active_impl = 'mamba_ssm.Mamba (thật)' if HAS_MAMBA else 'SimpleS6Block (fallback)'
    temporal_arch_ok = True

    # (24/9) `x_proj`/`dt_proj` là key RIÊNG CỦA MAMBA. Với encoder ATTENTION thì việc thiếu
    # chúng là BÌNH THƯỜNG — trước đây khối này báo ℹ kèm câu "checkpoint có thể thiếu cả nhánh
    # temporal" (gây hoang mang SAI) và đặt `temporal_arch_ok = False`.
    # Với ATTENTION, tiêu chí đúng là: `temporal_encoder.*` có được nạp ĐỦ hay không.
    _attn_enc = type(getattr(model, 'temporal_encoder', None)).__name__ == 'TemporalAttentionEncoder'

    if _attn_enc:
        temporal_arch_ok = (len(temporal_missing) == 0 and len(temporal_unexpected) == 0)
        if temporal_arch_ok:
            _n_tc = len([k for k in model_state if k.startswith('temporal_encoder.')])
            log(f"  [{tag}] ℹ️ Temporal encoder: **ATTENTION** — nạp đủ {_n_tc} keys "
                f"`temporal_encoder.*`. Bỏ qua kiểm tra `x_proj`/`dt_proj` (key riêng của Mamba).")
        else:
            _bad = []
            if temporal_missing:
                _bad.append(f"THIẾU {len(temporal_missing)} keys")
            if temporal_unexpected:
                _bad.append(f"THỪA {len(temporal_unexpected)} keys (có trong checkpoint, "
                            f"không có trong model)")
            log(f"  [{tag}] ❌ Temporal encoder: {' + '.join(_bad)} `temporal_encoder.*` → "
                f"trọng số temporal KHÔNG khớp kiến trúc (xem chi tiết bên dưới).")
    elif ck_x is None:
        # (24/9) Câu cũ — "checkpoint có thể thiếu cả nhánh temporal" — bị in CẢ KHI model là
        # ATTENTION, lúc đó nó VÔ NGHĨA (attention không có `x_proj`) và gây hoang mang.
        # Nay chỉ in trong nhánh MAMBA này, và nói đúng nguyên nhân khả dĩ nhất.
        log(f"  [{tag}] ℹ️ Model là MAMBA nhưng checkpoint KHÔNG có `{_xk}` → không xác định "
            f"được implementation temporal. Nhiều khả năng checkpoint được train bằng "
            f"`temporal_type` khác (ATTENTION?) hoặc thiếu hẳn nhánh temporal.")
        temporal_arch_ok = False
    else:
        if ck_x[0] == 33:
            ck_impl = 'SimpleS6Block (fallback)'
        elif ms_x is not None and ck_x[0] == ms_x[0]:
            ck_impl = active_impl
        else:
            ck_impl = f'KHÁC (x_proj={ck_x}, model={ms_x})'
        if ms_x is not None and ck_x != ms_x:
            temporal_arch_ok = False
            log(f"  [{tag}] ❌ CẢNH BÁO NGHIÊM TRỌNG: trọng số TEMPORAL KHÔNG khớp implementation!")
            log(f"      checkpoint: x_proj{ck_x}, dt_proj{ck_d}  ({ck_impl})")
            log(f"      model     : x_proj{ms_x}, dt_proj{_find_shape(model_state, _dk)}  ({active_impl})")
            log(f"      → `x_proj`/`dt_proj` bị bỏ do lệch shape → nhánh temporal chạy RANDOM.")
            log(f"      → Mọi kết luận về 'temporal/Mamba' đều vô hiệu. Phải cài đúng mamba_ssm "
                f"hoặc TRAIN LẠI với implementation đang dùng.")
        else:
            log(f"  [{tag}] ℹ️ Temporal encoder: cả checkpoint và model đều dùng **{ck_impl}** "
                f"(x_proj{ck_x}) → trọng số temporal khớp.")
            if not HAS_MAMBA and ck_impl == 'SimpleS6Block (fallback)':
                log(f"      ⚠️ Đây KHÔNG phải Mamba thật (`mamba_ssm` chưa cài). Muốn dùng Mamba thật:")
                log(f"         1) `pip install mamba-ssm causal-conv1d`; 2) **TRAIN LẠI** — không chuyển")
                log(f"         được trọng số vì `x_proj`/`dt_proj` khác shape. Trọng số còn lại (conv1d,")
                log(f"         A_log, D, in_proj, out_proj) thì tương thích.")

    # --- Giải thích các key classifier (BÌNH THƯỜNG, không phải lỗi) ---
    if classifier_missing or classifier_mismatch:
        ck_cls = [s[1][0] for s in classifier_mismatch if s[0].endswith('.weight')]
        md_cls = [s[2][0] for s in classifier_mismatch if s[0].endswith('.weight')]
        detail = ""
        if ck_cls and md_cls:
            detail = (f" (số lớp: checkpoint={ck_cls[0]} vs model={md_cls[0]})"
                      f" → model khởi tạo `num_identities` khác lúc train")
        # (24/9) Một key lệch shape nằm ở CẢ `missing` lẫn `skipped_shape`, nên câu cũ
        # "3 missing + 3 lệch shape" đọc như 6 vấn đề trong khi thật ra chỉ là 3 KEY. Tách rõ.
        _cls_mm = {s[0] for s in classifier_mismatch}
        _cls_only_missing = [k for k in classifier_missing if k not in _cls_mm]
        _parts = []
        if classifier_mismatch:
            _parts.append(f"{len(classifier_mismatch)} key lệch shape (bị bỏ, không nạp)")
        if _cls_only_missing:
            _parts.append(f"{len(_cls_only_missing)} key thiếu hẳn")
        log(f"  [{tag}] ℹ️ Đầu CLASSIFIER: " + " + ".join(_parts) + detail)
        log(f"      → KHÔNG phải lỗi: `head.classifier` chỉ dùng khi training "
            f"(eval trả `bn_feat` trước đó); `backbone.classifier_*` là đầu phân loại pretrain.")
        log(f"      → Đường trích feature (backbone trunk + temporal + bnneck) KHÔNG bị ảnh hưởng.")
        # Nếu muốn eval với đúng num_classes: truyền num_identities=<số lớp của checkpoint>.

    if backbone_missing or head_feature_missing or temporal_missing or temporal_unexpected:
        log(f"  [{tag}] ❌ CẢNH BÁO NGHIÊM TRỌNG: "
            f"{len(backbone_missing)} keys `backbone.*` + {len(head_feature_missing)} keys "
            f"feature của `head.*` + {len(temporal_missing)} keys `temporal_encoder.*` THIẾU "
            f"+ {len(temporal_unexpected)} keys `temporal_encoder.*` THỪA "
            f"→ trọng số ĐẶC TRƯNG không khớp checkpoint.")
        for k in (backbone_missing + head_feature_missing + temporal_missing
                  + temporal_unexpected)[:5]:
            log(f"      - {k}")
        log(f"      → Trọng số ĐẶC TRƯNG đang là init/pretrain, KHÔNG phải trọng số đã train.")
        log(f"      → Kết quả eval/infer sẽ nhiễu (visual feature không khớp temporal/head).")
        if temporal_missing or temporal_unexpected:
            log(f"      → RIÊNG `temporal_encoder.*`: kiến trúc dựng model KHÁC lúc train.")
            log(f"        Kiểm tra `train.temporal_type` và `train.temporal_pool` trong config "
                f"có khớp checkpoint không.")
        log(f"      → Kiểm tra: checkpoint có chứa các key này không? Có lệch tên/shape không?")

    return {
        'tag': tag,
        'path': str(checkpoint_path),
        'n_model_keys': n_model,
        'n_loaded_keys': n_loaded,
        'missing': list(missing),
        'unexpected': list(unexpected),
        'skipped_shape': [s[0] for s in skipped_shape],
        'backbone_missing': backbone_missing,
        'head_feature_missing': head_feature_missing,
        'temporal_missing': temporal_missing,
        'temporal_unexpected': temporal_unexpected,
        'classifier_missing': classifier_missing,
        'classifier_shape_mismatch': [s[0] for s in classifier_mismatch],
        'temporal_impl_checkpoint': ck_impl if ck_x is not None else None,
        'temporal_impl_active': active_impl,
        'temporal_arch_ok': temporal_arch_ok,
    }
