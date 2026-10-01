# Chạy E-ConvNeXt (phương án II) — từ GASNet tới ReID

Kiến trúc đã nằm trong code, chọn bằng cờ/key `econvnext` (opt-in). Thiếu cờ/key ⇒ DINOv3 gốc.
Thiết kế: `md/23thg9.md` §26–28.

---

## 0. Điểm mấu chốt: hai tầng phải cùng kiến trúc

```
Bước 1  gasnet/train.py   (VRU, phân loại)     -> gasnet_convnext_*.pth
Bước 2  train_reid.py     (UAV-Anti-UAV, ReID) -> best_model.pth
Bước 3  evaluate_reid.py + calibrate_threshold.py
```

`train_reid.py` nạp checkpoint của bước 1 vào `GASNet`. Nhưng `EConvNeXtBlock` dùng
`pointwise_conv1/2` là **Conv2d `[4C, C, 1, 1]`** còn DINOv3 dùng **Linear `[4C, C]`**, và
`layer_norm` vs `norm` khác tên. `model/reidnet.py` lọc key theo **shape khớp chính xác**, nên
checkpoint GASNet của DINOv3 nạp vào backbone E-ConvNeXt sẽ rụng gần hết và in ra hàng loạt
`⚠️ keys KHÔNG tìm thấy`.

⇒ **Muốn chạy E-ConvNeXt thì GASNet cũng phải train bằng `--econvnext`.**

**Tin tốt:** checkpoint GASNet baseline **đã có sẵn** — `config_colab.yaml` trỏ tới
`/content/drive/MyDrive/output/gasnet_convnext_v0.best.pth`. Nên **chỉ phải train GASNet cho
nhánh E-ConvNeXt**, baseline không cần chạy lại.

---

## 1. Dữ liệu cần có

**a. VRU** (cho bước 1) — tại `$DATA_ROOT` sao cho:

```
$DATA_ROOT/VRU/Pic/*.jpg
$DATA_ROOT/VRU/train_test_split/train_list.txt
$DATA_ROOT/VRU/train_test_split/test_list_1200.txt     Small
$DATA_ROOT/VRU/train_test_split/test_list_2400.txt     Medium
$DATA_ROOT/VRU/train_test_split/test_list_8000.txt     Big
```

Ba file test là **bắt buộc**: `--eval-every 10` tự bật `--run-eval` (`gasnet/train.py:1591`), mà
eval cần đủ 3 split (`evaluation.py:15-17`). Đổi lại, chính vì có eval nên mới sinh `*.best.pth`
để bước 2 dùng.

**b. UAV-Anti-UAV đã xử lý** (cho bước 2) — tại `paths.data_dir`:

```
query_train.json  gallery_train.json  query_test.json  gallery_test.json  train/  test/
```

**c. Trọng số DINOv3 pretrain** — tải tự động từ HuggingFace (cache lại sau lần đầu).

---

## 2. Bước 1 — Train GASNet

Recipe tái lập đã đóng thành script (trước đây nằm rải trong log, chính §11 đã ghi là còn thiếu):

```bash
./train_vru.sh econvnext      # -> $OUT_DIR/gasnet_convnext_e.pth và .best.pth
./train_vru.sh baseline       # -> $OUT_DIR/gasnet_convnext_v0.pth  (chỉ khi cần chạy lại)
```

`DATA_ROOT` mặc định `/content`, `OUT_DIR` mặc định `/content/drive/MyDrive/output`. Script kiểm
đủ dữ liệu trước khi chạy và cảnh báo nếu file đích đã tồn tại.

Recipe bên trong (120 epoch, bs 256, `--eval-every 10`, `--strong-aug`, `--pk-k 8`, `--use-gem`,
`--backbone dinov3_convnext`; hai variant chỉ khác `--econvnext` và đường dẫn ghi):

```bash
cd gasnet
python3 train.py --data-root /content --dataset vru \
  --epochs 120 --batch-size 256 --amp-dtype bf16 --grad-accum 1 \
  --num-workers 8 --prefetch-factor 4 --eval-every 10 \
  --save-path /content/drive/MyDrive/output/gasnet_convnext_e.pth \
  --strong-aug --backbone dinov3_convnext --pk-k 8 --use-gem \
  --econvnext
```

Đây là **120 epoch trên VRU** — chặng dài nhất của cả pipeline.

Kiểm sau khi xong: log phải có dòng `E-ConvNeXt backbone ready: 30 block copy nguyen, 6 block cat
bot kenh` (ba lần: train + mỗi lần eval dựng lại model).

---

## 3. Bước 2 — Trỏ config vào checkpoint mới rồi train ReID

Sửa `configs/config_econvnext.yaml`:

```yaml
paths:
  gasnet_weights: /content/drive/MyDrive/output/gasnet_convnext_e.best.pth   # <-- file E-ConvNeXt
  data_dir: ...
  gasnet_dir: ...
  checkpoint_dir: ...   # v4.0
  log_dir: ...          # v4.0
```

Rồi:

```bash
python3 train_reid.py --config configs/config_econvnext.yaml
```

Stage 1: 30 epoch (backbone đóng băng) → Stage 2: 30 epoch.

---

## 4. Pre-flight — chạy được ở bất cứ đâu, không cần GPU

Kiểm code trước khi tốn GPU. Lệnh này đã chạy thử và đạt:

```bash
cd /path/to/UAVAntiUAV
GASNET_PATH=$PWD/gasnet python3 - <<'EOF'
from model import UAVReIDNet
for tag, ec in (("DINOv3 goc (econvnext=False)", False),
                ("E-ConvNeXt II (econvnext=True)", True)):
    m = UAVReIDNet(num_identities=502, freeze_backbone=True,
                   backbone='dinov3_convnext', temporal_type='attention',
                   temporal_pool='attn', temporal_pe=True, econvnext=ec)
    n  = sum(p.numel() for p in m.parameters())
    bb = sum(p.numel() for p in m.backbone.parameters())
    print(f"  {tag:<34} tong {n/1e6:7.3f} M | backbone {bb/1e6:7.3f} M")
EOF
```

Phải ra:

```
  DINOv3 goc (econvnext=False)       tong  65.626 M | backbone  57.659 M
  E-ConvNeXt II (econvnext=True)     tong  64.803 M | backbone  56.836 M
```

| thấy gì | nghĩa là |
|---|---|
| `Sẽ sử dụng Dummy Network` | `GASNET_PATH` sai ⇒ model không thật, dừng lại |
| `0 block copy nguyen` | trọng số DINOv3 không nạp được |
| số params khác hẳn | code đã bị sửa khác |

---

## 5. Theo dõi bước 2

| | |
|---|---|
| log | `<paths.log_dir>/train_<YYYYmmdd_HHMMSS>.log` |
| checkpoint | `<paths.checkpoint_dir>/best_model.pth`, `last_model.pth` |
| validation | mỗi `val_freq: 5` epoch, rank-1 ở N = 8 / 12 / 16 |

Đầu Stage 2 phải thấy:

```
  Kien truc backbone: E-ConvNeXt (II)
  Stage 2 param groups: pretrained ... tensor, module mới 60 tensor (0.217 M = 0.45% backbone), random ... tensor
  Stage 2: đóng băng weight/bias + momentum=0.01 cho <N> BatchNorm2d
```

Ngay sau khi nạp `gasnet_weights`, phải **không** có `⚠️ keys KHÔNG tìm thấy`. Nếu có ⇒ checkpoint
GASNet không cùng kiến trúc (bước 1 thiếu `--econvnext`, hoặc config thiếu `econvnext: true`).

---

## 6. Bước 3 — Đo

```bash
python3 evaluate_reid.py       --config configs/config_econvnext.yaml
python3 calibrate_threshold.py --config configs/config_econvnext.yaml
```

`evaluate_reid.py` đọc `eval.model_path` — trỏ vào `best_model.pth` của run v4.0.
Phải calibrate lại threshold vì thang điểm của model mới khác.

---

## 7. Bảng A/B đầy đủ

| | GASNet | ReID config | `econvnext` |
|---|---|---|---|
| **Baseline** | `gasnet_convnext_v0.best.pth` (**đã có**) | `configs/config_baseline_v4.yaml` | không có key ⇒ `False` |
| **E-ConvNeXt** | `gasnet_convnext_e.best.pth` (**phải train**) | `configs/config_econvnext.yaml` | `true` |

Hai config này chỉ khác nhau **một key** `train.econvnext` và các đường dẫn ghi — siêu tham số
giống hệt, nên so trực tiếp được.

> **Đừng dùng `config_colab.yaml` để chạy baseline** — nó ghi vào thư mục `v3.9` và sẽ ghi đè
> checkpoint cũ.

---

## 8. Rủi ro đã biết

`CSPStage.conv3` của tác giả **luôn có GELU** (kể cả conv cuối stage). Nên `ga1`/`ga2` nhận đặc
trưng **không âm**, còn `ga3`/`ga4` vẫn nhận đặc trưng hậu-BN. Đây là hệ quả của việc bám code
gốc, không phải lựa chọn thiết kế. Nếu kết quả tệ hơn baseline rõ rệt thì đây là nghi phạm đầu
tiên — thử bỏ GELU ở `conv3`.

Số dự kiến của backbone: **7943.09 M MACs (−8.52 %)** và **48.630 M params (−1.66 %)** so với
DINOv3 gốc.
