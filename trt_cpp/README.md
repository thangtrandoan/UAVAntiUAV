# trt_cpp — chạy ReID bằng TensorRT C++

Chương trình C++ nối hai engine `*_backbone.engine` + `*_temporal.engine` thành một
đường trích embedding, thay cho `infer.py`/`evaluate_reid.py` khi cần tốc độ.

Không phụ thuộc Python, không dùng torch. Chỉ cần TensorRT, CUDA runtime và OpenCV
(OpenCV chỉ dùng để đọc file ảnh — lõi không cần).

---

## 1. Hợp đồng I/O của hai engine

Lấy trực tiếp từ `export_onnx.py` và kiểm lại trên file ONNX thật (`v1_*.onnx`,
opset 17, batch động):

| Engine | Tensor | Shape | dtype |
|---|---|---|---|
| backbone | `input_img` | `(B, 3, 224, 224)` | float32 |
| backbone | `cnn_feat` | `(B, 2560)` | float32 |
| temporal | `seq_feats` | `(B, N, 2560)` | float32 |
| temporal | `reid_embed` | `(B, 3072)` | float32, **đã L2-chuẩn-hoá** |

`2560 = 2048 (global) + 512 (fs)` của ResNet50-IBN; `3072 = 2560 + 512` của head.
Con số 2560/3072 được chương trình **đọc từ engine**, không hardcode — đổi
backbone (ví dụ DINOv3 với 960) vẫn chạy, miễn hai engine khớp nhau.

**Điểm mấu chốt:** `cnn_feat` là `(B·N, 2560)` và `seq_feats` là `(1, N, 2560)` —
**cùng thứ tự bộ nhớ, cùng số byte**. Nên chương trình bind **cùng một con trỏ
device** cho hai tensor: kết quả backbone đi thẳng vào temporal, không copy.

---

## 2. Build

Chạy **trên thiết bị đích** để link đúng TensorRT của JetPack.

```bash
cd trt_cpp
./build.sh                       # Orin = sm_87
CUDA_ARCH=86 ./build.sh          # Xavier/NX
```

Yêu cầu: TensorRT >= 8.5 (dùng API V3: `getNbIOTensors` / `enqueueV3`), CMake >= 3.20,
OpenCV 4.

`build.sh` chỉ là lớp mỏng quanh CMake; có thể gọi CMake trực tiếp nếu cần.

---

## 3. Dùng

### Chế độ bộ đệm vòng (online)

`--stream DIR` chạy đường **online** thay vì theo cụm. Mỗi bước chỉ đẩy **một**
frame qua backbone (batch 1) rồi ghi vector `[1, 2560]` vào bộ đệm vòng N slot;
temporal đọc cửa sổ `[1, N, 2560]` gồm N frame gần nhất. Khớp `SlidingWindowBuffer`
của `infer.py:58-105`, nên có embedding mới **ngay mỗi frame** thay vì chờ đủ N.

```bash
./build/uav_reid_trt $ENG --stream <dir> --bench 300
./build/uav_reid_trt $ENG --stream <dir> --stride 2 --dump emb.bin
```

Bộ đệm cấp `2*N` slot trên device: frame thứ p ghi vào slot `p%N` và `p%N + N`,
nên cửa sổ N frame gần nhất **luôn liền mạch** tại `[​(p-N+1)%N, +N)` — temporal đọc
thẳng. `infer.py` phải `torch.stack` lại mỗi lần; bản C++ không tốn lần nào.

| | kClip (mặc định) | kStream (`--stream`) |
|---|---|---|
| Backbone batch | `clips*N` | 1 |
| Có embedding mới sau | N frame | **mỗi frame** |
| Thông lượng / frame | tốt hơn (GPU bão hoà) | thấp hơn |

Đánh đổi này là thật, không phải chi tiết nhỏ: đo trong `tensorrt_result.txt`
(convnext) cho batch 4 = 10.24 ms và batch 12 = 7.38 ms, tức **2.56 ms/frame so với
0.62 ms/frame**. Nên **kClip cho offline/benchmark, kStream cho tracking trực tiếp**
— đừng dùng kStream để đo thông lượng tối đa.

`--stride K` tương ứng `SlidingWindowBuffer.should_extract()` (infer.py mặc định 2).

> Lưu ý: `infer.py:129-140` dùng `visual_plain` = mean thường cho head, còn
> weighted-mean theo sharpness chỉ dùng cho coarse score. Engine temporal tính
> `seq_feats.mean(dim=1)` đúng như `visual_plain` nên phần fused khớp; coarse score
> (nếu cần) phải tính **ngoài** engine vì nó cần điểm sharpness từng frame.

```bash
ENG="--backbone v1_backbone.engine --temporal v1_temporal.engine --num-frames 12"

# In cấu hình engine (không chạy suy luận)
./build/uav_reid_trt $ENG --info

# Đo tốc độ: 5 vòng khởi động, 200 vòng đo
./build/uav_reid_trt $ENG --bench 200 --warmup 5

# Embedding của một clip trong thư mục
./build/uav_reid_trt $ENG --clip processed/test/UAV-Anti-UAV_Test_000169_event_0/before

# Cosine giữa hai clip
./build/uav_reid_trt $ENG --match <dirA> <dirB>

# Hàng loạt: mỗi dòng "tên<TAB>path0,path1,...", ghi embedding float32 thô
./build/uav_reid_trt $ENG --manifest list.txt --dump emb.bin
```

Tuỳ chọn đáng chú ý:

| Cờ | Ý nghĩa |
|---|---|
| `--max-clips K` | xử lý K clip trong một lần gọi (batch backbone = K·N) |
| `--take-last` | lấy N frame **cuối** thay vì N frame đầu (giống `take_last=True` của `EvalDataset`) |
| `--order bgr\|rgb` | thứ tự kênh nguồn; `cv::imread` cho BGR nên mặc định `bgr` |
| `--dump FILE` | ghi float32 thô: `numpy.fromfile(f, np.float32).reshape(n, 3072)` |

`--max-clips` phải nằm trong optimization profile của engine backbone
(`K·N <= maxBatchSize`). Nếu vượt, chương trình in rõ min/max của từng profile rồi dừng.

---

## 4. Đã tối ưu những gì

| # | Việc | Vì sao có lợi |
|---|---|---|
| 1 | **Không còn Python** | Bỏ GIL, bỏ tạo tensor, bỏ `.to(device)` cho từng frame |
| 2 | **Tiền xử lý trên GPU** | Thay PIL/cv2 chạy CPU; một kernel cho cả batch N frame |
| 3 | **Cắt 224×224 ngay trong lúc truyền** | `cudaMemcpy2DAsync` chỉ gửi 150 528 B/frame thay vì 196 608 B |
| 4 | **Zero-copy giữa hai engine** | Bỏ một lần copy device-to-device toàn bộ `cnn_feat` |
| 5 | **Shape cố định, đặt một lần** | Không gọi `setInputShape` mỗi vòng ⇒ không suy lại shape |
| 6 | **Không cấp phát trong vòng nóng** | Arena một lần/engine, staging một lần, bảng trọng số cache theo kích thước nguồn |
| 7 | **Bảng trọng số resize nạp một lần** | Trước đây `cudaMalloc`/`cudaFree` mỗi lần gọi kèm một lần đồng bộ |
| 8 | **`measure=false` mặc định** | Chế độ chạy thật chỉ đồng bộ **một** lần ở cuối, không 3 lần mỗi vòng |

**Chưa làm** (cố ý, để đơn giản và đúng đắn trước):

- **CUDA Graph**: sẽ cắt được overhead launch, nhưng capture với shape động dễ vỡ.
  Đây là việc đáng làm tiếp nếu latency launch là đáng kể.
- **Đa stream / double-buffer**: backbone chiếm gần hết thời gian và đã bão hoà GPU,
  nên chồng lấn tiền xử lý với suy luận gần như không lợi.
- **Giải mã JPEG bằng nvJPEG**: hiện `cv::imread` giải mã trên CPU. Với camera thật
  nên đưa frame BGR thẳng vào, hoặc dùng nvJPEG/GStreamer.
- **Đo tốc độ**: máy build hiện tại **không truy cập được GPU** nên **chưa có số đo**.
  Chạy `--bench` trên thiết bị để lấy số thật. Đừng dùng con số của `trtexec` trong
  `tensorrt_result.txt` làm mốc cho toàn pipeline — nó chỉ đo riêng engine.

---

## 5. Độ chính xác tiền xử lý (đã đo)

Tham chiếu là `torchvision.transforms` đúng như `evaluate_reid.py:283-288`. Phép đo
so bản port Python trung thực của kernel với torchvision trên ảnh thật:

| Nguồn | max &#124;diff&#124; | mean &#124;diff&#124; | Ghi chú |
|---|---|---|---|
| 256×256 (`processed/`) | 2.53e-07 | 5.22e-08 | **bit-exact** (chỉ là epsilon float32) |
| 100×80 (phóng to) | 2.53e-07 | 6.26e-08 | **bit-exact** |
| 640×480 | 1.75e-02 | 1.05e-04 | lệch tối đa **1 mức xám** |
| 852×480 (video gốc) | 1.75e-02 | 8.21e-05 | lệch tối đa **1 mức xám** |
| 1920×1080 | 1.75e-02 | 5.02e-06 | lệch tối đa **1 mức xám** |

Vì sao vậy — và vì sao đây là mức tốt nhất làm được mà không viết lại số nguyên
dấu phẩy tĩnh của Pillow:

- `Resize((256,256))` của Pillow **antialias khi thu nhỏ**: nhân rộng nhân lọc tam
  giác theo `filterscale = max(1, in/256)`. Nội suy 2 điểm thường lệch tới **1.39**
  đơn vị đã chuẩn hoá — đủ làm lệch embedding. Bản này dựng lại đúng công thức
  `precompute_coeffs` của Pillow, đã kiểm bằng ảnh dò 1 chiều: hệ số khớp chính xác
  (`0.12549` / `0.37647`).
- Pillow ghi **uint8** giữa hai lượt (ngang rồi dọc) và tính bằng số nguyên
  `PRECISION_BITS = 22`. Chỗ làm tròn của nó khác `round()` ở đúng điểm hoà, nên
  lệch tối đa 1 mức xám trên một số ít pixel. Muốn khớp tuyệt đối phải tái tạo
  nguyên văn số học dấu phẩy tĩnh — không đáng, vì 1 mức xám tương đương nhiễu nén ảnh.

**Khuyến nghị:** nếu muốn **bit-exact tuyệt đối**, đưa vào ảnh đã đúng 256×256 như
`data_pipeline.py` sinh ra (`--crop-size 256`). Khi đó đường nhanh chạy và không có
nội suy nào cả.

### Kiểm chứng với Python

```bash
# C++ ghi embedding
./build/uav_reid_trt --backbone v1_backbone.engine --temporal v1_temporal.engine \
    --num-frames 12 --clip <dir> --dump cpp.bin

# Python cho cùng clip, so cosine
python - <<'PY'
import numpy as np, torch
a=np.fromfile('cpp.bin',np.float32)          # đã L2-chuẩn-hoá
# b = embedding lấy từ evaluate_reid.py cho đúng clip đó
b=np.asarray(torch.load('py_emb.pt')).ravel()
print('cos =', float(a@b/np.linalg.norm(a)/np.linalg.norm(b)))
PY
```

`cos > 0.999` là đạt. Lệch nhiều hơn thì kiểm lại `--order` và việc ảnh có đúng
`processed/` (đã crop bbox, 256×256) hay không.

---

## 6. Cấu trúc

```
trt_cpp/
├── CMakeLists.txt
├── build.sh
├── README.md
├── uav_reid_engine.py    cầu nối ctypes cho Python
├── infer_trt.py          chạy infer.py bằng engine (video vào, video ra)
├── compare_python.py     đối chiếu embedding C++ vs PyTorch
├── example_swap.py       ví dụ tối giản video -> video
├── include/uav_reid/
│   ├── cuda_check.hpp    macro báo lỗi CUDA/TRT, dừng kèm vị trí
│   ├── engine.hpp        bọc TensorRT: nạp, I/O, profile, arena, enqueueV3
│   ├── preprocess.hpp    khai báo kernel + bảng trọng số kiểu Pillow
│   └── pipeline.hpp      nối hai engine, zero-copy, shape cố định
└── src/
    ├── engine.cpp
    ├── preprocess.cu     kernel cắt/chuẩn hoá + hai lượt nội suy Pillow
    ├── pipeline.cpp
    ├── capi.cpp          C ABI cho Python (libuav_reid_engine.so)
    └── main.cpp          CLI: --info --bench --clip --match --manifest

`build.sh` tạo 3 đích: `uav_core` (static), `uav_reid_trt` (CLI) và
`libuav_reid_engine.so` (cho Python).
```

## 7. Ghi chú vận hành

- Engine build cho GPU/phiên bản TensorRT **khác** sẽ `deserialize` thất bại; thông
  báo lỗi in ra đường dẫn plan để dễ truy.
- Engine động cần optimization profile chứa shape yêu cầu. Chương trình tự chọn
  profile phù hợp và in min/max của mọi profile khi không có cái nào khớp.
- `--num-frames` phải đúng bằng lúc export temporal engine (mặc định pipeline là 12).
- **`--stream` cần profile của engine backbone nhận batch 1** (`minBatchSize = 1`).
  Nếu engine chỉ build cho batch 12/16 cố định thì kStream sẽ báo lỗi ngay khi
  `allocate()`. Export lại với `minShapes=input_img:1x3x224x224` là đủ; kClip và
  kStream dùng chung một engine được miễn profile phủ cả hai đầu.
- `--stream` cần `reid_embed` kiểu FLOAT và `cnn_feat` cùng kiểu với `seq_feats`
  (chương trình kiểm và báo rõ nếu không).
- Đầu ra **đã L2-chuẩn-hoá** bên trong engine, nên cosine = tích vô hướng.


## 8. Gọi từ Python — giữ `infer.py`, thay đúng 2 hàm

Bản thân CLI **không** có đầu ra video: nó chỉ cho embedding. Muốn có video, giữ
`infer.py` (đọc video, bbox GT, Memory Bank, state machine, vẽ, ghi mp4) và chỉ
thay chỗ gọi model. Như vậy chỉ có **một** nguồn sự thật cho logic tracking.

Không dùng pybind11 vì máy đích không có, và torch bản này không kèm header của
nó. Thay bằng **C ABI + ctypes** — cùng kiến trúc, không thêm phụ thuộc.

```python
from uav_reid_engine import ReIDEngine

eng = ReIDEngine('v1_backbone.engine', 'v1_temporal.engine', num_frames=16)
eng.visual_dim, eng.embed_dim, eng.num_frames   # 2560, 3072, 16

feat = eng.backbone(crop_bgr)          # (2560,)  mỗi frame
emb  = eng.temporal(seq)               # (3072,)  seq = (N, 2560)
```

### Hai chỗ cần thay trong `infer.py`

**1. `extract_cnn_feature`** — bỏ `transform()` và `model.backbone()`:

```python
# trước
tensor_frame = transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
feat_2560 = extract_cnn_feature(self.model, tensor_frame)

# sau — không cần transform, kernel lo resize 256 + crop 224 + chuẩn hoá + đổi RGB
feat_2560 = torch.from_numpy(eng.backbone(crop)).unsqueeze(0)
```

Điểm này cũng bỏ luôn phép `transform` trên CPU (PIL resize) — nằm trong
`metrics_prep_times`, giờ bằng 0.

**2. `compute_fused_vector`** — bỏ `temporal_encoder` + `head`:

```python
# trước
temporal_token, _ = model.temporal_encoder(seq_feats)
bn_feat = model.head(visual_plain, temporal_token)
fused_feat = F.normalize(bn_feat, p=2, dim=1)

# sau
fused_feat = torch.from_numpy(eng.temporal(seq_feats.squeeze(0).cpu().numpy()))
visual_plain = seq_feats.mean(dim=1)   # giữ nguyên, engine cũng dùng plain mean
```

### Giới hạn cần biết

Engine temporal **chỉ xuất `reid_embed`**. `FusedBundle` còn cần `temporal_token`
và `raw_feat` (dùng cho `fine_space: pre_bn` và để đo cosine trước/sau BatchNorm).
Nếu bạn chỉ dùng `fine_space: fused` thì để chúng `None` và bỏ khỏi Memory Bank là
đủ. Nếu cần `pre_bn`, phải export lại temporal với 3 đầu ra:

```python
class TemporalModel3(torch.nn.Module):
    def forward(self, seq_feats):
        visual_feat = seq_feats.mean(dim=1)
        temporal_token, _ = self.temporal_encoder(seq_feats)
        bn_feat = self.head(visual_feat, temporal_token)
        norm = lambda t: torch.nn.functional.normalize(t, p=2, dim=1)
        return (norm(bn_feat), temporal_token,
                norm(torch.cat([visual_feat, temporal_token], dim=-1)))

torch.onnx.export(TemporalModel3(...), dummy_seq, 'v1_temporal3.onnx',
                  output_names=['reid_embed', 'temporal_token', 'raw_feat'],
                  dynamic_axes={'seq_feats': {0: 'batch_size', 1: 'num_frames'},
                                'reid_embed': {0: 'batch_size'}}, ...)
```

### Hai API, đừng trộn

| Đường | Dùng khi |
|---|---|
| `backbone()` + `temporal(seq)` | cửa sổ do Python giữ — **dùng cho `infer.py`** |
| `push()` + `embed()` | bộ đệm vòng nằm trong C++ — dùng cho script độc lập |

Trộn hai đường là sai: `embed()` đọc bộ đệm C++, mà `backbone()` không ghi vào đó.

### Lỗi sẽ dừng tiến trình

Lõi C++ gọi `std::exit()` khi **cấu hình** sai (engine không khớp shape, sai N,
thiếu tensor). Đó là lỗi lúc mở engine, không phải lỗi dữ liệu, nên nó in
`file:dòng` rồi dừng chứ không ném exception — đừng bọc `try/except` mà trông đợi
bắt được.

### Video demo

```bash
cd trt_cpp
python example_swap.py --backbone v1_backbone.engine --temporal v1_temporal.engine \
    --num-frames 16 --video /duong/dan/video.mp4 --out demo.mp4
```

Đọc mp4, cắt theo `--bbox` (bỏ trống thì lấy cả khung), vẽ FPS/cosine, ghi mp4.
Đây là khuôn mẫu API, **không** chép lại logic SOFT/HARD LOCK.


## 9. Chạy đúng như `infer.py` (video vào, video ra)

`infer_trt.py` chạy nguyên pipeline của `infer.py` nhưng thay phần model bằng
engine. **Không sửa `infer.py`** — chỉ thay 4 điểm nối:

| Điểm nối | Thay bằng |
|---|---|
| `transform` (tham số của `process_frame`) | giữ ảnh gốc, bỏ resize/normalize PIL |
| `extract_cnn_feature` | `engine.backbone()` |
| `compute_fused_vector` | `engine.temporal()` |
| `TwoTierMemoryBank._max_sim` | chịu được query `None` |

Phần còn lại (đọc mp4, bbox GT, Memory Bank, SOFT/HARD LOCK, vẽ, `VideoWriter`)
vẫn là code của `infer.py`, nên chỉ có một nguồn sự thật.

```bash
cd trt_cpp
../uav_env/bin/python infer_trt.py \
    --engine-backbone ../v1_backbone.engine \
    --engine-temporal ../v1_temporal.engine \
    --config ../configs/config_jetson.yaml \
    --seq-dir ../data/UAV-Anti-UAV/Test/UAV-Anti-UAV_Test_000169 \
    --checkpoint ../best_model_conv.pth \
    --out-dir ../infer_output/trt_000169
```

`--seq-dir` là thư mục sequence, chứa `<tên_seq>.mp4`, `groundtruth_rect.txt`,
`absent.txt` (xem `infer.py:698-700`). Mọi cờ khác chuyển nguyên cho `infer.py`.

### BẮT BUỘC: N của engine phải bằng `num_frames` của infer

`config_jetson.yaml` dùng `num_frames = 12`, nhưng lệnh `trtexec` ghim `N=16`.
Đệm 12 lên 16 bằng cách lặp frame cuối **không** tương đương PyTorch chạy N=12:
`seq.mean(dim=1)` lấy trung bình 16 giá trị (có 4 bản trùng) và Mamba quét 16
bước thay vì 12. Cosine sẽ lệch, ngưỡng `0.6667` đã calibrate mất hiệu lực.

`infer_trt.py` **dừng lại** nếu phát hiện lệch (trừ khi thêm `--allow-pad`).
Build lại engine cho khớp:

```bash
trtexec --onnx=v1_temporal.onnx --saveEngine=v1_temporal.engine --fp16 \
  --minShapes=seq_feats:1x12x2560 --optShapes=seq_feats:1x12x2560 \
  --maxShapes=seq_feats:12x12x2560
```

Muốn N linh hoạt (1..16) thì mở dải, ONNX đã khai báo axis 1 là dynamic
(`export_onnx.py:64`), không phải export lại:

```bash
--minShapes=seq_feats:1x1x2560 --optShapes=seq_feats:1x12x2560 \
--maxShapes=seq_feats:12x16x2560
```

### Giới hạn còn lại

Engine temporal chỉ xuất `reid_embed`, nên `temporal_token` và `raw_feat` là
`None`. Hệ quả:

- `fine_space` **phải** là `fused` (mặc định). `pre_bn` sẽ cho điểm 0 và gate
  HARD LOCK không bao giờ mở.
- Với `debug_sim: true`, hai cột temporal/raw in ra `0.000` — đúng nghĩa "không
  có số liệu", không phải điểm thật bằng 0.

Muốn có đủ, export lại temporal với thêm đầu ra `temporal_token` (mục 8).
`raw_feat` không cần xuất: nó bằng `L2(cat(visual_plain, temporal_token))` và
`visual_plain` đã có sẵn ở phía Python.
