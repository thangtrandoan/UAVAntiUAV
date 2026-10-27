# tools/

Công cụ kiểm tra cho nhánh E-ConvNeXt. **Không nằm trong đường chạy train/eval/infer** —
xoá cả thư mục này thì pipeline vẫn chạy y nguyên.

## `audit_transfer.py` — chốt hồi quy

Kiểm việc chuyển trọng số DINOv3 → E-ConvNeXt có đầy đủ và đúng không:

- **A**: đối chiếu **từng tensor** nguồn với đích, in độ lệch lớn nhất (phải là `0.00e+00`).
- **A3**: liệt kê tensor nguồn **bỏ có chủ định**, kèm lý do.
- **B**: liệt kê tensor đích **không đến từ pretrain** (module mới + running stats).
- **C**: cho cùng đầu vào chạy qua `stage3`/`stage4` của cả hai, so độ lệch. Gồm cả
  **đường thật** của E-ConvNeXt.

Script **tự fail** (`exit 1`) nếu còn tensor nào chưa được phân loại, hoặc nếu
`stage3`/`stage4` lệch khỏi DINOv3. Nhờ vậy nó bắt được hai loại lỗi đã từng xảy ra:

- quên copy một tensor (từng bị với `depthwise_conv.bias`)
- đổi chuẩn hoá trong block thừa hưởng (LayerNorm → BatchNorm làm lệch 5×)

```
python3 tools/audit_transfer.py
```

## `measure_sigma_b.py` — nguồn của hằng số `SIGMA_B`

Đo `σ_b` = độ lệch chuẩn đầu ra `pointwise_conv2` (nhánh residual) trên DINOv3 pretrain,
bằng 8 ảnh thật.

Đây là **nguồn của hằng số `SIGMA_B` trong `gasnet/train.py`**. Phải chạy lại và cập nhật
`SIGMA_B` nếu:

- đổi cách cắt kênh ở `stages[0..1]`
- đổi kiến trúc block
- đổi backbone pretrain

```
python3 tools/measure_sigma_b.py
```

Script in ra luôn dòng `SIGMA_B = (...)` để dán thẳng vào `gasnet/train.py`.

## `check_gasnet_econvnext.py` — GASNet đầy đủ với E-ConvNeXt

Xác nhận bốn điều: GASNet dựng được với cả `econvnext=True/False`, forward chạy,
tham số truyền qua `GASNet -> DINOv3ConvNeXtBackbone` đúng, và **RGA vẫn nhận đúng
`96@56x56`** nên không raise.

Trích class cần thiết từ `gasnet/train.py` bằng `ast` rồi `exec`, để không phải
import cả module (chạy được cả khi thiếu dataset).

```
uav_env/bin/python tools/check_gasnet_econvnext.py
```

In ra: params gốc `57.659 M` -> E-ConvNeXt `56.824 M` (-1.45%), và so shape đầu ra.

## `smoke_econvnext.py` — smoke test backbone

Cùng cách `ast`-extract, nhưng chỉ kiểm backbone E-ConvNeXt dựng được và forward
ra đúng shape. Nhẹ hơn `check_gasnet_econvnext.py`.

```
uav_env/bin/python tools/smoke_econvnext.py
```

## `measure_rga.py` — `rga_c` có thật sự chọn kênh không

Đo **cổng (gate)** của `rga_c` trên ảnh thật. Nếu cổng gần như hằng số theo kênh
thì `rga_c` không làm gì theo kênh, và câu hỏi "ESE có bị thừa không" trở nên có
nghĩa. Đây là số liệu để trả lời câu đó, không phải suy đoán.

```
uav_env/bin/python tools/measure_rga.py
```

## Ghi chú

Các script chạy được từ bất kỳ thư mục nào (đường dẫn repo tự suy ra từ vị trí file).
Cần `uav_env/bin/python`; `audit_transfer.py` và `measure_sigma_b.py` cần
`processed/train/` có ảnh thật.

Thư mục `_ec/` cũ đã bỏ (27/10): ba script ở trên được chuyển sang `tools/`,
`cspconvnext.py` đã xoá — đó là bản copy code **PaddlePaddle** của paper, không
import được trong repo PyTorch này (`import paddle`, `from .convnext import ...`).
Bản gốc nằm ở PaddleClas; bản trong repo vẫn lấy lại được từ git history.
