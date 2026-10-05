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

## Ghi chú

Cả hai script chạy được từ bất kỳ thư mục nào (đường dẫn repo tự suy ra từ vị trí file).
Cần `uav_env/bin/python` và thư mục `processed/train/` có ảnh thật.

`_ec/` là thư mục cũ, đã nằm trong `.gitignore`; các script thí nghiệm một lần ở đó đã
được dọn, chỉ còn vài file cũ được git theo dõi từ trước.
