#!/usr/bin/env bash
# ============================================================================
# Train GASNet trên VRU — recipe tái lập (md/23thg9.md §11)
# ============================================================================
# Dùng:
#   ./train_vru.sh baseline    -> DINOv3 gốc, lưu gasnet_convnext_v0.pth(.best)
#   ./train_vru.sh econvnext   -> E-ConvNeXt,  lưu gasnet_convnext_e.pth(.best)
#
# Biến môi trường:
#   DATA_ROOT   thư mục CHỨA VRU/        (mặc định /content)
#   OUT_DIR     thư mục ghi checkpoint   (mặc định /content/drive/MyDrive/output)
#
# Dữ liệu phải có:
#   $DATA_ROOT/VRU/Pic/*.jpg
#   $DATA_ROOT/VRU/train_test_split/train_list.txt
#   $DATA_ROOT/VRU/train_test_split/test_list_1200.txt    (Small)
#   $DATA_ROOT/VRU/train_test_split/test_list_2400.txt    (Medium)
#   $DATA_ROOT/VRU/train_test_split/test_list_8000.txt    (Big)
#
# Ba file test là BẮT BUỘC: `--eval-every 10` tự bật `--run-eval`
# (gasnet/train.py:1591), mà eval cần đủ 3 split (evaluation.py:15-17).
# Đổi lại, chính vì có eval nên mới sinh ra `*.best.pth` để train_reid dùng.
#
# Lưu ý: hai variant ghi ra hai file KHÁC NHAU nên không ghi đè lẫn nhau.
# ============================================================================
set -euo pipefail

VARIANT="${1:-}"
case "$VARIANT" in
  baseline|econvnext) ;;
  *) echo "Dùng: $0 baseline|econvnext" >&2; exit 2 ;;
esac

DATA_ROOT="${DATA_ROOT:-/content}"
OUT_DIR="${OUT_DIR:-/content/drive/MyDrive/output}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$VARIANT" = "econvnext" ]; then
  SAVE="$OUT_DIR/gasnet_convnext_e.pth"
  EXTRA=(--econvnext)
else
  SAVE="$OUT_DIR/gasnet_convnext_v0.pth"
  EXTRA=()
fi

for f in "VRU/train_test_split/train_list.txt" \
         "VRU/train_test_split/test_list_1200.txt" \
         "VRU/train_test_split/test_list_2400.txt" \
         "VRU/train_test_split/test_list_8000.txt"; do
  if [ ! -f "$DATA_ROOT/$f" ]; then
    echo "THIẾU DỮ LIỆU: $DATA_ROOT/$f" >&2
    exit 1
  fi
done
if [ ! -d "$DATA_ROOT/VRU/Pic" ]; then
  echo "THIẾU DỮ LIỆU: $DATA_ROOT/VRU/Pic" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
if [ -f "$SAVE" ]; then
  echo "CẢNH BÁO: $SAVE đã tồn tại — sẽ bị ghi đè." >&2
fi

cd "$REPO/gasnet"

echo "=== GASNet [$VARIANT] ==="
echo "  data-root : $DATA_ROOT"
echo "  save-path : $SAVE"
echo "  log-path  : $OUT_DIR/gasnet_${VARIANT}.log"
if [ "$VARIANT" = "econvnext" ]; then
  echo "  kiến trúc : E-ConvNeXt (phương án II) — stem vb + CSPStage stages[0..1]"
else
  echo "  kiến trúc : DINOv3 gốc"
fi
echo

exec python3 train.py \
  --data-root "$DATA_ROOT" \
  --dataset vru \
  --epochs 120 --batch-size 256 --amp-dtype bf16 --grad-accum 1 \
  --num-workers 8 --prefetch-factor 4 --eval-every 10 \
  --save-path "$SAVE" \
  --log-path "$OUT_DIR/gasnet_${VARIANT}.log" \
  --strong-aug --backbone dinov3_convnext --pk-k 8 --use-gem \
  "${EXTRA[@]}"
