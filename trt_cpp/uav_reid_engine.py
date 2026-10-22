"""Cầu nối Python -> engine TensorRT C++ (qua ctypes).

Mục đích: thay đúng hai lời gọi PyTorch trong `infer.py` bằng engine TensorRT,
giữ nguyên toàn bộ phần còn lại (đọc video, bbox GT, Memory Bank, state machine
SOFT/HARD LOCK, vẽ, ghi mp4). Nhờ vậy chỉ có MỘT nguồn sự thật cho logic tracking.

Vì sao ctypes chứ không pybind11: máy đích không có pybind11, torch bản này cũng
không kèm header; ctypes là thư viện chuẩn, không thêm phụ thuộc và không phải
khớp ABI Python.

    from uav_reid_engine import ReIDEngine

    eng = ReIDEngine('v1_backbone.engine', 'v1_temporal.engine', num_frames=16)

    feat = eng.backbone(crop_bgr)          # (2560,) float32, mỗi frame
    emb  = eng.temporal(seq)               # (3072,) float32, seq = (N, 2560)

    # hoặc dùng bộ đệm vòng:
    feat = eng.push(crop_bgr)              # đẩy 1 frame, trả feature frame đó
    if eng.ready():
        emb = eng.embed()                  # cửa sổ N frame gần nhất

LƯU Ý: lõi C++ gọi `std::exit()` khi cấu hình sai (engine không khớp shape, sai
N, thiếu tensor). Đó là lỗi lúc MỞ engine, không phải lỗi dữ liệu — nó sẽ dừng
tiến trình kèm thông báo rõ ràng thay vì ném exception.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

_DEFAULT_LIB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "build", "libuav_reid_engine.so"
)


def _load(path: str) -> ctypes.CDLL:
    lib_path = os.environ.get("UAV_REID_LIB", path)
    if not os.path.exists(lib_path):
        raise OSError(
            f"không thấy {lib_path}. Chạy ./build.sh trong trt_cpp/ trước, "
            "hoặc đặt biến môi trường UAV_REID_LIB."
        )
    lib = ctypes.CDLL(lib_path)
    lib.uav_create.restype = ctypes.c_void_p
    lib.uav_create.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.uav_destroy.restype = None
    lib.uav_destroy.argtypes = [ctypes.c_void_p]
    for fn in ("uav_visual_dim", "uav_embed_dim", "uav_num_frames",
               "uav_temporal_n", "uav_ready"):
        getattr(lib, fn).restype = ctypes.c_int
        getattr(lib, fn).argtypes = [ctypes.c_void_p]
    lib.uav_backbone.restype = None
    lib.uav_backbone.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.uav_temporal.restype = None
    lib.uav_temporal.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.uav_push.restype = ctypes.c_int
    lib.uav_push.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.uav_embed.restype = None
    lib.uav_embed.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.uav_reset.restype = None
    lib.uav_reset.argtypes = [ctypes.c_void_p]
    return lib


class ReIDEngine:
    """Engine GASNet backbone + temporal trên TensorRT."""

    def __init__(
        self,
        backbone_plan: str,
        temporal_plan: str,
        num_frames: int = 16,
        device: int = 0,
        bgr: bool = True,
        lib_path: str = _DEFAULT_LIB,
    ) -> None:
        self._lib = _load(lib_path)
        self._handle = self._lib.uav_create(
            backbone_plan.encode(),
            temporal_plan.encode(),
            int(num_frames),
            int(device),
            1 if bgr else 0,
        )
        if not self._handle:
            raise RuntimeError("uav_create trả về null")
        self.visual_dim = int(self._lib.uav_visual_dim(self._handle))
        self.embed_dim = int(self._lib.uav_embed_dim(self._handle))
        self.num_frames = int(self._lib.uav_num_frames(self._handle))
        # N thật của engine temporal; khác num_frames nghĩa là engine build với N
        # khác config -> embedding lệch so với PyTorch.
        self.temporal_n = int(self._lib.uav_temporal_n(self._handle))

    # -- vòng đời --------------------------------------------------------
    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.uav_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "ReIDEngine":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"ReIDEngine(visual_dim={self.visual_dim}, embed_dim={self.embed_dim}, "
            f"num_frames={self.num_frames}, temporal_n={self.temporal_n})"
        )

    # -- một frame -------------------------------------------------------
    @staticmethod
    def _img(img: np.ndarray) -> np.ndarray:
        if img.dtype != np.uint8:
            raise TypeError(f"ảnh phải là uint8, nhận {img.dtype}")
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"ảnh phải là HxWx3, nhận {img.shape}")
        return np.ascontiguousarray(img)

    def backbone(self, img: np.ndarray) -> np.ndarray:
        """Ảnh HWC uint8 (BGR mặc định) -> vector đặc trưng (visual_dim,)."""
        a = self._img(img)
        out = np.empty(self.visual_dim, dtype=np.float32)
        self._lib.uav_backbone(
            self._handle,
            a.ctypes.data_as(ctypes.c_void_p),
            a.shape[1],
            a.shape[0],
            a.strides[0],
            out.ctypes.data_as(ctypes.c_void_p),
        )
        return out

    # -- một cửa sổ ------------------------------------------------------
    def temporal(self, seq: np.ndarray) -> np.ndarray:
        """(n, visual_dim) float32 với n <= num_frames -> (embed_dim,).

        n < num_frames được đệm bằng cách LẶP HÀNG CUỐI (bước thời gian = 0),
        đúng quy ước 'nhân bản frame cuối' của infer.py.
        """
        s = np.ascontiguousarray(seq, dtype=np.float32)
        if s.ndim != 2 or s.shape[1] != self.visual_dim:
            raise ValueError(
                f"seq phải là (n, {self.visual_dim}), nhận {s.shape}"
            )
        if not 1 <= s.shape[0] <= self.num_frames:
            raise ValueError(
                f"số frame {s.shape[0]} ngoài khoảng 1..{self.num_frames}"
            )
        out = np.empty(self.embed_dim, dtype=np.float32)
        self._lib.uav_temporal(
            self._handle,
            s.ctypes.data_as(ctypes.c_void_p),
            s.shape[0],
            out.ctypes.data_as(ctypes.c_void_p),
        )
        return out

    # -- bộ đệm vòng -----------------------------------------------------
    def push(self, img: np.ndarray) -> np.ndarray:
        """Đẩy một frame vào bộ đệm vòng, trả feature của frame đó."""
        a = self._img(img)
        out = np.empty(self.visual_dim, dtype=np.float32)
        self._lib.uav_push(
            self._handle,
            a.ctypes.data_as(ctypes.c_void_p),
            a.shape[1],
            a.shape[0],
            a.strides[0],
            out.ctypes.data_as(ctypes.c_void_p),
        )
        return out

    def embed(self) -> np.ndarray:
        """Chạy temporal trên N frame gần nhất. Gọi khi ready() == True."""
        out = np.empty(self.embed_dim, dtype=np.float32)
        self._lib.uav_embed(self._handle, out.ctypes.data_as(ctypes.c_void_p))
        return out

    def ready(self) -> bool:
        return bool(self._lib.uav_ready(self._handle))

    def reset(self) -> None:
        self._lib.uav_reset(self._handle)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("dùng: python uav_reid_engine.py v1_backbone.engine v1_temporal.engine")
        raise SystemExit(2)
    eng = ReIDEngine(sys.argv[1], sys.argv[2], num_frames=16)
    print(eng)
