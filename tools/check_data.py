#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_data.py — kiểm tra trùng lặp dữ liệu UAV-Anti-UAV ở mức file txt.

Bối cảnh (đọc từ ./md và data_pipeline.py):
    data/UAV_anti_UAV/
    ├── Train/
    │   └── <sequence_name>/
    │       ├── <sequence_name>.mp4
    │       ├── groundtruth_rect.txt   # mỗi dòng: x,y,w,h
    │       ├── absent.txt             # mỗi dòng: 0/1 (1 = UAV vắng mặt)
    │       ├── attributes.txt         # 15 cờ thuộc tính
    │       └── language.txt           # mô tả ngôn ngữ
    └── Test/
        └── ... (cấu trúc tương tự)

Mục tiêu: tìm sequence/file bị trùng toàn bộ hoặc trùng một phần. Script chỉ
đọc txt, không cần model/GPU, không cần OpenCV, không hash nội dung video.

Các loại trùng được báo cáo:
  1. Trùng TOÀN BỘ sequence : groundtruth_rect.txt + absent.txt giống hệt.
  2. Trùng groundtruth      : nội dung gt giống hệt nhưng absent khác/thiếu.
  3. Trùng MỘT PHẦN          : một đoạn bbox liên tiếp dài >= --min-block giống
                              hệt giữa hai sequence (kể cả khác split), hoặc
                              file ngắn nằm trọn trong file dài, hoặc nội dung
                              trùng rời rạc với tỉ lệ dòng chung rất cao.
  4. Lặp bất thường trong 1 file (bbox giống nhau quá nhiều) — chỉ là cảnh báo.
  5. Lỗi cấu trúc: thiếu file, gt rỗng/toàn 0, absent không phải 0/1, lệch số
     dòng gt/absent, thiếu video...

"Giống hệt" ở đây là giống sau CHUẨN HOÁ: bỏ khoảng trắng thừa, dấu phẩy/tab,
chuẩn hoá số (1.0 -> 1, -0 -> 0), không phân biệt hoa/thường với token chữ.
Vì vậy file copy nhưng đổi format vẫn bị bắt. Báo cáo vẫn ghi rõ file nào
byte-identical.

Cách dùng:
    python tools/check_data.py
    python tools/check_data.py --data-dir ./data/UAV_anti_UAV \
        --output ./tools/check_data_results.txt
    python tools/check_data.py --min-block 20 --max-pairs 0

Kết quả mặc định: ./tools/check_data_results.txt (UTF-8).
Mã thoát: 0 = chạy xong; 1 = lỗi không phân tích được; 2 = có trùng lặp và
được yêu cầu --fail-on-duplicates.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import sys
import time
import traceback
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Hằng số / cấu hình mặc định
# ---------------------------------------------------------------------------

MASK64 = (1 << 64) - 1
MOD64 = 1 << 64
HASH_BASE = 0x9E3779B97F4A7C15  # số lẻ, dùng cho rolling hash 64-bit

# MinHash đơn giản (bottom-k + banding) để bắt cả trường hợp nội dung bị đảo
# thứ tự / trộn dòng, khi phương pháp shingle liên tiếp không tìm được ứng viên.
MINHASH_MAX_VALUES = 64
MINHASH_ROWS = 4

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_REL = Path("data") / "UAV_anti_UAV"
DEFAULT_OUTPUT_REL = Path("tools") / "check_data_results.txt"

# Thứ tự ưu tiên khi một sequence có nhiều file cùng vai trò.
ROLE_ALIASES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("groundtruth", ("groundtruth_rect.txt", "groundtruth.txt", "gt.txt")),
    ("absent", ("absent.txt", "absence.txt")),
    ("attributes", ("attributes.txt",)),
    ("language", ("language.txt",)),
)
ROLE_ORDER = ("groundtruth", "absent", "attributes", "language")

# Token số: số nguyên/thực thông thường, có dấu, có exponent.
_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_SEPARATOR_RE = re.compile(r"[,;\t\[\]\(\)]+")
_SPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Chuẩn hoá nội dung
# ---------------------------------------------------------------------------

def _canonical_number(token: str) -> Optional[str]:
    """Trả về dạng chuẩn của một token số, hoặc None nếu không phải số."""
    if not _NUMBER_RE.match(token):
        return None
    try:
        value = float(token)
    except (ValueError, OverflowError):
        return None
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if value == 0.0:               # gồm cả -0.0
        return "0"
    if value.is_integer() and abs(value) < 1e16:
        return str(int(value))
    return repr(value)


def canonicalize_line(raw_line: str) -> str:
    """
    Chuẩn hoá một dòng để so sánh nội dung, không so byte.

    - Bỏ BOM/NBSP, khoảng trắng đầu/cuối.
    - Dấu phẩy/chấm phẩy/tab/ngoặc -> khoảng trắng (gt có thể là "x,y,w,h"
      hoặc "x y w h", có file bọc trong ngoặc).
    - Mỗi token số được đưa về cùng một biểu diễn (1.0 -> 1).
    - Token chữ được casefold.
    """
    text = raw_line.replace("\ufeff", "").replace("\u00a0", " ").strip()
    if not text:
        return ""
    text = _SEPARATOR_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    if not text:
        return ""
    out: List[str] = []
    for token in text.split(" "):
        number = _canonical_number(token)
        out.append(number if number is not None else token.casefold())
    return " ".join(out)


def parse_bbox(canonical_line: str) -> Optional[Tuple[float, float, float, float]]:
    """Lấy 4 số đầu tiên của dòng groundtruth; None nếu dòng không hợp lệ."""
    parts = canonical_line.split(" ")
    if len(parts) < 4:
        return None
    try:
        values = [float(parts[i]) for i in range(4)]
    except (ValueError, OverflowError):
        return None
    if not all(math.isfinite(v) for v in values):
        return None
    return values[0], values[1], values[2], values[3]


def _hash8(data: bytes) -> int:
    """Hash 64-bit ổn định giữa các lần chạy (không phụ thuộc hash randomization)."""
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{int(num_bytes)} B"


# ---------------------------------------------------------------------------
# Cấu trúc dữ liệu
# ---------------------------------------------------------------------------

@dataclass
class TextFile:
    """Thông tin một file txt đã đọc + các chỉ số phục vụ so trùng."""

    path: Path
    rel: str                 # đường dẫn tương đối so với data root, dạng POSIX
    role: str
    split: str
    seq_name: str
    seq_rel: str

    size: int = 0
    raw_sha: str = ""
    norm_sha: str = ""

    n_data_lines: int = 0
    n_blank: int = 0
    n_informative: int = 0       # số dòng bbox có w>0, h>0
    n_dup_informative: int = 0   # số dòng bbox bị lặp lại trong chính file
    n_bad_bbox: int = 0
    n_nonpos_bbox: int = 0       # w<=0 hoặc h<=0 (không tính all-zero)
    n_zero_bbox: int = 0         # đúng 0,0,0,0
    n_nonbinary: int = 0         # absent.txt có dòng khác 0/1

    max_static_run: int = 0      # chuỗi bbox giống nhau liên tiếp dài nhất
    max_static_start: int = 0    # dòng bắt đầu (1-based)

    bad_bbox_example: str = ""
    nonbinary_example: str = ""
    error: str = ""

    # Chỉ giữ chuỗi bbox "informative" để so trùng một phần.
    inf_hashes: array = field(default_factory=lambda: array("Q"))
    inf_orig: array = field(default_factory=lambda: array("I"))  # dòng gốc 1-based

    @property
    def usable(self) -> bool:
        return not self.error and self.n_data_lines > 0

    @property
    def dup_ratio(self) -> float:
        return (self.n_dup_informative / self.n_informative) if self.n_informative else 0.0


@dataclass
class SequenceInfo:
    """Một thư mục sequence và các file txt/video đi kèm."""

    path: Path
    rel: str
    split: str
    name: str
    files: Dict[str, TextFile] = field(default_factory=dict)
    video: Optional[Path] = None
    video_size: int = 0
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Đọc dữ liệu
# ---------------------------------------------------------------------------

def read_annotation(
    path: Path,
    role: str,
    split: str,
    seq_name: str,
    seq_rel: str,
    data_root: Path,
) -> TextFile:
    """Đọc một file txt, chuẩn hoá và tính các chỉ số cần thiết."""
    rel = path.relative_to(data_root).as_posix()
    tf = TextFile(
        path=path,
        rel=rel,
        role=role,
        split=split,
        seq_name=seq_name,
        seq_rel=seq_rel,
    )

    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        tf.error = f"{type(exc).__name__}: {exc}"
        return tf

    tf.size = len(raw_bytes)
    tf.raw_sha = _sha256(raw_bytes)

    try:
        if raw_bytes[:2] in (b"\xff\xfe", b"\xfe\xff"):
            text = raw_bytes.decode("utf-16", errors="replace")
        else:
            text = raw_bytes.decode("utf-8-sig", errors="replace")
    except Exception as exc:  # rất hiếm, nhưng không được làm sập cả audit
        tf.error = f"decode: {type(exc).__name__}: {exc}"
        return tf

    norm_hasher = hashlib.sha256()
    seen_inf: set = set()
    prev_inf_hash: Optional[int] = None
    prev_inf_line: Optional[int] = None
    static_run = 0
    best_run = 0
    best_start = 0

    for line_no, raw_line in enumerate(text.splitlines(), 1):
        canonical = canonicalize_line(raw_line)
        if not canonical:
            tf.n_blank += 1
            continue

        tf.n_data_lines += 1
        norm_hasher.update(canonical.encode("utf-8"))
        norm_hasher.update(b"\n")

        if role == "groundtruth":
            bbox = parse_bbox(canonical)
            if bbox is None:
                tf.n_bad_bbox += 1
                if not tf.bad_bbox_example:
                    tf.bad_bbox_example = raw_line.strip()[:160]
                continue

            x, y, w, h = bbox
            if w <= 0.0 or h <= 0.0:
                if x == 0.0 and y == 0.0 and w == 0.0 and h == 0.0:
                    tf.n_zero_bbox += 1
                else:
                    tf.n_nonpos_bbox += 1
                continue

            # Hash chỉ 4 toạ độ đầu -> không bị ảnh hưởng nếu file có cột phụ.
            bbox_key = " ".join(canonical.split(" ")[:4])
            line_hash = _hash8(bbox_key.encode("utf-8"))

            tf.n_informative += 1
            tf.inf_hashes.append(line_hash)
            tf.inf_orig.append(line_no)

            if line_hash in seen_inf:
                tf.n_dup_informative += 1
            else:
                seen_inf.add(line_hash)

            if (
                line_hash == prev_inf_hash
                and prev_inf_line is not None
                and line_no == prev_inf_line + 1
            ):
                static_run += 1
            else:
                static_run = 1
            if static_run > best_run:
                best_run = static_run
                best_start = line_no - static_run + 1
            prev_inf_hash = line_hash
            prev_inf_line = line_no

        elif role == "absent":
            # Chỉ kiểm tra miền giá trị; nội dung đã được hash ở norm_sha.
            if canonical not in ("0", "1"):
                tf.n_nonbinary += 1
                if not tf.nonbinary_example:
                    tf.nonbinary_example = raw_line.strip()[:160]

    tf.norm_sha = norm_hasher.hexdigest()
    tf.max_static_run = best_run
    tf.max_static_start = best_start
    return tf


def _split_and_rel(seq_path: Path, root: Path) -> Tuple[str, str]:
    """Suy ra split (Train/Test/...) và đường dẫn tương đối của sequence."""
    try:
        rel_path = seq_path.relative_to(root)
    except ValueError:
        rel_path = Path(seq_path.name)

    if rel_path == Path("."):
        return "(root)", "."
    parts = rel_path.parts
    split = parts[0] if len(parts) >= 2 else "(root)"
    return split, rel_path.as_posix()


def discover_sequences(root: Path) -> Tuple[List[SequenceInfo], List[str]]:
    """
    Quét đệ quy tìm các thư mục sequence (có groundtruth_rect.txt hoặc absent.txt).

    Trả về (danh sách sequence, danh sách lỗi đọc file). Không dừng ở Train/Test
    để còn dùng được nếu dataset được đặt tên split khác.
    """
    sequences: List[SequenceInfo] = []
    global_errors: List[str] = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in sorted(dirnames)
            if not d.startswith(".") and d.casefold() not in {"__pycache__"}
        ]

        lower_names = {name.casefold(): name for name in filenames}
        roles: Dict[str, Path] = {}
        for role, aliases in ROLE_ALIASES:
            for alias in aliases:
                real_name = lower_names.get(alias.casefold())
                if real_name is not None:
                    roles[role] = Path(dirpath) / real_name
                    break

        if "groundtruth" not in roles and "absent" not in roles:
            continue

        seq_path = Path(dirpath)
        split, seq_rel = _split_and_rel(seq_path, root)
        seq_name = seq_path.name if seq_path != root else root.name
        seq = SequenceInfo(
            path=seq_path,
            rel=seq_rel,
            split=split,
            name=seq_name,
        )

        # Video: ưu tiên <tên_thư_mục>.mp4, nếu không có thì lấy mp4 duy nhất.
        video_names = [name for name in filenames if name.casefold().endswith(".mp4")]
        expected = f"{seq_name}.mp4".casefold()
        chosen_video: Optional[str] = None
        for name in video_names:
            if name.casefold() == expected:
                chosen_video = name
                break
        if chosen_video is None and len(video_names) == 1:
            chosen_video = video_names[0]
        if chosen_video is None and len(video_names) > 1:
            seq.notes.append(f"có {len(video_names)} file .mp4, không xác định được video chính")
        if chosen_video is not None:
            seq.video = seq_path / chosen_video
            try:
                seq.video_size = seq.video.stat().st_size
            except OSError:
                seq.video_size = 0

        for role in ROLE_ORDER:
            file_path = roles.get(role)
            if file_path is None:
                continue
            tf = read_annotation(
                file_path,
                role=role,
                split=split,
                seq_name=seq_name,
                seq_rel=seq_rel,
                data_root=root,
            )
            seq.files[role] = tf
            if tf.error:
                global_errors.append(f"{tf.rel}: {tf.error}")

        sequences.append(seq)

    sequences.sort(key=lambda s: s.rel.casefold())
    return sequences, global_errors


# ---------------------------------------------------------------------------
# Gom nhóm trùng toàn bộ
# ---------------------------------------------------------------------------

def group_by_norm(files: Sequence[TextFile]) -> List[List[TextFile]]:
    """Gom các file có nội dung chuẩn hoá giống nhau; chỉ trả nhóm >= 2 file."""
    groups: Dict[str, List[TextFile]] = defaultdict(list)
    for tf in files:
        if tf.usable and tf.norm_sha:
            groups[tf.norm_sha].append(tf)
    result = [sorted(group, key=lambda x: x.rel.casefold()) for group in groups.values() if len(group) >= 2]
    result.sort(key=lambda group: group[0].rel.casefold())
    return result


def sequence_fingerprint(seq: SequenceInfo) -> Optional[Tuple[str, str]]:
    """Fingerprint sequence = (gt chuẩn hoá, absent chuẩn hoá); None nếu thiếu."""
    gt = seq.files.get("groundtruth")
    absent = seq.files.get("absent")
    if gt is None or not gt.usable or gt.n_informative == 0:
        return None
    if absent is None or not absent.usable:
        return None
    return gt.norm_sha, absent.norm_sha


def absent_signature(seq: SequenceInfo) -> str:
    absent = seq.files.get("absent")
    if absent is None:
        return "<thiếu absent>"
    if absent.error:
        return "<lỗi đọc>"
    if not absent.usable:
        return "<absent rỗng>"
    return absent.norm_sha[:10]


# ---------------------------------------------------------------------------
# So trùng một phần: rolling hash + LCS liên tiếp
# ---------------------------------------------------------------------------

def shingle_hashes(seq: Sequence[int], k: int, sample: int) -> List[int]:
    """
    Hash các cửa sổ k phần tử liên tiếp.

    sample > 1 chỉ lấy các shingle có hash % sample == 0 để giảm RAM; đổi lại
    có thể bỏ sót đoạn trùng ngắn. Mặc định sample=1 (không bỏ).
    """
    n = len(seq)
    if k <= 0 or n < k:
        return []

    power = pow(HASH_BASE, k - 1, MOD64)
    h = 0
    for idx in range(k):
        h = (h * HASH_BASE + seq[idx]) & MASK64

    out: List[int] = []
    if sample <= 1 or h % sample == 0:
        out.append(h)

    for i in range(1, n - k + 1):
        h = ((h - seq[i - 1] * power) & MASK64)
        h = (h * HASH_BASE + seq[i + k - 1]) & MASK64
        if sample <= 1 or h % sample == 0:
            out.append(h)
    return out


def _find_common_run(
    length: int,
    a: Sequence[int],
    b: Sequence[int],
) -> Optional[Tuple[int, int]]:
    """
    Tìm một đoạn chung liên tiếp dài đúng `length` giữa a và b.

    Dùng rolling hash để lập bảng đoạn của b, quét a; khi hash trùng thì so
    trực tiếp để loại false positive do collision.
    """
    if length <= 0 or length > len(a) or length > len(b):
        return None

    power = pow(HASH_BASE, length - 1, MOD64)

    h = 0
    for value in b[:length]:
        h = (h * HASH_BASE + value) & MASK64
    positions: Dict[int, List[int]] = {}
    last_b = len(b) - length
    for j in range(last_b + 1):
        positions.setdefault(h, []).append(j)
        if j < last_b:
            h = ((h - b[j] * power) & MASK64)
            h = (h * HASH_BASE + b[j + length]) & MASK64

    h = 0
    for value in a[:length]:
        h = (h * HASH_BASE + value) & MASK64
    last_a = len(a) - length
    for i in range(last_a + 1):
        candidates = positions.get(h)
        if candidates:
            a_slice = a[i:i + length]
            for j in candidates:
                if a_slice == b[j:j + length]:
                    return i, j
        if i < last_a:
            h = ((h - a[i] * power) & MASK64)
            h = (h * HASH_BASE + a[i + length]) & MASK64
    return None


def longest_common_run(
    a: Sequence[int],
    b: Sequence[int],
    min_length: int,
) -> Tuple[int, int, int]:
    """
    Độ dài đoạn chung liên tiếp dài nhất (>= min_length).

    Trả về (độ_dài, vị_trí_bắt_đầu_trong_a, vị_trí_bắt_đầu_trong_b);
    (0, -1, -1) nếu không có.
    """
    upper = min(len(a), len(b))
    if min_length < 1:
        min_length = 1
    if min_length > upper:
        return 0, -1, -1

    best_length = 0
    best_a = -1
    best_b = -1
    low, high = min_length, upper
    while low <= high:
        mid = (low + high) // 2
        found = _find_common_run(mid, a, b)
        if found is not None:
            best_length = mid
            best_a, best_b = found
            low = mid + 1
        else:
            high = mid - 1
    return best_length, best_a, best_b


def _signature_candidate_pairs(
    reps: Sequence[TextFile],
    max_files_per_band: int,
) -> Tuple[Counter, Dict[str, int]]:
    """
    MinHash bottom-k + banding để tìm ứng viên có tập dòng chung lớn.

    Bổ sung cho shingle liên tiếp: nếu nội dung bị đảo thứ tự hoặc trộn dòng
    thì không còn đoạn liên tiếp dài, nhưng tập dòng vẫn gần như trùng. Chỉ dùng
    để SINH ỨNG VIÊN; việc kết luận vẫn do compare_groundtruth_pair kiểm tra.
    """
    band_index: Dict[Tuple[int, ...], List[int]] = {}
    for file_id, tf in enumerate(reps):
        unique = sorted(set(tf.inf_hashes))
        # Quá ít dòng khác nhau thì banding không có ý nghĩa.
        if len(unique) < 2 * MINHASH_ROWS:
            continue
        signature = unique[:MINHASH_MAX_VALUES]
        for band in range(len(signature) // MINHASH_ROWS):
            start = band * MINHASH_ROWS
            chunk = tuple(signature[start:start + MINHASH_ROWS])
            band_index.setdefault((band,) + chunk, []).append(file_id)

    counts: Counter = Counter()
    skipped_bands = 0
    max_band_files = 0
    for ids in band_index.values():
        count = len(ids)
        if count < 2:
            continue
        if max_files_per_band and count > max_files_per_band:
            skipped_bands += 1
            max_band_files = max(max_band_files, count)
            continue
        for left in range(count - 1):
            file_a = ids[left]
            for right in range(left + 1, count):
                counts[(file_a, ids[right])] += 1

    return counts, {
        "bands": len(band_index),
        "signature_pairs": len(counts),
        "skipped_bands": skipped_bands,
        "max_band_files": max_band_files,
    }


def build_candidate_pairs(
    reps: Sequence[TextFile],
    k: int,
    sample: int,
    max_files_per_shingle: int,
) -> Tuple[List[Tuple[Tuple[int, int], int]], Dict[str, int]]:
    """
    Dùng shingle để tìm các cặp file CÓ THỂ trùng một phần, tránh so O(N^2).

    Một shingle là k dòng bbox liên tiếp. Chỉ cần một shingle chung là đủ đưa
    vào kiểm tra chính xác; việc lọc độ dài đoạn do compare_groundtruth_pair
    đảm nhiệm. Cặp có tập dòng chung lớn nhưng không còn đoạn liên tiếp được
    bổ sung bởi MinHash banding.
    """
    index: Dict[int, List[int]] = {}
    for file_id, tf in enumerate(reps):
        shingles = shingle_hashes(tf.inf_hashes, k, sample)
        if not shingles:
            continue
        for shingle in set(shingles):
            index.setdefault(shingle, []).append(file_id)

    pair_counts: Counter = Counter()
    ubiquitous = 0
    max_ubiquitous_files = 0
    for ids in index.values():
        count = len(ids)
        if count < 2:
            continue
        if max_files_per_shingle and count > max_files_per_shingle:
            ubiquitous += 1
            max_ubiquitous_files = max(max_ubiquitous_files, count)
            continue
        for left in range(count - 1):
            file_a = ids[left]
            for right in range(left + 1, count):
                pair_counts[(file_a, ids[right])] += 1

    # Chỉ cần MỘT shingle chung là đủ để đưa vào kiểm tra chính xác. Việc lọc
    # theo độ dài đoạn do compare_groundtruth_pair đảm nhiệm; đặt ngưỡng cao
    # hơn sẽ bỏ sót các đoạn trùng có bbox lặp lại.
    min_shared = 1
    shingle_candidates = [(pair, count) for pair, count in pair_counts.items() if count >= min_shared]

    signature_counts, signature_stats = _signature_candidate_pairs(reps, max_files_per_shingle)

    merged: Dict[Tuple[int, int], int] = {pair: count for pair, count in shingle_candidates}
    for pair, count in signature_counts.items():
        merged[pair] = merged.get(pair, 0) + count

    candidates = sorted(merged.items(), key=lambda item: (-item[1], item[0]))
    stats = {
        "shingles": len(index),
        "raw_candidate_pairs": len(pair_counts),
        "shingle_candidates": len(shingle_candidates),
        "signature_pairs": signature_stats["signature_pairs"],
        "signature_bands": signature_stats["bands"],
        "signature_skipped_bands": signature_stats["skipped_bands"],
        "max_band_files": signature_stats["max_band_files"],
        "candidate_pairs": len(candidates),
        "min_shared": min_shared,
        "ubiquitous_shingles": ubiquitous,
        "max_ubiquitous_files": max_ubiquitous_files,
    }
    return candidates, stats


def compare_groundtruth_pair(
    a: TextFile,
    b: TextFile,
    min_block: int,
    high_overlap: float,
) -> Optional[Dict[str, object]]:
    """Kiểm tra chính xác một cặp groundtruth; None nếu không đáng báo cáo."""
    len_a = len(a.inf_hashes)
    len_b = len(b.inf_hashes)
    if len_a < min_block or len_b < min_block:
        return None

    run_length, pos_a, pos_b = longest_common_run(a.inf_hashes, b.inf_hashes, min_block)

    set_a = set(a.inf_hashes)
    set_b = set(b.inf_hashes)
    shared = len(set_a & set_b)
    smaller_unique = min(len(set_a), len(set_b))
    overlap = (shared / smaller_unique) if smaller_unique else 0.0

    if run_length < min_block:
        # Không có đoạn dài đủ, nhưng nội dung có thể bị đảo/trộn: chỉ báo nếu
        # có đủ nhiều dòng KHÁC NHAU cùng xuất hiện và tỉ lệ dòng chung rất cao.
        if shared < min_block or overlap + 1e-12 < high_overlap:
            return None
        run_length, pos_a, pos_b = longest_common_run(a.inf_hashes, b.inf_hashes, 1)
        if run_length <= 0:
            return None
        category = "reordered"
    else:
        min_side = min(len_a, len_b)
        ratio = run_length / min_side if min_side else 0.0
        if run_length == min_side:
            category = "contained"
        elif ratio >= 0.90:
            category = "near-full"
        elif ratio >= 0.50:
            category = "partial-large"
        else:
            category = "partial-block"

    min_side = min(len_a, len_b)
    return {
        "a": a,
        "b": b,
        "run_length": run_length,
        "pos_a": pos_a,
        "pos_b": pos_b,
        "len_a": len_a,
        "len_b": len_b,
        "run_ratio": (run_length / min_side) if min_side else 0.0,
        "overlap": overlap,
        "shared_lines": shared,
        "category": category,
        "cross_split": a.split != b.split,
    }


# ---------------------------------------------------------------------------
# In báo cáo
# ---------------------------------------------------------------------------

class Report:
    """Ghi đồng thời ra stdout và file kết quả UTF-8."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "w", encoding="utf-8", newline="\n")

    def line(self, text: str = "") -> None:
        # Ghi file trước để báo cáo không bị thiếu dòng nếu console lỗi encoding.
        self._fh.write(text + "\n")
        try:
            print(text)
        except UnicodeEncodeError:
            print(text.encode("ascii", "replace").decode("ascii"))

    def section(self, title: str) -> None:
        self.line()
        self.line("=" * 100)
        self.line(title)
        self.line("=" * 100)

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> "Report":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def _splits_of(items: Sequence[object]) -> List[str]:
    splits = {getattr(item, "split", "?") for item in items}
    return sorted(splits)


def _cross_split(items: Sequence[object]) -> bool:
    return len(_splits_of(items)) > 1


def _split_text(items: Sequence[object]) -> str:
    splits = _splits_of(items)
    return ", ".join(splits) if splits else "?"


def _print_group_members(
    report: Report,
    members: Sequence[str],
    limit: int,
    indent: str = "        ",
) -> None:
    shown = members if limit <= 0 else members[:limit]
    for name in shown:
        report.line(f"{indent}- {name}")
    if limit > 0 and len(members) > limit:
        report.line(f"{indent}... và {len(members) - limit} mục nữa")


# ---------------------------------------------------------------------------
# Phân tích chính
# ---------------------------------------------------------------------------

def run_analysis(
    report: Report,
    args: argparse.Namespace,
    data_dir: Path,
    sequences: Sequence[SequenceInfo],
    read_errors: Sequence[str],
    started_at: float,
) -> Dict[str, object]:
    seq_by_rel: Dict[str, SequenceInfo] = {seq.rel: seq for seq in sequences}

    # ---------------------------- 1. TỔNG QUAN ----------------------------
    files_by_role: Counter = Counter()
    lines_by_role: Counter = Counter()
    bytes_by_role: Counter = Counter()
    for seq in sequences:
        for role, tf in seq.files.items():
            files_by_role[role] += 1
            lines_by_role[role] += tf.n_data_lines
            bytes_by_role[role] += tf.size

    all_videos = [seq for seq in sequences if seq.video is not None]
    any_video = bool(all_videos)
    total_txt = sum(bytes_by_role.values())

    report.section("1. TỔNG QUAN DỮ LIỆU")
    report.line(f"Dataset root : {data_dir}")
    report.line(f"Thời điểm    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.line(f"Số sequence  : {_fmt_int(len(sequences))}")
    report.line(f"Tổng file txt: {_fmt_int(sum(files_by_role.values()))} ({human_size(total_txt)})")
    report.line(f"Tổng video   : {_fmt_int(len(all_videos))} ({human_size(sum(s.video_size for s in all_videos))})")
    report.line("")
    report.line("Quy ước:")
    report.line("  * 'Toàn bộ' = nội dung sau chuẩn hoá giống hệt (không chỉ hash byte).")
    report.line("  * 'Một phần' = đoạn bbox liên tiếp giống hệt, dài >= --min-block dòng.")
    report.line("  * Chỉ phân tích nội dung txt; KHÔNG hash nội dung video.")
    report.line("")

    report.line(f"{'Split':<16}{'#seq':>8}{'#groundtruth':>14}{'#absent':>10}{'#video':>9}")
    for split in sorted({seq.split for seq in sequences}):
        subset = [seq for seq in sequences if seq.split == split]
        gt_count = sum(1 for seq in subset if "groundtruth" in seq.files)
        ab_count = sum(1 for seq in subset if "absent" in seq.files)
        video_count = sum(1 for seq in subset if seq.video is not None)
        report.line(f"{split:<16}{len(subset):>8,}{gt_count:>14,}{ab_count:>10,}{video_count:>9,}")

    report.line("")
    report.line(f"{'Vai trò':<16}{'Số file':>9}{'Số dòng':>13}{'Ghi chú':>34}")
    for role in ROLE_ORDER:
        note = ""
        if role == "groundtruth":
            note = f"{_fmt_int(sum(tf.n_informative for s in sequences for tf in s.files.values() if tf.role == role))} dòng bbox hợp lệ"
        elif role == "absent":
            note = "0/1: 1 = vắng mặt"
        report.line(f"{role:<16}{files_by_role[role]:>9,}{lines_by_role[role]:>13,}{note:>34}")

    missing_gt = [seq for seq in sequences if "groundtruth" not in seq.files]
    missing_ab = [seq for seq in sequences if "absent" not in seq.files]
    report.line("")
    report.line(f"Thiếu groundtruth: {len(missing_gt)} sequence | Thiếu absent: {len(missing_ab)} sequence"
                f" | Thiếu video: {len(sequences) - len(all_videos) if any_video else 'không kiểm tra (không có mp4)'}")
    if read_errors:
        report.line(f"Lỗi đọc file    : {len(read_errors)} (xem mục 6)")

    # ------------------- 2. TRÙNG TOÀN BỘ SEQUENCE ------------------------
    report.section("2. TRÙNG TOÀN BỘ SEQUENCE (groundtruth + absent giống hệt)")

    fingerprint_groups: Dict[Tuple[str, str], List[SequenceInfo]] = defaultdict(list)
    for seq in sequences:
        fingerprint = sequence_fingerprint(seq)
        if fingerprint is not None:
            fingerprint_groups[fingerprint].append(seq)

    full_groups: List[List[SequenceInfo]] = [
        sorted(group, key=lambda s: s.rel.casefold())
        for group in fingerprint_groups.values()
        if len(group) >= 2
    ]
    full_groups.sort(key=lambda group: group[0].rel.casefold())
    full_sets = [{seq.rel for seq in group} for group in full_groups]
    full_member_rels = {seq.rel for group in full_groups for seq in group}

    if not full_groups:
        report.line("Không phát hiện sequence nào trùng cả groundtruth + absent (sau chuẩn hoá).")
    else:
        cross_count = sum(1 for group in full_groups if _cross_split(group))
        report.line(f"Phát hiện {len(full_groups)} nhóm trùng toàn bộ, "
                    f"{len(full_member_rels)} sequence liên quan"
                    f"{f', trong đó {cross_count} nhóm KHÁC SPLIT' if cross_count else ''}.")
        report.line("")
        limit = args.max_groups
        shown_groups = full_groups if limit <= 0 else full_groups[:limit]
        for idx, group in enumerate(shown_groups, 1):
            group_gt_files = [seq.files["groundtruth"] for seq in group]
            byte_same = len({tf.raw_sha for tf in group_gt_files}) == 1
            absent_hashes = {seq.files["absent"].norm_sha for seq in group}
            absent_same = len(absent_hashes) == 1
            cross = _cross_split(group)
            report.line(
                f"[F{idx:03d}] {len(group)} sequence | gt {'byte-identical' if byte_same else 'giống sau chuẩn hoá'}"
                f" | absent {'giống nhau' if absent_same else 'khác nhau'}"
                f" | split: {_split_text(group)}"
                + ("  <-- CROSS-SPLIT" if cross else "")
            )
            _print_group_members(report, [seq.rel for seq in group], args.max_members)
        if limit > 0 and len(full_groups) > limit:
            report.line(f"... còn {len(full_groups) - limit} nhóm nữa (dùng --max-groups 0 để xem tất cả).")

    # ------------------- 3. TRÙNG GROUNDTRUTH -----------------------------
    report.section("3. TRÙNG GROUNDTRUTH (nhưng chưa chắc trùng absent)")

    gt_files_all = [
        seq.files["groundtruth"]
        for seq in sequences
        if "groundtruth" in seq.files
    ]
    # File toàn 0 / không có bbox hợp lệ không phải "dữ liệu trùng" đáng phân
    # tích; chúng được báo riêng ở mục 6.
    degenerate_gt = sum(1 for tf in gt_files_all if tf.usable and tf.n_informative == 0)
    gt_files = [tf for tf in gt_files_all if tf.n_informative > 0]
    gt_groups = group_by_norm(gt_files)

    gt_only_groups: List[List[TextFile]] = []
    for group in gt_groups:
        rels = {tf.seq_rel for tf in group}
        if any(rels <= full_set for full_set in full_sets):
            continue
        gt_only_groups.append(group)

    if not gt_only_groups:
        report.line("Không có nhóm groundtruth trùng ngoài các nhóm đã báo ở mục 2.")
    else:
        cross_count = sum(
            1 for group in gt_only_groups
            if len({tf.split for tf in group}) > 1
        )
        report.line(f"Phát hiện {len(gt_only_groups)} nhóm groundtruth giống hệt"
                    f"{f', trong đó {cross_count} nhóm KHÁC SPLIT' if cross_count else ''}.")
        report.line("")
        limit = args.max_groups
        shown_groups = gt_only_groups if limit <= 0 else gt_only_groups[:limit]
        for idx, group in enumerate(shown_groups, 1):
            byte_same = len({tf.raw_sha for tf in group}) == 1
            absent_hashes = {absent_signature(seq_by_rel[tf.seq_rel]) for tf in group}
            absent_same = len(absent_hashes) == 1
            cross = len({tf.split for tf in group}) > 1
            report.line(
                f"[G{idx:03d}] {len(group)} file | {'byte-identical' if byte_same else 'giống sau chuẩn hoá'}"
                f" | absent {'giống nhau' if absent_same else 'khác nhau'}"
                f" | split: {', '.join(sorted({tf.split for tf in group}))}"
                + ("  <-- CROSS-SPLIT" if cross else "")
            )
            _print_group_members(report, [tf.rel for tf in group], args.max_members)
        if limit > 0 and len(gt_only_groups) > limit:
            report.line(f"... còn {len(gt_only_groups) - limit} nhóm nữa (dùng --max-groups 0 để xem tất cả).")

    # ------------------- 4. TRÙNG MỘT PHẦN -------------------------------
    report.section("4. TRÙNG MỘT PHẦN (đoạn bbox liên tiếp / file nằm trong file khác)")

    partial_records: List[Dict[str, object]] = []
    if args.no_partial:
        report.line("Bỏ qua do dùng --no-partial.")
    else:
        reps_by_hash: Dict[str, TextFile] = {}
        splits_by_hash: Dict[str, set] = defaultdict(set)
        skipped_short = 0
        for tf in sorted(gt_files, key=lambda x: x.rel.casefold()):
            if not tf.usable or len(tf.inf_hashes) < args.min_block:
                skipped_short += 1
                continue
            splits_by_hash[tf.norm_sha].add(tf.split)
            reps_by_hash.setdefault(tf.norm_sha, tf)
        reps = sorted(reps_by_hash.values(), key=lambda x: x.rel.casefold())

        exact_copies = max(0, len([tf for tf in gt_files if tf.usable])
                           - skipped_short - len(reps))
        report.line(f"Đã gộp {exact_copies:,} file trùng tuyệt đối; bỏ qua {skipped_short:,} file "
                    f"có < {args.min_block} dòng bbox — còn {len(reps):,} đại diện để so cặp.")
        if degenerate_gt:
            report.line(f"Đã loại {degenerate_gt:,} file groundtruth không có bbox hợp lệ khỏi "
                        f"phân tích trùng (xem mục 6).")
        report.line(f"Tiêu chí: >= {args.min_block} dòng bbox liên tiếp giống hệt, hoặc "
                    f">= {args.high_overlap * 100:.0f}% dòng chung (trùng rời rạc).")
        report.line("")

        if len(reps) < 2:
            report.line("Không đủ 2 đại diện để so trùng một phần.")
        else:
            k = min(args.shingle_size, args.min_block)
            candidates, stats = build_candidate_pairs(
                reps,
                k=k,
                sample=args.shingle_sample,
                max_files_per_shingle=args.max_files_per_shingle,
            )
            report.line(
                f"Shingle: k={k}, sample={args.shingle_sample}, "
                f"min_shared={stats['min_shared']} | "
                f"{stats['shingles']:,} shingle, {stats['raw_candidate_pairs']:,} cặp thô, "
                f"{stats['shingle_candidates']:,} cặp từ shingle."
            )
            report.line(
                f"MinHash: {stats['signature_bands']:,} band, "
                f"{stats['signature_pairs']:,} cặp bổ sung | "
                f"tổng ứng viên: {stats['candidate_pairs']:,} cặp."
            )
            if stats["ubiquitous_shingles"]:
                report.line(
                    f"Lưu ý: bỏ qua {stats['ubiquitous_shingles']:,} shingle xuất hiện ở > "
                    f"{args.max_files_per_shingle} file (tối đa {stats['max_ubiquitous_files']:,} file)."
                )
            if stats["signature_skipped_bands"]:
                report.line(
                    f"Lưu ý: bỏ qua {stats['signature_skipped_bands']:,} band MinHash xuất hiện ở > "
                    f"{args.max_files_per_shingle} file (tối đa {stats['max_band_files']:,} file)."
                )
            if args.shingle_sample > 1:
                report.line("Lưu ý: --shingle-sample > 1 có thể bỏ sót đoạn trùng ngắn.")
            report.line("")

            verify_limit = args.max_verify_pairs
            to_verify = candidates if verify_limit <= 0 else candidates[:verify_limit]
            if verify_limit > 0 and len(candidates) > verify_limit:
                report.line(f"Chỉ kiểm tra {verify_limit:,}/{len(candidates):,} cặp ứng viên "
                            f"(dùng --max-verify-pairs 0 để kiểm tra tất cả).")

            for (idx_a, idx_b), _shared in to_verify:
                record = compare_groundtruth_pair(
                    reps[idx_a],
                    reps[idx_b],
                    min_block=args.min_block,
                    high_overlap=args.high_overlap,
                )
                if record is not None:
                    # Đại diện của một nhóm trùng tuyệt đối có thể nằm ở split
                    # khác thành viên còn lại; đánh dấu cross nếu BẤT KỲ cặp
                    # thành viên nào khác split.
                    splits_a = splits_by_hash.get(record["a"].norm_sha, set())
                    splits_b = splits_by_hash.get(record["b"].norm_sha, set())
                    record["cross_split"] = any(
                        split_a != split_b
                        for split_a in splits_a
                        for split_b in splits_b
                    )
                    partial_records.append(record)

            partial_records.sort(
                key=lambda rec: (
                    0 if rec["cross_split"] else 1,
                    -int(rec["run_length"]),
                    -float(rec["overlap"]),
                )
            )

            if not partial_records:
                report.line("Không phát hiện cặp nào trùng đoạn đủ dài hoặc trùng rời rạc đáng kể.")
            else:
                cross_count = sum(1 for rec in partial_records if rec["cross_split"])
                report.line(f"Phát hiện {len(partial_records)} cặp trùng một phần"
                            f"{f', trong đó {cross_count} cặp KHÁC SPLIT' if cross_count else ''}:")
                report.line("[DUP] = cùng split; [LEAK] = khác split. "
                            "'informative' = dòng bbox có w>0,h>0 (bỏ dòng 0/không hợp lệ).")
                report.line("")

                limit = args.max_pairs
                shown = partial_records if limit <= 0 else partial_records[:limit]
                category_names = {
                    "contained": "CHỨA TRỌN FILE NGẮN",
                    "near-full": "GẦN NHƯ TOÀN BỘ",
                    "partial-large": "MỘT PHẦN LỚN",
                    "partial-block": "MỘT ĐOẠN",
                    "reordered": "TRÙNG RỜI RẠC",
                }
                for idx, rec in enumerate(shown, 1):
                    a = rec["a"]
                    b = rec["b"]
                    pos_a = int(rec["pos_a"])
                    pos_b = int(rec["pos_b"])
                    run_length = int(rec["run_length"])
                    len_a = int(rec["len_a"])
                    len_b = int(rec["len_b"])
                    orig_a_start = int(a.inf_orig[pos_a])
                    orig_a_end = int(a.inf_orig[pos_a + run_length - 1])
                    orig_b_start = int(b.inf_orig[pos_b])
                    orig_b_end = int(b.inf_orig[pos_b + run_length - 1])
                    category = category_names.get(str(rec["category"]), str(rec["category"]))
                    prefix = "[LEAK]" if rec["cross_split"] else "[DUP] "
                    if rec["category"] == "reordered":
                        report.line(
                            f"{prefix} [{idx:03d}] {category} | dòng chung={rec['overlap'] * 100:.1f}%"
                            f" | đoạn liên tiếp dài nhất={run_length:,} dòng"
                        )
                    else:
                        report.line(
                            f"{prefix} [{idx:03d}] {category} | run={run_length:,} dòng"
                            f" | {rec['run_ratio'] * 100:.1f}% của file ngắn hơn"
                            f" | dòng chung={rec['overlap'] * 100:.1f}%"
                        )
                    report.line(f"        A: {a.rel}  ({len_a:,} dòng informative, {a.split})")
                    report.line(f"           đoạn dòng {orig_a_start}..{orig_a_end}"
                                + ("" if orig_a_end - orig_a_start + 1 == run_length
                                   else f" (có {orig_a_end - orig_a_start + 1 - run_length} dòng không tính)"))
                    report.line(f"        B: {b.rel}  ({len_b:,} dòng informative, {b.split})")
                    report.line(f"           đoạn dòng {orig_b_start}..{orig_b_end}"
                                + ("" if orig_b_end - orig_b_start + 1 == run_length
                                   else f" (có {orig_b_end - orig_b_start + 1 - run_length} dòng không tính)"))
                if limit > 0 and len(partial_records) > limit:
                    report.line(f"... còn {len(partial_records) - limit} cặp nữa "
                                f"(dùng --max-pairs 0 để xem tất cả).")

    # ------------------- 5. LẶP BẤT THƯỜNG TRONG FILE --------------------
    report.section("5. LẶP BẤT THƯỜNG TRONG MỘT FILE (cảnh báo, không kết luận trùng)")

    within_flags: List[TextFile] = []
    for seq in sequences:
        tf = seq.files.get("groundtruth")
        if tf is None or not tf.usable or tf.n_informative == 0:
            continue
        if tf.max_static_run >= args.within_run_warn:
            within_flags.append(tf)
        elif tf.n_informative >= 50 and tf.dup_ratio >= 0.50:
            within_flags.append(tf)
    within_flags.sort(key=lambda tf: (-tf.max_static_run, -tf.dup_ratio))

    if not within_flags:
        report.line(f"Không có file nào có >= {args.within_run_warn} bbox giống nhau liên tiếp"
                    " hoặc tỉ lệ dòng lặp nội bộ >= 50%.")
    else:
        report.line(f"{len(within_flags)} file có dấu hiệu lặp bất thường "
                    f"(ngưỡng: chuỗi lặp >= {args.within_run_warn} dòng, hoặc >= 50% dòng bị lặp).")
        report.line("Đây có thể là UAV đứng yên / tracker bị kẹt / annotation copy — cần xem video.")
        report.line("")
        for idx, tf in enumerate(within_flags[:50], 1):
            report.line(
                f"[W{idx:02d}] {tf.rel} | informative={tf.n_informative:,}"
                f" | lặp nội bộ={tf.n_dup_informative:,} ({tf.dup_ratio * 100:.1f}%)"
                f" | chuỗi giống nhau dài nhất={tf.max_static_run:,} dòng"
                f" (bắt đầu dòng {tf.max_static_start})"
            )
        if len(within_flags) > 50:
            report.line(f"... còn {len(within_flags) - 50} file nữa.")

    # ------------------- 6. LỖI / CHẤT LƯỢNG ANNOTATION ------------------
    report.section("6. LỖI CẤU TRÚC & CHẤT LƯỢNG ANNOTATION")

    empty_gt = [seq for seq in sequences if "groundtruth" in seq.files
                and (not seq.files["groundtruth"].usable)]
    empty_absent = [seq for seq in sequences if "absent" in seq.files
                    and (not seq.files["absent"].usable)]
    no_inf = [seq for seq in sequences if "groundtruth" in seq.files
              and seq.files["groundtruth"].usable and seq.files["groundtruth"].n_informative == 0]
    bad_absent = [seq for seq in sequences if "absent" in seq.files
                  and seq.files["absent"].n_nonbinary > 0]
    mismatched = []
    for seq in sequences:
        gt = seq.files.get("groundtruth")
        ab = seq.files.get("absent")
        if gt is not None and ab is not None and gt.usable and ab.usable:
            if gt.n_data_lines != ab.n_data_lines:
                mismatched.append((seq, gt.n_data_lines, ab.n_data_lines))
    missing_videos = [seq for seq in sequences if any_video and seq.video is None]

    report.line(f"Thiếu groundtruth : {len(missing_gt):,}")
    report.line(f"Thiếu absent      : {len(missing_ab):,}")
    report.line(f"groundtruth rỗng  : {len(empty_gt):,}")
    report.line(f"absent rỗng       : {len(empty_absent):,}")
    report.line(f"gt không có bbox hợp lệ: {len(no_inf):,}")
    report.line(f"absent có giá trị khác 0/1: {len(bad_absent):,}")
    report.line(f"Lệch số dòng gt/absent: {len(mismatched):,}")
    report.line(f"Tổng bbox all-zero / w,h<=0: "
                f"{sum(seq.files['groundtruth'].n_zero_bbox for seq in sequences if 'groundtruth' in seq.files):,} / "
                f"{sum(seq.files['groundtruth'].n_nonpos_bbox for seq in sequences if 'groundtruth' in seq.files):,}")
    if any_video:
        report.line(f"Thiếu video       : {len(missing_videos):,}")
    if read_errors:
        report.line(f"Lỗi đọc file      : {len(read_errors):,}")
    report.line("")

    def _examples(title: str, items: Sequence[str], limit: int = 10) -> None:
        if not items:
            return
        report.line(f"{title} (hiển thị tối đa {limit}):")
        for item in items[:limit]:
            report.line(f"    - {item}")
        if len(items) > limit:
            report.line(f"    ... và {len(items) - limit} mục nữa")
        report.line("")

    _examples("Sequence thiếu groundtruth", [seq.rel for seq in missing_gt])
    _examples("Sequence thiếu absent", [seq.rel for seq in missing_ab])
    _examples("groundtruth rỗng/không đọc được",
              [f"{seq.rel} ({seq.files['groundtruth'].error or '0 dòng'})" for seq in empty_gt])
    _examples("absent rỗng/không đọc được",
              [f"{seq.rel} ({seq.files['absent'].error or '0 dòng'})" for seq in empty_absent])
    _examples("groundtruth không có bbox w>0,h>0",
              [f"{seq.rel} ({seq.files['groundtruth'].n_data_lines} dòng)" for seq in no_inf])
    _examples("absent có giá trị khác 0/1",
              [f"{seq.rel} ({seq.files['absent'].n_nonbinary} dòng; ví dụ: {seq.files['absent'].nonbinary_example!r})"
               for seq in bad_absent])
    _examples("Lệch số dòng groundtruth/absent",
              [f"{seq.rel}: gt={gt_n:,}, absent={ab_n:,}" for seq, gt_n, ab_n in mismatched])
    _examples("Thiếu video",
              [f"{seq.rel} (có {seq.files.get('groundtruth').n_data_lines:,} dòng gt)"
               if seq.files.get("groundtruth") else seq.rel for seq in missing_videos])
    _examples("Lỗi đọc file", list(read_errors))
    _examples("groundtruth có dòng bbox không parse được",
              [f"{seq.rel} ({seq.files['groundtruth'].n_bad_bbox} dòng; ví dụ: "
               f"{seq.files['groundtruth'].bad_bbox_example!r})"
               for seq in sequences
               if "groundtruth" in seq.files and seq.files["groundtruth"].n_bad_bbox > 0])
    _examples("Ghi chú sequence",
              [f"{seq.rel}: {'; '.join(seq.notes)}" for seq in sequences if seq.notes])

    # ------------------- 7. TRÙNG TÊN SEQUENCE ---------------------------
    report.section("7. TRÙNG TÊN SEQUENCE / VIDEO (khác thư mục)")

    by_name: Dict[str, List[SequenceInfo]] = defaultdict(list)
    for seq in sequences:
        by_name[seq.name].append(seq)
    name_collisions = [
        sorted(group, key=lambda s: s.rel.casefold())
        for group in by_name.values()
        if len(group) >= 2
    ]
    name_collisions.sort(key=lambda group: group[0].name.casefold())

    if not name_collisions:
        report.line("Không có tên sequence nào xuất hiện ở nhiều thư mục.")
    else:
        cross = sum(1 for group in name_collisions if _cross_split(group))
        report.line(f"{len(name_collisions)} tên sequence bị dùng ở nhiều thư mục"
                    f"{f', trong đó {cross} tên KHÁC SPLIT' if cross else ''}:")
        for group in name_collisions[:args.max_groups if args.max_groups > 0 else len(name_collisions)]:
            report.line(f"    - {group[0].name}: " + ", ".join(seq.rel for seq in group)
                        + ("  <-- CROSS-SPLIT" if _cross_split(group) else ""))
        if args.max_groups > 0 and len(name_collisions) > args.max_groups:
            report.line(f"    ... và {len(name_collisions) - args.max_groups} tên nữa.")

    by_video_name: Dict[str, List[SequenceInfo]] = defaultdict(list)
    for seq in sequences:
        if seq.video is not None:
            by_video_name[seq.video.name.casefold()].append(seq)
    video_collisions = [
        sorted(group, key=lambda s: s.rel.casefold())
        for group in by_video_name.values()
        if len(group) >= 2
    ]
    video_collisions.sort(key=lambda group: group[0].video.name.casefold() if group[0].video else "")
    if video_collisions:
        report.line("")
        report.line(f"{len(video_collisions)} tên file video bị dùng ở nhiều thư mục:")
        for group in video_collisions[:args.max_groups if args.max_groups > 0 else len(video_collisions)]:
            filename = group[0].video.name if group[0].video else "?"
            report.line(f"    - {filename}: " + ", ".join(seq.rel for seq in group)
                        + ("  <-- CROSS-SPLIT" if _cross_split(group) else ""))
        if args.max_groups > 0 and len(video_collisions) > args.max_groups:
            report.line(f"    ... và {len(video_collisions) - args.max_groups} tên video nữa.")

    # ------------------- 8. ABSENT / METADATA ----------------------------
    report.section("8. TRÙNG ABSENT VÀ METADATA (chỉ tham khảo)")

    absent_files = [seq.files["absent"] for seq in sequences if "absent" in seq.files]
    absent_groups = group_by_norm(absent_files)
    absent_only = []
    for group in absent_groups:
        rels = {tf.seq_rel for tf in group}
        if any(rels <= full_set for full_set in full_sets):
            continue
        absent_only.append(group)

    report.line(f"absent.txt giống hệt: {len(absent_groups)} nhóm"
                f" | trong đó groundtruth khác (không phải trùng toàn bộ): {len(absent_only)} nhóm.")
    report.line("Lưu ý: cùng lịch vắng mặt KHÔNG đủ kết luận trùng dữ liệu; phải xem groundtruth/video.")
    if absent_only:
        report.line("Ví dụ:")
        for group in absent_only[:10]:
            report.line(f"    - {len(group)} sequence: " + ", ".join(tf.seq_rel for tf in group[:8])
                        + (" ..." if len(group) > 8 else ""))
    report.line("")

    metadata_groups: Dict[str, List[List[TextFile]]] = {}
    for role in ("attributes", "language"):
        role_files = [seq.files[role] for seq in sequences if role in seq.files]
        metadata_groups[role] = group_by_norm(role_files)
        affected = sum(len(group) for group in metadata_groups[role])
        report.line(f"{role}.txt: {affected:,} file nằm trong {len(metadata_groups[role]):,} nhóm giống hệt.")
    report.line("Metadata thường dùng chung template nên trùng là bình thường; không tính vào kết luận.")
    if args.include_metadata:
        report.line("")
        for role in ("attributes", "language"):
            if metadata_groups[role]:
                report.line(f"Các nhóm {role}.txt giống hệt:")
                for group in metadata_groups[role][:args.max_groups if args.max_groups > 0 else len(metadata_groups[role])]:
                    report.line(f"    - {len(group)} file: " + ", ".join(tf.rel for tf in group[:10])
                                + (" ..." if len(group) > 10 else ""))

    # ------------------- 9. KẾT LUẬN ------------------------------------
    report.section("9. KẾT LUẬN")

    partial_cross = sum(1 for rec in partial_records if rec["cross_split"])
    full_cross = sum(1 for group in full_groups if _cross_split(group))
    gt_cross = sum(1 for group in gt_only_groups if len({tf.split for tf in group}) > 1)
    name_cross = sum(1 for group in name_collisions if _cross_split(group))

    report.line(f"Trùng toàn bộ sequence : {len(full_groups):,} nhóm / {len(full_member_rels):,} sequence"
                f"{f' ({full_cross} nhóm khác split)' if full_cross else ''}")
    report.line(f"Trùng groundtruth      : {len(gt_only_groups):,} nhóm"
                f"{f' ({gt_cross} nhóm khác split)' if gt_cross else ''}")
    report.line(f"Trùng một phần         : {len(partial_records):,} cặp"
                f"{f' ({partial_cross} cặp khác split)' if partial_cross else ''}")
    report.line(f"Tên sequence trùng     : {len(name_collisions):,} tên"
                f"{f' ({name_cross} tên khác split)' if name_cross else ''}")
    report.line(f"Lặp bất thường nội bộ  : {len(within_flags):,} file")
    report.line(f"Lỗi/annotation bất thường: {len(empty_gt) + len(empty_absent) + len(no_inf) + len(bad_absent) + len(mismatched) + len(read_errors):,} mục")
    report.line("")

    has_duplicate = bool(full_groups or gt_only_groups or partial_records)
    has_cross = bool(full_cross or gt_cross or partial_cross)
    if has_cross:
        report.line("KẾT LUẬN: CÓ trùng lặp, trong đó có cặp KHÁC SPLIT — nguy cơ rò rỉ Train/Test.")
        report.line("          Cần xử lý các nhóm/cặp đánh dấu CROSS-SPLIT trước khi train/evaluate.")
    elif has_duplicate:
        report.line("KẾT LUẬN: CÓ trùng lặp nội dung annotation. Cần giữ một bản hoặc loại bỏ bản trùng.")
    elif name_cross:
        report.line("KẾT LUẬN: Không phát hiện trùng nội dung, NHƯNG có tên sequence trùng khác split.")
        report.line("          Cần kiểm tra mục 7: có thể là cùng một sequence bị đặt ở cả Train và Test.")
    else:
        report.line("KẾT LUẬN: Không phát hiện trùng lặp nội dung txt theo các tiêu chí trên.")

    report.line("")
    report.line("Gợi ý xử lý:")
    report.line("  1. Mở trực tiếp các file/đoạn dòng ở mục 2-4 để xác nhận (đặc biệt cặp chỉ trùng một đoạn).")
    report.line("  2. Trùng toàn bộ cùng split: giữ 1 bản, xoá/rename bản còn lại.")
    report.line("  3. Trùng khác split: chuyển tất cả về cùng một split trước khi chia train/test.")
    report.line("  4. Nếu chỉ trùng một đoạn: kiểm tra xem có phải cùng video bị cắt thành nhiều sequence.")
    report.line("  5. Các file ở mục 5-6 cần xem video/annotation gốc; không tự động xoá dựa trên cảnh báo.")
    report.line("")
    report.line("Lưu ý: trùng txt không chắc trùng nội dung video, và ngược lại hai video khác nhau có thể")
    report.line("có annotation giống nhau do dữ liệu tổng hợp. Báo cáo này là bằng chứng để kiểm tra, không")
    report.line("phải lệnh xoá tự động.")
    report.line(f"Thời gian chạy: {time.time() - started_at:.1f}s")
    report.line(f"Report: {report.path}")

    return {
        "full_groups": full_groups,
        "gt_only_groups": gt_only_groups,
        "partial_records": partial_records,
        "within_flags": within_flags,
        "has_duplicate": has_duplicate,
        "has_cross": has_cross,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Kiểm tra trùng lặp dữ liệu UAV-Anti-UAV ở mức file txt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help=f"Thư mục dataset (mặc định: {DEFAULT_DATA_REL.as_posix()}).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"File kết quả (mặc định: {DEFAULT_OUTPUT_REL.as_posix()}).",
    )
    parser.add_argument(
        "--min-block",
        type=int,
        default=12,
        help="Số dòng bbox liên tiếp tối thiểu để coi là trùng một phần.",
    )
    parser.add_argument(
        "--shingle-size",
        type=int,
        default=8,
        help="Số dòng mỗi shingle khi tìm ứng viên.",
    )
    parser.add_argument(
        "--shingle-sample",
        type=int,
        default=1,
        help="Chỉ index 1/N shingle (1 = không bỏ). Tăng để giảm RAM, đổi lại dễ sót đoạn ngắn.",
    )
    parser.add_argument(
        "--high-overlap",
        type=float,
        default=0.90,
        help="Tỉ lệ dòng chung tối thiểu để báo trùng rời rạc (0-1).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=200,
        help="Số cặp trùng một phần tối đa in ra (0 = không giới hạn).",
    )
    parser.add_argument(
        "--max-verify-pairs",
        type=int,
        default=5000,
        help="Số cặp ứng viên tối đa đem kiểm tra chính xác (0 = không giới hạn).",
    )
    parser.add_argument(
        "--max-files-per-shingle",
        type=int,
        default=50,
        help="Bỏ qua shingle xuất hiện ở nhiều hơn N file để tránh bùng nổ tổ hợp (0 = không giới hạn).",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=100,
        help="Số nhóm trùng tối đa in ra ở mỗi mục (0 = không giới hạn).",
    )
    parser.add_argument(
        "--max-members",
        type=int,
        default=30,
        help="Số thành viên tối đa in ra trong một nhóm (0 = không giới hạn).",
    )
    parser.add_argument(
        "--within-run-warn",
        type=int,
        default=30,
        help="Cảnh báo nếu một file có chuỗi bbox giống nhau liên tiếp >= N dòng.",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Liệt kê cả nhóm attributes.txt/language.txt giống hệt (mặc định chỉ đếm).",
    )
    parser.add_argument(
        "--no-partial",
        action="store_true",
        help="Bỏ qua bước so trùng một phần (chỉ kiểm tra trùng toàn bộ).",
    )
    parser.add_argument(
        "--fail-on-duplicates",
        action="store_true",
        help="Trả mã thoát 2 nếu phát hiện trùng lặp toàn bộ/một phần.",
    )
    return parser


def _resolve_data_dir(explicit: Optional[str]) -> Tuple[Path, str]:
    """
    Xác định data dir.

    Nếu người dùng truyền --data-dir: tôn trọng tuyệt đối (resolve theo CWD).
    Nếu dùng mặc định: thử ./data/UAV_anti_UAV trước, sau đó tự dò các biến thể
    tên (UAV-Anti-UAV, ...) để phù hợp repo hiện tại.
    """
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve(), ""

    default = REPO_ROOT / DEFAULT_DATA_REL
    if default.exists():
        return default, ""

    data_parent = REPO_ROOT / "data"
    if data_parent.is_dir():
        candidates = []
        try:
            children = sorted(data_parent.iterdir(), key=lambda p: p.name.casefold())
        except OSError:
            children = []
        for child in children:
            if not child.is_dir():
                continue
            normalized = child.name.casefold().replace("-", "_")
            if normalized == "uav_anti_uav":
                candidates.append(child)
        if candidates:
            chosen = candidates[0]
            return chosen, (
                f"Không thấy {DEFAULT_DATA_REL.as_posix()}; đã tự dùng {chosen} "
                f"(đổi tên thư mục hoặc truyền --data-dir để chắc chắn)."
            )
    return default, ""


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        # Giữ encoding console hiện tại (Windows console thường là UTF-8, còn khi
        # pipe có thể là cp1252). Chỉ thay lỗi bằng '?' để không crash; file kết
        # quả vẫn luôn UTF-8.
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    parser = build_parser()
    args = parser.parse_args(argv)

    # Kiểm tra tham số.
    if args.min_block < 2:
        parser.error("--min-block phải >= 2")
    if args.shingle_size < 2:
        parser.error("--shingle-size phải >= 2")
    if args.shingle_sample < 1:
        parser.error("--shingle-sample phải >= 1")
    if not 0.0 < args.high_overlap <= 1.0:
        parser.error("--high-overlap phải trong khoảng (0, 1]")
    if args.max_pairs < 0 or args.max_verify_pairs < 0 or args.max_groups < 0 or args.max_members < 0:
        parser.error("các tham số max-* phải >= 0")
    if args.max_files_per_shingle < 0:
        parser.error("--max-files-per-shingle phải >= 0")
    if args.within_run_warn < 1:
        parser.error("--within-run-warn phải >= 1")

    started_at = time.time()
    data_dir, data_note = _resolve_data_dir(args.data_dir)

    if args.output:
        output_path = Path(args.output).expanduser()
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        output_path = output_path.resolve()
    else:
        output_path = (REPO_ROOT / DEFAULT_OUTPUT_REL).resolve()

    try:
        report = Report(output_path)
    except OSError as exc:
        print(f"[check_data] LỖI: không tạo được file kết quả {output_path}: {exc}", file=sys.stderr)
        return 1

    with report:
        report.section("KIỂM TRA TRÙNG LẶP DỮ LIỆU UAV-ANTI-UAV (TXT)")
        report.line(f"Dataset      : {data_dir}")
        report.line(f"Report       : {output_path}")
        report.line(f"Cấu hình     : min_block={args.min_block}, shingle_size={args.shingle_size}, "
                    f"shingle_sample={args.shingle_sample}, high_overlap={args.high_overlap}")
        report.line(f"               max_pairs={args.max_pairs}, max_verify_pairs={args.max_verify_pairs}, "
                    f"max_files_per_shingle={args.max_files_per_shingle}")
        if data_note:
            report.line("")
            report.line(f"LƯU Ý ĐƯỜNG DẪN: {data_note}")

        if not data_dir.exists():
            report.section("LỖI")
            report.line(f"Không tìm thấy dataset: {data_dir}")
            report.line("")
            report.line("Cấu trúc mong đợi:")
            report.line("    data/UAV_anti_UAV/Train/<sequence>/groundtruth_rect.txt + absent.txt + ...")
            report.line("    data/UAV_anti_UAV/Test/<sequence>/...")
            report.line("")
            report.line("Truyền đúng đường dẫn bằng --data-dir, ví dụ:")
            report.line("    python tools/check_data.py --data-dir ./data/UAV_anti_UAV")
            report.line(f"Thời gian chạy: {time.time() - started_at:.1f}s")
            return 1

        if not data_dir.is_dir():
            report.section("LỖI")
            report.line(f"Đường dẫn không phải thư mục: {data_dir}")
            return 1

        print(f"[check_data] Quét {data_dir} ...", file=sys.stderr, flush=True)
        try:
            sequences, read_errors = discover_sequences(data_dir)
        except Exception as exc:
            report.section("LỖI KHI QUÉT DỮ LIỆU")
            report.line(f"{type(exc).__name__}: {exc}")
            report.line(traceback.format_exc())
            return 1

        print(f"[check_data] Đọc xong {len(sequences):,} sequence, "
              f"{len(read_errors):,} lỗi đọc file.", file=sys.stderr, flush=True)

        if not sequences:
            report.section("LỖI")
            report.line("Không tìm thấy sequence nào có groundtruth_rect.txt hoặc absent.txt.")
            report.line("Kiểm tra lại cấu trúc dataset (Train/Test/<sequence>/...).")
            report.line("")
            report.line("Các thư mục con cấp 1 đang có:")
            try:
                for child in sorted(data_dir.iterdir(), key=lambda p: p.name.casefold())[:30]:
                    report.line(f"    - {child.name}{'/' if child.is_dir() else ''}")
            except OSError as exc:
                report.line(f"    (không đọc được: {exc})")
            report.line(f"Thời gian chạy: {time.time() - started_at:.1f}s")
            return 1

        try:
            findings = run_analysis(
                report=report,
                args=args,
                data_dir=data_dir,
                sequences=sequences,
                read_errors=read_errors,
                started_at=started_at,
            )
        except Exception as exc:
            report.section("LỖI KHI PHÂN TÍCH")
            report.line(f"{type(exc).__name__}: {exc}")
            report.line(traceback.format_exc())
            return 1

        if args.fail_on_duplicates and findings.get("has_duplicate"):
            return 2
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
