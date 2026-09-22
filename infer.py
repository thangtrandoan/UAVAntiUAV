import os
import sys
import time
import argparse
import yaml
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import builtins
from collections import namedtuple
from torchvision import transforms

from model import UAVReIDNet, load_checkpoint_verbose

def extract_cnn_feature(model, tensor_frame):
    with torch.no_grad():
        feats = model.backbone(tensor_frame)
        if isinstance(feats, tuple):
            if isinstance(feats[0], tuple):
                global_feat = feats[0][0]
                fs_feat = feats[0][1]
            else:
                global_feat = feats[0]
                fs_feat = feats[1]
            feats = torch.cat([global_feat, fs_feat], dim=-1)
    return feats

def compute_reid_embedding(model, seq_feats, visual_feat=None):
    with torch.no_grad():
        if visual_feat is None:
            visual_feat = seq_feats.mean(dim=1)
        temporal_token, _ = model.temporal_encoder(seq_feats)
        bn_feat = model.head(visual_feat, temporal_token)
        bn_feat = F.normalize(bn_feat, p=2, dim=1)
    return bn_feat

def compute_sharpness(crop_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

class SlidingWindowBuffer:
    def __init__(self, window_size: int = 16, stride: int = 2):
        self.window_size = window_size
        self.stride = stride
        self.features = []
        self.sharpness_scores = []
        self._frame_counter = 0
    
    def should_extract(self) -> bool:
        result = (self._frame_counter % self.stride == 0)
        self._frame_counter += 1
        return result
    
    def add(self, feat: torch.Tensor, sharpness: float):
        self.features.append(feat)
        self.sharpness_scores.append(sharpness)
        if len(self.features) > self.window_size:
            self.features.pop(0)
            self.sharpness_scores.pop(0)
    
    def is_ready(self) -> bool:
        return len(self.features) >= self.window_size
    
    def get_sequence(self) -> torch.Tensor:
        return torch.stack(self.features, dim=1)

    # 🛠️ (15/9) PHÂN VAI stride:
    #   SOFT LOCK  → thu LIÊN TỤC (stride=1): chỉ lọc thô bằng `visual`, không cần
    #                bước thời gian, thu liên tục để phản ứng nhanh.
    #   HARD LOCK  → LẤY CÁCH QUÃNG (`stride = frame_stride`): bước thời gian của cửa sổ
    #                temporal PHẢI khớp lúc train (`data_pipeline.frame_stride`) và khớp
    #                Memory Bank (dựng từ `sliding_window`, cũng stride = frame_stride).
    def get_strided_sequence(self, stride: int = 1) -> torch.Tensor:
        return torch.stack(self.features[::stride], dim=1)
    
    def get_weighted_visual_mean(self, stride: int = 1) -> torch.Tensor:
        feats = self.features[::stride]
        weights = torch.tensor(self.sharpness_scores[::stride], dtype=torch.float32)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = torch.ones_like(weights) / len(weights)
        
        stacked = torch.stack([f.squeeze(0) for f in feats])
        return (stacked * weights.unsqueeze(1).to(stacked.device)).sum(dim=0, keepdim=True)
    
    def clear(self):
        self.features.clear()
        self.sharpness_scores.clear()
        self._frame_counter = 0

# 🛠️ DEBUG (14/9): bundle trả về đủ mọi tầng của phép fusion để đo được
# cosine TRƯỚC BatchNorm (raw_feat) vs SAU BatchNorm (fused_feat).
#   visual_mean    : weighted mean theo sharpness — chỉ dùng cho coarse score
#   visual_plain   : plain mean — đúng như lúc train, là input của head
#   temporal_token : đầu ra Mamba
#   raw_feat       : L2-normalize( cat(visual_plain, temporal_token) )  ← TRƯỚC bnneck
#   fused_feat     : L2-normalize( bnneck(cat(...)) )                   ← SAU bnneck (fine score)
FusedBundle = namedtuple(
    'FusedBundle',
    ['visual_mean', 'visual_plain', 'temporal_token', 'raw_feat', 'fused_feat']
)


def compute_fused_vector(model, sliding_window, stride: int = 1):
    # 🛠️ (15/9) `stride`: HARD LOCK truyền `frame_stride` để cửa sổ temporal có bước
    # thời gian khớp lúc train + khớp Memory Bank. SOFT LOCK để mặc định 1 (liên tục).
    seq_feats = sliding_window.get_strided_sequence(stride)
    
    # 1. BẮT BUỘC dùng mean để đưa vào khối Fusion Head (vì lúc train model học bằng mean)
    # Nếu đưa 1 frame vào Fusion Head, phân phối (variance) bị sai lệch dẫn đến Mamba tính sai bét
    #
    # 🛠️ FIX: Fused Feature (qua head) phải dùng PLAIN MEAN giống training
    # (model.py: visual_feat = feats.mean(dim=1)). Weighted mean (theo sharpness)
    # chỉ nên dùng cho COARSE score (backbone feature), KHÔNG đưa vào head,
    # vì head được train với plain mean → weighted mean làm fused feature lệch → fine score thấp.
    visual_mean = sliding_window.get_weighted_visual_mean(stride)   # dùng cho coarse score (không qua head)
    visual_plain = seq_feats.mean(dim=1)                     # giống hệt training → cho head
    
    # Tính temporal_token + fused_feat MỘT LẦN (không gọi temporal_encoder 2 lần)
    with torch.no_grad():
        temporal_token, _ = model.temporal_encoder(seq_feats)
        # Đầu vào THÔ của head (trước bnneck) — chính là vector bị BatchNorm1d biến đổi
        feat = torch.cat([visual_plain, temporal_token], dim=-1)
        raw_feat = F.normalize(feat, p=2, dim=1)
        
        bn_feat = model.head(visual_plain, temporal_token)
        fused_feat = F.normalize(bn_feat, p=2, dim=1)
    
    return FusedBundle(visual_mean, visual_plain, temporal_token, raw_feat, fused_feat)

class TwoTierMemoryBank:
    def __init__(self, max_anchor: int = 10, max_recent: int = 30):
        self.max_anchor = max_anchor
        self.max_recent = max_recent
        self.anchor_bank = []
        self.recent_bank = []
    
    @staticmethod
    def _make_entry(visual_feat, fused_feat, temporal_token=None,
                    visual_plain=None, raw_feat=None):
        """Lưu đủ 5 tầng vector để debug: coarse(weighted) / plain / temporal / PRE-BN / POST-BN."""
        def _norm(t):
            return F.normalize(t, p=2, dim=1) if t is not None else None
        return {
            "visual": _norm(visual_feat),
            "visual_plain": _norm(visual_plain),
            "fused": _norm(fused_feat),
            "raw": _norm(raw_feat),
            "temporal": _norm(temporal_token),
        }
    
    def add_anchor(self, visual_feat: torch.Tensor, fused_feat: torch.Tensor,
                   temporal_token: torch.Tensor = None, visual_plain: torch.Tensor = None,
                   raw_feat: torch.Tensor = None):
        if len(self.anchor_bank) < self.max_anchor:
            self.anchor_bank.append(self._make_entry(
                visual_feat, fused_feat, temporal_token, visual_plain, raw_feat))
    
    def add_recent(self, visual_feat: torch.Tensor, fused_feat: torch.Tensor,
                   temporal_token: torch.Tensor = None, visual_plain: torch.Tensor = None,
                   raw_feat: torch.Tensor = None):
        self.recent_bank.append(self._make_entry(
            visual_feat, fused_feat, temporal_token, visual_plain, raw_feat))
        if len(self.recent_bank) > self.max_recent:
            self.recent_bank.pop(0)
    
    def _max_sim(self, query_feat: torch.Tensor, key: str) -> float:
        """Cosine similarity lớn nhất giữa query và mọi entry có trường `key`."""
        query = F.normalize(query_feat, p=2, dim=1)
        max_sim = 0.0
        for entry in self.anchor_bank + self.recent_bank:
            ref = entry.get(key)
            if ref is not None:
                sim = torch.mm(query, ref.t()).item()
                max_sim = max(max_sim, sim)
        return max_sim
    
    def coarse_score(self, query_feat: torch.Tensor) -> float:
        """Coarse: visual weighted-mean (KHÔNG qua head)."""
        return self._max_sim(query_feat, "visual")
    
    # DEBUG: so sánh riêng visual_plain (đúng input của head) với anchor
    def visual_plain_score(self, query_visual_plain: torch.Tensor) -> float:
        return self._max_sim(query_visual_plain, "visual_plain")
    
    # DEBUG: so sánh riêng temporal token với anchor
    def temporal_score(self, query_temporal: torch.Tensor) -> float:
        return self._max_sim(query_temporal, "temporal")
    
    # DEBUG: cosine TRƯỚC BatchNorm1d (cat thô đã L2-normalize)
    def raw_score(self, query_raw: torch.Tensor) -> float:
        return self._max_sim(query_raw, "raw")
    
    # Fine: cosine SAU BatchNorm1d — đây là score pipeline đang dùng để HARD LOCK
    def fine_score(self, query_fused: torch.Tensor) -> float:
        return self._max_sim(query_fused, "fused")
    
    def is_empty(self) -> bool:
        return len(self.anchor_bank) == 0 and len(self.recent_bank) == 0
    
    def size_info(self) -> str:
        return f"Anchor: {len(self.anchor_bank)}/{self.max_anchor} | Recent: {len(self.recent_bank)}/{self.max_recent}"

def parse_args():
    parser = argparse.ArgumentParser(description="Sequence Inference for UAV ReID (OOP Pipeline)")
    parser.add_argument("--seq-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/config_jetson.yaml")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    return parser.parse_args()

def crop_and_pad(frame, bbox, padding):
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox
    
    pad_w, pad_h = int(bw * padding), int(bh * padding)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w, x + bw + pad_w)
    y2 = min(h, y + bh + pad_h)
    
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]

class SeqReIDPipeline:
    T0_INIT = "T0_INIT"
    T1_LOST = "T1_LOST"
    T2_SEARCH = "T2_SEARCH"
    T3_VERIFIED = "T3_VERIFIED"
    
    def __init__(self, model, device, cfg):
        self.model = model
        self.device = device
        self.state = self.T0_INIT
        
        self.stride = cfg.get('stride', 2)
        self.num_frames = cfg.get('num_frames', 16)
        self.soft_lock_threshold = cfg.get('soft_lock_threshold', 0.50)
        self.reid_threshold = cfg.get('reid_threshold', 0.75)
        self.hijack_threshold = cfg.get('hijack_threshold', 0.40)
        self.hijack_check_count = cfg.get('hijack_check_count', 5)
        self.update_interval_sec = cfg.get('update_interval_sec', 2.0)
        self.bbox_padding = cfg.get('bbox_padding', 0.2)
        # 🛠️ DEBUG (14/9): in tách cosine TRƯỚC BN (raw) vs SAU BN (fused).
        # Bật/tắt bằng `debug_sim` trong block `infer` của config.
        self.debug_sim = cfg.get('debug_sim', True)
        # 🛠️ (14/9): không gian dùng cho gate HARD LOCK: 'fused' (mặc định, hành vi cũ)
        # hoặc 'pre_bn' (đầu vào bnneck). Đổi sang 'pre_bn' thì PHẢI đổi `reid_threshold`
        # theo `calibrated_threshold.json -> thresholds.pre_bn` (hai thang điểm khác nhau).
        self.fine_space = cfg.get('fine_space', 'fused')
        
        self.memory_bank = TwoTierMemoryBank(
            max_anchor=cfg.get('max_anchor_size', 10),
            max_recent=cfg.get('max_recent_size', 30)
        )
        # 🛠️ (15/9) PHÂN VAI stride (theo yêu cầu: soft lock thu liên tục, hard lock mới stride):
        #   - `sliding_window` (tracking + Memory Bank): lấy CÁCH QUÃNG `frame_stride`
        #   - `soft_lock_buffer` (T2_SEARCH): thu LIÊN TỤC (stride=1) để phản ứng nhanh,
        #     nhưng chứa đủ `(num_frames-1)*stride + 1` frame để khi HARD LOCK thì LẤY
        #     CÁCH QUÃNG ra đúng `num_frames` mẫu với bước thời gian = frame_stride.
        # Trước đây soft_lock thu liên tục rồi đưa NGUYÊN chuỗi liên tục vào head
        # → bước thời gian = 1 so với bank bước = frame_stride → temporal token lệch.
        # (xem md/15thg9.md §13)
        self.sliding_window = SlidingWindowBuffer(self.num_frames, self.stride)
        # 🛠️ (22/9) HAI CỬA SỔ SONG SONG (theo yêu cầu):
        #   (1) SOFT LOCK  — `num_frames` frame LIÊN TỤC (bước 1). NHANH, chỉ để CHỌN ỨNG VIÊN
        #       / xem điểm. Khi tái xuất có nhiều mục tiêu, mục tiêu có điểm coarse cao nhất
        #       được chọn để đem đi HARD LOCK. KHÔNG dùng để chốt.
        #   (2) HARD LOCK  — `num_frames` MẪU, mỗi mẫu cách nhau `frame_stride`. CHÍNH XÁC,
        #       mới là cửa sổ đem so với Memory Bank (bank cũng bước `frame_stride`).
        #   (3) Memory Bank / tracking (`sliding_window`) — bước `frame_stride`.
        # Hệ quả số học: HARD LOCK cần `(num_frames-1)*stride+1` frame video (12/4 -> 45),
        # còn SOFT LOCK chỉ cần `num_frames` frame (-> 12). Soft lock KHÔNG phải chờ 45.
        self.soft_lock_capacity = self.num_frames
        self.soft_lock_buffer = SlidingWindowBuffer(self.num_frames, stride=1)
        self.hard_lock_capacity = (self.num_frames - 1) * self.stride + 1
        self.hard_lock_buffer = SlidingWindowBuffer(self.num_frames, self.stride)
        self._soft_lock_announced = False
        self._soft_lock_passed = False
        # ⚠️ RÀNG BUỘC ỨNG VIÊN (22/9): cửa sổ HARD LOCK phải thuộc ĐÚNG ứng viên mà soft lock
        # đã chọn — nếu không, nó trộn frame của hai vật khác nhau và điểm hard lock vô nghĩa.
        # Ở pipeline NÀY bbox là GT nên chỉ có MỘT ứng viên (chính target), và ứng viên đó tồn
        # tại ngay từ frame tái xuất -> thu từ frame tái xuất là đúng, không cần reset.
        # NẾU mở rộng sang nhiều mục tiêu: phải `hard_lock_buffer.clear()` mỗi khi ứng viên đổi
        # (xem `phan_rang/infer_realworld.py`, chỗ `soft_lock_id != best_tid`).

        # 🛠️ (22/9) NỚI RESET khi target vắng mặt trong T2_SEARCH (md/22thg9.md §4, §8.3).
        # `soft_lock_buffer` cần `(num_frames-1)*stride+1` frame LIÊN TỤC (12/4 -> 45), mà
        # GT trong video này có lần tái xuất chỉ ~10 frame -> reset ở 1 frame vắng làm event
        # đó VĨNH VIỄN không thể lock. Vắng ngắn (detection dropout) chỉ SKIP frame và GIỮ
        # feature đã thu; chỉ vắng LIÊN TIẾP quá `gap_tolerance` mới reset thật.
        # Khoảng vắng làm cửa sổ có "lỗ" thời gian. Điều này KHÔNG hoàn toàn xa lạ với
        # train: `data_pipeline.py` bỏ qua frame có bbox không hợp lệ, nên danh sách frame
        # trong JSON cũng có thể có các mốc cách xa hơn `frame_stride` (xem §4.2).
        # Khác biệt còn lại: bên train là THIẾU mẫu, còn đây là mẫu bắc QUA một khoảng vắng.
        self.gap_tolerance = cfg.get('t2_search_gap_tolerance', 2)
        self._absent_streak = 0
        
        self.last_update_time = 0.0
        self._hijack_checks_remaining = 0
        
        self.metrics_cnn_times = []
        self.metrics_mamba_times = []
        self.false_alarms = 0
        self.reid_latency_frames = []
        self.reappeared_frame_idx = -1
        # Tích luỹ để tổng hợp cuối sequence (trả lời câu hỏi: BN có phá cosine không?)
        self.debug_pre_bn_scores = []
        self.debug_post_bn_scores = []
        # 🛠️ (14/9): tách theo tag — nếu gộp chung thì "tỉ lệ cửa sổ vượt ngưỡng" bị lẫn giữa
        # cửa sổ re-acquire (T2_SEARCH) và kiểm tra anti-hijack (T3_VERIFIED), hai thứ khác bản chất.
        self.debug_tag_counts = {'re-acquire': 0, 'anti-hijack': 0}
        
    def _log_sim_breakdown(self, frame_idx, bundle, fused_score, tag="sim"):
        """
        In breakdown cosine của cùng một truy vấn (cùng cửa sổ) với Memory Bank:
          visual_shot  : weighted mean (coarse)      — ngoài head
          visual_plain : plain mean (input head)     — ngoài head
          temporal     : token Mamba                 — input head
          PRE-BN raw   : cat(visual_plain, temporal) — ĐẦU VÀO bnneck
          POST-BN fused: bnneck(...)                 — đầu ra bnneck = fine score
        Cách đọc: raw cao (>=0.8) mà fused thấp hơn `reid_threshold` → bnneck là mắt xích đang
                  chặn HARD LOCK (nhưng CHƯA nói được BN có hại hay không — cần TAR@FAR).
                  raw thấp sẵn (~ temporal)          → vấn đề nằm ở feature (temporal/dữ liệu train).
        """
        if not self.debug_sim:
            return
        vis = self.memory_bank.coarse_score(bundle.visual_mean)
        vis_plain = self.memory_bank.visual_plain_score(bundle.visual_plain)
        temp = self.memory_bank.temporal_score(bundle.temporal_token)
        raw = self.memory_bank.raw_score(bundle.raw_feat)
        self.debug_pre_bn_scores.append(raw)
        self.debug_post_bn_scores.append(fused_score)
        if tag in self.debug_tag_counts:
            self.debug_tag_counts[tag] += 1
        print(f"[{frame_idx}] DEBUG {tag}: visual_shot={vis:.3f} | visual_plain={vis_plain:.3f} | "
              f"temporal={temp:.3f} || PRE-BN raw={raw:.3f} -> POST-BN fused={fused_score:.3f} "
              f"(BN delta={raw - fused_score:+.3f})")
        
    def _transition_to_lost(self, frame_idx):
        print(f"[{frame_idx}] Target LOST! -> T1_LOST")
        self.state = self.T1_LOST
        self.sliding_window.clear()
        self.soft_lock_buffer.clear()
        self.hard_lock_buffer.clear()
        self._soft_lock_announced = False
        self._soft_lock_passed = False
        
    def process_frame(self, frame, bbox, is_absent, frame_idx, transform):
        valid_bbox = bbox[2] > 0 and bbox[3] > 0
        current_time = time.time()
        
        if self.state in [self.T0_INIT, self.T3_VERIFIED]:
            if is_absent or not valid_bbox:
                if len(self.sliding_window.features) > 0:
                    # 🛠️ (15/9) BỎ pad bằng cách NHÂN BẢN frame cuối (bước thời gian = 0).
                    # Cửa sổ đã thu với bước = frame_stride; nếu chưa đủ `num_frames` thì
                    # tính trên ĐÚNG số frame đã có → MỌI bước chuyển tiếp vẫn = frame_stride.
                    # (temporal_encoder dùng pos_embed[:, :N, :] và conv1d cắt về L nên N nhỏ OK)
                        
                    if self.device.type == 'cuda': torch.cuda.synchronize()
                    t0 = time.time()
                    bundle = compute_fused_vector(self.model, self.sliding_window)
                    if self.device.type == 'cuda': torch.cuda.synchronize()
                    self.metrics_mamba_times.append((time.time() - t0) * 1000)
                    if len(self.memory_bank.anchor_bank) < self.memory_bank.max_anchor:
                        self.memory_bank.add_anchor(bundle.visual_mean, bundle.fused_feat, bundle.temporal_token,
                                                    bundle.visual_plain, bundle.raw_feat)
                    else:
                        self.memory_bank.add_recent(bundle.visual_mean, bundle.fused_feat, bundle.temporal_token,
                                                    bundle.visual_plain, bundle.raw_feat)
                    print(f"[{frame_idx}] Last-moment Memory Bank update before LOST. Bank: {self.memory_bank.size_info()}")
                self._transition_to_lost(frame_idx)
                return
                
            crop = crop_and_pad(frame, bbox, self.bbox_padding)
            if crop is not None and self.sliding_window.should_extract():
                sharpness = compute_sharpness(crop)
                tensor_frame = transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
                if self.device.type == 'cuda': torch.cuda.synchronize()
                t0 = time.time()
                feat_2560 = extract_cnn_feature(self.model, tensor_frame)
                if self.device.type == 'cuda': torch.cuda.synchronize()
                self.metrics_cnn_times.append((time.time() - t0) * 1000)
                self.sliding_window.add(feat_2560, sharpness)
                
            time_elapsed = current_time - self.last_update_time
            if self.sliding_window.is_ready() and time_elapsed >= self.update_interval_sec:
                if self.device.type == 'cuda': torch.cuda.synchronize()
                t0 = time.time()
                bundle = compute_fused_vector(self.model, self.sliding_window)
                if self.device.type == 'cuda': torch.cuda.synchronize()
                self.metrics_mamba_times.append((time.time() - t0) * 1000)
                
                # Anti-Hijack: so sánh với bank CŨ trước khi thêm vector hiện tại vào bank.
                # 🛠️ (14/9): gate này giữ NGUYÊN trên không gian `fused` + `hijack_threshold` riêng
                # (đây là câu hỏi "còn đúng vật thể không?", khác gate HARD LOCK), nên `fine_space`
                # KHÔNG ảnh hưởng tới nó.
                if self.state == self.T3_VERIFIED and self._hijack_checks_remaining > 0:
                    hijack_score = self.memory_bank.fine_score(bundle.fused_feat)
                    self._hijack_checks_remaining -= 1
                    print(f"[{frame_idx}] Anti-Hijack check #{self.hijack_check_count - self._hijack_checks_remaining}: score={hijack_score:.3f}")
                    self._log_sim_breakdown(frame_idx, bundle, hijack_score, tag="anti-hijack")
                    if hijack_score < self.hijack_threshold:
                        print(f"[{frame_idx}] WARNING: HIJACK DETECTED! -> T1_LOST")
                        self._transition_to_lost(frame_idx)
                        return
                
                if len(self.memory_bank.anchor_bank) < self.memory_bank.max_anchor:
                    self.memory_bank.add_anchor(bundle.visual_mean, bundle.fused_feat, bundle.temporal_token,
                                                bundle.visual_plain, bundle.raw_feat)
                    print(f"[{frame_idx}] Anchor updated. {self.memory_bank.size_info()}")
                else:
                    self.memory_bank.add_recent(bundle.visual_mean, bundle.fused_feat, bundle.temporal_token,
                                                bundle.visual_plain, bundle.raw_feat)
                    print(f"[{frame_idx}] Recent updated. {self.memory_bank.size_info()}")
                
                self.last_update_time = current_time

        elif self.state == self.T1_LOST:
            if not is_absent and valid_bbox:
                print(f"[{frame_idx}] UAV reappeared from GT. -> T2_SEARCH")
                self.state = self.T2_SEARCH
                self.reappeared_frame_idx = frame_idx
                self.soft_lock_buffer.clear()
                self.hard_lock_buffer.clear()
                self._soft_lock_announced = False
                self._soft_lock_passed = False
                self._absent_streak = 0
                
        elif self.state == self.T2_SEARCH:
            if is_absent or not valid_bbox:
                # 🛠️ (22/9) NỚI RESET (md/22thg9.md §4, §8.3): vắng NGẮN thì bỏ qua frame,
                # GIỮ nguyên feature đã thu; chỉ vắng LIÊN TIẾP > `gap_tolerance` mới reset.
                self._absent_streak += 1
                if self._absent_streak <= self.gap_tolerance:
                    print(f"[{frame_idx}] T2_SEARCH: vang {self._absent_streak}/{self.gap_tolerance} frame"
                          f" -> BO QUA, giu {len(self.soft_lock_buffer.features)}/{self.soft_lock_capacity} feature")
                    return
                print(f"[{frame_idx}] UAV lost during T2_SEARCH "
                      f"(vang {self._absent_streak} > {self.gap_tolerance}). -> T1_LOST")
                self._transition_to_lost(frame_idx)
                return

            self._absent_streak = 0
                
            crop = crop_and_pad(frame, bbox, self.bbox_padding)
            if crop is not None:
                sharpness = compute_sharpness(crop)
                tensor_frame = transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
                if self.device.type == 'cuda': torch.cuda.synchronize()
                t0 = time.time()
                feat_2560 = extract_cnn_feature(self.model, tensor_frame)
                if self.device.type == 'cuda': torch.cuda.synchronize()
                self.metrics_cnn_times.append((time.time() - t0) * 1000)
                
                # Nếu đang trong quá trình thu thập Soft Lock, tiếp tục thu thập vô điều kiện
                if len(self.soft_lock_buffer.features) > 0:
                    self.soft_lock_buffer.add(feat_2560, sharpness)
                    print(f"[{frame_idx}] Soft Lock collecting: {len(self.soft_lock_buffer.features)}/{self.num_frames}")
                else:
                    if self.memory_bank.is_empty():
                        # Chưa có target identity (sequence bắt đầu bằng absent): bbox từ GT chính là target
                        coarse_score = 1.0
                    else:
                        coarse_score = self.memory_bank.coarse_score(feat_2560)
                    if coarse_score >= self.soft_lock_threshold:
                        self.soft_lock_buffer.add(feat_2560, sharpness)
                        print(f"[{frame_idx}] Soft Lock collecting: 1/{self.num_frames} (coarse={coarse_score:.3f})")
                    else:
                        print(f"[{frame_idx}] Coarse FAILED! (coarse={coarse_score:.3f} < {self.soft_lock_threshold})")
                
                # (1) SOFT LOCK — đủ `num_frames` frame LIÊN TỤC -> TÍNH ĐIỂM.
                #     Soft lock KHÔNG quyết định danh tính. Nó chỉ là CỔNG CHẶN: điểm cao
                #     quá ngưỡng thì mới cho phép tính HARD LOCK; không cao thì quay lại
                #     T1_LOST và tính soft lock lại từ đầu.
                #     Điểm = cosine COARSE (visual-only, `coarse_score`) giữa TRUNG BÌNH
                #     `num_frames` frame liên tục và Memory Bank. Chọn coarse vì (a) nó chỉ
                #     dùng `feat_2560` nên KHÔNG dính lỗi temporal theo N (đã đo: visual_plain
                #     0.986–0.994 ở MỌI N), (b) lấy trung bình N frame nên chống nhiễu per-frame.
                if self.soft_lock_buffer.is_ready() and not self._soft_lock_announced:
                    self._soft_lock_announced = True
                    mean_visual = torch.stack(list(self.soft_lock_buffer.features)).mean(dim=0)
                    soft_score = self.memory_bank.coarse_score(mean_visual)
                    if soft_score >= self.soft_lock_threshold:
                        self._soft_lock_passed = True
                        print(f"[{frame_idx}] SOFT LOCK PASS (soft={soft_score:.3f} >= "
                              f"{self.soft_lock_threshold}) -> MOI duoc tinh HARD LOCK")
                    else:
                        print(f"[{frame_idx}] SOFT LOCK FAIL (soft={soft_score:.3f} < "
                              f"{self.soft_lock_threshold}) -> quay lai T1_LOST, tinh lai soft lock")
                        self._transition_to_lost(frame_idx)
                        return

                # (2) HARD LOCK — cửa sổ `num_frames` MẪU cách nhau `frame_stride`.
                #     ⚠️ CHỈ BẮT ĐẦU THU sau khi soft lock đã PASS. KHÔNG thu song song trước đó:
                #     khi chưa có điểm soft lock thì CHƯA BIẾT phải thu frame của MỤC TIÊU NÀO.
                #     Data hiện tại chỉ có 1 mục tiêu nên thu sớm cũng "đúng", nhưng pipeline sẽ
                #     SAI khi có nhiều mục tiêu -> thu TUẦN TỰ cho đúng pipeline.
                #     Hệ quả thời gian: t_hard_lock = N + (N−1)×stride ≈ 12 + 44 = 56 frame.
                _new_hard_sample = False
                if self._soft_lock_passed:
                    _new_hard_sample = self.hard_lock_buffer.should_extract()
                    if _new_hard_sample:
                        self.hard_lock_buffer.add(feat_2560, sharpness)

                # Đủ `num_frames` mẫu cách quãng VÀ soft lock đã pass -> chạy Lọc Tinh
                if self._soft_lock_passed and _new_hard_sample and self.hard_lock_buffer.is_ready():
                    if self.device.type == 'cuda': torch.cuda.synchronize()
                    t0 = time.time()
                    # Cửa sổ đã cách quãng sẵn (`should_extract`) -> đưa vào head với stride=1.
                    bundle = compute_fused_vector(self.model, self.hard_lock_buffer)
                    if self.device.type == 'cuda': torch.cuda.synchronize()
                    mamba_time = (time.time() - t0) * 1000
                    self.metrics_mamba_times.append(mamba_time)
                    
                    if self.memory_bank.is_empty():
                        fine_score = 1.0
                    else:
                        # 🛠️ (14/9): `fine_space` chọn không gian cho gate HARD LOCK.
                        #   'fused'  (mặc định, hành vi cũ) : qua ReIDHead (BatchNorm1d)
                        #   'pre_bn'                        : cat(visual, temporal) — ĐẦU VÀO bnneck
                        # ĐỔI SANG 'pre_bn' THÌ PHẢI ĐỔI LUÔN `reid_threshold` = threshold calibrate
                        # cho không gian đó (`calibrated_threshold.json` -> thresholds.pre_bn),
                        # vì hai không gian có thang điểm khác hẳn nhau.
                        # Lưu ý: cột debug POST-BN luôn phải là fused THẬT, nên tính riêng.
                        fused_score = self.memory_bank.fine_score(bundle.fused_feat)
                        if self.fine_space == 'pre_bn':
                            fine_score = self.memory_bank.raw_score(bundle.raw_feat)
                        else:
                            fine_score = fused_score
                        # 🛠️ DEBUG (14/9): breakdown đầy đủ, đặc biệt là PRE-BN raw vs POST-BN fused
                        self._log_sim_breakdown(frame_idx, bundle, fused_score, tag="re-acquire")
                    if fine_score >= self.reid_threshold:
                        latency = frame_idx - self.reappeared_frame_idx
                        self.reid_latency_frames.append(latency)
                        print(f"[{frame_idx}] HARD LOCK! (fine={fine_score:.3f} >= {self.reid_threshold}) Latency: {latency} frames")
                        self.state = self.T3_VERIFIED
                        self._hijack_checks_remaining = self.hijack_check_count
                        self.last_update_time = time.time()
                        
                        if len(self.memory_bank.anchor_bank) < self.memory_bank.max_anchor:
                            self.memory_bank.add_anchor(bundle.visual_mean, bundle.fused_feat, bundle.temporal_token,
                                                        bundle.visual_plain, bundle.raw_feat)
                        else:
                            self.memory_bank.add_recent(bundle.visual_mean, bundle.fused_feat, bundle.temporal_token,
                                                        bundle.visual_plain, bundle.raw_feat)
                            
                        # 🛠️ (22/9) Nạp `sliding_window` từ ĐÚNG cửa sổ HARD LOCK vừa dùng
                        # (`hard_lock_buffer` đã cách quãng sẵn, không cần `[::stride]`).
                        self.sliding_window = SlidingWindowBuffer(self.num_frames, self.stride)
                        for _f, _s in zip(self.hard_lock_buffer.features,
                                          self.hard_lock_buffer.sharpness_scores):
                            self.sliding_window.add(_f, _s)
                        self.soft_lock_buffer = SlidingWindowBuffer(self.num_frames, stride=1)
                        self.hard_lock_buffer = SlidingWindowBuffer(self.num_frames, self.stride)
                        self._soft_lock_announced = False
                        self._soft_lock_passed = False
                    else:
                        self.false_alarms += 1
                        print(f"[{frame_idx}] Fine FAILED! (fine={fine_score:.3f} < {self.reid_threshold}) -> Rolling Window...")
                        # 🛠️ (22/9) Rolling window TỰ ĐỘNG: `hard_lock_buffer` được nuôi bằng
                        # `should_extract()` nên cứ mỗi `frame_stride` frame nó nhận 1 mẫu mới và
                        # đẩy mẫu CŨ NHẤT ra → cửa sổ kế tiếp lệch đúng 1 mẫu (= frame_stride frame).
                        # Không cần pop tay (pop tay là cách của buffer thu liên tục trước đây).
                        pass

    def draw_ui(self, display_frame, bbox, frame_idx):
        color = (0, 0, 255)
        text = "LOST"
        if self.state in [self.T0_INIT, self.T3_VERIFIED]:
            color = (0, 255, 0)
            text = "TRACKING (HARD LOCK)"
        elif self.state == self.T2_SEARCH:
            color = (255, 255, 0)
            text = (f"SEARCHING (soft {len(self.soft_lock_buffer.features)}/{self.num_frames}"
                    f" | hard {len(self.hard_lock_buffer.features)}/{self.num_frames})")
            
        cv2.putText(display_frame, f"State: {self.state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display_frame, f"Bank: {self.memory_bank.size_info()}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if bbox[2] > 0 and bbox[3] > 0 and self.state != self.T1_LOST:
            x, y, bw, bh = bbox
            cv2.rectangle(display_frame, (x, y), (x+bw, y+bh), color, 2)
            cv2.putText(display_frame, text, (x, max(0, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def format_bn_debug_lines(pre_scores, post_scores, thr, tag_counts=None, n_false_alarms=None):
    """
    🛠️ (14/9): dựng khối báo cáo PRE-BN vs POST-BN — DÙNG CHUNG cho cả khối per-sequence
    (`run_sequence`) lẫn khối `=== AGGREGATED METRICS ===` (`main`), để hai nơi không lệch nhau.

    Trả về [] nếu chưa có dữ liệu debug.

    Cách đọc: raw cao (>=0.8) mà fused thấp hơn `thr` -> bnneck là mắt xích đang chặn HARD LOCK
              (nhưng CHƯA nói được BN có hại hay không — cần TAR@FAR).
              raw thấp sẵn (~ temporal)        -> vấn đề nằm ở feature (temporal/dữ liệu train).

    tag_counts / n_false_alarms (tuỳ chọn): tách cửa sổ re-acquire (T2_SEARCH, chính là gate
    HARD LOCK) khỏi kiểm tra anti-hijack (T3_VERIFIED, gate khác, ngưỡng khác) — nếu không tách
    thì "tỉ lệ vượt ngưỡng" trộn hai thứ khác bản chất.
    """
    if pre_scores is None or post_scores is None or len(pre_scores) == 0 or len(post_scores) == 0:
        return []
    pre = np.asarray(pre_scores, dtype=np.float64)
    post = np.asarray(post_scores, dtype=np.float64)
    pre_mean, post_mean = float(pre.mean()), float(post.mean())
    delta = pre_mean - post_mean

    if pre_mean < 0.70:
        verdict = "raw cũng thấp sẵn -> lỗi nằm ở feature (temporal/dữ liệu train), KHÔNG phải BN"
    elif delta < 0.05:
        verdict = "BN gần như không ảnh hưởng cosine"
    elif pre_mean >= thr > post_mean:
        verdict = (f"BN nén cosine xuống dưới ngưỡng {thr:.2f} -> BN là mắt xích đang chặn HARD LOCK. "
                   f"LƯU Ý: raw cao hơn KHÔNG chứng minh phân biệt tốt hơn (impostor cũng cao hơn); "
                   f"cần TAR@FAR: calibrate_threshold.py rồi evaluate_reid.py --space pre_bn")
    else:
        side = "trên" if post_mean >= thr else "dưới"
        verdict = (f"BN nén cosine {delta:.3f}, nhưng cả raw lẫn fused đều đang {side} ngưỡng {thr:.2f} "
                   f"-> BN không phải nút thắt của sequence này")

    pass_pre = float(np.mean(pre >= thr)) * 100.0
    pass_post = float(np.mean(post >= thr)) * 100.0
    lines = [
        f"Sim PRE-BN  (raw concat)   : {pre_mean:.3f} (n={len(pre)})",
        f"Sim POST-BN (fused/fine)   : {post_mean:.3f}",
        f"BN degradation (pre - post) : {delta:+.3f} -> {verdict}",
        f"Cua so vuot nguong {thr:.2f}    : raw {pass_pre:.1f}%  |  fused {pass_post:.1f}%"
        f"   (toan bo {len(pre)} cua so; KHONG phai FAR: chua co nhan genuine/impostor)",
    ]
    if tag_counts:
        n_re = int(tag_counts.get('re-acquire', 0))
        n_ah = int(tag_counts.get('anti-hijack', 0))
        extra = f"Phan bo theo tag           : re-acquire {n_re} | anti-hijack {n_ah}"
        if n_false_alarms is not None and n_re > 0:
            n_lock = max(0, n_re - int(n_false_alarms))
            extra += (f"   -> HARD LOCK {n_lock}/{n_re} = {100.0 * n_lock / n_re:.1f}% "
                      f"(chi tinh cua so re-acquire)")
        lines.append(extra)
    return lines


def run_sequence(seq_dir, model, device, transform, cfg, inf_cfg, out_base=None):
    seq_name = os.path.basename(os.path.normpath(seq_dir))
    video_path = os.path.join(seq_dir, f"{seq_name}.mp4")
    gt_path = os.path.join(seq_dir, "groundtruth_rect.txt")
    absent_path = os.path.join(seq_dir, "absent.txt")
    
    if out_base:
        out_dir = os.path.join(out_base, seq_name)
    else:
        out_dir = inf_cfg.get('out_dir', f"infer_output/{seq_name}")
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "config_used.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    
    metrics_path = os.path.join(out_dir, "metrics.txt")
    metrics_file = open(metrics_path, "w")
    _orig_print = builtins.print
    def custom_print(*args_p, **kwargs_p):
        msg = " ".join(str(a) for a in args_p)
        _orig_print(msg, **kwargs_p)
        if not metrics_file.closed:
            metrics_file.write(msg + "\n")
            metrics_file.flush()
    builtins.print = custom_print

    output_video_name = inf_cfg.get('output_video', 'output.mp4')
    final_output_path = os.path.join(out_dir, os.path.basename(output_video_name))
    
    # Removed incorrect model initialization
    
    bboxes = []
    if os.path.exists(gt_path):
        with open(gt_path, "r") as f:
            for line in f:
                parts = line.strip().replace(',', ' ').split()
                if len(parts) >= 4:
                    bboxes.append([int(float(p)) for p in parts[:4]])
                else:
                    bboxes.append([0, 0, 0, 0])
                    
    absent = []
    if os.path.exists(absent_path):
        with open(absent_path, "r") as f:
            absent = [int(line.strip()) for line in f if line.strip().isdigit()]
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        metrics_file.close()
        builtins.print = _orig_print
        return None

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(final_output_path, fourcc, fps_video, (width, height))
    
    pipeline = SeqReIDPipeline(model, device, inf_cfg)
    
    frame_idx = 0
    print(f"Starting OOP Sequence Inference Stream for {seq_name}...")
    
    total_processing_time = 0.0
    
    # Cảnh báo nếu absent.txt bị cắt ngắn — trước đây mặc định True (coi là "mất") 
    # làm pipeline KHÔNG BAO GIỜ re-acquire → latency N/A âm thầm.
    if absent and len(absent) < len(bboxes):
        print(f" ⚠️ CẢNH BÁO: absent.txt có {len(absent)} dòng < GT {len(bboxes)} frame. "
              f"Các frame thiếu sẽ được coi là PRESENT (is_absent=False) để pipeline có thể re-acquire.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        # Mặc định is_absent = False (present) khi absent.txt thiếu dòng.
        # Trước đây là True (absent) → target bị coi là mất vĩnh viễn ở các frame bị cắt → kết quả tệ âm thầm.
        is_absent = (absent[frame_idx] == 1) if frame_idx < len(absent) else False
        bbox = bboxes[frame_idx] if frame_idx < len(bboxes) else [0,0,0,0]
        
        t_start = time.time()
        display_frame = frame.copy()
        
        pipeline.process_frame(frame, bbox, is_absent, frame_idx, transform)
        pipeline.draw_ui(display_frame, bbox, frame_idx)
        
        out_vid.write(display_frame)
        if device.type == 'cuda': torch.cuda.synchronize()
        total_processing_time += time.time() - t_start
        
        frame_idx += 1

    cap.release()
    out_vid.release()
    print("Inference completed!")
    
    # Generate Performance Metrics
    metrics_report = ["\n--- PERFORMANCE METRICS ---"]
    avg_cnn = 0.0
    avg_mamba = 0.0
    throughput = 0.0
    
    if pipeline.metrics_cnn_times:
        avg_cnn = np.mean(pipeline.metrics_cnn_times)
        metrics_report.append(f"Avg CNN Feature Extraction : {avg_cnn:.2f} ms")
        
    throughput = frame_idx / total_processing_time if total_processing_time > 0 else 0.0
    metrics_report.append(f"Avg System Throughput      : {throughput:.2f} FPS")
        
    if pipeline.metrics_mamba_times:
        avg_mamba = np.mean(pipeline.metrics_mamba_times)
        metrics_report.append(f"Avg Mamba + Head Time      : {avg_mamba:.2f} ms")
        
    if pipeline.reid_latency_frames:
        avg_lat = np.mean(pipeline.reid_latency_frames)
        metrics_report.append(f"Re-acquisition Latency     : {avg_lat:.2f} frames")
    else:
        metrics_report.append("Re-acquisition Latency     : N/A")
        
    metrics_report.append(f"False Alarms (Fine Fails)  : {pipeline.false_alarms}")
    
    # 🛠️ DEBUG (14/9): tổng hợp PRE-BN vs POST-BN để trả lời câu hỏi
    # "BatchNorm1d trong ReIDHead có phá cosine similarity không?" — dùng chung helper với main().
    metrics_report.extend(format_bn_debug_lines(
        pipeline.debug_pre_bn_scores, pipeline.debug_post_bn_scores, pipeline.reid_threshold,
        tag_counts=pipeline.debug_tag_counts, n_false_alarms=pipeline.false_alarms))
    
    print("\n".join(metrics_report))
    metrics_file.close()
    builtins.print = _orig_print
    
    mean_latency = np.mean(pipeline.reid_latency_frames) if pipeline.reid_latency_frames else -1.0
    return (avg_cnn, avg_mamba, throughput, mean_latency, pipeline.false_alarms,
            pipeline.debug_pre_bn_scores, pipeline.debug_post_bn_scores,
            pipeline.debug_tag_counts)

def main():
    args = parse_args()
    cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
            
    inf_cfg = cfg.get('infer', {})

    # 🛠️ (15/9) ĐỒNG BỘ frame_stride — NGUỒN DUY NHẤT là `data_pipeline.frame_stride`.
    # Lý do: bước thời gian giữa 2 frame liên tiếp trong cửa sổ temporal phải GIỐNG NHAU
    # ở data → train → infer. Nếu infer lấy dày hơn (stride nhỏ hơn frame_stride) thì
    # temporal token lệch phân phối so với lúc train → fine score tụt dù `visual` vẫn khớp.
    # (xem md/15thg9.md §13)
    _dp_cfg = cfg.get('data_pipeline', {}) or {}
    _frame_stride = _dp_cfg.get('frame_stride')
    if _frame_stride is not None:
        _old_stride = inf_cfg.get('stride')
        if _old_stride is not None and _old_stride != _frame_stride:
            print(f"⚠️  infer.stride={_old_stride} != data_pipeline.frame_stride={_frame_stride}"
                  f" → DÙNG frame_stride={_frame_stride} (đồng bộ toàn pipeline).")
        inf_cfg['stride'] = _frame_stride
        print(f"🔗 frame_stride đồng bộ = {_frame_stride} (lấy từ data_pipeline.frame_stride)")

    seq_dir_arg = args.seq_dir or inf_cfg.get('seq_dir')
    if not seq_dir_arg:
        print("Error: --seq-dir must be provided.")
        return

    print(f"Initializing Mamba ReID Model...")
    backbone_type = inf_cfg.get('backbone', 'resnet50_ibn')
    model = UAVReIDNet(backbone=backbone_type)
    model_path = args.checkpoint or inf_cfg.get('model_path', './best_model.pth')
    if os.path.exists(model_path):
        # 🛠️ (14/9): báo cáo đầy đủ missing/unexpected/shape-mismatch thay vì "Loaded" mù quáng.
        # Cảnh báo nghiêm trọng nếu `backbone.*` không được nạp (visual branch chạy pretrain,
        # trong khi temporal/head được train trên feature khác → mọi score đều đáng ngờ).
        load_checkpoint_verbose(model, model_path, tag="infer")
        print("Model loaded successfully.")
    else:
        print(f"Warning: Checkpoint {model_path} not found. Running with random weights.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if seq_dir_arg.lower() == "all":
        base_test_dir = "./data/UAV-Anti-UAV/Test"
        all_dirs = [os.path.join(base_test_dir, d) for d in sorted(os.listdir(base_test_dir)) if os.path.isdir(os.path.join(base_test_dir, d))]
        valid_seqs = []
        for d in all_dirs:
            absent_path = os.path.join(d, "absent.txt")
            if os.path.exists(absent_path):
                with open(absent_path, "r") as f:
                    absent = [int(line.strip()) for line in f if line.strip().isdigit()]
                if 1 in absent:
                    valid_seqs.append(d)
        print(f"Found {len(valid_seqs)} sequences with disappearance events.")
        
        all_cnn = []
        all_mamba = []
        all_throughput = []
        all_latency = []
        all_false_alarms = []
        all_pre_bn = []
        all_post_bn = []
        all_tag_counts = {'re-acquire': 0, 'anti-hijack': 0}
        
        base_out_dir = args.out_dir or inf_cfg.get('out_dir', './infer_output')
        print(f"Batch processing: Results will be saved in base directory: {base_out_dir}")
        for sdir in valid_seqs:
            res = run_sequence(sdir, model, device, transform, cfg, inf_cfg, out_base=base_out_dir)
            if res:
                c, m, t, l, f, pre_bn, post_bn, tag_counts = res
                all_cnn.append(c)
                all_mamba.append(m)
                all_throughput.append(t)
                if l >= 0:
                    all_latency.append(l)
                all_false_alarms.append(f)
                all_pre_bn.extend(pre_bn)
                all_post_bn.extend(post_bn)
                for _k, _v in (tag_counts or {}).items():
                    all_tag_counts[_k] = all_tag_counts.get(_k, 0) + _v
                
        # Calculate averages
        avg_cnn = np.mean(all_cnn) if all_cnn else 0.0
        avg_mamba = np.mean(all_mamba) if all_mamba else 0.0
        avg_throughput = np.mean(all_throughput) if all_throughput else 0.0
        avg_latency = np.mean(all_latency) if all_latency else 0.0
        sum_false_alarms = int(np.sum(all_false_alarms)) if all_false_alarms else 0
        
        # 🛠️ (14/9): dùng CHUNG helper với run_sequence -> verdict + tỉ lệ vượt ngưỡng cũng in ở đây.
        bn_lines = format_bn_debug_lines(
            all_pre_bn, all_post_bn, inf_cfg.get('reid_threshold', 0.75),
            tag_counts=all_tag_counts, n_false_alarms=sum_false_alarms)
        
        print("\n=== AGGREGATED METRICS ===")
        print(f"Avg CNN Feature Extraction : {avg_cnn:.2f} ms")
        print(f"Avg System Throughput      : {avg_throughput:.2f} FPS")
        print(f"Avg Mamba + Head Time      : {avg_mamba:.2f} ms")
        print(f"Avg Re-acquisition Latency : {avg_latency:.2f} frames")
        print(f"Total False Alarms         : {sum_false_alarms}")
        for line in bn_lines:
            print(line)
        
        # Save to summary text file
        os.makedirs(base_out_dir, exist_ok=True)
        with open(os.path.join(base_out_dir, "summary_metrics.txt"), "w") as sf:
            sf.write("=== AGGREGATED METRICS ===\n")
            sf.write(f"Avg CNN Feature Extraction : {avg_cnn:.2f} ms\n")
            sf.write(f"Avg System Throughput      : {avg_throughput:.2f} FPS\n")
            sf.write(f"Avg Mamba + Head Time      : {avg_mamba:.2f} ms\n")
            sf.write(f"Avg Re-acquisition Latency : {avg_latency:.2f} frames\n")
            sf.write(f"Total False Alarms         : {sum_false_alarms}\n")
            for line in bn_lines:
                sf.write(line + "\n")
            
    else:
        run_sequence(seq_dir_arg, model, device, transform, cfg, inf_cfg, out_base=args.out_dir)

if __name__ == "__main__":
    main()
