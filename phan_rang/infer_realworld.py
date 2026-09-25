import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import time
import argparse
import builtins

from torchvision import transforms
from model import UAVReIDNet
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Real-world Inference with Multi-UAV Coarse-to-Fine ReID")
    parser.add_argument('--config', type=str, default='configs/config_jetson.yaml')
    return parser.parse_args()

def extract_cnn_feature(model, tensor_frame):
    """Trích xuất 2560-dim feature tĩnh (GASNet)."""
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

def compute_reid_embedding(model, seq_feats):
    """Trích xuất 3072-dim fused feature (Visual + Mamba Temporal)."""
    with torch.no_grad():
        visual_feat = seq_feats.mean(dim=1)
        temporal_token, _ = model.temporal_encoder(seq_feats)
        bn_feat = model.head(visual_feat, temporal_token)
        bn_feat = F.normalize(bn_feat, p=2, dim=1)
    return bn_feat

def compute_sharpness(crop_bgr: np.ndarray) -> float:
    """Tính điểm độ nét của ảnh crop bằng Laplacian variance."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def crop_and_pad(frame, box, bbox_padding):
    """Crop và pad bounding box từ frame, trả về (crop_bgr, center_xy)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    bw, bh = x2 - x1, y2 - y1
    center = (x1 + bw / 2.0, y1 + bh / 2.0)
    
    pad_w, pad_h = int(bw * bbox_padding), int(bh * bbox_padding)
    x1_p = max(0, x1 - pad_w)
    y1_p = max(0, y1 - pad_h)
    x2_p = min(w, x2 + pad_w)
    y2_p = min(h, y2 + pad_h)
    
    crop = frame[y1_p:y2_p, x1_p:x2_p]
    return crop, center

def find_target_box(filtered_boxes, target_id):
    """Tìm bounding box của target trong danh sách detection."""
    for box in filtered_boxes:
        if box.id is not None and int(box.id[0]) == target_id:
            return box
    return None

class SlidingWindowBuffer:
    """Cửa sổ trượt quản lý k frame features với stride sampling."""
    def __init__(self, window_size: int = 16, stride: int = 2):
        self.window_size = window_size
        self.stride = stride
        self.features = []
        self.sharpness_scores = []
        self._frame_counter = 0
    
    def should_extract(self) -> bool:
        """Kiểm tra frame hiện tại có nên chạy GASNet không (dựa trên stride)."""
        result = (self._frame_counter % self.stride == 0)
        self._frame_counter += 1
        return result
    
    def add(self, feat: torch.Tensor, sharpness: float):
        """Thêm feature vào cửa sổ."""
        self.features.append(feat)
        self.sharpness_scores.append(sharpness)
        if len(self.features) > self.window_size:
            self.features.pop(0)
            self.sharpness_scores.pop(0)
    
    def is_ready(self) -> bool:
        """Đã đủ k frame chưa?"""
        return len(self.features) >= self.window_size
    
    def get_sequence(self) -> torch.Tensor:
        """Trả về tensor [1, k, 2560] để đưa vào Mamba."""
        return torch.stack(self.features, dim=1)

    # 🛠️ (15/9) PHÂN VAI stride: SOFT LOCK thu LIÊN TỤC (stride=1) để phản ứng nhanh;
    # HARD LOCK mới LẤY CÁCH QUÃNG (`stride = frame_stride`) để bước thời gian khớp
    # lúc train và khớp Memory Bank.
    def get_strided_sequence(self, stride: int = 1) -> torch.Tensor:
        """Trả về tensor [1, k, 2560] với bước thời gian `stride`."""
        return torch.stack(self.features[::stride], dim=1)
    
    def get_weighted_visual_mean(self, stride: int = 1) -> torch.Tensor:
        """Tính trung bình có trọng số theo sharpness -> vector 2560-dim."""
        feats = self.features[::stride]
        weights = torch.tensor(self.sharpness_scores[::stride], dtype=torch.float32)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = torch.ones_like(weights) / len(weights)
        
        stacked = torch.stack([f.squeeze(0) for f in feats])  # [k, 2560]
        return (stacked * weights.unsqueeze(1).to(stacked.device)).sum(dim=0, keepdim=True)  # [1, 2560]
    
    def clear(self):
        self.features.clear()
        self.sharpness_scores.clear()
        self._frame_counter = 0

def compute_fused_vector(model, sliding_window: SlidingWindowBuffer, stride: int = 1) -> tuple:
    """Từ sliding window đầy đủ k frames, tính ra cặp (visual_2560, fused_3072).

    🛠️ (15/9) `stride`: HARD LOCK truyền `frame_stride` (lấy cách quãng), SOFT LOCK để 1.
    """
    visual_mean = sliding_window.get_weighted_visual_mean(stride)  # [1, 2560]
    seq = sliding_window.get_strided_sequence(stride)  # [1, k, 2560]
    fused_feat = compute_reid_embedding(model, seq)  # [1, 3072]
    return visual_mean, fused_feat

class TwoTierMemoryBank:
    """Ngân Hàng Ký Ức 2 tầng: Anchor (bất biến) + Recent (cửa sổ trượt)."""
    def __init__(self, max_anchor: int = 10, max_recent: int = 30):
        self.max_anchor = max_anchor
        self.max_recent = max_recent
        self.anchor_bank = []
        self.recent_bank = []
    
    def add_anchor(self, visual_feat: torch.Tensor, fused_feat: torch.Tensor):
        if len(self.anchor_bank) < self.max_anchor:
            self.anchor_bank.append({
                "visual": F.normalize(visual_feat, p=2, dim=1),
                "fused": F.normalize(fused_feat, p=2, dim=1)
            })
    
    def add_recent(self, visual_feat: torch.Tensor, fused_feat: torch.Tensor):
        self.recent_bank.append({
            "visual": F.normalize(visual_feat, p=2, dim=1),
            "fused": F.normalize(fused_feat, p=2, dim=1)
        })
        if len(self.recent_bank) > self.max_recent:
            self.recent_bank.pop(0)
    
    def coarse_score(self, query_visual: torch.Tensor) -> float:
        query = F.normalize(query_visual, p=2, dim=1)
        max_sim = 0.0
        for entry in self.anchor_bank + self.recent_bank:
            sim = torch.mm(query, entry["visual"].t()).item()
            max_sim = max(max_sim, sim)
        return max_sim
    
    def fine_score(self, query_fused: torch.Tensor) -> float:
        query = F.normalize(query_fused, p=2, dim=1)
        max_sim = 0.0
        for entry in self.anchor_bank + self.recent_bank:
            sim = torch.mm(query, entry["fused"].t()).item()
            max_sim = max(max_sim, sim)
        return max_sim
    
    def is_empty(self) -> bool:
        return len(self.anchor_bank) == 0 and len(self.recent_bank) == 0
    
    def size_info(self) -> str:
        return f"Anchor: {len(self.anchor_bank)}/{self.max_anchor} | Recent: {len(self.recent_bank)}/{self.max_recent}"

class ReIDPipeline:
    """State Machine quản lý toàn bộ luồng ReID."""
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
        self.lost_threshold = cfg.get('lost_threshold', 10)
        self.bbox_padding = cfg.get('bbox_padding', 0.2)
        
        self.memory_bank = TwoTierMemoryBank(
            max_anchor=cfg.get('max_anchor_size', 10),
            max_recent=cfg.get('max_recent_size', 30)
        )
        self.sliding_window = SlidingWindowBuffer(
            window_size=self.num_frames,
            stride=self.stride
        )
        
        self.target_track_id = None
        self.original_target_id = None
        self.last_target_center = None
        self.lost_count = 0
        self.last_update_time = 0.0
        self._hijack_checks_remaining = 0
        
        self.soft_lock_id = None
        # 🛠️ (15/9) PHÂN VAI stride: soft_lock THU LIÊN TỤC (stride=1) để phản ứng nhanh,
        # HARD LOCK mới LẤY CÁCH QUÃNG `frame_stride` (khớp train + Memory Bank).
        # 🛠️ (22/9) HAI CỬA SỔ SONG SONG: soft lock = `num_frames` frame LIÊN TỤC (nhanh,
        # chỉ để chọn ứng viên / xem điểm), hard lock = `num_frames` MẪU cách nhau `frame_stride`
        # (chính xác, mới đem so Memory Bank). Xem infer.py để biết chi tiết.
        self.soft_lock_buffer = SlidingWindowBuffer(window_size=self.num_frames, stride=1)
        self.hard_lock_buffer = SlidingWindowBuffer(self.num_frames, self.stride)
        self._soft_lock_announced = False
        self._soft_lock_passed = False
        self.candidate_scores = {}

        # 🛠️ (22/9) NỚI RESET khi track soft-lock vắng mặt (md/22thg9.md §4, §8.3).
        # Cần `(num_frames-1)*stride+1` frame LIÊN TỤC; mất track 1 frame mà reset ngay thì
        # hầu hết event không bao giờ đủ frame. Vắng ngắn -> GIỮ id + buffer, chỉ vắng
        # LIÊN TIẾP > `gap_tolerance` mới reset thật.
        self.gap_tolerance = cfg.get('t2_search_gap_tolerance', 2)
        self._absent_streak = 0
        
    def _transition_to_lost(self):
        print(f"Target {self.target_track_id} LOST! -> T1_LOST")
        self.state = self.T1_LOST
        self.target_track_id = None
        self.lost_count = 0
        self.sliding_window.clear()
        self.soft_lock_id = None
        self.soft_lock_buffer.clear()
        self.hard_lock_buffer.clear()
        self._soft_lock_announced = False
        self._soft_lock_passed = False
        self.candidate_scores.clear()
        
    def process_tracking(self, frame, filtered_boxes, frame_idx, transform):
        target_box = find_target_box(filtered_boxes, self.target_track_id)
        if target_box is not None:
            self.lost_count = 0
            crop, center = crop_and_pad(frame, target_box, self.bbox_padding)
            self.last_target_center = center
            
            if crop.size > 0 and self.sliding_window.should_extract():
                sharpness = compute_sharpness(crop)
                tensor_frame = transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
                feat_2560 = extract_cnn_feature(self.model, tensor_frame)
                self.sliding_window.add(feat_2560, sharpness)
            
            current_time = time.time()
            time_elapsed = current_time - self.last_update_time
            
            if self.sliding_window.is_ready() and time_elapsed >= self.update_interval_sec:
                visual_mean, fused_feat = compute_fused_vector(self.model, self.sliding_window)
                
                if len(self.memory_bank.anchor_bank) < self.memory_bank.max_anchor:
                    self.memory_bank.add_anchor(visual_mean, fused_feat)
                    print(f"[{frame_idx}] Anchor updated. {self.memory_bank.size_info()}")
                else:
                    self.memory_bank.add_recent(visual_mean, fused_feat)
                    print(f"[{frame_idx}] Recent updated. {self.memory_bank.size_info()}")

                if self.state == self.T0_INIT:
                    if len(self.memory_bank.anchor_bank) >= 1:
                        self.state = self.T3_VERIFIED
                        self._hijack_checks_remaining = 0
                        print(f"[{frame_idx}] T0 -> T3_VERIFIED. Target locked.")
                elif self.state == self.T3_VERIFIED:
                    if self._hijack_checks_remaining > 0:
                        hijack_score = self.memory_bank.fine_score(fused_feat)
                        self._hijack_checks_remaining -= 1
                        print(f"[{frame_idx}] Anti-Hijack check #{self.hijack_check_count - self._hijack_checks_remaining}: score={hijack_score:.3f}")
                        if hijack_score < self.hijack_threshold:
                            print(f"[{frame_idx}] WARNING: HIJACK DETECTED! score={hijack_score:.3f} < {self.hijack_threshold}. -> T1_LOST")
                            self._transition_to_lost()
                            return
                
                self.last_update_time = current_time
        else:
            self.lost_count += 1
            if self.lost_count >= self.lost_threshold:
                if len(self.sliding_window.features) > 0:
                    # 🛠️ (15/9) BỎ pad NHÂN BẢN frame cuối (bước thời gian = 0) — xem infer.py
                        
                    visual_mean, fused_feat = compute_fused_vector(self.model, self.sliding_window)
                    if len(self.memory_bank.anchor_bank) < self.memory_bank.max_anchor:
                        self.memory_bank.add_anchor(visual_mean, fused_feat)
                    else:
                        self.memory_bank.add_recent(visual_mean, fused_feat)
                    print(f"[{frame_idx}] Last-moment Memory Bank update before LOST.")
                self._transition_to_lost()

    def process_lost(self, filtered_boxes, frame_idx):
        if len(filtered_boxes) > 0 and not self.memory_bank.is_empty():
            self.state = self.T2_SEARCH
            self.candidate_scores.clear()
            print(f"[{frame_idx}] UAV detected during LOST state. -> T2_SEARCH")

    def process_search(self, frame, filtered_boxes, frame_idx, transform):
        new_candidates = []
        tensor_frames = []
        
        for box in filtered_boxes:
            tid = int(box.id[0])
            if tid in self.candidate_scores:
                continue
            
            crop, center = crop_and_pad(frame, box, self.bbox_padding)
            if crop.size == 0:
                continue
            
            tensor_frame = transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).to(self.device)
            tensor_frames.append(tensor_frame)
            new_candidates.append(tid)
            
        if len(tensor_frames) > 0:
            t0 = time.time()
            batch_tensors = torch.stack(tensor_frames, dim=0)
            batch_feats = extract_cnn_feature(self.model, batch_tensors)
            batch_feats_norm = F.normalize(batch_feats, p=2, dim=1)
            
            for i, tid in enumerate(new_candidates):
                feat_norm_i = batch_feats_norm[i:i+1]
                coarse_score = self.memory_bank.coarse_score(feat_norm_i)
                self.candidate_scores[tid] = coarse_score
                print(f"[{frame_idx}] Coarse: ID:{tid} score={coarse_score:.3f}")
                if coarse_score < self.soft_lock_threshold:
                    print(f"[{frame_idx}]   -> LOAI (score < {self.soft_lock_threshold})")
            print(f"[{frame_idx}] BATCHED CNN Extraction for {len(tensor_frames)} candidates in {(time.time() - t0)*1000:.1f}ms")

        valid_candidates = {
            tid: score for tid, score in self.candidate_scores.items()
            if score >= self.soft_lock_threshold
        }
        
        if valid_candidates:
            best_tid = max(valid_candidates, key=valid_candidates.get)
            if self.soft_lock_id != best_tid:
                self.soft_lock_id = best_tid
                self.soft_lock_buffer.clear()
                # 🛠️ (22/9) BẮT BUỘC: cửa sổ HARD LOCK phải thuộc ĐÚNG ứng viên đang được
                # soft lock chọn. Đổi ứng viên mà không xoá -> cửa sổ temporal trộn frame của
                # HAI vật khác nhau -> điểm hard lock vô nghĩa (nó không còn là "điểm của thằng
                # có soft lock cao nhất" nữa).
                self.hard_lock_buffer.clear()
                self._soft_lock_announced = False
                self._soft_lock_passed = False
                print(f"[{frame_idx}] SOFT LOCK -> ID:{best_tid} (coarse={valid_candidates[best_tid]:.3f})"
                      f" [reset cua so HARD LOCK]")
        
        if self.soft_lock_id is not None:
            soft_lock_box = find_target_box(filtered_boxes, self.soft_lock_id)
            if soft_lock_box is not None:
                crop, center = crop_and_pad(frame, soft_lock_box, self.bbox_padding)
                if crop.size > 0:
                    sharpness = compute_sharpness(crop)
                    tensor_frame = transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
                    feat_2560 = extract_cnn_feature(self.model, tensor_frame)
                    
                    # Thu thập vô điều kiện - không kiểm tra lại Coarse
                    self._absent_streak = 0
                    if not self._soft_lock_announced:
                        self.soft_lock_buffer.add(feat_2560, sharpness)
                        print(f"[{frame_idx}] Soft Lock ID:{self.soft_lock_id} collecting: {len(self.soft_lock_buffer.features)}/{self.num_frames}")

                    # (1) SOFT LOCK — đủ `num_frames` frame LIÊN TỤC -> TÍNH ĐIỂM (CỔNG CHẶN).
                    #     Soft lock KHÔNG quyết định danh tính: điểm >= ngưỡng thì mới cho phép
                    #     tính HARD LOCK; < ngưỡng thì quay lại T1_LOST và tính soft lock lại.
                    #     Điểm = cosine COARSE (visual-only) giữa TRUNG BÌNH `num_frames` frame
                    #     liên tục và Memory Bank (coarse chỉ dùng `feat_2560` nên KHÔNG dính lỗi
                    #     temporal theo N; lấy trung bình N frame nên chống nhiễu per-frame).
                    if self.soft_lock_buffer.is_ready() and not self._soft_lock_announced:
                        self._soft_lock_announced = True
                        mean_visual = torch.stack(list(self.soft_lock_buffer.features)).mean(dim=0)
                        soft_score = self.memory_bank.coarse_score(mean_visual)
                        if soft_score >= self.soft_lock_threshold:
                            self._soft_lock_passed = True
                            print(f"[{frame_idx}] SOFT LOCK PASS ID:{self.soft_lock_id} (soft={soft_score:.3f} >= "
                                  f"{self.soft_lock_threshold}) -> MOI duoc tinh HARD LOCK")
                        else:
                            print(f"[{frame_idx}] SOFT LOCK FAIL ID:{self.soft_lock_id} (soft={soft_score:.3f} < "
                                  f"{self.soft_lock_threshold}) -> quay lai T1_LOST")
                            self._transition_to_lost()
                            return

                    # (2) HARD LOCK — ⚠️ CHỈ BẮT ĐẦU THU sau khi soft lock đã PASS. Không thu
                    # song song trước đó: chưa có điểm soft lock thì CHƯA BIẾT thu frame của
                    # MỤC TIÊU NÀO. Hệ quả: t_hard_lock ≈ N + (N−1)×stride = 56 frame.
                    _new_hard_sample = False
                    if self._soft_lock_passed:
                        _new_hard_sample = self.hard_lock_buffer.should_extract()
                        if _new_hard_sample:
                            self.hard_lock_buffer.add(feat_2560, sharpness)
                            print(f"[{frame_idx}] HARD LOCK ID:{self.soft_lock_id} collecting: "
                                  f"{len(self.hard_lock_buffer.features)}/{self.num_frames} "
                                  f"(moi {self.stride} frame)")

                    if self._soft_lock_passed and _new_hard_sample and self.hard_lock_buffer.is_ready():
                        # Cửa sổ đã cách quãng sẵn (`should_extract`) -> stride=1.
                        visual_mean, fused_feat = compute_fused_vector(self.model, self.hard_lock_buffer)
                        fine_score = self.memory_bank.fine_score(fused_feat)
                        print(f"[{frame_idx}] Fine ReID: ID:{self.soft_lock_id} score={fine_score:.3f}")
                        
                        if fine_score >= self.reid_threshold:
                            print(f"[{frame_idx}] HARD LOCK! ID:{self.soft_lock_id} (fine={fine_score:.3f} >= {self.reid_threshold})")
                            self.state = self.T3_VERIFIED
                            self.target_track_id = self.soft_lock_id
                            self.last_target_center = center
                            self._hijack_checks_remaining = self.hijack_check_count
                            self.last_update_time = time.time()
                            
                            self.memory_bank.add_recent(visual_mean, fused_feat)
                            
                            # 🛠️ (15/9) Nạp `sliding_window` từ ĐÚNG cửa sổ đã dùng để HARD LOCK
                            # (bản lấy cách quãng), KHÔNG phải toàn bộ buffer thu liên tục.
                            # 🛠️ (22/9) Nạp từ ĐÚNG cửa sổ HARD LOCK (đã cách quãng sẵn).
                            self.sliding_window = SlidingWindowBuffer(self.num_frames, self.stride)
                            for _f, _s in zip(self.hard_lock_buffer.features,
                                              self.hard_lock_buffer.sharpness_scores):
                                self.sliding_window.add(_f, _s)
                            
                            self.soft_lock_id = None
                            self.soft_lock_buffer = SlidingWindowBuffer(window_size=self.num_frames, stride=1)
                            self.hard_lock_buffer = SlidingWindowBuffer(self.num_frames, self.stride)
                            self._soft_lock_announced = False
                            self._soft_lock_passed = False
                            self.candidate_scores.clear()
                        else:
                            print(f"[{frame_idx}] Fine FAILED! ID:{self.soft_lock_id} (fine={fine_score:.3f} < {self.reid_threshold}) -> T1_LOST")
                            self._transition_to_lost()
            else:
                # 🛠️ (22/9) NỚI RESET (md/22thg9.md §4, §8.3): giữ id + feature đã thu.
                self._absent_streak += 1
                if self._absent_streak <= self.gap_tolerance:
                    print(f"[{frame_idx}] Soft Lock ID:{self.soft_lock_id} vang "
                          f"{self._absent_streak}/{self.gap_tolerance} frame -> BO QUA, "
                          f"giu soft {len(self.soft_lock_buffer.features)}/{self.num_frames}"
                          f" | hard {len(self.hard_lock_buffer.features)}/{self.num_frames} feature")
                else:
                    print(f"[{frame_idx}] Soft Lock ID:{self.soft_lock_id} lost from frame "
                          f"(vang {self._absent_streak} > {self.gap_tolerance}). Resetting.")
                    self.soft_lock_id = None
                    self.soft_lock_buffer.clear()
                    self.hard_lock_buffer.clear()
                    self._soft_lock_announced = False
                    self._soft_lock_passed = False
                    self._absent_streak = 0
        
        if not valid_candidates and self.soft_lock_id is None:
            if len(filtered_boxes) == 0:
                self.state = self.T1_LOST
                
        # Clean up candidates that are no longer in the frame
        active_ids = [int(box.id[0]) for box in filtered_boxes]
        lost_tids = [tid for tid in self.candidate_scores.keys() if tid not in active_ids]
        for ltid in lost_tids:
            del self.candidate_scores[ltid]
            # 🛠️ (22/9) KHÔNG clear soft-lock ở đây: việc reset do nhánh `else` phía trên
            # quyết định (có nới `gap_tolerance`). Xoá khỏi `candidate_scores` vẫn cần, để
            # khi track quay lại nó được CHẤM LẠI coarse score.
                
    def draw_ui(self, display_frame, filtered_boxes):
        status_color = (0, 0, 255) if self.state == self.T1_LOST else (0, 255, 0)
        cv2.putText(display_frame, f"State: {self.state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        if not self.memory_bank.is_empty():
            cv2.putText(display_frame, f"Bank: {self.memory_bank.size_info()}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for box in filtered_boxes:
            tid = int(box.id[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            b_x, b_y, b_w, b_h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            
            if self.state in [self.T0_INIT, self.T3_VERIFIED] and tid == self.target_track_id:
                cv2.rectangle(display_frame, (b_x, b_y), (b_x+b_w, b_y+b_h), (0, 255, 0), 2)
                display_id = self.original_target_id if self.original_target_id is not None else tid
                cv2.putText(display_frame, f"TARGET ID:{display_id}", (b_x, max(0, b_y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif self.state == self.T2_SEARCH:
                if tid == self.soft_lock_id:
                    cv2.rectangle(display_frame, (b_x, b_y), (b_x+b_w, b_y+b_h), (255, 255, 0), 2)
                    cv2.putText(display_frame, f"SOFT LOCK ID:{tid} ({len(self.soft_lock_buffer.features)}/{self.num_frames})", (b_x, max(0, b_y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                elif tid in self.candidate_scores and self.candidate_scores[tid] >= self.soft_lock_threshold:
                    cv2.rectangle(display_frame, (b_x, b_y), (b_x+b_w, b_y+b_h), (0, 255, 255), 2)
                    cv2.putText(display_frame, f"Wait ID:{tid}", (b_x, max(0, b_y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                else:
                    cv2.rectangle(display_frame, (b_x, b_y), (b_x+b_w, b_y+b_h), (128, 128, 128), 1)

def main():
    args = parse_args()
    cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
            
    inf_cfg = cfg.get('infer_realworld', {})
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    video_path = inf_cfg.get('video', '')
    video_name = os.path.basename(video_path).split('.')[0] if video_path else 'output'
    
    out_dir = inf_cfg.get('out_dir', f"infer_realworld/{video_name}")
    os.makedirs(out_dir, exist_ok=True)
    
    config_dump_path = os.path.join(out_dir, "config_used.yaml")
    with open(config_dump_path, "w") as f:
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
    globals()['print'] = custom_print

    output_video_name = inf_cfg.get('output_video', 'output_realworld.mp4')
    final_output_path = os.path.join(out_dir, os.path.basename(output_video_name))
    
    backbone_type = inf_cfg.get('backbone', 'resnet50_ibn')
    # 🛠️ (24/9) FALLBACK về `train.temporal_type` (giống `evaluate_reid.py`).
    # Section `infer_realworld` thường không khai báo key này -> trước đây luôn mặc định
    # 'mamba'. Với checkpoint ATTENTION thì `strict=False` bỏ hết `temporal_encoder.transformer.*`
    # -> temporal encoder chạy random, kết quả vô nghĩa mà không có cảnh báo nghiêm trọng.
    temporal_type = inf_cfg.get(
        'temporal_type', cfg.get('train', {}).get('temporal_type', 'mamba'))
    # 🛠️ (24/9) `temporal_pool`/`temporal_pe` phải khớp lúc TRAIN, nếu không `attn_pool`
    # giữ zero-init -> mất hệ số `sqrt(N)` (chi tiết: calibrate_threshold.py).
    temporal_pool = inf_cfg.get(
        'temporal_pool', cfg.get('train', {}).get('temporal_pool', 'attn'))
    temporal_pe = bool(inf_cfg.get(
        'temporal_pe', cfg.get('train', {}).get('temporal_pe', True)))
    print(f"Initializing ReID Model with {backbone_type} backbone, "
          f"temporal={temporal_type}, pool={temporal_pool}, pe={temporal_pe}...")
    model = UAVReIDNet(backbone=backbone_type, temporal_type=temporal_type,
                       temporal_pool=temporal_pool, temporal_pe=temporal_pe)
    model_path = inf_cfg.get('model_path', './best_model.pth')
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        new_state_dict = {}
        model_state = model.state_dict()
        for k, v in state_dict.items():
            new_k = k.replace('_orig_mod.', '') if k.startswith('_orig_mod.') else k
            if new_k in model_state and v.shape != model_state[new_k].shape:
                continue
            new_state_dict[new_k] = v
        model.load_state_dict(new_state_dict, strict=False)
        print("Model loaded successfully.")
    else:
        print(f"Warning: Checkpoint {model_path} not found. Running with random weights.")
    
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    yolo_model_path = inf_cfg.get('yolo_model', 'yolov8n.pt')
    print(f"Initializing YOLOv8 Detector: {yolo_model_path}")
    yolo_model = YOLO(yolo_model_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Video not found at {video_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    start_sec = float(inf_cfg.get('start_sec', 0.0))
    if start_sec > 0:
        print(f"Skipping to {start_sec} seconds...")
        cap.set(cv2.CAP_PROP_POS_MSEC, int(start_sec * 1000))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(final_output_path, fourcc, fps, (width, height))
    
    pipeline = ReIDPipeline(model, device, inf_cfg)
    query_img_path = inf_cfg.get('query_img', '')
    
    if query_img_path and os.path.exists(query_img_path):
        print(f"Loading Query Image for Target Initialization: {query_img_path}")
        q_img = cv2.imread(query_img_path)
        q_img = cv2.cvtColor(q_img, cv2.COLOR_BGR2RGB)
        q_tensor = transform(q_img).unsqueeze(0).to(device)
        q_feat_2560 = extract_cnn_feature(model, q_tensor)
        # Vì chỉ có 1 ảnh, nhân bản ra k frames cho Mamba
        q_seq = q_feat_2560.unsqueeze(1).repeat(1, pipeline.num_frames, 1)
        q_feat_3072 = compute_reid_embedding(model, q_seq)
        
        pipeline.memory_bank.add_anchor(q_feat_2560, q_feat_3072)
        pipeline.state = ReIDPipeline.T1_LOST
        pipeline.original_target_id = "QUERY"
        print("Query Feature added to Anchor Bank. System entering SEARCH mode.")
    else:
        pipeline.state = ReIDPipeline.T0_INIT
        print("No Query Image provided. Auto-Locking on first detected object.")

    frame_idx = 0
    detector_conf = inf_cfg.get('detector_conf', 0.15)
    detector_imgsz = inf_cfg.get('detector_imgsz', 1088)
    detector_classes = inf_cfg.get('detector_classes', None)

    print("Starting Multi-UAV Coarse-to-Fine Inference Stream (New Pipeline)...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        results = yolo_model.track(frame, persist=True, conf=detector_conf, imgsz=detector_imgsz, classes=detector_classes, verbose=False)
        
        filtered_boxes = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                if box.id is not None:
                    cls_id = int(box.cls[0])
                    if detector_classes is None or cls_id in detector_classes:
                        filtered_boxes.append(box)

        display_frame = frame.copy()
        
        if pipeline.state == ReIDPipeline.T0_INIT:
            if pipeline.target_track_id is None:
                if len(filtered_boxes) > 0:
                    best_box = max(filtered_boxes, key=lambda b: float(b.conf[0]))
                    pipeline.target_track_id = int(best_box.id[0])
                    pipeline.original_target_id = pipeline.target_track_id
                    print(f"[{frame_idx}] Auto-Locked onto Target Track ID: {pipeline.target_track_id}")
                    pipeline.process_tracking(frame, filtered_boxes, frame_idx, transform)
            else:
                pipeline.process_tracking(frame, filtered_boxes, frame_idx, transform)
            
        elif pipeline.state == ReIDPipeline.T1_LOST:
            pipeline.process_lost(filtered_boxes, frame_idx)
            
        elif pipeline.state == ReIDPipeline.T2_SEARCH:
            pipeline.process_search(frame, filtered_boxes, frame_idx, transform)
            
        elif pipeline.state == ReIDPipeline.T3_VERIFIED:
            pipeline.process_tracking(frame, filtered_boxes, frame_idx, transform)

        pipeline.draw_ui(display_frame, filtered_boxes)
        out.write(display_frame)
        frame_idx += 1

    cap.release()
    out.release()
    metrics_file.close()

    print("Inference completed!")
    print(f"Output video saved to {final_output_path}")

if __name__ == '__main__':
    main()
