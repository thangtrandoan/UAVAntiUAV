import os
import sys
import time
import argparse
import yaml
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
import builtins
from torchvision import transforms

from model import UAVReIDNet

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

def compute_reid_embedding(model, seq_feats):
    with torch.no_grad():
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

    # 🛠️ (15/9) PHÂN VAI stride: SOFT LOCK thu LIÊN TỤC (stride=1) để phản ứng nhanh;
    # HARD LOCK mới LẤY CÁCH QUÃNG (`stride = frame_stride`) để bước thời gian khớp
    # lúc train và khớp Memory Bank (dựng từ `sliding_window`, cũng stride = frame_stride).
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

def compute_fused_vector(model, sliding_window: SlidingWindowBuffer, stride: int = 1) -> tuple:
    # 🛠️ (15/9) `stride`: HARD LOCK truyền `frame_stride` (lấy cách quãng), SOFT LOCK để 1.
    visual_mean = sliding_window.get_weighted_visual_mean(stride)
    seq = sliding_window.get_strided_sequence(stride)
    fused_feat = compute_reid_embedding(model, seq)
    return visual_mean, fused_feat

class TwoTierMemoryBank:
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

def sample_imposters(data_root, exclude_seq_name, num_imposters, model, transform, device, num_frames, bbox_padding, stride=1):
    """Samples random frames from OTHER videos to act as false positives

    🛠️ (15/9) `stride`: lấy 1 frame mỗi `stride` frame để bước thời gian khớp
    `data_pipeline.frame_stride` (giống train/infer).
    """
    imposter_feats = []
    
    if not os.path.exists(data_root):
        print(f"Warning: data-root {data_root} does not exist. Cannot sample imposters.")
        return []
        
    seqs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and d != exclude_seq_name]
    if not seqs: return []
        
    sampled_seqs = random.sample(seqs, min(num_imposters, len(seqs)))
    print(f"  -> Sampling {len(sampled_seqs)} imposters from: {sampled_seqs}")
    
    for seq in sampled_seqs:
        seq_path = os.path.join(data_root, seq)
        video_path = os.path.join(seq_path, f"{seq}.mp4")
        gt_path = os.path.join(seq_path, "groundtruth_rect.txt")
        absent_path = os.path.join(seq_path, "absent.txt")
        
        if not os.path.exists(video_path) or not os.path.exists(gt_path): continue
            
        absent = []
        if os.path.exists(absent_path):
            with open(absent_path, 'r') as f:
                absent = [int(line.strip()) for line in f if line.strip().isdigit()]
                
        bboxes = []
        with open(gt_path, 'r') as f:
             for line in f:
                parts = line.strip().replace(',', ' ').split()
                if len(parts) >= 4: bboxes.append([int(float(p)) for p in parts[:4]])
                else: bboxes.append([0, 0, 0, 0])
                
        valid_starts = []
        span = num_frames * stride          # 🛠️ (15/9) cửa sổ trải `num_frames*stride` frame gốc
        for i in range(max(0, len(bboxes) - span)):
            is_valid = True
            for j in range(i, i + span):
                if j < len(absent) and absent[j] == 1:
                    is_valid = False; break
                if j < len(bboxes) and (bboxes[j][2] <= 0 or bboxes[j][3] <= 0):
                    is_valid = False; break
            if is_valid: valid_starts.append(i)
                
        if not valid_starts: continue
            
        start_idx = random.choice(valid_starts)
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        
        buffer = SlidingWindowBuffer(num_frames, stride=1)   # ta tự chọn frame nào đưa vào
        for j in range(num_frames * stride):
            ret, frame = cap.read()
            if not ret: break
            if j % stride != 0: continue    # 🛠️ (15/9) chỉ lấy 1 frame mỗi `stride`
            
            bbox = bboxes[start_idx + j]
            crop = crop_and_pad(frame, bbox, bbox_padding)
            if crop is not None:
                sharpness = compute_sharpness(crop)
                tensor_frame = transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
                feat = extract_cnn_feature(model, tensor_frame)
                buffer.add(feat, sharpness)
        
        cap.release()
        
        if buffer.is_ready():
            visual_mean, fused_feat = compute_fused_vector(model, buffer)
            imposter_feats.append(fused_feat)
            
    return imposter_feats

class SeqRobustnessPipeline:
    T0_INIT = "T0_INIT"
    T1_LOST = "T1_LOST"
    T2_SEARCH = "T2_SEARCH"
    T3_VERIFIED = "T3_VERIFIED"
    
    def __init__(self, model, device, cfg, data_root, seq_name):
        self.model = model
        self.device = device
        self.data_root = data_root
        self.seq_name = seq_name
        self.state = self.T0_INIT
        
        self.stride = cfg.get('stride', 2)
        self.num_frames = cfg.get('num_frames', 16)
        self.soft_lock_threshold = cfg.get('soft_lock_threshold', 0.50)
        self.reid_threshold = cfg.get('reid_threshold', 0.75)
        self.hijack_threshold = cfg.get('hijack_threshold', 0.40)
        self.hijack_check_count = cfg.get('hijack_check_count', 5)
        self.update_interval_sec = cfg.get('update_interval_sec', 2.0)
        self.time_source = cfg.get('time_source', 'video')  # 'video' | 'wall'
        self.bbox_padding = cfg.get('bbox_padding', 0.2)
        self.num_imposters = cfg.get('num_imposters', 5)
        
        self.memory_bank = TwoTierMemoryBank(
            max_anchor=cfg.get('max_anchor_size', 10),
            max_recent=cfg.get('max_recent_size', 30)
        )
        # 🛠️ (15/9) PHÂN VAI stride: soft_lock THU LIÊN TỤC (stride=1) để phản ứng nhanh,
        # HARD LOCK mới LẤY CÁCH QUÃNG `frame_stride` (khớp train + Memory Bank).
        self.sliding_window = SlidingWindowBuffer(self.num_frames, self.stride)
        # 🛠️ (22/9) HAI CỬA SỔ SONG SONG: soft lock = `num_frames` frame LIÊN TỤC (nhanh,
        # chỉ để chọn ứng viên / xem điểm), hard lock = `num_frames` MẪU cách nhau `frame_stride`
        # (chính xác, mới đem so Memory Bank). Xem infer.py để biết chi tiết.
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
        # Cần `(num_frames-1)*stride+1` frame LIÊN TỤC; vắng ngắn chỉ SKIP frame và GIỮ
        # feature đã thu, chỉ vắng LIÊN TIẾP > `gap_tolerance` mới reset thật.
        # Khoảng vắng làm cửa sổ có "lỗ" thời gian; train cũng có mốc cách xa hơn
        # `frame_stride` (data_pipeline bỏ frame bbox không hợp lệ), nhưng bên train là
        # THIẾU mẫu còn đây là mẫu bắc QUA một khoảng vắng.
        self.gap_tolerance = cfg.get('t2_search_gap_tolerance', 2)
        self._absent_streak = 0
        
        self.last_update_time = 0.0
        self._hijack_checks_remaining = 0
        
        self.reappeared_frame_idx = -1
        self.all_genuine_scores = []
        self.all_imposter_scores = []
        
        self.last_verification_text = ""
        self.last_verification_color = (255, 255, 255)
        self.overlay_timer = 0
        
    def _transition_to_lost(self, frame_idx):
        print(f"[{frame_idx}] Target LOST! -> T1_LOST")
        self.state = self.T1_LOST
        self.sliding_window.clear()
        self.soft_lock_buffer.clear()
        self.hard_lock_buffer.clear()
        self._soft_lock_announced = False
        self._soft_lock_passed = False
        if hasattr(self.model, 'cached_imposters'):
            self.model.cached_imposters = []
        
    def process_frame(self, frame, bbox, is_absent, frame_idx, transform, video_time=None):
        valid_bbox = bbox[2] > 0 and bbox[3] > 0
                # 🛠️ (22/9) ĐỒNG HỒ: mặc định dùng **THỜI GIAN CỦA VIDEO** (`frame_idx / fps`),
        # KHÔNG dùng `time.time()`. `update_interval_sec` nghĩa là "bao lâu (theo video) thì
        # cập nhật Memory Bank một lần" — đó là đại lượng CỦA VIDEO, không phải của máy.
        # Dùng đồng hồ thực làm số lần update phụ thuộc TỐC ĐỘ XỬ LÝ -> CÙNG MỘT VIDEO,
        # máy khác nhau cho kết quả khác nhau (Colab T4 vs Jetson vs GPU nhanh).
        # `time_source='wall'` để quay lại hành vi cũ (chỉ nên dùng cho stream thời gian thực).
        if video_time is not None and self.time_source == 'video':
            current_time = video_time
        else:
            current_time = time.time()
        
        if self.state in [self.T0_INIT, self.T3_VERIFIED]:
            if is_absent or not valid_bbox:
                if len(self.sliding_window.features) > 0:
                    # 🛠️ (15/9) BỎ pad NHÂN BẢN frame cuối (bước thời gian = 0) — xem infer.py
                        
                    visual_mean, fused_feat = compute_fused_vector(self.model, self.sliding_window)
                    if len(self.memory_bank.anchor_bank) < self.memory_bank.max_anchor:
                        self.memory_bank.add_anchor(visual_mean, fused_feat)
                    else:
                        self.memory_bank.add_recent(visual_mean, fused_feat)
                    print(f"[{frame_idx}] Last-moment Memory Bank update before LOST.")
                self._transition_to_lost(frame_idx)
                return
                
            crop = crop_and_pad(frame, bbox, self.bbox_padding)
            if crop is not None and self.sliding_window.should_extract():
                sharpness = compute_sharpness(crop)
                tensor_frame = transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
                feat_2560 = extract_cnn_feature(self.model, tensor_frame)
                self.sliding_window.add(feat_2560, sharpness)
                
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
                        if hijack_score < self.hijack_threshold:
                            print(f"[{frame_idx}] WARNING: HIJACK DETECTED! -> T1_LOST")
                            self._transition_to_lost(frame_idx)
                            return
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
                # 🛠️ (22/9) NỚI RESET (md/22thg9.md §4, §8.3).
                self._absent_streak += 1
                if self._absent_streak <= self.gap_tolerance:
                    print(f"[{frame_idx}] T2_SEARCH: vang {self._absent_streak}/{self.gap_tolerance} frame"
                          f" -> BO QUA, giu soft {len(self.soft_lock_buffer.features)}/{self.num_frames}"
                          f" | hard {len(self.hard_lock_buffer.features)}/{self.num_frames} feature")
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
                feat_2560 = extract_cnn_feature(self.model, tensor_frame)
                coarse_score = self.memory_bank.coarse_score(feat_2560)

                # Thu thập soft lock VÔ ĐIỀU KIỆN một khi đã bắt đầu (giống infer.py): cổng chặn
                # thật là ĐIỂM TRÊN CẢ CỬA SỔ, không phải coarse từng frame.
                # 🛠️ (22/9) DỪNG thu soft lock ngay khi đã có ĐIỂM soft lock (xem infer.py).
                if (not self._soft_lock_announced
                        and (len(self.soft_lock_buffer.features) > 0
                             or coarse_score >= self.soft_lock_threshold)):
                    self.soft_lock_buffer.add(feat_2560, sharpness)
                    print(f"[{frame_idx}] Soft Lock collecting: {len(self.soft_lock_buffer.features)}/{self.num_frames} (coarse={coarse_score:.3f})")

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
                            print(f"[{frame_idx}] SOFT LOCK PASS (soft={soft_score:.3f} >= "
                                  f"{self.soft_lock_threshold}) -> MOI duoc tinh HARD LOCK")
                        else:
                            print(f"[{frame_idx}] SOFT LOCK FAIL (soft={soft_score:.3f} < "
                                  f"{self.soft_lock_threshold}) -> quay lai T1_LOST")
                            self._transition_to_lost()
                            return

                    # (2) HARD LOCK — ⚠️ CHỈ BẮT ĐẦU THU sau khi soft lock đã PASS. Không thu
                    # song song trước đó: chưa có điểm soft lock thì CHƯA BIẾT thu frame của
                    # MỤC TIÊU NÀO (data 1 mục tiêu nên thu sớm cũng "đúng", nhưng pipeline sẽ
                    # SAI khi nhiều mục tiêu). Hệ quả: t_hard_lock ≈ N + (N−1)×stride = 56 frame.
                    _new_hard_sample = False
                    if self._soft_lock_passed:
                        _new_hard_sample = self.hard_lock_buffer.should_extract()
                        if _new_hard_sample:
                            self.hard_lock_buffer.add(feat_2560, sharpness)
                            print(f"[{frame_idx}] HARD LOCK collecting: "
                                  f"{len(self.hard_lock_buffer.features)}/{self.num_frames} "
                                  f"(moi {self.stride} frame)")

                    # Đủ `num_frames` mẫu cách quãng VÀ soft lock đã PASS -> chạy Lọc Tinh.
                    if self._soft_lock_passed and _new_hard_sample and self.hard_lock_buffer.is_ready():
                        print(f"\n--- [Frame {frame_idx}] ROBUSTNESS TEST (MULTIPLE UAV SIMULATION) ---")
                        visual_mean, fused_feat = compute_fused_vector(self.model, self.hard_lock_buffer)
                        
                        best_genuine = self.memory_bank.fine_score(fused_feat)
                        self.all_genuine_scores.append(best_genuine)
                        
                        if not hasattr(self.model, 'cached_imposters') or len(self.model.cached_imposters) == 0:
                            self.model.cached_imposters = sample_imposters(self.data_root, self.seq_name, self.num_imposters, self.model, transform, self.device, self.num_frames, self.bbox_padding, self.stride)
                        
                        imposter_best_sims = []
                        for imp_fused_feat in self.model.cached_imposters:
                            imp_best = self.memory_bank.fine_score(imp_fused_feat)
                            imposter_best_sims.append(imp_best)
                            
                        self.all_imposter_scores.extend(imposter_best_sims)
                        max_imposter = max(imposter_best_sims) if imposter_best_sims else 0.0
                        
                        print(f"Genuine Sim: {best_genuine:.3f}")
                        for i, s in enumerate(imposter_best_sims):
                            print(f"Imposter {i+1} Sim: {s:.3f}")
                            
                        self.overlay_timer = 90
                        if max_imposter > self.reid_threshold and max_imposter > best_genuine:
                            print(f"Result: HIJACKED BY IMPOSTER! Imp: {max_imposter:.3f} > Gen: {best_genuine:.3f}")
                            self.last_verification_text = f"HIJACKED! | Gen: {best_genuine:.2f} < Imp: {max_imposter:.2f}"
                            self.last_verification_color = (255, 0, 255)
                            # 🛠️ (22/9) Rolling window TỰ ĐỘNG: `hard_lock_buffer` nuôi bằng
                            # `should_extract()` -> mỗi `frame_stride` frame nhận 1 mẫu mới và
                            # đẩy mẫu cũ nhất ra. Không pop tay.
                            pass
                        elif best_genuine >= self.reid_threshold:
                            print(f"Result: HARD LOCK! (fine={best_genuine:.3f} >= {self.reid_threshold})")
                            self.state = self.T3_VERIFIED
                            self._hijack_checks_remaining = self.hijack_check_count
                            self.last_update_time = current_time
                            
                            self.last_verification_text = f"ReID Pass | Gen: {best_genuine:.2f} | Imp: {max_imposter:.2f}"
                            self.last_verification_color = (0, 255, 0)
                            
                            if len(self.memory_bank.anchor_bank) < self.memory_bank.max_anchor:
                                self.memory_bank.add_anchor(visual_mean, fused_feat)
                            else:
                                self.memory_bank.add_recent(visual_mean, fused_feat)
                                
                            # 🛠️ (15/9) Nạp `sliding_window` từ ĐÚNG cửa sổ đã dùng để HARD LOCK
                            # (bản lấy cách quãng), KHÔNG phải toàn bộ buffer thu liên tục.
                            # 🛠️ (22/9) Nạp từ ĐÚNG cửa sổ HARD LOCK (đã cách quãng sẵn).
                            self.sliding_window = SlidingWindowBuffer(self.num_frames, self.stride)
                            for _f, _s in zip(self.hard_lock_buffer.features,
                                              self.hard_lock_buffer.sharpness_scores):
                                self.sliding_window.add(_f, _s)
                            self.soft_lock_buffer = SlidingWindowBuffer(self.num_frames, stride=1)
                            self.hard_lock_buffer = SlidingWindowBuffer(self.num_frames, self.stride)
                            self._soft_lock_announced = False
                            self._soft_lock_passed = False
                        else:
                            print(f"Result: Fine FAILED! (fine={best_genuine:.3f} < {self.reid_threshold})")
                            self.last_verification_text = f"ReID Fail | Gen: {best_genuine:.2f} | Imp: {max_imposter:.2f}"
                            self.last_verification_color = (0, 0, 255)
                            # 🛠️ (22/9) Rolling window TỰ ĐỘNG (như trên).
                            pass
                else:
                    pass  # coarse tụt khi CHƯA bắt đầu thu -> không làm gì

    def draw_ui(self, display_frame, bbox, frame_idx):
        h, w = display_frame.shape[:2]
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
            
        if self.overlay_timer > 0:
            cv2.putText(display_frame, self.last_verification_text, (50, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.last_verification_color, 2)
            self.overlay_timer -= 1

def parse_args():
    parser = argparse.ArgumentParser(description="Real-time Inference with Robustness Evaluation (False Positives)")
    parser.add_argument("--seq-dir", type=str, default=None)
    parser.add_argument("--data-root", type=str, default="./data/UAV-Anti-UAV/Test")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--config", type=str, default="configs/config_jetson.yaml")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = {}
    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: Config file not found at '{args.config}'")
            return
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
            
    inf_cfg = cfg.get('infer_robustness', cfg.get('infer', {}))
    seq_dir = args.seq_dir or inf_cfg.get('seq_dir')
    if not seq_dir:
        print("Error: --seq-dir must be provided.")
        return

    seq_name = os.path.basename(os.path.normpath(seq_dir))
    video_path = os.path.join(seq_dir, f"{seq_name}.mp4")
    gt_path = os.path.join(seq_dir, "groundtruth_rect.txt")
    absent_path = os.path.join(seq_dir, "absent.txt")
    
    out_dir = inf_cfg.get('out_dir', f"infer_robustness/{seq_name}")
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "config_used.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
        
    output_video_name = inf_cfg.get('output_video', 'output_robustness.mp4')
    final_output_path = os.path.join(out_dir, os.path.basename(output_video_name))
    log_path = os.path.join(out_dir, "robustness_log.txt")
    log_file = open(log_path, "w")
    
    _orig_print = builtins.print
    def custom_print(*args_p, **kwargs_p):
        msg = " ".join(str(a) for a in args_p)
        _orig_print(msg, **kwargs_p)
        if not log_file.closed:
            log_file.write(msg + "\n")
            log_file.flush()
    globals()['print'] = custom_print
    
    bboxes = []
    if os.path.exists(gt_path):
        with open(gt_path, "r") as f:
            for line in f:
                parts = line.strip().replace(',', ' ').split()
                if len(parts) >= 4: bboxes.append([int(float(p)) for p in parts[:4]])
                else: bboxes.append([0, 0, 0, 0])
                    
    absent = []
    if os.path.exists(absent_path):
        with open(absent_path, "r") as f:
            absent = [int(line.strip()) for line in f if line.strip().isdigit()]

    print(f"Initializing Model...")
    backbone_type = inf_cfg.get('backbone', 'resnet50_ibn')
    temporal_type = inf_cfg.get('temporal_type', 'mamba')
    model = UAVReIDNet(freeze_backbone=False, backbone=backbone_type, temporal_type=temporal_type)
    model_path = inf_cfg.get('model_path', args.checkpoint)
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
        print(f"Loaded weights from {model_path}")
    else:
        print(f"WARNING: Checkpoint not found at {model_path}. Running with random weights!")
    
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

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(final_output_path, fourcc, fps_video, (width, height))
    
    data_root = inf_cfg.get('data_root', args.data_root)
    pipeline = SeqRobustnessPipeline(model, device, inf_cfg, data_root, seq_name)
    
    print(f"Starting OOP Robustness Inference on {seq_name}...")
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        is_absent = absent[frame_idx] == 1 if frame_idx < len(absent) else True
        bbox = bboxes[frame_idx] if frame_idx < len(bboxes) else [0,0,0,0]
        
        display_frame = frame.copy()
        
        # 🛠️ (22/9) THỜI GIAN THẬT CỦA VIDEO = frame_idx / fps (không phụ thuộc tốc độ máy).
        video_time = frame_idx / fps_video if (fps_video and fps_video > 0) else frame_idx / 30.0
        pipeline.process_frame(frame, bbox, is_absent, frame_idx, transform, video_time=video_time)
        pipeline.draw_ui(display_frame, bbox, frame_idx)
        
        out_vid.write(display_frame)
        frame_idx += 1

    cap.release()
    out_vid.release()
    
    print("\n================ ROBUSTNESS REPORT ================")
    print(f"Total Verifications (Genuine Queries)  : {len(pipeline.all_genuine_scores)}")
    print(f"Total Imposter Queries Evaluated       : {len(pipeline.all_imposter_scores)}")
    
    if pipeline.all_genuine_scores and pipeline.all_imposter_scores:
        avg_gen = np.mean(pipeline.all_genuine_scores)
        avg_imp = np.mean(pipeline.all_imposter_scores)
        print(f"\nAverage Genuine Similarity : {avg_gen:.4f}")
        print(f"Average Imposter Similarity: {avg_imp:.4f}")
        print(f"Average Margin (Gen - Imp) : {avg_gen - avg_imp:.4f}")
        
        reid_threshold = pipeline.reid_threshold
        frr = sum(1 for x in pipeline.all_genuine_scores if x < reid_threshold) / len(pipeline.all_genuine_scores)
        far = sum(1 for x in pipeline.all_imposter_scores if x >= reid_threshold) / len(pipeline.all_imposter_scores)
        print(f"\nAt Threshold {reid_threshold}:")
        print(f"False Rejection Rate (FRR) : {frr*100:.2f}% (Target UAV not recognized)")
        print(f"False Acceptance Rate (FAR): {far*100:.2f}% (Imposter UAV accepted)")
        
        plt.figure(figsize=(10, 6))
        plt.hist(pipeline.all_genuine_scores, bins=20, alpha=0.6, label='Genuine (True UAV)', color='green')
        plt.hist(pipeline.all_imposter_scores, bins=20, alpha=0.6, label='Imposter (False UAV)', color='red')
        plt.axvline(reid_threshold, color='blue', linestyle='dashed', linewidth=2, label=f'Threshold ({reid_threshold})')
        plt.xlabel('Cosine Similarity Score')
        plt.ylabel('Frequency')
        plt.title('ReID Robustness: Genuine vs Imposter Similarity Distribution')
        plt.legend()
        
        plot_path = os.path.join(out_dir, "score_distribution.png")
        plt.savefig(plot_path)
        print(f"\nScore distribution plot saved to: {plot_path}")
        
    print(f"Detailed logs saved to: {log_path}")
    print(f"Output video saved to: {final_output_path}")
    log_file.close()

if __name__ == "__main__":
    main()
