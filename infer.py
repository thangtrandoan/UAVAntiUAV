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
        temporal_token = model.temporal_encoder(seq_feats)
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
    
    def get_weighted_visual_mean(self) -> torch.Tensor:
        weights = torch.tensor(self.sharpness_scores, dtype=torch.float32)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = torch.ones_like(weights) / len(weights)
        
        stacked = torch.stack([f.squeeze(0) for f in self.features])
        return (stacked * weights.unsqueeze(1).to(stacked.device)).sum(dim=0, keepdim=True)
    
    def clear(self):
        self.features.clear()
        self.sharpness_scores.clear()
        self._frame_counter = 0

def compute_fused_vector(model, sliding_window: SlidingWindowBuffer) -> tuple:
    visual_mean = sliding_window.get_weighted_visual_mean()
    seq = sliding_window.get_sequence()
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
    
    def coarse_score(self, query_feat: torch.Tensor) -> float:
        query = F.normalize(query_feat, p=2, dim=1)
        max_sim = 0.0
        for entry in self.anchor_bank + self.recent_bank:
            sim = torch.mm(query, entry["fused"].t()).item()
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

def parse_args():
    parser = argparse.ArgumentParser(description="Sequence Inference for UAV ReID (OOP Pipeline)")
    parser.add_argument("--seq-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--config", type=str, default="configs/config_jetson.yaml")
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
        
        self.memory_bank = TwoTierMemoryBank(
            max_anchor=cfg.get('max_anchor_size', 10),
            max_recent=cfg.get('max_recent_size', 30)
        )
        self.sliding_window = SlidingWindowBuffer(self.num_frames, self.stride)
        self.soft_lock_buffer = SlidingWindowBuffer(self.num_frames, stride=1)
        
        self.last_update_time = 0.0
        self._hijack_checks_remaining = 0
        
        self.metrics_cnn_times = []
        self.metrics_mamba_times = []
        self.false_alarms = 0
        self.reid_latency_frames = -1
        self.reappeared_frame_idx = -1
        
    def _transition_to_lost(self, frame_idx):
        print(f"[{frame_idx}] Target LOST! -> T1_LOST")
        self.state = self.T1_LOST
        self.sliding_window.clear()
        self.soft_lock_buffer.clear()
        
    def process_frame(self, frame, bbox, is_absent, frame_idx, transform):
        valid_bbox = bbox[2] > 0 and bbox[3] > 0
        current_time = time.time()
        
        if self.state in [self.T0_INIT, self.T3_VERIFIED]:
            if is_absent or not valid_bbox:
                if len(self.sliding_window.features) > 0:
                    # Pad the sliding window if it's not ready
                    while not self.sliding_window.is_ready():
                        self.sliding_window.add(self.sliding_window.features[-1], self.sliding_window.sharpness_scores[-1])
                        
                    t0 = time.time()
                    visual_mean, fused_feat = compute_fused_vector(self.model, self.sliding_window)
                    self.metrics_mamba_times.append((time.time() - t0) * 1000)
                    if len(self.memory_bank.anchor_bank) < self.memory_bank.max_anchor:
                        self.memory_bank.add_anchor(visual_mean, fused_feat)
                    else:
                        self.memory_bank.add_recent(visual_mean, fused_feat)
                    print(f"[{frame_idx}] Last-moment Memory Bank update before LOST. Bank: {self.memory_bank.size_info()}")
                self._transition_to_lost(frame_idx)
                return
                
            crop = crop_and_pad(frame, bbox, self.bbox_padding)
            if crop is not None and self.sliding_window.should_extract():
                sharpness = compute_sharpness(crop)
                tensor_frame = transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
                t0 = time.time()
                feat_2560 = extract_cnn_feature(self.model, tensor_frame)
                self.metrics_cnn_times.append((time.time() - t0) * 1000)
                self.sliding_window.add(feat_2560, sharpness)
                
            time_elapsed = current_time - self.last_update_time
            if self.sliding_window.is_ready() and time_elapsed >= self.update_interval_sec:
                t0 = time.time()
                visual_mean, fused_feat = compute_fused_vector(self.model, self.sliding_window)
                self.metrics_mamba_times.append((time.time() - t0) * 1000)
                
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
                
        elif self.state == self.T2_SEARCH:
            if is_absent or not valid_bbox:
                print(f"[{frame_idx}] UAV lost during T2_SEARCH. -> T1_LOST")
                self._transition_to_lost(frame_idx)
                return
                
            crop = crop_and_pad(frame, bbox, self.bbox_padding)
            if crop is not None:
                sharpness = compute_sharpness(crop)
                tensor_frame = transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
                t0 = time.time()
                feat_2560 = extract_cnn_feature(self.model, tensor_frame)
                self.metrics_cnn_times.append((time.time() - t0) * 1000)
                
                # Nếu đang trong quá trình thu thập Soft Lock, tiếp tục thu thập vô điều kiện
                if len(self.soft_lock_buffer.features) > 0:
                    self.soft_lock_buffer.add(feat_2560, sharpness)
                    print(f"[{frame_idx}] Soft Lock collecting: {len(self.soft_lock_buffer.features)}/{self.num_frames}")
                else:
                    # Chưa vào Soft Lock -> kiểm tra Coarse để quyết định có bắt đầu thu thập không
                    # Nhân bản feat_2560 ra k frames để lấy vector 3072-dim qua BNNeck
                    seq_feat_coarse = feat_2560.unsqueeze(1).expand(-1, self.num_frames, -1)
                    t0_mamba = time.time()
                    feat_3072_coarse = compute_reid_embedding(self.model, seq_feat_coarse)
                    self.metrics_mamba_times.append((time.time() - t0_mamba) * 1000)
                    
                    coarse_score = self.memory_bank.coarse_score(feat_3072_coarse)
                    if coarse_score >= self.soft_lock_threshold:
                        self.soft_lock_buffer.add(feat_2560, sharpness)
                        print(f"[{frame_idx}] Soft Lock collecting: 1/{self.num_frames} (coarse={coarse_score:.3f})")
                    else:
                        print(f"[{frame_idx}] Coarse FAILED! (coarse={coarse_score:.3f} < {self.soft_lock_threshold})")
                
                # Đủ k frame -> chạy Lọc Tinh
                if self.soft_lock_buffer.is_ready():
                    t0 = time.time()
                    visual_mean, fused_feat = compute_fused_vector(self.model, self.soft_lock_buffer)
                    mamba_time = (time.time() - t0) * 1000
                    self.metrics_mamba_times.append(mamba_time)
                    
                    fine_score = self.memory_bank.fine_score(fused_feat)
                    if fine_score >= self.reid_threshold:
                        self.reid_latency_frames = frame_idx - self.reappeared_frame_idx
                        print(f"[{frame_idx}] HARD LOCK! (fine={fine_score:.3f} >= {self.reid_threshold}) Latency: {self.reid_latency_frames} frames")
                        self.state = self.T3_VERIFIED
                        self._hijack_checks_remaining = self.hijack_check_count
                        self.last_update_time = time.time()
                        
                        if len(self.memory_bank.anchor_bank) < self.memory_bank.max_anchor:
                            self.memory_bank.add_anchor(visual_mean, fused_feat)
                        else:
                            self.memory_bank.add_recent(visual_mean, fused_feat)
                            
                        self.sliding_window = self.soft_lock_buffer
                        self.sliding_window.stride = self.stride
                        self.soft_lock_buffer = SlidingWindowBuffer(self.num_frames, stride=1)
                    else:
                        self.false_alarms += 1
                        print(f"[{frame_idx}] Fine FAILED! (fine={fine_score:.3f} < {self.reid_threshold}) -> Rolling Window...")
                        self.soft_lock_buffer.features.pop(0)
                        self.soft_lock_buffer.sharpness_scores.pop(0)

    def draw_ui(self, display_frame, bbox, frame_idx):
        color = (0, 0, 255)
        text = "LOST"
        if self.state in [self.T0_INIT, self.T3_VERIFIED]:
            color = (0, 255, 0)
            text = "TRACKING (HARD LOCK)"
        elif self.state == self.T2_SEARCH:
            color = (255, 255, 0)
            text = f"SEARCHING ({len(self.soft_lock_buffer.features)}/{self.num_frames})"
            
        cv2.putText(display_frame, f"State: {self.state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display_frame, f"Bank: {self.memory_bank.size_info()}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if bbox[2] > 0 and bbox[3] > 0 and self.state != self.T1_LOST:
            x, y, bw, bh = bbox
            cv2.rectangle(display_frame, (x, y), (x+bw, y+bh), color, 2)
            cv2.putText(display_frame, text, (x, max(0, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


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
    
    backbone_type = inf_cfg.get('backbone', 'resnet50_ibn')
    print(f"Initializing Mamba ReID Model with {backbone_type} backbone...")
    model = UAVReIDNet(backbone=backbone_type)
    
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
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        is_absent = absent[frame_idx] == 1 if frame_idx < len(absent) else True
        bbox = bboxes[frame_idx] if frame_idx < len(bboxes) else [0,0,0,0]
        
        display_frame = frame.copy()
        
        pipeline.process_frame(frame, bbox, is_absent, frame_idx, transform)
        pipeline.draw_ui(display_frame, bbox, frame_idx)
        
        out_vid.write(display_frame)
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
        throughput = 1000.0 / avg_cnn if avg_cnn > 0 else 0.0
        metrics_report.append(f"Avg CNN Feature Extraction : {avg_cnn:.2f} ms")
        metrics_report.append(f"Avg System Throughput      : {throughput:.2f} FPS")
    if pipeline.metrics_mamba_times:
        avg_mamba = np.mean(pipeline.metrics_mamba_times)
        metrics_report.append(f"Avg Mamba + Head Time      : {avg_mamba:.2f} ms")
    metrics_report.append(f"Re-acquisition Latency     : {pipeline.reid_latency_frames} frames" if pipeline.reid_latency_frames >= 0 else "Re-acquisition Latency     : N/A")
    metrics_report.append(f"False Alarms (Fine Fails)  : {pipeline.false_alarms}")
    
    print("\n".join(metrics_report))
    metrics_file.close()
    builtins.print = _orig_print
    
    return avg_cnn, avg_mamba, throughput, pipeline.reid_latency_frames, pipeline.false_alarms

def main():
    args = parse_args()
    cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
            
    inf_cfg = cfg.get('infer', {})
    seq_dir_arg = args.seq_dir or inf_cfg.get('seq_dir')
    if not seq_dir_arg:
        print("Error: --seq-dir must be provided.")
        return

    print(f"Initializing Mamba ReID Model...")
    model = UAVReIDNet()
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
        
        base_out_dir = inf_cfg.get('out_dir', './infer_output')
        print(f"Batch processing: Results will be saved in base directory: {base_out_dir}")
        for sdir in valid_seqs:
            res = run_sequence(sdir, model, device, transform, cfg, inf_cfg, out_base=base_out_dir)
            if res:
                c, m, t, l, f = res
                all_cnn.append(c)
                all_mamba.append(m)
                all_throughput.append(t)
                if l >= 0:
                    all_latency.append(l)
                all_false_alarms.append(f)
                
        # Calculate averages
        avg_cnn = np.mean(all_cnn) if all_cnn else 0.0
        avg_mamba = np.mean(all_mamba) if all_mamba else 0.0
        avg_throughput = np.mean(all_throughput) if all_throughput else 0.0
        avg_latency = np.mean(all_latency) if all_latency else 0.0
        sum_false_alarms = int(np.sum(all_false_alarms)) if all_false_alarms else 0
        
        print("\n=== AGGREGATED METRICS ===")
        print(f"Avg CNN Feature Extraction : {avg_cnn:.2f} ms")
        print(f"Avg System Throughput      : {avg_throughput:.2f} FPS")
        print(f"Avg Mamba + Head Time      : {avg_mamba:.2f} ms")
        print(f"Avg Re-acquisition Latency : {avg_latency:.2f} frames")
        print(f"Total False Alarms         : {sum_false_alarms}")
        
        # Save to summary text file
        os.makedirs(base_out_dir, exist_ok=True)
        with open(os.path.join(base_out_dir, "summary_metrics.txt"), "w") as sf:
            sf.write("=== AGGREGATED METRICS ===\n")
            sf.write(f"Avg CNN Feature Extraction : {avg_cnn:.2f} ms\n")
            sf.write(f"Avg System Throughput      : {avg_throughput:.2f} FPS\n")
            sf.write(f"Avg Mamba + Head Time      : {avg_mamba:.2f} ms\n")
            sf.write(f"Avg Re-acquisition Latency : {avg_latency:.2f} frames\n")
            sf.write(f"Total False Alarms         : {sum_false_alarms}\n")
            
    else:
        run_sequence(seq_dir_arg, model, device, transform, cfg, inf_cfg, out_base=None)

if __name__ == "__main__":
    main()
