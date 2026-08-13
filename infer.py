import os
import sys
import time
import argparse
import yaml
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

from model import UAVReIDNet

def extract_cnn_feature(model, frame_tensor):
    with torch.no_grad():
        feats = model.backbone(frame_tensor)
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

def parse_args():
    parser = argparse.ArgumentParser(description="Real-time Inference for UAV ReID")
    parser.add_argument("--seq-dir", type=str, default=None, help="Path to the sequence directory (e.g., ../UAV-Anti-UAV/Test/video_001)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to yaml config (e.g., configs/jetson.yaml)")
    parser.add_argument("--output-video", type=str, default="output_reid.mp4", help="Output video path")
    parser.add_argument("--threshold", type=float, default=0.7, help="ReID cosine similarity threshold")
    parser.add_argument("--num-frames", type=int, default=16, help="Number of frames for temporal modeling")
    parser.add_argument("--bbox-padding", type=float, default=0.2, help="BBox padding ratio")
    return parser.parse_args()

def crop_and_resize(frame, bbox, padding, crop_size=256, final_size=224):
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox
    
    pad_w = bw * padding
    pad_h = bh * padding
    
    x1 = int(max(0, x - pad_w))
    y1 = int(max(0, y - pad_h))
    x2 = int(min(w, x + bw + pad_w))
    y2 = int(min(h, y + bh + pad_h))
    
    if x2 <= x1 or y2 <= y1:
        return None
        
    crop = frame[y1:y2, x1:x2]
    try:
        # Resize to 256 first, then we will center crop to 224 later
        resized = cv2.resize(crop, (crop_size, crop_size))
        return resized
    except Exception as e:
        return None

def main():
    args = parse_args()
    
    # Load config if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
            print(f"Loaded config from {args.config}")
            if 'infer' in cfg:
                inf = cfg['infer']
                args.seq_dir = inf.get('seq_dir', args.seq_dir)
                args.checkpoint = inf.get('model_path', args.checkpoint)
                args.output_video = inf.get('output_video', args.output_video)
                args.threshold = inf.get('reid_threshold', args.threshold)
                args.num_frames = inf.get('num_frames', args.num_frames)
                args.bbox_padding = inf.get('bbox_padding', args.bbox_padding)
            elif 'eval' in cfg and 'model_path' in cfg['eval']:
                args.checkpoint = cfg['eval']['model_path']
                
    if not args.seq_dir:
        print("Error: --seq-dir must be provided either via command line or inside the config file under 'infer: seq_dir'.")
        return

    seq_name = os.path.basename(os.path.normpath(args.seq_dir))
    video_path = os.path.join(args.seq_dir, f"{seq_name}.mp4")
    gt_path = os.path.join(args.seq_dir, "groundtruth_rect.txt")
    absent_path = os.path.join(args.seq_dir, "absent.txt")
    
    # Create output directory: infer/{seq_name}
    out_dir = os.path.join("infer", seq_name)
    os.makedirs(out_dir, exist_ok=True)
    vid_name = os.path.basename(args.output_video)
    final_output_path = os.path.join(out_dir, vid_name)
    
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return
        
    # Read Groundtruth (BBoxes and Absent flags)
    # We simulate a detector/tracker by using the ground truth bounding boxes when the drone is visible.
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

    print(f"Initializing Model: {args.checkpoint}...")
    model = UAVReIDNet(freeze_backbone=False)
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
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
        print(f"Warning: Checkpoint {args.checkpoint} not found. Running with random weights.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
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
    
    # State machine variables
    memory_bank = []
    feature_buffer = []
    
    # Metrics
    cnn_times = []
    mamba_times = []
    false_alarms = 0
    reid_latency_frames = -1
    reappeared_frame_idx = -1
    
    status = "INITIAL_TRACKING"
    assigned_id = "DRONE_01"
    reid_confidence = 0.0
    
    frame_idx = 0
    print("Starting Video Inference Stream...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        is_absent = absent[frame_idx] == 1 if frame_idx < len(absent) else True
        bbox = bboxes[frame_idx] if frame_idx < len(bboxes) else [0,0,0,0]
        valid_bbox = bbox[2] > 0 and bbox[3] > 0
        
        display_frame = frame.copy()
        
        if status == "INITIAL_TRACKING":
            if is_absent:
                if len(feature_buffer) > 0:
                    while len(feature_buffer) < args.num_frames:
                        feature_buffer.append(feature_buffer[-1])
                    if len(feature_buffer) > args.num_frames:
                        indices = np.linspace(0, len(feature_buffer)-1, args.num_frames).astype(int)
                        feature_buffer = [feature_buffer[i] for i in indices]
                        
                    seq_feats = torch.stack(feature_buffer, dim=1) # [1, 16, 2560]
                    t0 = time.time()
                    gallery_feat = compute_reid_embedding(model, seq_feats)
                    mamba_times.append((time.time() - t0) * 1000)
                    
                    memory_bank.append(gallery_feat)
                    if len(memory_bank) > 50: memory_bank.pop(0)
                    print(f"[{frame_idx}] Drone disappeared. Final Gallery feature added. Memory Bank Size: {len(memory_bank)}")
                
                status = "LOST"
                feature_buffer = []
            elif valid_bbox:
                # SPARSE SAMPLING (Gallery Collection): Tiết kiệm GPU, lấy ngắt quãng (3 frame lấy 1)
                if frame_idx % 3 == 0:
                    crop = crop_and_resize(frame, bbox, args.bbox_padding)
                    if crop is not None:
                        tensor_frame = transform(crop).unsqueeze(0).to(device)
                        t0 = time.time()
                        feat = extract_cnn_feature(model, tensor_frame)
                        cnn_times.append((time.time() - t0) * 1000)
                        feature_buffer.append(feat)
                        
                        if len(feature_buffer) > args.num_frames:
                            feature_buffer.pop(0)
                            
                        # MEMORY BANK: Tự động chụp lại đặc trưng vào album mỗi 2 giây (60 frames)
                        if frame_idx % 60 == 0 and len(feature_buffer) == args.num_frames:
                            seq_feats = torch.stack(feature_buffer, dim=1)
                            t0 = time.time()
                            gallery_feat = compute_reid_embedding(model, seq_feats)
                            mamba_times.append((time.time() - t0) * 1000)
                            memory_bank.append(gallery_feat)
                            if len(memory_bank) > 50: memory_bank.pop(0)
                        
                cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255, 0, 0), 2)
                cv2.putText(display_frame, f"ID: {assigned_id} (Tracking)", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        elif status == "LOST":
            cv2.putText(display_frame, "STATUS: TARGET LOST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            if not is_absent and valid_bbox:
                status = "REAPPEARED_VERIFYING"
                reappeared_frame_idx = frame_idx
                feature_buffer = []
                print(f"[{frame_idx}] Object reappeared. Collecting frames for ReID verification...")

        elif status == "REAPPEARED_VERIFYING":
            cv2.putText(display_frame, "STATUS: VERIFYING ID...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
            
            if is_absent:
                status = "LOST"
                feature_buffer = []
            elif valid_bbox:
                # DENSE SAMPLING (Query Collection): Lấy mọi frame liên tục để xác minh siêu tốc trong 0.5s
                crop = crop_and_resize(frame, bbox, args.bbox_padding)
                if crop is not None:
                    tensor_frame = transform(crop).unsqueeze(0).to(device)
                    t0 = time.time()
                    feat = extract_cnn_feature(model, tensor_frame)
                    cnn_times.append((time.time() - t0) * 1000)
                    feature_buffer.append(feat)
                    
                cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 165, 255), 2)
                
                if len(feature_buffer) == args.num_frames:
                    seq_feats = torch.stack(feature_buffer, dim=1) # [1, 16, 2560]
                    t0 = time.time()
                    query_feature = compute_reid_embedding(model, seq_feats)
                    mamba_time = (time.time() - t0) * 1000
                    mamba_times.append(mamba_time)
                    
                    # ROLLING VERIFICATION & MULTI-GALLERY MATCHING
                    sims = [torch.mm(query_feature, g_feat.t()).item() for g_feat in memory_bank]
                    best_sim = max(sims) if sims else 0.0
                    reid_confidence = best_sim
                    
                    if best_sim > args.threshold:
                        status = "REID_SUCCESS"
                        reid_latency_frames = frame_idx - reappeared_frame_idx
                        print(f"[{frame_idx}] ReID SUCCESS! Sim: {best_sim:.3f} > {args.threshold}. Latency: {reid_latency_frames} frames (Mamba time: {mamba_time:.1f} ms)")
                    else:
                        false_alarms += 1
                        print(f"[{frame_idx}] ReID FAILED! Sim: {best_sim:.3f} <= {args.threshold}. Rolling window next frame...")
                        # ROLLING QUERY: pop the oldest, stay in VERIFYING state for the next frame
                        feature_buffer.pop(0)

        elif status == "REID_SUCCESS":
            if is_absent:
                # Nếu drone lại biến mất lần 2, lần 3... Cập nhật lại Gallery Feature mới nhất!
                if len(feature_buffer) > 0:
                    while len(feature_buffer) < args.num_frames:
                        feature_buffer.append(feature_buffer[-1])
                    if len(feature_buffer) > args.num_frames:
                        indices = np.linspace(0, len(feature_buffer)-1, args.num_frames).astype(int)
                        feature_buffer = [feature_buffer[i] for i in indices]
                        
                    seq_feats = torch.stack(feature_buffer, dim=1) # [1, 16, 2560]
                    t0 = time.time()
                    gallery_feat = compute_reid_embedding(model, seq_feats)
                    mamba_times.append((time.time() - t0) * 1000)
                    
                    memory_bank.append(gallery_feat)
                    if len(memory_bank) > 50: memory_bank.pop(0)
                    print(f"[{frame_idx}] Drone disappeared AGAIN. Memory Bank Size: {len(memory_bank)}")
                    
                status = "LOST"
                feature_buffer = []
            elif valid_bbox:
                # SPARSE SAMPLING: Đã nhận diện thành công, trở lại lấy ngắt quãng để dưỡng sức GPU
                if frame_idx % 3 == 0:
                    crop = crop_and_resize(frame, bbox, args.bbox_padding)
                    if crop is not None:
                        tensor_frame = transform(crop).unsqueeze(0).to(device)
                        t0 = time.time()
                        feat = extract_cnn_feature(model, tensor_frame)
                        cnn_times.append((time.time() - t0) * 1000)
                        feature_buffer.append(feat)
                        
                        if len(feature_buffer) > args.num_frames:
                            feature_buffer.pop(0)
                            
                        # MEMORY BANK: Tự động chụp lại đặc trưng vào album mỗi 2 giây (60 frames)
                        if frame_idx % 60 == 0 and len(feature_buffer) == args.num_frames:
                            seq_feats = torch.stack(feature_buffer, dim=1)
                            t0 = time.time()
                            gallery_feat = compute_reid_embedding(model, seq_feats)
                            mamba_times.append((time.time() - t0) * 1000)
                            memory_bank.append(gallery_feat)
                            if len(memory_bank) > 50: memory_bank.pop(0)
                        
                cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
                cv2.putText(display_frame, f"ID: {assigned_id} (ReID Track)", (bbox[0], bbox[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_frame, f"ReID Conf: {reid_confidence:.2f}", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw HUD overlay
        if len(cnn_times) > 0:
            avg_cnn_time = np.mean(cnn_times)
            fps = 1000.0 / avg_cnn_time if avg_cnn_time > 0 else 0
            cv2.putText(display_frame, f"Model FPS: {fps:.1f}", (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"CNN Time : {avg_cnn_time:.1f} ms", (10, height - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        cv2.putText(display_frame, f"False Alarms: {false_alarms}", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        out_vid.write(display_frame)
        frame_idx += 1

    cap.release()
    out_vid.release()
    print("Inference completed!")
    print(f"Output video saved to {final_output_path}")
    
    # Generate Performance Metrics
    metrics_report = []
    metrics_report.append("--- PERFORMANCE METRICS ---")
    if len(cnn_times) > 0:
        metrics_report.append(f"Average CNN Feature Extraction (per frame) : {np.mean(cnn_times):.2f} ms")
        avg_mamba = np.mean(mamba_times) if len(mamba_times) > 0 else 0.0
        metrics_report.append(f"Average Mamba + Head Time (per 16 frames)  : {avg_mamba:.2f} ms")
        metrics_report.append(f"Average FPS (System throughput)            : {1000.0 / np.mean(cnn_times):.2f} FPS")
    if reid_latency_frames >= 0:
        metrics_report.append(f"Re-acquisition Latency                          : {reid_latency_frames} frames")
    else:
        metrics_report.append(f"Re-acquisition Latency                          : N/A (Never recovered)")
    metrics_report.append(f"Total False Alarms / Ignored Detections         : {false_alarms}")
    
    report_text = "\n".join(metrics_report)
    print(f"\n{report_text}")
    
    # Save to metrics.txt
    metrics_path = os.path.join(out_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(report_text)
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
