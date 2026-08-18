import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import time
import math
import argparse
import builtins
from collections import defaultdict
from PIL import Image

from torchvision import transforms
from model import UAVReIDNet
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Real-world Inference with Multi-UAV Coarse-to-Fine ReID")
    parser.add_argument('--config', type=str, default='configs/config_jetson.yaml')
    return parser.parse_args()

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

def expand_coarse_to_fine(feat):
    # Dùng 1 frame nhân bản thành 16 frame để tương thích với Mamba Temporal Encoder
    return feat.unsqueeze(1).repeat(1, 16, 1)

def main():
    args = parse_args()
    cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
            
    inf_cfg = cfg.get('infer_realworld', {})
    class Args: pass
    args_obj = Args()
    args_obj.video = inf_cfg.get('video', '')
    args_obj.yolo_model = inf_cfg.get('yolo_model', 'yolov8n.pt')
    args_obj.detector_conf = inf_cfg.get('detector_conf', 0.15)
    args_obj.detector_imgsz = inf_cfg.get('detector_imgsz', 1080)
    args_obj.detector_classes = inf_cfg.get('detector_classes', None)
    args_obj.lost_threshold = inf_cfg.get('lost_threshold', 10)
    args_obj.model_path = inf_cfg.get('model_path', './best_model.pth')
    args_obj.output_video = inf_cfg.get('output_video', 'output_realworld.mp4')
    args_obj.threshold = inf_cfg.get('reid_threshold', 0.75)
    args_obj.coarse_threshold = inf_cfg.get('coarse_threshold', 0.50)
    args_obj.num_frames = inf_cfg.get('num_frames', 16)
    args_obj.bbox_padding = inf_cfg.get('bbox_padding', 0.2)
    args_obj.max_anchor_size = inf_cfg.get('max_anchor_size', 10)
    args_obj.max_recent_size = inf_cfg.get('max_recent_size', 30)
    args_obj.query_img = inf_cfg.get('query_img', '')
    args_obj.start_sec = float(inf_cfg.get('start_sec', 0.0))
    args_obj.spatial_weight = inf_cfg.get('spatial_weight', 0.30)
    args = args_obj

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    video_name = os.path.basename(args.video).split('.')[0]
    out_dir = os.path.join(os.path.dirname(args.output_video), f"infer_realworld/{video_name}")
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

    final_output_path = os.path.join(out_dir, args.output_video)
    
    print(f"Initializing Mamba ReID Model...")
    model = UAVReIDNet()
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("Model loaded successfully.")
    else:
        print(f"Warning: Checkpoint {args.model_path} not found. Running with random weights.")
    
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"Initializing YOLOv8 Detector: {args.yolo_model}")
    yolo_model = YOLO(args.yolo_model)
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Video not found at {args.video}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_diagonal = math.sqrt(width**2 + height**2)
    
    if args.start_sec > 0:
        print(f"Skipping to {args.start_sec} seconds...")
        cap.set(cv2.CAP_PROP_POS_MSEC, int(args.start_sec * 1000))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(final_output_path, fourcc, fps, (width, height))

    anchor_bank = []
    recent_bank = []
    
    def update_memory_banks(feat):
        if len(anchor_bank) > 0:
            sim_to_anchors = [torch.mm(feat, g_feat.t()).item() for g_feat in anchor_bank]
            if max(sim_to_anchors) < 0.40:
                return True # Lỗi Hijacked
                
        if len(anchor_bank) < args.max_anchor_size:
            anchor_bank.append(feat)
        else:
            all_feats = anchor_bank + recent_bank
            sims = [torch.mm(feat, g_feat.t()).item() for g_feat in all_feats]
            if not sims or max(sims) < 0.95:
                recent_bank.append(feat)
                if len(recent_bank) > args.max_recent_size:
                    recent_bank.pop(0)
        return False
        
    # Xử lý Khởi tạo mục tiêu
    if args.query_img and os.path.exists(args.query_img):
        print(f"Loading Query Image for Target Initialization: {args.query_img}")
        q_img = cv2.imread(args.query_img)
        q_img = cv2.cvtColor(q_img, cv2.COLOR_BGR2RGB)
        q_tensor = transform(q_img).unsqueeze(0).to(device)
        q_feat = extract_cnn_feature(model, q_tensor)
        q_seq = expand_coarse_to_fine(q_feat)
        q_bn = compute_reid_embedding(model, q_seq)
        update_memory_banks(q_bn)
        status = "LOST"
        target_track_id = None
        print("Query Feature added to Anchor Bank. System entering SEARCH mode.")
    else:
        status = "INITIAL_TRACKING"
        target_track_id = None
        print("No Query Image provided. Auto-Locking on first detected object.")

    candidate_buffers = {}
    candidate_scores = {}
    tracking_buf = []
    blacklisted_ids = set()
    lost_count = 0
    frame_idx = 0
    mamba_times = []
    last_target_center = None
    
    print("Starting Multi-UAV Coarse-to-Fine Inference Stream...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        results = yolo_model.track(frame, persist=True, conf=args.detector_conf, imgsz=args.detector_imgsz, classes=args.detector_classes, verbose=False)
        
        filtered_boxes = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                if box.id is not None:
                    cls_id = int(box.cls[0])
                    if args.detector_classes is None or cls_id in args.detector_classes:
                        filtered_boxes.append(box)

        display_frame = frame.copy()
        target_box = None
        
        # 1. BẮT MÙ (Nếu không có Query Image)
        if status == "INITIAL_TRACKING" and target_track_id is None:
            if len(filtered_boxes) > 0:
                best_box = max(filtered_boxes, key=lambda b: float(b.conf[0]))
                target_track_id = int(best_box.id[0])
                candidate_buffers[target_track_id] = []
                print(f"[{frame_idx}] Auto-Locked onto Target Track ID: {target_track_id}")
                
        # 2. THEO DÕI VÀ XÁC THỰC (TRACKING)
        if status in ["INITIAL_TRACKING", "TRACKING"]:
            for box in filtered_boxes:
                if int(box.id[0]) == target_track_id:
                    target_box = box
                    break
                    
            if target_box is not None:
                lost_count = 0
                x1, y1, x2, y2 = target_box.xyxy[0].cpu().numpy().astype(int)
                b_x, b_y, b_w, b_h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                last_target_center = (b_x + b_w/2.0, b_y + b_h/2.0)
                
                pad_w, pad_h = int(b_w * args.bbox_padding), int(b_h * args.bbox_padding)
                x1_p, y1_p = max(0, b_x - pad_w), max(0, b_y - pad_h)
                x2_p, y2_p = min(width, b_x + b_w + pad_w), min(height, b_y + b_h + pad_h)
                
                crop = frame[y1_p:y2_p, x1_p:x2_p]
                if crop.size > 0:
                    crop_img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    tensor_frame = transform(crop_img).unsqueeze(0).to(device)
                    feat = extract_cnn_feature(model, tensor_frame)
                    
                    if status == "INITIAL_TRACKING":
                        candidate_buffers[target_track_id].append(feat)
                        if len(candidate_buffers[target_track_id]) == args.num_frames:
                            seq_feats = torch.stack(candidate_buffers[target_track_id], dim=1)
                            gallery_feat = compute_reid_embedding(model, seq_feats)
                            update_memory_banks(gallery_feat)
                            status = "TRACKING"
                            tracking_buf = candidate_buffers[target_track_id].copy()
                            print(f"[{frame_idx}] Initial Anchor Built. Transitioning to TRACKING.")
                            del candidate_buffers[target_track_id]
                    else: # CONTINUOUS VERIFICATION
                        tracking_buf.append(feat)
                        if len(tracking_buf) > args.num_frames: tracking_buf.pop(0)
                            
                        if frame_idx % 60 == 0 and len(tracking_buf) == args.num_frames:
                            seq_feats = torch.stack(tracking_buf, dim=1)
                            gallery_feat = compute_reid_embedding(model, seq_feats)
                            if update_memory_banks(gallery_feat):
                                print(f"[{frame_idx}] WARNING: ByteTrack Hijack Detected! ReID Score dropped < 0.40. Abandoning Target!")
                                status = "LOST"
                                blacklisted_ids.add(target_track_id)
                                target_track_id = None
                                candidate_buffers.clear()
                                candidate_scores.clear()
                                tracking_buf.clear()
                            else:
                                print(f"[{frame_idx}] Verification Passed.")
                            
                cv2.rectangle(display_frame, (b_x, b_y), (b_x+b_w, b_y+b_h), (0, 255, 0), 2)
                cv2.putText(display_frame, f"TARGET ID:{target_track_id}", (b_x, max(0, b_y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                lost_count += 1
                if lost_count >= args.lost_threshold:
                    print(f"[{frame_idx}] Target {target_track_id} LOST! Transitioning to SEARCH mode.")
                    status = "LOST"
                    target_track_id = None
                    candidate_buffers.clear()
                    candidate_scores.clear()
                    tracking_buf.clear()
                    
        # 3. TÌM KIẾM COARSE-TO-FINE (Nhiều UAV cùng lúc)
        if status == "LOST":
            active_ids = []
            for box in filtered_boxes:
                tid = int(box.id[0])
                if tid in blacklisted_ids:
                    continue
                    
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                b_x, b_y, b_w, b_h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                
                # Tính Phạt Không gian (Spatial Penalty)
                spatial_penalty = 0.0
                if last_target_center is not None:
                    cx, cy = b_x + b_w/2.0, b_y + b_h/2.0
                    dist = math.sqrt((cx - last_target_center[0])**2 + (cy - last_target_center[1])**2)
                    dist_norm = min(1.0, dist / frame_diagonal)
                    spatial_penalty = args.spatial_weight * dist_norm
                
                pad_w, pad_h = int(b_w * args.bbox_padding), int(b_h * args.bbox_padding)
                x1_p, y1_p = max(0, b_x - pad_w), max(0, b_y - pad_h)
                x2_p, y2_p = min(width, b_x + b_w + pad_w), min(height, b_y + b_h + pad_h)
                crop = frame[y1_p:y2_p, x1_p:x2_p]
                if crop.size == 0: continue
                
                crop_img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensor_frame = transform(crop_img).unsqueeze(0).to(device)
                feat = extract_cnn_feature(model, tensor_frame)
                
                # A. COARSE REID (Lọc 1 Frame)
                if tid not in candidate_buffers:
                    seq_coarse = expand_coarse_to_fine(feat)
                    bn_coarse = compute_reid_embedding(model, seq_coarse)
                    
                    anc_sims = [torch.mm(bn_coarse, g_feat.t()).item() for g_feat in anchor_bank]
                    max_anc = max(anc_sims) if anc_sims else 0.0
                    final_coarse = max_anc - spatial_penalty
                    
                    if final_coarse >= args.coarse_threshold:
                        candidate_buffers[tid] = [feat]
                        candidate_scores[tid] = final_coarse
                        print(f"[{frame_idx}] Coarse Match! ID:{tid} ReID:{max_anc:.2f} - Pen:{spatial_penalty:.2f} = {final_coarse:.2f} >= {args.coarse_threshold}. Collecting Fine frames...")
                
                # B. FINE REID (Kiểm chứng 16 Frames)
                else:
                    candidate_buffers[tid].append(feat)
                    if len(candidate_buffers[tid]) == args.num_frames:
                        seq_feats = torch.stack(candidate_buffers[tid], dim=1)
                        t0 = time.time()
                        gallery_feat = compute_reid_embedding(model, seq_feats)
                        mamba_times.append((time.time() - t0) * 1000)
                        
                        anc_sims = [torch.mm(gallery_feat, g_feat.t()).item() for g_feat in anchor_bank]
                        rec_sims = [torch.mm(gallery_feat, g_feat.t()).item() for g_feat in recent_bank]
                        max_anc = max(anc_sims) if anc_sims else 0.0
                        max_rec = max(rec_sims) if rec_sims else 0.0
                        best_sim = max(max_anc, max_rec)
                        final_sim = best_sim - spatial_penalty
                        
                        if final_sim >= args.threshold:
                            print(f"[{frame_idx}] FINE SUCCESS! ID:{tid} (ReID:{best_sim:.2f} - Pen:{spatial_penalty:.2f}) = {final_sim:.2f} >= {args.threshold}. LOCKING!")
                            status = "TRACKING"
                            target_track_id = tid
                            tracking_buf = candidate_buffers[tid].copy()
                            candidate_buffers.clear()
                            candidate_scores.clear()
                            lost_count = 0
                            update_memory_banks(gallery_feat)
                            break
                        else:
                            print(f"[{frame_idx}] FINE FAILED! ID:{tid} Score {final_sim:.2f} < {args.threshold}. Blacklisting ID.")
                            blacklisted_ids.add(tid)
                            del candidate_buffers[tid]
                            if tid in candidate_scores: del candidate_scores[tid]
                
                # Tìm Soft Lock (TID có điểm Coarse cao nhất)
                soft_lock_tid = max(candidate_scores, key=candidate_scores.get) if candidate_scores else None
                
                # Vẽ khung chờ / Soft Lock
                if tid in candidate_buffers:
                    if tid == soft_lock_tid:
                        # Soft Lock: Màu Cyan (Xanh lơ)
                        cv2.rectangle(display_frame, (b_x, b_y), (b_x+b_w, b_y+b_h), (255, 255, 0), 2)
                        cv2.putText(display_frame, f"SOFT LOCK ID:{tid} ({len(candidate_buffers[tid])}/16)", (b_x, max(0, b_y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    else:
                        # Candidate bình thường: Màu Vàng
                        cv2.rectangle(display_frame, (b_x, b_y), (b_x+b_w, b_y+b_h), (0, 255, 255), 2)
                        cv2.putText(display_frame, f"Verify ID:{tid} ({len(candidate_buffers[tid])}/16)", (b_x, max(0, b_y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                else:
                    cv2.rectangle(display_frame, (b_x, b_y), (b_x+b_w, b_y+b_h), (128, 128, 128), 1)
                    
            active_ids = [int(b.id[0]) for b in filtered_boxes]
            lost_candidates = [tid for tid in candidate_buffers.keys() if tid not in active_ids]
            for ltid in lost_candidates:
                del candidate_buffers[ltid]
                if ltid in candidate_scores: del candidate_scores[ltid]

        cv2.putText(display_frame, f"Status: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if status == "LOST" else (0, 255, 0), 2)
        if len(anchor_bank) > 0:
            cv2.putText(display_frame, f"Bank: {len(anchor_bank)} Anchor | {len(recent_bank)} Recent", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(display_frame)
        frame_idx += 1

    cap.release()
    out.release()
    metrics_file.close()

    print("Inference completed!")
    print(f"Output video saved to {final_output_path}")

if __name__ == '__main__':
    main()
