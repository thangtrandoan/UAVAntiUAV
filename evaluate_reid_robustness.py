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
import random
import matplotlib.pyplot as plt
import builtins

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
    parser = argparse.ArgumentParser(description="Real-time Inference with Robustness Evaluation (False Positives)")
    parser.add_argument("--seq-dir", type=str, default=None, help="Path to the target sequence directory")
    parser.add_argument("--data-root", type=str, default="./data/UAV-Anti-UAV/Test", help="Root dir of dataset to sample imposters from")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to yaml config")
    parser.add_argument("--output-video", type=str, default="output_robustness.mp4", help="Output video path")
    parser.add_argument("--threshold", type=float, default=0.7, help="ReID cosine similarity threshold")
    parser.add_argument("--num-frames", type=int, default=16, help="Number of frames for temporal modeling")
    parser.add_argument("--bbox-padding", type=float, default=0.2, help="BBox padding ratio")
    parser.add_argument("--num-imposters", type=int, default=5, help="Number of imposter videos to sample per verification")
    parser.add_argument("--max-anchor-size", type=int, default=5)
    parser.add_argument("--max-recent-size", type=int, default=15)
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
        resized = cv2.resize(crop, (crop_size, crop_size))
        return resized
    except Exception as e:
        return None

def sample_imposters(data_root, exclude_seq_name, num_imposters, model, transform, device, args):
    """Samples random 16-frame queries from OTHER videos to act as false positives"""
    imposter_feats = []
    
    if not os.path.exists(data_root):
        print(f"Warning: data-root {data_root} does not exist. Cannot sample imposters.")
        return []
        
    seqs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and d != exclude_seq_name]
    if not seqs:
        return []
        
    sampled_seqs = random.sample(seqs, min(num_imposters, len(seqs)))
    print(f"  -> Sampling {len(sampled_seqs)} imposters from: {sampled_seqs}")
    
    for seq in sampled_seqs:
        seq_path = os.path.join(data_root, seq)
        video_path = os.path.join(seq_path, f"{seq}.mp4")
        gt_path = os.path.join(seq_path, "groundtruth_rect.txt")
        absent_path = os.path.join(seq_path, "absent.txt")
        
        if not os.path.exists(video_path) or not os.path.exists(gt_path):
            continue
            
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
                
        # Find valid 16-frame windows
        valid_starts = []
        for i in range(len(bboxes) - args.num_frames):
            is_valid = True
            for j in range(i, i + args.num_frames):
                if j < len(absent) and absent[j] == 1:
                    is_valid = False; break
                if j < len(bboxes) and (bboxes[j][2] <= 0 or bboxes[j][3] <= 0):
                    is_valid = False; break
            if is_valid:
                valid_starts.append(i)
                
        if not valid_starts:
            continue
            
        start_idx = random.choice(valid_starts)
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        
        feature_buffer = []
        for j in range(args.num_frames):
            ret, frame = cap.read()
            if not ret: break
            
            bbox = bboxes[start_idx + j]
            crop = crop_and_resize(frame, bbox, args.bbox_padding)
            if crop is not None:
                tensor_frame = transform(crop).unsqueeze(0).to(device)
                feat = extract_cnn_feature(model, tensor_frame)
                feature_buffer.append(feat)
        
        cap.release()
        
        if len(feature_buffer) == args.num_frames:
            seq_feats = torch.stack(feature_buffer, dim=1)
            query_feat = compute_reid_embedding(model, seq_feats)
            imposter_feats.append(query_feat)
            
    return imposter_feats

def main():
    args = parse_args()
    
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
            if 'infer_robustness' in cfg:
                inf = cfg['infer_robustness']
                args.seq_dir = inf.get('seq_dir', args.seq_dir)
                args.data_root = inf.get('data_root', args.data_root)
                args.checkpoint = inf.get('model_path', args.checkpoint)
                args.output_video = inf.get('output_video', args.output_video)
                args.threshold = inf.get('reid_threshold', args.threshold)
                args.num_frames = inf.get('num_frames', args.num_frames)
                args.bbox_padding = inf.get('bbox_padding', args.bbox_padding)
                args.num_imposters = inf.get('num_imposters', args.num_imposters)
                args.max_anchor_size = inf.get('max_anchor_size', args.max_anchor_size)
                args.max_recent_size = inf.get('max_recent_size', args.max_recent_size)
            elif 'infer' in cfg:
                inf = cfg['infer']
                args.seq_dir = inf.get('seq_dir', args.seq_dir)
                args.checkpoint = inf.get('model_path', args.checkpoint)
                args.threshold = inf.get('reid_threshold', args.threshold)
                args.num_frames = inf.get('num_frames', args.num_frames)
                args.bbox_padding = inf.get('bbox_padding', args.bbox_padding)
                
    if not args.seq_dir:
        print("Error: --seq-dir must be provided.")
        return

    seq_name = os.path.basename(os.path.normpath(args.seq_dir))
    video_path = os.path.join(args.seq_dir, f"{seq_name}.mp4")
    gt_path = os.path.join(args.seq_dir, "groundtruth_rect.txt")
    absent_path = os.path.join(args.seq_dir, "absent.txt")
    
    out_dir = os.path.join("infer_robustness", seq_name)
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "config_used.yaml"), "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False)
        
    final_output_path = os.path.join(out_dir, args.output_video)
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
    
    log_file.write("--- CSV DATA: Frame, Genuine_Sim, Imposter_Sims ---\n")
    
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return
        
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

    print(f"Initializing Model: {args.checkpoint}...")
    model = UAVReIDNet(freeze_backbone=False)
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        new_state_dict = {}
        model_state = model.state_dict()
        for k, v in state_dict.items():
            new_k = k.replace('_orig_mod.', '') if k.startswith('_orig_mod.') else k
            if new_k in model_state and v.shape != model_state[new_k].shape: continue
            new_state_dict[new_k] = v
        model.load_state_dict(new_state_dict, strict=False)
    else:
        print(f"Warning: Checkpoint not found. Random weights.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
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
    
    anchor_bank = []
    recent_bank = []
    feature_buffer = []
    
    def update_memory_banks(feat):
        if len(anchor_bank) > 0:
            sim_to_anchors = [torch.mm(feat, g_feat.t()).item() for g_feat in anchor_bank]
            if max(sim_to_anchors) < 0.40:
                return True # Hijacked
                
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
    
    status = "INITIAL_TRACKING"
    frame_idx = 0
    
    # Store metrics for final report
    all_genuine_scores = []
    all_imposter_scores = []
    
    print(f"Starting Robustness Inference on {seq_name}...")
    
    # Variables for visualization overlay
    last_verification_text = ""
    last_verification_color = (255, 255, 255)
    overlay_timer = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
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
                        
                    seq_feats = torch.stack(feature_buffer, dim=1)
                    gallery_feat = compute_reid_embedding(model, seq_feats)
                    update_memory_banks(gallery_feat)
                status = "LOST"
                feature_buffer = []
            elif valid_bbox:
                if frame_idx % 3 == 0:
                    crop = crop_and_resize(frame, bbox, args.bbox_padding)
                    if crop is not None:
                        tensor_frame = transform(crop).unsqueeze(0).to(device)
                        feat = extract_cnn_feature(model, tensor_frame)
                        feature_buffer.append(feat)
                        if len(feature_buffer) > args.num_frames: feature_buffer.pop(0)
                        
                        if frame_idx % 60 == 0 and len(feature_buffer) == args.num_frames:
                            seq_feats = torch.stack(feature_buffer, dim=1)
                            gallery_feat = compute_reid_embedding(model, seq_feats)
                            if update_memory_banks(gallery_feat):
                                status = "LOST"
                                feature_buffer = []
                        
                if status != "LOST":
                    cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255, 0, 0), 2)
                    cv2.putText(display_frame, "Tracking (Building Gallery)", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        elif status == "LOST":
            cv2.putText(display_frame, "TARGET LOST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            if not is_absent and valid_bbox:
                status = "REAPPEARED_VERIFYING"
                feature_buffer = []
                if hasattr(model, 'cached_imposters'): model.cached_imposters = []
                print(f"[{frame_idx}] Re-appeared! Collecting frames...")

        elif status == "REAPPEARED_VERIFYING":
            cv2.putText(display_frame, "VERIFYING ID...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
            
            if is_absent:
                status = "LOST"
                feature_buffer = []
            elif valid_bbox:
                crop = crop_and_resize(frame, bbox, args.bbox_padding)
                if crop is not None:
                    tensor_frame = transform(crop).unsqueeze(0).to(device)
                    feat = extract_cnn_feature(model, tensor_frame)
                    feature_buffer.append(feat)
                    
                cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 165, 255), 2)
                
                if len(feature_buffer) == args.num_frames:
                    print(f"\n--- [Frame {frame_idx}] ROBUSTNESS TEST INITIATED ---")
                    # 1. Genuine Query
                    seq_feats = torch.stack(feature_buffer, dim=1)
                    query_feature = compute_reid_embedding(model, seq_feats)
                    
                    anchor_sims = [torch.mm(query_feature, g_feat.t()).item() for g_feat in anchor_bank]
                    recent_sims = [torch.mm(query_feature, g_feat.t()).item() for g_feat in recent_bank]
                    max_anc = max(anchor_sims) if anchor_sims else 0.0
                    max_rec = max(recent_sims) if recent_sims else 0.0
                    best_genuine = max(max_anc, max_rec)
                    
                    all_genuine_scores.append(best_genuine)
                    
                    # 2. Imposter Queries (Cached during rolling window)
                    if not hasattr(model, 'cached_imposters') or len(model.cached_imposters) == 0:
                        model.cached_imposters = sample_imposters(args.data_root, seq_name, args.num_imposters, model, transform, device, args)
                    
                    imposter_best_sims = []
                    for imp_feat in model.cached_imposters:
                        imp_anc_sims = [torch.mm(imp_feat, g_feat.t()).item() for g_feat in anchor_bank]
                        imp_rec_sims = [torch.mm(imp_feat, g_feat.t()).item() for g_feat in recent_bank]
                        i_max_anc = max(imp_anc_sims) if imp_anc_sims else 0.0
                        i_max_rec = max(imp_rec_sims) if imp_rec_sims else 0.0
                        imp_best = max(i_max_anc, i_max_rec)
                        imposter_best_sims.append(imp_best)
                    
                    all_imposter_scores.extend(imposter_best_sims)
                    
                    imp_str = ",".join([f"{s:.3f}" for s in imposter_best_sims])
                    log_file.write(f"{frame_idx}, {best_genuine:.3f}, {imp_str}\n")
                    log_file.flush()
                    
                    print(f"Genuine Sim: {best_genuine:.3f}")
                    for i, s in enumerate(imposter_best_sims):
                        print(f"Imposter {i+1} Sim: {s:.3f}")
                        
                    max_imposter = max(imposter_best_sims) if imposter_best_sims else 0.0
                    
                    overlay_timer = 90 # show result for ~3 seconds (at 30fps)
                    if max_imposter > args.threshold and max_imposter > best_genuine:
                        print(f"Result: HIJACKED BY IMPOSTER!")
                        feature_buffer.pop(0)
                        last_verification_text = f"HIJACKED! | Gen: {best_genuine:.2f} (A:{max_anc:.2f}|R:{max_rec:.2f}) < Imp: {max_imposter:.2f}"
                        last_verification_color = (255, 0, 255) # Purple (BGR)
                    elif best_genuine > args.threshold:
                        status = "REID_SUCCESS"
                        print(f"Result: REID SUCCESS (Threshold {args.threshold})")
                        last_verification_text = f"ReID Pass | Gen: {best_genuine:.2f} (A:{max_anc:.2f}|R:{max_rec:.2f}) | Imp: {max_imposter:.2f}"
                        last_verification_color = (0, 255, 0)
                    else:
                        print(f"Result: REID FAILED. Rolling window...")
                        feature_buffer.pop(0)
                        last_verification_text = f"ReID Fail | Gen: {best_genuine:.2f} (A:{max_anc:.2f}|R:{max_rec:.2f}) | Imp: {max_imposter:.2f}"
                        last_verification_color = (0, 0, 255)

        elif status == "REID_SUCCESS":
            if is_absent:
                if len(feature_buffer) > 0:
                    while len(feature_buffer) < args.num_frames: feature_buffer.append(feature_buffer[-1])
                    if len(feature_buffer) > args.num_frames:
                        indices = np.linspace(0, len(feature_buffer)-1, args.num_frames).astype(int)
                        feature_buffer = [feature_buffer[i] for i in indices]
                    seq_feats = torch.stack(feature_buffer, dim=1)
                    gallery_feat = compute_reid_embedding(model, seq_feats)
                    update_memory_banks(gallery_feat)
                status = "LOST"
                feature_buffer = []
            elif valid_bbox:
                if frame_idx % 3 == 0:
                    crop = crop_and_resize(frame, bbox, args.bbox_padding)
                    if crop is not None:
                        tensor_frame = transform(crop).unsqueeze(0).to(device)
                        feat = extract_cnn_feature(model, tensor_frame)
                        feature_buffer.append(feat)
                        if len(feature_buffer) > args.num_frames: feature_buffer.pop(0)
                        
                        if frame_idx % 60 == 0 and len(feature_buffer) == args.num_frames:
                            seq_feats = torch.stack(feature_buffer, dim=1)
                            gallery_feat = compute_reid_embedding(model, seq_feats)
                            if update_memory_banks(gallery_feat):
                                status = "LOST"
                                feature_buffer = []
                        
                if status != "LOST":
                    cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
                    cv2.putText(display_frame, "ReID Tracked", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw Robustness Overlay
        if overlay_timer > 0:
            cv2.putText(display_frame, last_verification_text, (50, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, last_verification_color, 2)
            overlay_timer -= 1
            
        out_vid.write(display_frame)
        frame_idx += 1

    cap.release()
    out_vid.release()
    
    print("\n================ ROBUSTNESS REPORT ================")
    print(f"Total Verifications (Genuine Queries)  : {len(all_genuine_scores)}")
    print(f"Total Imposter Queries Evaluated       : {len(all_imposter_scores)}")
    
    if all_genuine_scores and all_imposter_scores:
        avg_gen = np.mean(all_genuine_scores)
        avg_imp = np.mean(all_imposter_scores)
        print(f"\nAverage Genuine Similarity : {avg_gen:.4f}")
        print(f"Average Imposter Similarity: {avg_imp:.4f}")
        print(f"Average Margin (Gen - Imp) : {avg_gen - avg_imp:.4f}")
        
        # Calculate FAR and FRR based on current threshold
        frr = sum(1 for x in all_genuine_scores if x < args.threshold) / len(all_genuine_scores)
        far = sum(1 for x in all_imposter_scores if x >= args.threshold) / len(all_imposter_scores)
        print(f"\nAt Threshold {args.threshold}:")
        print(f"False Rejection Rate (FRR) : {frr*100:.2f}% (Target UAV not recognized)")
        print(f"False Acceptance Rate (FAR): {far*100:.2f}% (Imposter UAV accepted)")
        
        # Generate Histogram Plot
        plt.figure(figsize=(10, 6))
        plt.hist(all_genuine_scores, bins=20, alpha=0.6, label='Genuine (True UAV)', color='green')
        plt.hist(all_imposter_scores, bins=20, alpha=0.6, label='Imposter (False UAV)', color='red')
        plt.axvline(args.threshold, color='blue', linestyle='dashed', linewidth=2, label=f'Threshold ({args.threshold})')
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
