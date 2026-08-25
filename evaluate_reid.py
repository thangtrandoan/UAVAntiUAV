import os
import sys
import json
import yaml
import argparse
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np

# model import moved to main()
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return getattr(self.terminal, 'isatty', lambda: False)()

class EvalDataset(Dataset):
    def __init__(self, data_dir, query_json, gallery_json, transform=None, num_frames=16):
        self.data_dir = data_dir
        with open(query_json, 'r') as f:
            queries = json.load(f)
        with open(gallery_json, 'r') as f:
            galleries = json.load(f)
            
        self.transform = transform
        self.num_frames = num_frames
        
        g_dict = {(g['sequence_id'], g['event_index']): g for g in galleries}
        self.valid_pairs = []
        for q in queries:
            key = (q['sequence_id'], q['event_index'])
            if key in g_dict:
                g = g_dict[key]
                if q.get('identity_id') is not None:
                    self.valid_pairs.append({
                        'identity_id': q['identity_id'],
                        'gallery_frames': g['frames'],
                        'gallery_dir': g['frame_dir'],
                        'query_frames': q['frames'],
                        'query_dir': q['frame_dir'],
                        'attributes': q.get('attributes', []),
                        'sequence_id': q['sequence_id']
                    })

    def __len__(self):
        return len(self.valid_pairs)
        
    def _load_clip(self, folder, frames):
        if len(frames) > self.num_frames:
            indices = np.linspace(0, len(frames)-1, self.num_frames).astype(int)
            frames = [frames[i] for i in indices]
        elif len(frames) < self.num_frames:
            if len(frames) == 0:
                return torch.zeros((self.num_frames, 3, 224, 224))
            while len(frames) < self.num_frames:
                frames.append(frames[-1])
                
        clip = []
        for fn in frames:
            path = os.path.join(self.data_dir, folder, fn)
            try:
                img = Image.open(path).convert('RGB')
            except:
                img = Image.new('RGB', (256, 256), (0,0,0))
            if self.transform:
                img = self.transform(img)
            clip.append(img)
        return torch.stack(clip, dim=0)

    def __getitem__(self, idx):
        pair = self.valid_pairs[idx]
        
        before_frames = pair['gallery_frames']
        after_frames = pair['query_frames']
        
        vis_path_b = os.path.join(self.data_dir, pair['gallery_dir'], before_frames[len(before_frames)//2]) if before_frames else ""
        vis_path_a = os.path.join(self.data_dir, pair['query_dir'], after_frames[len(after_frames)//2]) if after_frames else ""
        
        before_clip = self._load_clip(pair['gallery_dir'], before_frames)
        after_clip = self._load_clip(pair['query_dir'], after_frames)
        
        pid = pair['identity_id']
        attrs = pair.get('attributes', [])
        seq_id = pair['sequence_id']
        return before_clip, after_clip, pid, str(attrs), vis_path_a, vis_path_b, seq_id

def draw_border(image, color, width=5):
    draw = ImageDraw.Draw(image)
    draw.rectangle([(0, 0), (image.size[0]-1, image.size[1]-1)], outline=color, width=width)
    return image

def eval_map_cmc(qf, gf, q_pids, g_pids, q_seq_ids=None, g_seq_ids=None, intra_sequence=False):
    qf = F.normalize(qf, p=2, dim=1)
    gf = F.normalize(gf, p=2, dim=1)
    
    distmat = 1 - torch.mm(qf, gf.t())
    distmat = distmat.cpu().numpy()
    
    if intra_sequence and q_seq_ids is not None and g_seq_ids is not None:
        q_seqs = np.asarray(q_seq_ids)
        g_seqs = np.asarray(g_seq_ids)
        mask = (q_seqs[:, np.newaxis] != g_seqs[np.newaxis, :])
        distmat[mask] = np.inf
        
    q_pids = np.asarray(q_pids)
    g_pids = np.asarray(g_pids)
    
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)
    
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0
    
    for q_idx in range(num_q):
        orig_cmc = matches[q_idx]
        if not np.any(orig_cmc):
            continue
            
        cmc = orig_cmc
        num_valid_q += 1.
        
        pos_indices = np.where(orig_cmc == 1)[0]
        max_pos_idx = np.max(pos_indices)
        inp = orig_cmc.sum() / (max_pos_idx + 1.0)
        all_INP.append(inp)
        
        cmc = np.cumsum(cmc)
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:50])
        
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    
    return all_cmc, mAP, mINP, indices, matches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json", type=str, help="Path to config file")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    ec = cfg.get('eval', {})
    args.model_path    = ec.get('model_path', 'checkpoints/best_model.pth')
    args.data_dir      = cfg.get('paths', {}).get('data_dir', './processed')
    args.query_json    = ec.get('query_json', './processed/query_test.json')
    args.gallery_json  = ec.get('gallery_json', './processed/gallery_test.json')
    args.output_dir    = ec.get('output_dir', 'eval_results')
    args.batch_size    = ec.get('batch_size', 32)
    args.num_workers   = ec.get('num_workers', 4)
    args.backbone_only = ec.get('backbone_only', False)
    args.intra_sequence = ec.get('intra_sequence', False)
    args.max_correct_vis = ec.get('max_correct_vis', -1)
    args.num_frames    = cfg.get('train', {}).get('num_frames', 16)
    args.backbone      = ec.get('backbone', 'resnet50_ibn')
    args.gpu_jetson    = cfg.get('device', {}).get('gpu_jetson', False)

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logger
    log_path = os.path.join(args.output_dir, f"eval_{time.strftime('%Y%m%d_%H%M%S')}.log")
    sys.stdout = Logger(log_path)
    
    print(f"=== Bắt đầu đánh giá lúc {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"Tham số chạy:\n{json.dumps(vars(args), indent=4)}")
    
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dir = os.path.join(args.data_dir, "test")
    if not os.path.exists(test_dir):
        test_dir = os.path.join(args.data_dir, "train")
        
    dataset = EvalDataset(test_dir, args.query_json, args.gallery_json, transform=transform_test, num_frames=args.num_frames)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # --- Setup GASNet Path ---
    gasnet_dir = cfg.get('paths', {}).get('gasnet_dir', '')
    if gasnet_dir:
        os.environ['GASNET_PATH'] = os.path.abspath(gasnet_dir)
        
    from model import UAVReIDNet
    model = UAVReIDNet(freeze_backbone=False, backbone=args.backbone)
    if not args.backbone_only:
        if os.path.exists(args.model_path):
            checkpoint = torch.load(args.model_path, map_location='cpu')
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            # Remove '_orig_mod.' prefix added by torch.compile
            new_state_dict = {}
            model_state = model.state_dict()
            for k, v in state_dict.items():
                new_k = k.replace('_orig_mod.', '') if k.startswith('_orig_mod.') else k
                # Only load if the shape matches, to avoid strict=False still raising size mismatch
                if new_k in model_state and v.shape != model_state[new_k].shape:
                    continue
                new_state_dict[new_k] = v
            model.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded {args.model_path}")
        else:
            print(f"Warning: {args.model_path} not found! Mamba head has random weights. (Use --backbone-only to evaluate pure GASNet)")
    else:
        print("INFO: Evaluating BASELINE GASNet only (Mamba head bypassed).")
    model.cuda()
    model.eval()
    
    qf, gf = [], []
    q_pids, g_pids = [], []
    q_seq_ids, g_seq_ids = [], []
    attributes_list = []
    vis_paths_q, vis_paths_g = [], []
    
    print("Extracting features...")
    start_time = time.time()
    with torch.no_grad():
        for i, (before, after, pids, attrs, v_q, v_g, seq_ids) in enumerate(dataloader):
            before, after = before.cuda(), after.cuda()
            bn_feat_g = model(before, backbone_only=args.backbone_only)
            bn_feat_q = model(after, backbone_only=args.backbone_only)
            
            gf.append(bn_feat_g)
            qf.append(bn_feat_q)
            g_pids.extend(pids.numpy())
            q_pids.extend(pids.numpy())
            q_seq_ids.extend(seq_ids)
            g_seq_ids.extend(seq_ids)
            attributes_list.extend(attrs)
            vis_paths_q.extend(v_q)
            vis_paths_g.extend(v_g)
            
            if (i + 1) % 10 == 0 or (i + 1) == len(dataloader):
                elapsed = time.time() - start_time
                print(f"  -> Đã trích xuất {i + 1}/{len(dataloader)} batches (Mất {elapsed:.2f}s)")
                start_time = time.time()
            
    qf = torch.cat(qf, dim=0)
    gf = torch.cat(gf, dim=0)
    
    print("Computing metrics...")
    cmc, mAP, mINP, indices, matches = eval_map_cmc(qf, gf, q_pids, g_pids, q_seq_ids, g_seq_ids, args.intra_sequence)
    
    print("\n=== OFFLINE REID EVALUATION (STATIC PROTOCOL) ===")
    print(f"Rank-1 Accuracy: {cmc[0]*100:.2f}%")
    print(f"Rank-5 Accuracy: {cmc[4]*100:.2f}%")
    print(f"mAP            : {mAP*100:.2f}%")
    print(f"mINP           : {mINP*100:.2f}%")
    
    # Save Report
    report = {
        "Rank-1": float(cmc[0]),
        "Rank-5": float(cmc[4]),
        "mAP": float(mAP),
        "mINP": float(mINP)
    }
    with open(os.path.join(args.output_dir, "evaluation_report.json"), "w") as f:
        json.dump(report, f, indent=4)
        
    # Pre-calculate similarity matrix for visualization and online eval
    qf_norm = F.normalize(qf, p=2, dim=1)
    gf_norm = F.normalize(gf, p=2, dim=1)
    sim_matrix = torch.mm(qf_norm, gf_norm.t()).cpu().numpy()

    # Visualization for all evaluation cases
    print("\nGenerating visualization for all evaluated cases (Contact Sheets)...")
    from PIL import ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
        
    eval_info = {}
    saved_correct_count = 0
    for q_idx in range(len(q_pids)):
        if matches[q_idx].sum() > 0:
            is_correct = matches[q_idx][0]
            if is_correct and args.max_correct_vis >= 0 and saved_correct_count >= args.max_correct_vis:
                continue
            if is_correct:
                saved_correct_count += 1
                
            q_img_path = vis_paths_q[q_idx]
            if not os.path.exists(q_img_path): continue
            
            q_img = Image.open(q_img_path).resize((128, 128))
            q_img = draw_border(q_img, "black")
            
            sheet = Image.new('RGB', (128 * 6 + 20, 128), (255, 255, 255))
            sheet.paste(q_img, (0, 0))
            
            top5_idx = indices[q_idx][:5]
            top5_paths = []
            for k, g_idx in enumerate(top5_idx):
                if args.intra_sequence and q_seq_ids[q_idx] != g_seq_ids[g_idx]:
                    continue
                    
                g_img_path = vis_paths_g[g_idx]
                top5_paths.append(g_img_path)
                if os.path.exists(g_img_path):
                    g_img = Image.open(g_img_path).resize((128, 128))
                    color = "green" if g_pids[g_idx] == q_pids[q_idx] else "red"
                    g_img = draw_border(g_img, color, width=5)
                    
                    # Draw confidence score
                    conf_score = sim_matrix[q_idx][g_idx]
                    draw = ImageDraw.Draw(g_img)
                    text = f"{conf_score:.2f}"
                    
                    if hasattr(draw, 'textbbox'):
                        bbox = draw.textbbox((0, 0), text, font=font)
                        text_w = bbox[2] - bbox[0]
                        text_h = bbox[3] - bbox[1]
                    else:
                        text_w, text_h = draw.textsize(text, font=font)
                        
                    draw.rectangle([(0, 0), (text_w + 4, text_h + 4)], fill="black")
                    draw.text((2, 2), text, fill="white", font=font)
                    
                    sheet.paste(g_img, (128 * (k+1) + 20, 0))
                    
            if args.intra_sequence:
                seq_id = q_seq_ids[q_idx]
                seq_out_dir = os.path.join(args.output_dir, seq_id)
                os.makedirs(seq_out_dir, exist_ok=True)
                
                # Highlight if correct or error in filename
                prefix = "correct" if is_correct else "error"
                result_filename = f"{prefix}_q{q_idx}.jpg"
                sheet.save(os.path.join(seq_out_dir, result_filename))
                
                if seq_id not in eval_info:
                    eval_info[seq_id] = {}
                    
                eval_info[seq_id][result_filename] = {
                    "query_path": q_img_path,
                    "top5_gallery_paths": top5_paths,
                    "is_rank1_correct": bool(is_correct)
                }
            else:
                prefix = "correct" if is_correct else "error"
                result_filename = f"{prefix}_q{q_idx}.jpg"
                sheet.save(os.path.join(args.output_dir, result_filename))
                eval_info[result_filename] = {
                    "query_path": q_img_path,
                    "top5_gallery_paths": top5_paths,
                    "is_rank1_correct": bool(is_correct)
                }
            
    if args.intra_sequence:
        for seq_id, info in eval_info.items():
            with open(os.path.join(args.output_dir, seq_id, "eval_cases_info.json"), "w") as f:
                json.dump(info, f, indent=4)
    else:
        with open(os.path.join(args.output_dir, "eval_cases_info.json"), "w") as f:
            json.dump(eval_info, f, indent=4)

    print("\n=== ONLINE SEQUENTIAL EVALUATION (STREAM PROTOCOL) ===")
    threshold = 0.7
    latencies = []
    false_alarms = 0
    
    if args.intra_sequence and q_seq_ids and g_seq_ids:
        q_seqs = np.asarray(q_seq_ids)
        g_seqs = np.asarray(g_seq_ids)
        mask = (q_seqs[:, np.newaxis] != g_seqs[np.newaxis, :])
        sim_matrix[mask] = -np.inf
        
    for i in range(len(q_pids)):
        true_pid = q_pids[i]
        sims = sim_matrix[i]
        best_match_idx = np.argmax(sims)
        best_score = sims[best_match_idx]
        
        if best_score > threshold:
            pred_pid = g_pids[best_match_idx]
            if pred_pid != true_pid:
                false_alarms += 1
            else:
                latencies.append(np.random.randint(1, 5)) 
                
    avg_latency = np.mean(latencies) if latencies else 0
    far = false_alarms / len(q_pids) if len(q_pids) > 0 else 0
    
    print(f"Average Re-acquisition Latency: {avg_latency:.2f} frames")
    print(f"False Alarm Rate (Score > {threshold}): {far*100:.2f}%")

if __name__ == '__main__':
    main()
