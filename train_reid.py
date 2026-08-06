import os
import sys
import json
import yaml
import argparse
import time
import random
import math
import numpy as np
from collections import defaultdict
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms
# (Removed deprecated torch.cuda.amp import)

# model import moved to main() to allow config parsing first
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

# ==========================================
# 1. LOSS FUNCTIONS
# ==========================================

class LabelSmoothCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        log_probs = self.log_softmax(inputs)
        targets_oh = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_oh = (1 - self.epsilon) * targets_oh + self.epsilon / num_classes
        loss = (-targets_oh * log_probs).sum(1).mean()
        return loss

class HardTripletLoss(nn.Module):
    """
    Online Hard Mining Triplet Loss
    """
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        inputs = F.normalize(inputs, p=2, dim=1)
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()

        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            if len(dist[i][mask[i] == 0]) > 0:
                dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
            else:
                dist_an.append(torch.tensor([0.0], device=dist.device))

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        if torch.all(dist_an == 0):
            return torch.tensor(0.0, requires_grad=True, device=dist.device)

        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

class CenterLoss(nn.Module):
    """
    Kéo các feature thuộc cùng 1 identity về cùng 1 center
    """
    def __init__(self, num_classes=1000, feat_dim=2816, lr_center=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        self.lr_center = lr_center

    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)
        
        classes = torch.arange(self.num_classes).long().to(x.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss

class TemporalConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, temporal_features):
        if len(temporal_features.shape) == 2 or temporal_features.size(1) < 2:
            return torch.tensor(0.0).to(temporal_features.device)
            
        sim = F.cosine_similarity(temporal_features[:, :-1, :], temporal_features[:, 1:, :], dim=-1)
        loss = 1.0 - sim.mean()
        return loss

# ==========================================
# 2. DATASET & DATALOADER
# ==========================================

class UAVReIDDataset(Dataset):
    def __init__(self, data_dir, json_path, transform=None, num_frames=16):
        self.data_dir = data_dir
        with open(json_path, 'r') as f:
            self.pairs = json.load(f)
        self.transform = transform
        self.num_frames = num_frames
        
        self.valid_pairs = [p for p in self.pairs if p['identity_id'] is not None]
        self.identities = sorted(list(set(p['identity_id'] for p in self.valid_pairs)))
        self.id_to_idx = {pid: i for i, pid in enumerate(self.identities)}
        self.num_identities = len(self.identities)

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
        seq_event = f"{pair['sequence_id']}_event_{pair['event_index']}"
        before_dir = os.path.join(seq_event, "before")
        after_dir = os.path.join(seq_event, "after")
        
        before_clip = self._load_clip(before_dir, pair['before_frames'])
        after_clip = self._load_clip(after_dir, pair['after_frames'])
        
        pid = self.id_to_idx[pair['identity_id']]
        return before_clip, after_clip, pid

class ReIDBatchSampler(Sampler):
    """
    PK Sampler đảm bảo mỗi batch có P identities, mỗi identity có K instances
    để hỗ trợ Hard Triplet Loss.
    """
    def __init__(self, dataset, batch_size, num_instances=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        
        self.index_dic = defaultdict(list)
        for index, pair in enumerate(self.dataset.valid_pairs):
            pid = self.dataset.id_to_idx[pair['identity_id']]
            self.index_dic[pid].append(index)
            
        self.pids = list(self.index_dic.keys())
        self.length = 0
        for pid in self.pids:
            num = len(self.index_dic[pid])
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances
            
    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = self.index_dic[pid].copy()
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
            random.shuffle(idxs)
            
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []
                    
        avai_pids = self.pids.copy()
        final_idxs = []
        
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)
                    
        batches = []
        for i in range(0, len(final_idxs), self.batch_size):
            batch = final_idxs[i:i + self.batch_size]
            if len(batch) == self.batch_size:
                batches.append(batch)
                
        return iter(batches)

    def __len__(self):
        return self.length // self.batch_size

# ==========================================
# 3. TRAINING LOOP
# ==========================================

def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        else:
            progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json", type=str, help="Path to config file")
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    args.data_dir       = cfg.get('paths', {}).get('data_dir', 'processed')
    args.pairs_json     = os.path.join(args.data_dir, 'pairs_train.json')
    args.gasnet_weights = cfg.get('paths', {}).get('gasnet_weights', '')
    args.checkpoint_dir = cfg.get('paths', {}).get('checkpoint_dir', 'checkpoints')
    args.log_dir        = cfg.get('paths', {}).get('log_dir', 'logs')
    
    args.gpu_jetson     = cfg.get('device', {}).get('gpu_jetson', False)
    
    # --- A100/H100/Colab Optimizations ---
    if not args.gpu_jetson:
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high') # TF32 for Ampere+ GPUs
            
    tc = cfg.get('train', {})
    args.resume         = tc.get('resume', '')
    args.batch_size     = tc.get('batch_size', 32)
    args.num_instances  = tc.get('num_instances', 4)
    args.num_frames     = tc.get('num_frames', 16)
    args.num_workers    = tc.get('num_workers', 4)
    args.use_amp        = tc.get('use_amp', False)
    args.pin_memory     = tc.get('pin_memory', False)
    args.use_compile    = tc.get('use_compile', False) # Thêm cờ bật/tắt compile
    args.val_freq       = tc.get('val_freq', 5) # Đọc số epoch đánh giá từ config (mặc định 5)

    
    args.epochs_stage1      = tc.get('stage1', {}).get('epochs', 30)
    args.lr_stage1          = float(tc.get('stage1', {}).get('lr', 3.5e-4))
    args.weight_decay       = float(tc.get('stage1', {}).get('weight_decay', 5e-4))
    
    args.epochs_stage2      = tc.get('stage2', {}).get('epochs', 20)
    args.lr_stage2_backbone = float(tc.get('stage2', {}).get('lr_backbone', 1e-5))
    args.lr_stage2_temporal = float(tc.get('stage2', {}).get('lr_temporal', 1e-4))
    args.lr_stage2_head     = float(tc.get('stage2', {}).get('lr_head', 1e-4))
    
    lc = tc.get('loss', {})
    args.lam1           = float(lc.get('lam1', 1.0))
    args.lam2           = float(lc.get('lam2', 0.5))
    args.lam3           = float(lc.get('lam3', 0.05))
    args.lr_center      = float(lc.get('lr_center', 0.5))

    train_dir = os.path.join(args.data_dir, "train")
    
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log")
    sys.stdout = Logger(log_path)
    
    print(f"=== Bắt đầu huấn luyện lúc {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"Tham số chạy: {json.dumps(vars(args), indent=4)}")
    
    
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    ])

    dataset = UAVReIDDataset(train_dir, args.pairs_json, transform=transform_train, num_frames=args.num_frames)
    num_identities = dataset.num_identities
    
    batch_size = args.batch_size
    num_instances = args.num_instances
    sampler = ReIDBatchSampler(dataset, batch_size=batch_size, num_instances=num_instances)
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=args.pin_memory)

    # --- Setup Validation ---
    try:
        from evaluate_reid import EvalDataset, eval_map_cmc
        transform_test = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_dir = os.path.join(args.data_dir, "test")
        if not os.path.exists(test_dir):
            test_dir = os.path.join(args.data_dir, "train")
        val_pairs_json = args.pairs_json.replace('pairs_train', 'pairs_test')
        val_dataset = EvalDataset(test_dir, val_pairs_json, transform=transform_test, num_frames=args.num_frames)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        has_val = True
    except Exception as e:
        print(f"Cảnh báo: Không thể setup validation, sẽ bỏ qua bước này. Lỗi: {e}")
        has_val = False

    def validate(model, loader):
        print(" --- Đang chạy Validation trên tập Test ---")
        model.eval()
        qf, gf, q_pids, g_pids = [], [], [], []
        start_t = time.time()
        with torch.no_grad():
            for before, after, pids, _, _, _ in loader:
                before, after = before.cuda(), after.cuda()
                bn_feat_g = model(before)
                bn_feat_q = model(after)
                gf.append(bn_feat_g)
                qf.append(bn_feat_q)
                g_pids.extend(pids.numpy())
                q_pids.extend(pids.numpy())
        qf = torch.cat(qf, dim=0)
        gf = torch.cat(gf, dim=0)
        cmc, mAP, mINP, _, _ = eval_map_cmc(qf, gf, q_pids, g_pids)
        print(f" -> Val Time: {time.time() - start_t:.2f}s | Rank-1: {cmc[0]*100:.2f}% | mAP: {mAP*100:.2f}%")
        model.train()
        return float(cmc[0]) # Trả về Rank-1


    # --- Setup GASNet Path ---
    gasnet_dir = cfg.get('paths', {}).get('gasnet_dir', '')
    if gasnet_dir:
        os.environ['GASNET_PATH'] = os.path.abspath(gasnet_dir)
        
    from model import UAVReIDNet
    model = UAVReIDNet(gasnet_weights_path=args.gasnet_weights or None, num_identities=num_identities, freeze_backbone=True)
    model.cuda()
    
    # --- torch.compile cho tốc độ tối đa trên A100/H100 ---
    if not args.gpu_jetson and args.use_compile and hasattr(torch, 'compile'):
        print("Bật torch.compile() để tối ưu model...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"Cảnh báo: torch.compile() thất bại: {e}. Sẽ chạy mode bình thường.")

    criterion_id = LabelSmoothCrossEntropy()
    criterion_triplet = HardTripletLoss(margin=0.3)
    criterion_center = CenterLoss(num_classes=num_identities, feat_dim=3072).cuda()

    # Stage 1 Optimizer (Backbone frozen)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                  lr=args.lr_stage1, weight_decay=args.weight_decay)
    optimizer_center = torch.optim.SGD(criterion_center.parameters(), lr=args.lr_center)
    
    scaler = torch.amp.GradScaler('cuda', enabled=args.use_amp)

    epochs_stage1 = args.epochs_stage1
    epochs_stage2 = args.epochs_stage2
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    def train_epoch(epoch, stage_name, optim, lr_scheduler=None, lam1=1.0, lam2=0.5, lam3=0.005):
        model.train()
        start_time = time.time()
        epoch_loss = 0.0
        
        for i, (before, after, pids) in enumerate(dataloader):
            before, after, pids = before.cuda(), after.cuda(), pids.cuda()
            
            optim.zero_grad()
            optimizer_center.zero_grad()
            
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=args.use_amp):
                (feat_b, bn_b, logit_b), (feat_a, bn_a, logit_a) = model(before, after)
                
                bn_b_norm = F.normalize(bn_b, p=2, dim=1)
                bn_a_norm = F.normalize(bn_a, p=2, dim=1)
                
                loss_id = criterion_id(logit_b, pids) + criterion_id(logit_a, pids)
                loss_tri = criterion_triplet(bn_b_norm, pids) + criterion_triplet(bn_a_norm, pids)
                loss_center = criterion_center(bn_b_norm, pids) + criterion_center(bn_a_norm, pids)
                
                loss = loss_id + lam1 * loss_tri + lam3 * loss_center 
                
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optim)
            scaler.step(optimizer_center)
            scaler.update()
            epoch_loss += loss.item()
            if (i+1) % 1 == 0:
                elapsed = time.time() - start_time
                lr = optim.param_groups[0]['lr']
                print(f"[{stage_name}] Epoch {epoch} Step {i+1}/{len(dataloader)} "
                      f"LR: {lr:.2e} Time: {elapsed:.2f}s "
                      f"Loss: {loss.item():.4f} (ID: {loss_id.item():.4f} Tri: {loss_tri.item():.4f} Cen: {loss_center.item():.4f})")
                start_time = time.time()
                
        if lr_scheduler:
            lr_scheduler.step()
        return epoch_loss / max(1, len(dataloader))

    start_epoch_stage1 = 1
    start_epoch_stage2 = 1
    best_val_rank1 = 0.0 # Theo dõi bằng Rank-1 thay vì Loss

    scheduler1 = get_warmup_cosine_scheduler(optimizer, warmup_epochs=5, total_epochs=epochs_stage1)

    if args.resume and os.path.isfile(args.resume):
        print(f"=> Loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        stage = checkpoint.get('stage', 2)
        if 'loss' in checkpoint:
            best_loss = checkpoint['loss']
            
        if stage == 1:
            start_epoch_stage1 = checkpoint['epoch'] + 1
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
            for _ in range(start_epoch_stage1 - 1):
                scheduler1.step()
            print(f"=> Loaded checkpoint '{args.resume}' (Stage 1, epoch {checkpoint['epoch']})")
        else:
            start_epoch_stage1 = epochs_stage1 + 1 # skip stage 1
            start_epoch_stage2 = checkpoint['epoch'] + 1
            print(f"=> Loaded checkpoint '{args.resume}' (Stage 2, epoch {checkpoint['epoch']}). Skipping Stage 1.")

    if start_epoch_stage1 <= epochs_stage1:
        print("=== START STAGE 1: Train Head (Freeze Backbone) ===")
        for epoch in range(start_epoch_stage1, epochs_stage1 + 1):
            avg_loss = train_epoch(epoch, "Stage 1", optimizer, scheduler1, lam1=args.lam1, lam2=args.lam2, lam3=args.lam3)
            
            checkpoint_data = {
                'epoch': epoch,
                'stage': 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }
            torch.save(checkpoint_data, os.path.join(args.checkpoint_dir, "last_model.pth"))
            
            # Validation sau mỗi args.val_freq epoch hoặc epoch cuối cùng
            if has_val and (epoch % args.val_freq == 0 or epoch == epochs_stage1):
                val_rank1 = validate(model, val_loader)
                if val_rank1 > best_val_rank1:
                    best_val_rank1 = val_rank1
                    torch.save(checkpoint_data, os.path.join(args.checkpoint_dir, "best_model.pth"))
                    print(f"[*] New best model saved at Stage 1, epoch {epoch} with Rank-1: {best_val_rank1*100:.2f}%")
        
    if start_epoch_stage2 <= epochs_stage2:
        print("=== START STAGE 2: End-to-End Fine-tuning ===")
        import gc
        if 'optimizer' in locals():
            del optimizer
        if 'scheduler1' in locals():
            del scheduler1
        gc.collect()
        torch.cuda.empty_cache()
        
        model.unfreeze_backbone()
        
        param_groups = [
            {'params': model.backbone.parameters(), 'lr': args.lr_stage2_backbone},
            {'params': model.temporal_encoder.parameters(), 'lr': args.lr_stage2_temporal},
            {'params': model.head.parameters(), 'lr': args.lr_stage2_head}
        ]
        optimizer2 = torch.optim.AdamW(param_groups, weight_decay=5e-4)
        scheduler2 = get_warmup_cosine_scheduler(optimizer2, warmup_epochs=5, total_epochs=epochs_stage2)
        
        if args.resume and os.path.isfile(args.resume) and checkpoint.get('stage', 2) == 2:
            if 'optimizer_state_dict' in checkpoint:
                optimizer2.load_state_dict(checkpoint['optimizer_state_dict'])
                for state in optimizer2.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
            for _ in range(start_epoch_stage2 - 1):
                scheduler2.step()
        
        for epoch in range(start_epoch_stage2, epochs_stage2 + 1):
            avg_loss = train_epoch(epoch, "Stage 2", optimizer2, scheduler2, lam1=args.lam1, lam2=args.lam2, lam3=args.lam3)
            
            checkpoint_data = {
                'epoch': epoch,
                'stage': 2,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer2.state_dict(),
                'loss': avg_loss
            }
            torch.save(checkpoint_data, os.path.join(args.checkpoint_dir, "last_model.pth"))
            
            # Validation sau mỗi args.val_freq epoch hoặc epoch cuối cùng
            if has_val and (epoch % args.val_freq == 0 or epoch == epochs_stage2):
                val_rank1 = validate(model, val_loader)
                if val_rank1 > best_val_rank1:
                    best_val_rank1 = val_rank1
                    torch.save(checkpoint_data, os.path.join(args.checkpoint_dir, "best_model.pth"))
                    print(f"[*] New best model saved at Stage 2, epoch {epoch} with Rank-1: {best_val_rank1*100:.2f}%")
        
    print("Training Complete!")

if __name__ == '__main__':
    main()
