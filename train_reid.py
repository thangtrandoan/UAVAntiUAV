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

    def isatty(self):
        return getattr(self.terminal, 'isatty', lambda: False)()

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
        # Khởi tạo centers = 0 thay vì randn:
        # centers randn → ||c||² ≈ feat_dim (vd 1472) → center loss ban đầu cực lớn,
        # cộng thêm ||x||² của feature thô sẽ dominate gradient, phá vỡ training.
        self.centers = nn.Parameter(torch.zeros(self.num_classes, self.feat_dim))
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
    """🛠️ (24/9) SỬA LỖI NGHIỆM TẦM THƯỜNG — nguyên nhân thảm hoạ epoch 20-30 (run 24/9).

    BẢN CŨ: `loss = 1 - cos(x_t, x_{t+1}).mean()`.
    ❌ LỖI: `cos` KHÔNG quan tâm độ lớn. Nghiệm tầm thường để `cos -> 1` là làm MỌI
       `x_t` GIỐNG HỆT NHAU (hằng số theo thời gian). Khi đó loss -> 0 nhưng nhánh
       temporal MẤT HẾT thông tin chuyển động. Chuẩn hoá phương sai KHÔNG cứu được
       (hằng số chia eps vẫn là hằng số -> cos vẫn = 1).

    BẰNG CHỨNG từ log run 24/9:
       epoch 20-30: Temporal loss = 0.0001  ĐÚNG LÚC Rank-1 sập về 0.04%
                    (chance = 2.90% -> thấp hơn chance 70 lần)
       epoch 33-35: Temporal loss TĂNG LẠI 0.0043-0.0050 ĐÚNG LÚC hồi phục 61.6%
       => tương quan hoàn hảo giữa "loss temporal = 0" và "feature sụp".

    BẢN MỚI: RANKING LOSS theo KHOẢNG CÁCH THỜI GIAN.
       Yêu cầu: cặp LIỀN KỀ giống nhau HƠN cặp XA NHẤT ít nhất `margin`.
       ✅ Chuỗi HẰNG SỐ cho `sim_near = sim_far = 1` -> loss = margin > 0
          => NGHIỆM TẦM THƯỜNG BỊ LOẠI BỎ về mặt toán học.
       ✅ Đúng inductive bias: encoder thời gian phải tạo chuỗi có CẤU TRÚC
          (similarity GIẢM theo khoảng cách), tức giữ thông tin chuyển động.
    """
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, temporal_features):
        if len(temporal_features.shape) == 2 or temporal_features.size(1) < 3:
            return torch.tensor(0.0, device=temporal_features.device,
                                dtype=temporal_features.dtype)
        x = temporal_features.float()                      # ổn định dưới autocast
        sim_near = F.cosine_similarity(x[:, :-1, :], x[:, 1:, :], dim=-1).mean()
        sim_far = F.cosine_similarity(x[:, 0, :], x[:, -1, :], dim=-1).mean()
        return F.relu(sim_far - sim_near + self.margin)

# ==========================================
# 2. DATASET & DATALOADER
# ==========================================

class UAVReIDDataset(Dataset):
    def __init__(self, data_dir, query_json, gallery_json, transform=None, num_frames=16,
                 n_values=None):
        self.data_dir = data_dir
        with open(query_json, 'r') as f:
            queries = json.load(f)
        with open(gallery_json, 'r') as f:
            galleries = json.load(f)
            
        self.transform = transform
        self.num_frames = num_frames
        # 🛠️ (22/9) N RANDOM THEO BATCH (md/22thg9.md §16). `num_frames` giữ vai trò
        # "N mặc định / N dùng cho validation". `n_values` là danh sách N sẽ random.
        # CHỈ SỐ ĐƯỢC MÃ HOÁ N:  idx = k * L + g   (k = chỉ số sample, g = chỉ số N, L = len(n_values))
        # Nhờ vậy `__getitem__` suy ra N TỪ CHÍNH INDEX -> stateless, an toàn với num_workers>0
        # (không phải set attribute trên dataset từ sampler, thứ không truyền được qua fork).
        self.n_values = list(n_values) if n_values else [num_frames]
        
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
                        'query_dir': q['frame_dir']
                    })
        
        self.identities = sorted(list(set(p['identity_id'] for p in self.valid_pairs)))
        self.id_to_idx = {pid: i for i, pid in enumerate(self.identities)}
        self.num_identities = len(self.identities)

    def __len__(self):
        return len(self.valid_pairs) * len(self.n_values)
        
    def _load_clip(self, folder, frames, take_last=False, n=None):
        n = n or self.num_frames
        # 🛠️ (22/9) COPY list: code cũ `frames.append(...)` khi clip ngắn làm list TRONG
        # `valid_pairs` phình ra vĩnh viễn. Với N cố định thì chỉ là rác; với N RANDOM thì
        # thành LỖI (pad tới n=4 rồi lần sau đọc n=16 -> toàn frame lặp).
        frames = list(frames)
        # 🛠️ (15/9) ĐỒNG BỘ frame_stride — BỎ `np.linspace`.
        # `np.linspace` lấy mẫu TRẢI ĐỀU cả danh sách → bước thời gian hiệu dụng
        #     = frame_stride × (len-1)/(num_frames-1)   >   frame_stride
        # trong khi infer lấy ĐÚNG 1 frame mỗi `frame_stride` → lệch phân phối temporal.
        # Cắt CONTIGUOUS để bước thời gian giữa 2 frame liên tiếp = ĐÚNG frame_stride
        # ở cả train và infer:
        #   - before (gallery): lấy `num_frames` frame CUỐI → sát t1 (lúc mất dấu)
        #   - after  (query)  : lấy `num_frames` frame ĐẦU → sát t2 (lúc tái xuất)
        # Kết quả Y HỆT việc sinh lại data với num_before/after_frames = num_frames,
        # nhưng KHÔNG cần chạy lại data_pipeline.py.
        if len(frames) > n:
            frames = frames[-n:] if take_last else frames[:n]
        elif len(frames) < n:
            if len(frames) == 0:
                return torch.zeros((n, 3, 224, 224))
            while len(frames) < n:
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
        L = len(self.n_values)
        g, k = idx % L, idx // L          # g = chỉ số N, k = chỉ số sample
        n = self.n_values[g]
        pair = self.valid_pairs[k]

        # CÙNG một N cho gallery (before) và query (after) — bắt buộc, vì loss so sánh chúng.
        before_clip = self._load_clip(pair['gallery_dir'], pair['gallery_frames'],
                                      take_last=True, n=n)
        after_clip = self._load_clip(pair['query_dir'], pair['query_frames'],
                                     take_last=False, n=n)
        
        pid = self.id_to_idx[pair['identity_id']]
        return before_clip, after_clip, pid

class ReIDBatchSampler(Sampler):
    """
    PK Sampler đảm bảo mỗi batch có P identities, mỗi identity có K instances
    để hỗ trợ Hard Triplet Loss.
    """
    def __init__(self, dataset, batch_size, num_instances=4, n_values=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        # 🛠️ (22/9) N random theo BATCH (không theo sample — không collate được (B,N,C)).
        self.n_values = list(n_values) if n_values else list(getattr(dataset, 'n_values', [dataset.num_frames]))
        self.L = len(self.n_values)
        
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
            
    def _pk_batches(self):
        """Sinh batch PK trên chỉ số SAMPLE (k), CHƯA gắn N. Y hệt logic cũ."""
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
        return batches

    def __iter__(self):
        base = self._pk_batches()
        # Mỗi batch rút MỘT N. Dùng pool xoay vòng để MỘT epoch trải đều mọi N
        # (thay vì rút ngẫu nhiên thuần, có thể lệch), nhưng KHÔNG làm epoch dài thêm.
        gs = (list(range(self.L)) * (len(base) // self.L + 1))[:len(base)]
        random.shuffle(gs)
        # idx = k * L + g  -> mọi index trong batch cùng g -> cùng N -> collate được.
        return iter([[k * self.L + g for k in b] for b, g in zip(base, gs)])

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
    args.query_json     = os.path.join(args.data_dir, 'query_train.json')
    args.gallery_json   = os.path.join(args.data_dir, 'gallery_train.json')
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
    # 🛠️ (24/9) ÉP best toàn cục khi resume. Checkpoint CŨ (trước 24/9) KHÔNG có trường
    # `best_val_rank1` -> restore 0.0 -> validation đầu tiên sẽ ghi đè `best_model.pth`
    # dù tệ hơn. Đặt = -1 để tắt (dùng giá trị trong checkpoint).
    args.resume_best_rank1 = float(tc.get('resume_best_rank1', -1.0))
    args.batch_size     = tc.get('batch_size', 32)
    args.num_instances  = tc.get('num_instances', 4)
    args.num_frames     = tc.get('num_frames', 16)
    args.num_workers    = tc.get('num_workers', 4)
    args.use_amp        = tc.get('use_amp', False)
    args.pin_memory     = tc.get('pin_memory', False)
    args.use_compile    = tc.get('use_compile', False) # Thêm cờ bật/tắt compile
    args.val_freq       = tc.get('val_freq', 5) # Đọc số epoch đánh giá từ config (mặc định 5)
    # 🛠️ (22/9) ĐO NHIỀU N (md/22thg9.md §25). Tiêu chí đạt là "fused >= 0.85 ở CẢ 3 N và
    # lệch < 0.05", nhưng validation cũ chỉ đo MỘT N (args.num_frames) -> không thấy được
    # độ lệch theo N -> train mù. `val_n_list` là danh sách N sẽ đo mỗi lần validation.
    args.val_n_list     = tc.get('val_n_list') or [args.num_frames]
    # 🛠️ Early stopping. patience = số LẦN VALIDATION liên tiếp không cải thiện (0 = tắt).
    args.early_stop_patience = int(tc.get('early_stop_patience', 0))
    args.early_stop_min_delta = float(tc.get('early_stop_min_delta', 1e-4))
    args.stop_on_target = bool(tc.get('stop_on_target', True))
    args.target_rank1   = float(tc.get('target_rank1', 0.85))
    args.target_spread  = float(tc.get('target_spread', 0.05))
    # 🛠️ (22/9) torch.compile + N RANDOM: mỗi giá trị N cho một SHAPE khác nhau
    # (`extract_features` reshape thành [B*N, C, H, W]) -> compile mặc định (dynamic=False)
    # sẽ BIÊN DỊCH LẠI cho từng N. Với 7 giá trị N đó là 7 lần biên dịch (mỗi lần hàng
    # chục giây tới vài phút). `dynamic=True` xử lý shape thay đổi mà không biên dịch lại.
    args.compile_dynamic = bool(tc.get('compile_dynamic', True))
    args.backbone       = tc.get('backbone', 'resnet50_ibn')

    
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

    # 🛠️ (22/9) N random theo batch (md/22thg9.md §16). Bỏ key / 1 phần tử = hành vi cũ.
    n_values = tc.get('n_frames_choices') or [args.num_frames]
    if len(n_values) > 1:
        print(f" 🔀 N RANDOM theo batch: {n_values} (cùng N cho query+gallery trong 1 batch)")
    dataset = UAVReIDDataset(train_dir, args.query_json, args.gallery_json, transform=transform_train, num_frames=args.num_frames, n_values=n_values)
    num_identities = dataset.num_identities
    
    batch_size = args.batch_size
    num_instances = args.num_instances
    sampler = ReIDBatchSampler(dataset, batch_size=batch_size, num_instances=num_instances, n_values=n_values)
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
            raise FileNotFoundError(
                f"Không tìm thấy test dir tại {test_dir}. "
                "KHÔNG được fallback về train dir để validate (sẽ cho Rank-1 ảo cao). "
                "Chạy data_pipeline trước để tạo test set + query_test.json/gallery_test.json."
            )
        val_query_json = args.query_json.replace('query_train', 'query_test')
        val_gallery_json = args.gallery_json.replace('gallery_train', 'gallery_test')
        # 🛠️ (22/9) Một EvalDataset cho MỖI N trong `val_n_list`.
        val_loaders = {}
        for _n in args.val_n_list:
            _ds = EvalDataset(test_dir, val_query_json, val_gallery_json,
                              transform=transform_test, num_frames=_n)
            val_loaders[_n] = DataLoader(_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
        has_val = True
        print(f" ✅ Validation đo {len(args.val_n_list)} N: {args.val_n_list}")
    except Exception as e:
        print(f"Cảnh báo: Không thể setup validation, sẽ bỏ qua bước này. Lỗi: {e}")
        has_val = False
        val_loaders = {}
    if not has_val:
        print(" ⚠️ KHÔNG có validation -> KHÔNG kiểm chứng được tiêu chí 3 N và KHÔNG early-stop được.")

    def validate(model, loader, tag=""):
        print(f" --- Validation trên tập Test {tag} ---")
        model.eval()
        qf, gf, q_pids, g_pids = [], [], [], []
        start_t = time.time()
        with torch.no_grad():
            for batch in loader:
                before, after, pids = batch[0], batch[1], batch[2]
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

    def validate_multi_n(model, loaders, n_list, stage_name=""):
        """🛠️ (22/9) Đo Rank-1 (`fused`) ở NHIỀU N và trả về ĐIỂM = MIN qua các N.

        Vì sao MIN mà không phải TRUNG BÌNH: tiêu chí đạt là "mọi N >= 0.85". Trung bình
        CHE MẤT đúng cái đang hỏng — một N sụp xuống 0.15 vẫn cho trung bình cao nếu các N
        khác tốt (đúng tình trạng hiện tại: N=8 là 0.157 nhưng N=16 là 0.849). Tối đa hoá
        MIN = tối ưu hoá ca XẤU NHẤT = đúng mục tiêu.
        """
        per_n = {}
        for _n in n_list:
            per_n[_n] = validate(model, loaders[_n], tag=f"(N={_n})")
        spread = max(per_n.values()) - min(per_n.values())
        worst = min(per_n.values())
        line = " | ".join(f"N={_n}:{per_n[_n]*100:6.2f}%" for _n in n_list)
        print(f" ==> [{stage_name}] {line} | LỆCH={spread*100:5.2f}% | MIN={worst*100:6.2f}%")
        return worst, per_n, spread


    # --- Setup GASNet Path ---
    gasnet_dir = cfg.get('paths', {}).get('gasnet_dir', '')
    if gasnet_dir:
        os.environ['GASNET_PATH'] = os.path.abspath(gasnet_dir)
        
    from model import UAVReIDNet
    # 🛠️ (22/9) Pooling + PE của temporal encoder (md/22thg9.md §31).
    # Mặc định 'attn' + PE: attention CHỌN được vị trí nên PE trở nên có ích (§30.5).
    # `infer.py`/`evaluate_reid_robustness.py` tạo model bằng DEFAULT nên PHẢI khớp.
    model = UAVReIDNet(
        gasnet_weights_path=args.gasnet_weights or None,
        num_identities=num_identities,
        freeze_backbone=True,
        backbone=args.backbone,
        temporal_pool=tc.get('temporal_pool', 'attn'),
        temporal_pe=bool(tc.get('temporal_pe', True))
    )
    print(f" Temporal encoder: pool={tc.get('temporal_pool', 'attn')}, pe={bool(tc.get('temporal_pe', True))}")
    model.cuda()
    
    # --- torch.compile cho tốc độ tối đa trên A100/H100 ---
    if not args.gpu_jetson and args.use_compile and hasattr(torch, 'compile'):
        print(f"Bật torch.compile() để tối ưu model (dynamic={args.compile_dynamic})...")
        try:
            model = torch.compile(model, dynamic=args.compile_dynamic)
        except Exception as e:
            print(f"Cảnh báo: torch.compile() thất bại: {e}. Sẽ chạy mode bình thường.")

    # 🛠️ (22/9) Đọc từ config — trước đây 2 key `label_smooth`/`triplet_margin` CÓ trong
    # yaml nhưng code HARDCODE, sửa yaml không có tác dụng (bẫy âm thầm).
    criterion_id = LabelSmoothCrossEntropy(epsilon=float(lc.get('label_smooth', 0.1)))
    criterion_triplet = HardTripletLoss(margin=float(lc.get('triplet_margin', 0.3)))
    feat_dim = 1472 if args.backbone == "dinov3_convnext" else 3072
    criterion_center = CenterLoss(num_classes=num_identities, feat_dim=feat_dim).cuda()
    criterion_temporal = TemporalConsistencyLoss()

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
        epoch_loss_id = 0.0
        epoch_loss_tri = 0.0
        epoch_loss_center = 0.0
        epoch_loss_temp = 0.0
        
        from tqdm import tqdm
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"[{stage_name}] Epoch {epoch}", leave=False, dynamic_ncols=True)
        for i, (before, after, pids) in pbar:
            before, after, pids = before.cuda(), after.cuda(), pids.cuda()
            
            optim.zero_grad()
            optimizer_center.zero_grad()
            
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=args.use_amp):
                (feat_b, bn_b, logit_b), (feat_a, bn_a, logit_a), (t_seq_b, t_seq_a) = model(before, after)
                
                # ID Loss (dùng logits)
                loss_id = criterion_id(logit_b, pids) + criterion_id(logit_a, pids)
                
                # Triplet Loss (dùng features đã L2-normalize)
                bn_b_norm = F.normalize(bn_b, p=2, dim=1)
                bn_a_norm = F.normalize(bn_a, p=2, dim=1)
                loss_tri = criterion_triplet(bn_b_norm, pids) + criterion_triplet(bn_a_norm, pids)
                
                # Center Loss (dùng features THÔ chưa normalize — đúng theo paper!)
                # Giảm lam3 từ 0.05 xuống 0.001 để tránh center loss dominate gradient
                loss_center = criterion_center(bn_b, pids) + criterion_center(bn_a, pids)
                
                # Temporal Consistency Loss (lam2)
                loss_temp = criterion_temporal(t_seq_b) + criterion_temporal(t_seq_a)
                
                loss = loss_id + lam1 * loss_tri + lam2 * loss_temp + lam3 * loss_center 
                
            # 🛠️ FIX NaN: nếu loss không finite (NaN/inf) → BỎ QUA bước này,
            # không backward/step để tránh đầu độc toàn bộ weights.
            # (Backbone vừa unfreeze + BN batch stats có thể tạo 1 batch xấu → inf grad)
            if not torch.isfinite(loss):
                print(f"[{i}] ⚠️ Bỏ qua batch (loss={loss.item():.3e} không finite).")
                continue
                
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optim)
            scaler.step(optimizer_center)
            scaler.update()
            epoch_loss += loss.item()
            epoch_loss_id += loss_id.item()
            epoch_loss_tri += loss_tri.item()
            epoch_loss_center += loss_center.item()
            epoch_loss_temp += loss_temp.item()
            if (i+1) % 1 == 0:
                lr = optim.param_groups[0]['lr']
                pbar.set_postfix({
                    'Loss': f"{loss.item():.3f}",
                    'ID': f"{loss_id.item():.3f}",
                    'Tri': f"{loss_tri.item():.3f}",
                    'Cen': f"{loss_center.item():.3f}",
                    'Tmp': f"{loss_temp.item():.3f}",
                    'LR': f"{lr:.1e}"
                })
                
        if lr_scheduler:
            lr_scheduler.step()
            
        num_batches = max(1, len(dataloader))
        print(f"\n---> TỔNG KẾT [{stage_name}] Epoch {epoch}: "
              f"Tổng Loss = {epoch_loss/num_batches:.4f} | "
              f"ID = {epoch_loss_id/num_batches:.4f} | "
              f"Triplet = {epoch_loss_tri/num_batches:.4f} | "
              f"Temporal = {epoch_loss_temp/num_batches:.4f} | "
              f"Center = {epoch_loss_center/num_batches:.4f}\n")
              
        return epoch_loss / num_batches

    start_epoch_stage1 = 1
    start_epoch_stage2 = 1
    best_val_rank1 = 0.0 # Best TOÀN CỤC (MIN qua các N) — dùng để lưu best_model.pth
    # 🛠️ (22/9) Early stopping theo TỪNG STAGE: `best` riêng để Stage 2 không bị chặn bởi
    # thành tích của Stage 1 (nếu so với best toàn cục thì Stage 2 gần như luôn "không cải
    # thiện" ngay từ lần validation đầu -> early stop sai).
    _es = {1: {'best': -1.0, 'bad': 0}, 2: {'best': -1.0, 'bad': 0}}

    scheduler1 = get_warmup_cosine_scheduler(optimizer, warmup_epochs=5, total_epochs=epochs_stage1)

    if args.resume and os.path.isfile(args.resume):
        print(f"=> Loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        stage = checkpoint.get('stage', 2)
        if 'loss' in checkpoint:
            best_loss = checkpoint['loss']
        # 🛠️ (24/9) Khôi phục best TOÀN CỤC — xem giải thích ở `checkpoint_data`.
        best_val_rank1 = float(checkpoint.get('best_val_rank1', 0.0))
        if args.resume_best_rank1 >= 0:
            print(f"=> ÉP best_val_rank1 = {args.resume_best_rank1*100:.2f}% "
                  f"(config `resume_best_rank1`; checkpoint ghi {best_val_rank1*100:.2f}%)")
            best_val_rank1 = args.resume_best_rank1
        else:
            print(f"=> best_val_rank1 khôi phục = {best_val_rank1*100:.2f}% "
                  f"(0.00% = checkpoint cũ chưa lưu trường này — nên đặt `resume_best_rank1`)")
            
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
                'loss': avg_loss,
                # 🛠️ (24/9) PHẢI lưu: nếu không, resume xong `best_val_rank1` reset về 0.0
                # -> validation ĐẦU TIÊN sau resume sẽ ghi đè `best_model.pth` bằng model
                # CÓ THỂ TỆ HƠN (run 24/9: epoch 55 = 40.91% sẽ đè epoch 35 = 61.56%).
                'best_val_rank1': best_val_rank1
            }
            torch.save(checkpoint_data, os.path.join(args.checkpoint_dir, "last_model.pth"))
            
            # Validation sau mỗi args.val_freq epoch hoặc epoch cuối cùng
            if has_val and (epoch % args.val_freq == 0 or epoch == epochs_stage1):
                score, per_n, spread = validate_multi_n(model, val_loaders, args.val_n_list, "Stage 1")
                met = (min(per_n.values()) >= args.target_rank1) and (spread <= args.target_spread)
                if met:
                    print(f"[✓] ĐẠT TIÊU CHÍ (epoch {epoch}): mọi N >= {args.target_rank1:.2f} "
                          f"và lệch {spread*100:.2f}% <= {args.target_spread*100:.0f}%")
                # 🛠️ (24/9) CẢNH BÁO SỤP: run 24/9 sập về 0.04% (chance=2.90%) mà log chỉ
                # hiện "[i] Không cải thiện" bình thường -> rất dễ bỏ qua.
                if best_val_rank1 > 0 and score < 0.3 * best_val_rank1:
                    print(f"🚨 [CẢNH BÁO SỤP] MIN={score*100:.2f}% < 30% của best "
                          f"({best_val_rank1*100:.2f}%). Feature có thể đã hỏng — "
                          f"kiểm tra Temporal loss và cân nhắc resume từ best_model.pth.")
                # Cải thiện theo TỪNG STAGE -> điều khiển early stop
                if score > _es[1]['best'] + args.early_stop_min_delta:
                    _es[1]['best'] = score
                    _es[1]['bad'] = 0
                    # Nhưng chỉ LƯU khi tốt hơn best TOÀN CỤC
                    if score > best_val_rank1:
                        best_val_rank1 = score
                        checkpoint_data['best_val_rank1'] = best_val_rank1  # best_model tự ghi điểm của nó
                        torch.save(checkpoint_data, os.path.join(args.checkpoint_dir, "best_model.pth"))
                        print(f"[*] Best mới (MIN qua N) ở Stage 1, epoch {epoch}: {best_val_rank1*100:.2f}%")
                else:
                    _es[1]['bad'] += 1
                    if args.early_stop_patience > 0:
                        print(f"[i] Không cải thiện: {_es[1]['bad']}/{args.early_stop_patience}")
                if met and args.stop_on_target:
                    print(f"→ DỪNG Stage 1 vì đã ĐẠT TIÊU CHÍ.")
                    break
                if args.early_stop_patience > 0 and _es[1]['bad'] >= args.early_stop_patience:
                    print(f"→ EARLY STOP Stage 1: {args.early_stop_patience} lần validation liên tiếp không cải thiện.")
                    break
        
    if start_epoch_stage2 <= epochs_stage2:
        print("=== START STAGE 2: End-to-End Fine-tuning ===")
        import gc
        if 'optimizer' in locals():
            del optimizer
        if 'scheduler1' in locals():
            del scheduler1
        gc.collect()
        torch.cuda.empty_cache()
        
        # 🛠️ FIX NaN: GradScaler phải được TẠO MỚI ở Stage 2.
        # - Scale factor cũ đã tích lũy qua 30 epoch Stage 1 (lớn dần ×2 mỗi 2000 steps),
        #   khiến gradient scaled của backbone (mạng sâu vừa unfreeze) bị overflow → inf → NaN.
        # - State cũ của optimizer Stage 1 vẫn nằm trong scaler, làm scaler.update() hoạt động sai.
        # - torch.compile cũng cần graph mới khi backbone chuyển requires_grad=True → reset dynamo.
        scaler = torch.amp.GradScaler('cuda', enabled=args.use_amp)
        if args.use_compile and hasattr(torch, '_dynamo'):
            try:
                torch._dynamo.reset()
                print("  Đã reset torch._dynamo trước Stage 2 (graph cũ chứa backbone frozen).")
            except Exception as e:
                print(f"  Cảnh báo: không reset được dynamo: {e}")
        
        model.unfreeze_backbone()
        
        # Phân tách trọng số có sẵn (pretrained) và trọng số random (GA, FS, Head...)
        pretrained_params = []
        random_params = []
        
        for name, param in model.backbone.named_parameters():
            # convnext_backbone + ga1-4 + fs1-2 đều có weights từ GASNet đã train VRU
            # → thuộc nhóm pretrained (lr thấp 1e-5, fine-tune nhẹ lên domain UAV)
            if "convnext_backbone" in name or "swin_backbone" in name or "base" in name or "ga" in name or "fs" in name:
                pretrained_params.append(param)
            else:
                # bnneck, classifier... là random/thay đổi theo num_identities UAV
                random_params.append(param)
                
        param_groups = [
            {'params': pretrained_params, 'lr': args.lr_stage2_backbone},              # 1e-5
            {'params': random_params, 'lr': args.lr_stage2_temporal},                  # 1e-4 (dùng chung mức LR to)
            {'params': model.temporal_encoder.parameters(), 'lr': args.lr_stage2_temporal}, # 1e-4
            {'params': model.head.parameters(), 'lr': args.lr_stage2_head}             # 1e-4
        ]
        # 🛠️ (22/9) Trước đây hardcode 5e-4 -> key `stage2.weight_decay` trong yaml VÔ HIỆU.
        _wd2 = float(tc.get('stage2', {}).get('weight_decay', 5e-4))
        optimizer2 = torch.optim.AdamW(param_groups, weight_decay=_wd2)
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
                'loss': avg_loss,
                # 🛠️ (24/9) PHẢI lưu: nếu không, resume xong `best_val_rank1` reset về 0.0
                # -> validation ĐẦU TIÊN sau resume sẽ ghi đè `best_model.pth` bằng model
                # CÓ THỂ TỆ HƠN (run 24/9: epoch 55 = 40.91% sẽ đè epoch 35 = 61.56%).
                'best_val_rank1': best_val_rank1
            }
            torch.save(checkpoint_data, os.path.join(args.checkpoint_dir, "last_model.pth"))
            
            # Validation sau mỗi args.val_freq epoch hoặc epoch cuối cùng
            if has_val and (epoch % args.val_freq == 0 or epoch == epochs_stage2):
                score, per_n, spread = validate_multi_n(model, val_loaders, args.val_n_list, "Stage 2")
                met = (min(per_n.values()) >= args.target_rank1) and (spread <= args.target_spread)
                if met:
                    print(f"[✓] ĐẠT TIÊU CHÍ (epoch {epoch}): mọi N >= {args.target_rank1:.2f} "
                          f"và lệch {spread*100:.2f}% <= {args.target_spread*100:.0f}%")
                # 🛠️ (24/9) CẢNH BÁO SỤP: run 24/9 sập về 0.04% (chance=2.90%) mà log chỉ
                # hiện "[i] Không cải thiện" bình thường -> rất dễ bỏ qua.
                if best_val_rank1 > 0 and score < 0.3 * best_val_rank1:
                    print(f"🚨 [CẢNH BÁO SỤP] MIN={score*100:.2f}% < 30% của best "
                          f"({best_val_rank1*100:.2f}%). Feature có thể đã hỏng — "
                          f"kiểm tra Temporal loss và cân nhắc resume từ best_model.pth.")
                # Cải thiện theo TỪNG STAGE -> điều khiển early stop
                if score > _es[2]['best'] + args.early_stop_min_delta:
                    _es[2]['best'] = score
                    _es[2]['bad'] = 0
                    # Nhưng chỉ LƯU khi tốt hơn best TOÀN CỤC
                    if score > best_val_rank1:
                        best_val_rank1 = score
                        checkpoint_data['best_val_rank1'] = best_val_rank1  # best_model tự ghi điểm của nó
                        torch.save(checkpoint_data, os.path.join(args.checkpoint_dir, "best_model.pth"))
                        print(f"[*] Best mới (MIN qua N) ở Stage 2, epoch {epoch}: {best_val_rank1*100:.2f}%")
                else:
                    _es[2]['bad'] += 1
                    if args.early_stop_patience > 0:
                        print(f"[i] Không cải thiện: {_es[2]['bad']}/{args.early_stop_patience}")
                if met and args.stop_on_target:
                    print(f"→ DỪNG Stage 2 vì đã ĐẠT TIÊU CHÍ.")
                    break
                if args.early_stop_patience > 0 and _es[2]['bad'] >= args.early_stop_patience:
                    print(f"→ EARLY STOP Stage 2: {args.early_stop_patience} lần validation liên tiếp không cải thiện.")
                    break
        
    print("Training Complete!")

if __name__ == '__main__':
    main()
