"""Do sigma_b = std kenh cua nhanh residual (dau ra pointwise_conv2) tren DINOv3 goc.

sigma_b la do loi ma norm2 (BatchNorm) xoa mat khi bi chen vao nhanh residual. Dung
de khoi tao norm2.weight = 2 * gamma * sigma_b, khoi phuc dung do loi pretrain.

DAY LA NGUON CUA HANG SO SIGMA_B trong gasnet/train.py. Neu doi cach cat kenh, doi
kien truc block, hoac doi backbone thi phai chay lai script nay va cap nhat SIGMA_B.

Chay tu bat ky thu muc nao:
    python3 tools/measure_sigma_b.py
"""
import sys, glob, os, torch
from PIL import Image
import torchvision.transforms as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gasnet"))
from train import DINOv3ConvNeXtBackbone

torch.manual_seed(0)
files = []
for d in sorted(glob.glob(os.path.join(ROOT, "processed", "train", "*"))):
    fs = sorted(glob.glob(os.path.join(d, "**", "*.jpg"), recursive=True))
    if fs:
        files.append(fs[len(fs) // 2])
    if len(files) == 8:
        break
tf = T.Compose([T.Resize((256, 256)), T.CenterCrop((224, 224)), T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
x = torch.stack([tf(Image.open(f).convert("RGB")) for f in files])
print(f"  {len(files)} anh", flush=True)

bb = DINOv3ConvNeXtBackbone(pretrained=True, econvnext=False)
bb.train()
stages = [bb.stage1, bb.stage2, bb.stage3, bb.stage4]

print(f"\n  {'stage':<8} {'so block':>8} {'sigma_b TB':>12} {'sigma_b min':>12} "
      f"{'max':>9} {'|gamma| TB':>11} {'gamma*sigma_b':>14}", flush=True)

out = {}
for si, st in enumerate(stages):
    caps = {}
    hs = []
    for i, blk in enumerate(st.layers):
        hs.append(blk.pointwise_conv2.register_forward_hook(
            lambda m, i_, o, k=i: caps.__setitem__(k, o.detach())))
    with torch.no_grad():
        h = x
        for s in stages[:si + 1]:
            h = s(h)
    for hk in hs:
        hk.remove()
    # std per-channel, roi lay trung binh qua cac block
    per_block = torch.stack([caps[i].std(dim=(0, 2, 3)).mean() for i in sorted(caps)])
    gam = torch.stack([b.gamma.data.abs().mean() for b in st.layers])
    sb = per_block.mean().item()
    print(f"  stage{si+1:<3} {len(st.layers):>8} {sb:>12.5f} {per_block.min().item():>12.5f} "
          f"{per_block.max().item():>9.5f} {gam.mean().item():>11.5f} "
          f"{(per_block*gam).mean().item():>14.6f}", flush=True)
    out[f"stage{si+1}"] = sb
print("\n  => sigma_b dung cho cong thuc norm2.weight = 2*gamma*sigma_b", flush=True)
vals = tuple(round(out[f"stage{i}"], 3) for i in range(1, 5))
print(f"  => SIGMA_B = {vals}   (dan vao gasnet/train.py)", flush=True)
