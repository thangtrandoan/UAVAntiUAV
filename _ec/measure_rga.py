"""Do xem RGAChannel (rga_c) co thuc su chon kenh khong.

Cau hoi: da co rga_c roi thi ESE co bi thua khong?
Do: cong (gate) cua rga_c tren anh that. Neu cong gan nhu hang so
theo kenh => rga_c khong lam gi theo kenh => ESE con dat dat.
"""
from __future__ import annotations

import ast
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "gasnet" / "train.py"
CKPT = ROOT / "best_model_conv.pth"

NEEDED = [
    "_concat_relation",
    "RGASpatial",
    "RGAChannel",
    "RGABlock",
    "Lite3x3",
    "ChannelGate",
    "OSBlockFS",
    "BNNeck",
    "GeMPool",
    "DINOv3ConvNeXtBackbone",
    "GASNet",
]


def load_classes():
    tree = ast.parse(SRC.read_text())
    body = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in NEEDED
    ]
    ns = {"torch": torch, "nn": nn, "F": F}
    exec(compile(ast.Module(body=body, type_ignores=[]), "<gasnet>", "exec"), ns)
    return ns


def build_backbone(ns):
    """Dung GASNet (day la 'backbone' trong checkpoint) va nap trong so da train."""
    bb = ns["GASNet"](
        num_classes=502,
        use_pretrained=False,
        backbone="dinov3_convnext",
    )
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    pref = "_orig_mod.backbone."
    sub = {k[len(pref) :]: v for k, v in sd.items() if k.startswith(pref)}
    print(f"  checkpoint: epoch={ck.get('epoch')} stage={ck.get('stage')} loss={ck.get('loss'):.4f}")
    print(f"  key 'backbone' trong checkpoint: {len(sub)}")
    missing, unexpected = bb.load_state_dict(sub, strict=False)
    print(f"  missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"    missing vi du: {missing[:4]}")
    if unexpected:
        print(f"    unexpected vi du: {unexpected[:4]}")
    bb.eval()
    return bb


def load_images(n, size=224):
    root = ROOT / "data" / "UAV-Anti-UAV" / "Test"
    dirs = sorted([d for d in root.iterdir() if d.is_dir()])[:n]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    out = []
    for d in dirs:
        f = next(d.glob("*.jpg"))
        im = Image.open(f).convert("RGB").resize((size, size))
        x = torch.from_numpy(np_array(im)).permute(2, 0, 1).float() / 255.0
        out.append((x - mean) / std)
    return torch.stack(out), [d.name for d in dirs]


def np_array(im):
    import numpy as np

    return np.asarray(im, dtype="uint8")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    ns = load_classes()
    print("Dung backbone...")
    bb = build_backbone(ns)

    gates = {}

    def hook(name):
        def fn(_m, _i, o):
            gates.setdefault(name, []).append(o.detach())

        return fn

    hooks = []
    for tag in ("ga1", "ga2", "ga3", "ga4"):
        ga = getattr(bb, tag)
        hooks.append(ga.rga_c.sigmoid.register_forward_hook(hook(f"{tag}.rga_c")))
        hooks.append(ga.rga_s.sigmoid.register_forward_hook(hook(f"{tag}.rga_s")))

    x, names = load_images(n)
    print(f"  {n} anh: {names[:3]} ...")
    t0 = time.time()
    with torch.no_grad():
        out = bb(x)
    dt = time.time() - t0
    shp = [tuple(o.shape) for o in out] if isinstance(out, (tuple, list)) else tuple(out.shape)
    print(f"  forward xong trong {dt:.1f} s ({dt / n:.1f} s/anh)   out={shp}")
    for h in hooks:
        h.remove()

    print()
    print("  " + "=" * 78)
    print("  CONG CUA RGA (gate) TREN ANH THAT")
    print("  " + "=" * 78)
    for name in ("ga1.rga_c", "ga2.rga_c", "ga3.rga_c", "ga4.rga_c"):
        if name not in gates:
            continue
        g = torch.cat(gates[name]).squeeze()  # [B, C]
        if g.dim() == 1:
            g = g.unsqueeze(0)
        b, c = g.shape
        cm = g.mean(0)  # [C] trung binh theo anh
        dyn = g - cm  # phan phu thuoc anh
        print(f"\n  {name}:  gate shape [B={b}, C={c}]")
        print(f"    gate trung binh toan bo     = {g.mean():.4f}   (hang so 0.5 = khong chon gi)")
        print(f"    std toan bo                 = {g.std():.4f}")
        print(f"    std theo KENH (tinh)        = {cm.std():.4f}   <-- chon kenh tinh")
        print(f"    std theo ANH (dong)         = {dyn.std():.4f}   <-- dieu bien theo du lieu")
        print(f"    kenh: min={cm.min():.4f}  max={cm.max():.4f}  bien do={cm.max()-cm.min():.4f}")
        tot = g.var()
        print(f"    ty le phuong sai tinh/dong  = {cm.var() / tot:.3f} / {dyn.var() / tot:.3f}")

    print()
    print("  " + "-" * 78)
    print("  CONG CUA RGASpatial (de so, gate [B,1,H,W])")
    print("  " + "-" * 78)
    for name in ("ga1.rga_s", "ga4.rga_s"):
        if name not in gates:
            continue
        g = torch.cat(gates[name])
        print(
            f"    {name}: mean={g.mean():.4f}  std={g.std():.4f}  "
            f"min={g.min():.4f}  max={g.max():.4f}"
        )


if __name__ == "__main__":
    main()
