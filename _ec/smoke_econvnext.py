"""Smoke test cho backbone E-ConvNeXt trong gasnet/train.py.

Khong import duoc gasnet.train (thieu dataset.py / evaluation.py / utils.py o goc
repo), nen AST-extract rieng cac class can thiet roi exec.

Chay: uav_env/bin/python _ec/smoke_econvnext.py
"""
from __future__ import annotations

import ast
import os
import sys

import torch
import torch.nn as nn

SRC = os.path.join(os.path.dirname(__file__), "..", "gasnet", "train.py")
NEEDED = [
    "EffectiveSELayer",
    "EConvNeXtBlock",
    "ConvBNGELU",
    "EConvNeXtStem",
    "CSPStage",
    "EConvNeXtStage",
    "DINOv3ConvNeXtBackbone",
]


def load_classes(src_path: str, names: list[str]) -> dict:
    tree = ast.parse(open(src_path, encoding="utf-8").read())
    picked = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    missing = set(names) - {n.name for n in picked}
    if missing:
        raise SystemExit(f"Thieu class trong {src_path}: {sorted(missing)}")
    mod = ast.Module(body=picked, type_ignores=[])
    ast.fix_missing_locations(mod)
    ns: dict = {"torch": torch, "nn": nn, "os": os}
    exec(compile(mod, src_path, "exec"), ns)
    return ns


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def main() -> None:
    ns = load_classes(SRC, NEEDED)
    Backbone = ns["DINOv3ConvNeXtBackbone"]

    print("=" * 74)
    print("1. Kien truc E-ConvNeXt (econvnext=True)")
    print("=" * 74)
    bb = Backbone(pretrained=True, econvnext=True).eval()
    for name, mod in (("stem", bb.stem), ("stage1", bb.stage1), ("stage2", bb.stage2),
                      ("stage3", bb.stage3), ("stage4", bb.stage4)):
        print(f"  {name:<8} {type(mod).__name__:<20} {count_params(mod)/1e6:7.3f} M params")
    print(f"  {'TONG':<8} {'':<20} {count_params(bb)/1e6:7.3f} M params")

    x = torch.randn(2, 3, 224, 224)
    want = [("stem", (2, 64, 112, 112)), ("stage1", (2, 96, 56, 56)),
            ("stage2", (2, 192, 28, 28)), ("stage3", (2, 384, 14, 14)),
            ("stage4", (2, 768, 7, 7))]
    print("\n2. Kich thuoc tensor tung chang (input 224x224)")
    ok = True
    with torch.no_grad():
        for name, exp in want:
            x = getattr(bb, name)(x)
            good = tuple(x.shape) == exp
            ok &= good
            print(f"  {name:<8} {tuple(x.shape)}  {'OK' if good else 'SAI, doi ' + str(exp)}")

    print("\n3. Rang buoc RGA: ga1 can 96 @ 56x56")
    with torch.no_grad():
        h = bb.stem(torch.randn(1, 3, 224, 224))
        h = bb.stage1(h)
    print(f"  stage1 output = {tuple(h.shape)}  -> ga1 nhan 96 @ 56x56: "
          f"{'OK' if tuple(h.shape) == (1, 96, 56, 56) else 'SAI'}")

    print("\n4. Kiem tra chuyen gamma -> norm2.weight = 2*gamma")
    for tag, blk in (("stage3.layers[0]", bb.stage3.layers[0]),
                     ("stage4.layers[0]", bb.stage4.layers[0]),
                     ("stage1.blocks[0]", bb.stage1.blocks[0]),
                     ("stage2.blocks[0]", bb.stage2.blocks[0])):
        w = blk.norm2.weight.data
        print(f"  {tag:<18} dim={w.numel():<4} mean={w.mean():+.4f} std={w.std():.4f} "
              f"min={w.min():+.2f} max={w.max():+.2f}")
    print("  (gamma pretrain: mean ~0, std ~1.428 => 2*gamma std ~2.856)")

    print("\n5. So sanh voi kien truc DINOv3 goc (econvnext=False)")
    bb0 = Backbone(pretrained=True, econvnext=False)
    print(f"  goc     : {count_params(bb0)/1e6:7.3f} M params")
    print(f"  E-Conv  : {count_params(bb)/1e6:7.3f} M params "
          f"({100*(count_params(bb)-count_params(bb0))/count_params(bb0):+.2f}%)")

    print("\n6. Trong so nao la moi (khong lay tu pretrain)")
    known = ("depthwise_conv", "norm.weight", "norm.bias", "pointwise_conv1",
             "pointwise_conv2", "norm2.weight")
    fresh = 0
    for n, p in bb.named_parameters():
        if not any(k in n for k in known):
            fresh += p.numel()
    print(f"  params moi (stem/conv1-3/attn/down): {fresh/1e6:.3f} M "
          f"= {100*fresh/count_params(bb):.2f}% tong")

    print("\n" + ("TAT CA OK" if ok else "CO LOI KICH THUOC"))


if __name__ == "__main__":
    sys.exit(main())
