"""Kiem tra GASNet day du (backbone + RGA + head) voi backbone E-ConvNeXt.

Xac nhan: (1) GASNet dung duoc voi ca econvnext=True/False, (2) forward chay,
(3) tham so truyen qua GASNet -> DINOv3ConvNeXtBackbone dung, (4) RGA van nhan
dung 96@56x56 nen khong raise.

Chay: uav_env/bin/python _ec/check_gasnet_econvnext.py
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "gasnet" / "train.py"

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
    # E-ConvNeXt
    "EffectiveSELayer",
    "EConvNeXtBlock",
    "ConvBNGELU",
    "EConvNeXtStem",
    "CSPStage",
    "EConvNeXtStage",
    "DINOv3ConvNeXtBackbone",
    "GASNet",
]


def load_classes() -> dict:
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    body = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in NEEDED
    ]
    missing = set(NEEDED) - {n.name for n in body}
    if missing:
        raise SystemExit(f"Thieu: {sorted(missing)}")
    ns = {"torch": torch, "nn": nn, "F": F, "os": os}
    exec(compile(ast.Module(body=body, type_ignores=[]), "<gasnet>", "exec"), ns)
    return ns


def nparams(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def main() -> None:
    ns = load_classes()
    GASNet = ns["GASNet"]

    out = {}
    for tag, ec in (("DINOv3 gốc", False), ("E-ConvNeXt II", True)):
        print("=" * 74)
        print(f"{tag}  (econvnext={ec})")
        print("=" * 74)
        net = GASNet(num_classes=502, use_pretrained=True, backbone="dinov3_convnext",
                     econvnext=ec).eval()
        print(f"  GASNet params (nuoc): {nparams(net):,}  ({nparams(net)/1e6:.3f} M)")
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            try:
                y = net(x)
                shapes = []
                def walk(o):
                    if isinstance(o, torch.Tensor):
                        shapes.append(tuple(o.shape))
                    elif isinstance(o, (tuple, list)):
                        for i in o:
                            walk(i)
                walk(y)
                print(f"  forward OK, output shapes: {shapes}")
                out[tag] = {"params": nparams(net), "shapes": shapes, "ok": True}
            except Exception as e:  # noqa: BLE001
                print(f"  forward LỖI: {type(e).__name__}: {e}")
                out[tag] = {"ok": False}
        del net

    print("=" * 74)
    g, e = out.get("DINOv3 gốc"), out.get("E-ConvNeXt II")
    if g and e and g.get("ok") and e.get("ok"):
        print(f"  params: gốc {g['params']/1e6:.3f} M -> E-ConvNeXt {e['params']/1e6:.3f} M "
              f"({100*(e['params']-g['params'])/g['params']:+.2f}%)")
        print(f"  shape ra giống hệt: {g['shapes'] == e['shapes']}")
        print("\n  KET LUAN: GASNet chay duoc voi backbone moi, RGA khong raise.")
    else:
        print("  KET LUAN: CO LOI")
        sys.exit(1)


if __name__ == "__main__":
    main()
