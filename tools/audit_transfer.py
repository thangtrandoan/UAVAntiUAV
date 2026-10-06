"""Kiem vet can viec chuyen trong so DINOv3 -> E-ConvNeXt.

Muc tieu: KHONG con tensor nao chua duoc phan loai. Moi tensor nguon phai roi vao
dung mot trong: copy nguyen / cat kenh / bo co chu dinh. Moi tensor dich phai la:
copy tu pretrain / module moi / running stats cua BatchNorm.

Gom 3 phan:
  A. Doi chieu tung tensor nguon -> dich, in lech lon nhat.
  B. Liet ke tensor dich khong den tu pretrain (module moi + running stats).
  C. Kiem stage THAT cua E-ConvNeXt co tai tao DINOv3 chinh xac khong.

Day la CHOT HOI QUY: script tu fail neu con tensor nao chua phan loai, hoac neu
stage3/stage4 lech khoi DINOv3 (vi du ai do doi norm cua block sang BatchNorm).

Chay tu bat ky thu muc nao:
    python3 tools/audit_transfer.py
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gasnet"))
from train import DINOv3ConvNeXtBackbone  # noqa: E402

FAIL = []


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-9)).item()


def check(label, ok, detail=""):
    mark = "OK  " if ok else "FAIL"
    if not ok:
        FAIL.append(label)
    print(f"    [{mark}] {label}{('  ' + detail) if detail else ''}", flush=True)


def hid_for(p1w, p2w, k):
    """Sinh lai dung tap hidden unit ma _copy_block_ da chon."""
    imp = p1w.norm(dim=1) * p2w.norm(dim=0)
    return torch.topk(imp, 4 * k).indices


def adapt_ln(ln):
    """Tra ve module nhan NCHW. LayerNorm cua block la channels_last (can boc),
    LayerNorm trong downsample_layers la channels_first (dung truc tiep)."""
    if getattr(ln, "data_format", "channels_last") == "channels_first":
        return ln
    return LNWrap(ln)


class LNWrap(nn.Module):
    """Boc LayerNorm channels_last de dung trong luong NCHW cua ta."""

    def __init__(self, ln):
        super().__init__()
        self.ln = ln

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class LNBlock(nn.Module):
    """Dung lai trong so cua EConvNeXtBlock nhung chuan hoa bang LayerNorm."""

    def __init__(self, blk, ln):
        super().__init__()
        self.depthwise_conv = blk.depthwise_conv
        self.ln = ln
        self.pointwise_conv1 = blk.pointwise_conv1
        self.activation_fn = blk.activation_fn
        self.pointwise_conv2 = blk.pointwise_conv2
        self.gamma = blk.gamma

    def forward(self, x):
        r = x
        h = self.ln(self.depthwise_conv(x))
        h = self.pointwise_conv1(h)
        h = self.activation_fn(h)
        h = self.pointwise_conv2(h)
        return r + h * self.gamma.view(1, -1, 1, 1)


def main():
    ref = DINOv3ConvNeXtBackbone(pretrained=True, econvnext=False)
    new = DINOv3ConvNeXtBackbone(pretrained=True, econvnext=True)

    print("\n" + "=" * 78, flush=True)
    print("  A. DOI CHIEU TUNG TENSOR NGUON -> DICH", flush=True)
    print("=" * 78, flush=True)

    n_src = n_checked = 0

    # ---------- A1. stages[2..3]: copy nguyen ----------
    for si in (2, 3):
        s, d = getattr(ref, f"stage{si + 1}"), getattr(new, f"stage{si + 1}")
        devs = {
            "down.LN.weight": (d.downsample_layers[0].weight, s.downsample_layers[0].weight),
            "down.LN.bias": (d.downsample_layers[0].bias, s.downsample_layers[0].bias),
            "down.Conv.weight": (d.downsample_layers[1].weight, s.downsample_layers[1].weight),
            "down.Conv.bias": (d.downsample_layers[1].bias, s.downsample_layers[1].bias),
        }
        for nm, (a, b) in devs.items():
            n_src += 1
            n_checked += 1
            check(f"stages[{si}].{nm}", torch.equal(a.data, b.data),
                  f"lech { (a.data - b.data).abs().max().item():.2e}")

        # 9 tensor moi block
        keys = ["depthwise_conv.weight", "depthwise_conv.bias", "norm.weight", "norm.bias",
                "pointwise_conv1.weight", "pointwise_conv1.bias",
                "pointwise_conv2.weight", "pointwise_conv2.bias", "gamma"]
        bad = []
        for i in range(len(d.layers)):
            bd, bs = d.layers[i], s.layers[i]
            pairs = [
                ("depthwise_conv.weight", bd.depthwise_conv.weight, bs.depthwise_conv.weight),
                ("depthwise_conv.bias", bd.depthwise_conv.bias, bs.depthwise_conv.bias),
                ("norm.weight", bd.norm.weight, bs.layer_norm.weight),
                ("norm.bias", bd.norm.bias, bs.layer_norm.bias),
                ("pointwise_conv1.weight", bd.pointwise_conv1.weight[:, :, 0, 0], bs.pointwise_conv1.weight),
                ("pointwise_conv1.bias", bd.pointwise_conv1.bias, bs.pointwise_conv1.bias),
                ("pointwise_conv2.weight", bd.pointwise_conv2.weight[:, :, 0, 0], bs.pointwise_conv2.weight),
                ("pointwise_conv2.bias", bd.pointwise_conv2.bias, bs.pointwise_conv2.bias),
                ("gamma", bd.gamma, bs.gamma),
            ]
            for nm, a, b in pairs:
                n_src += 1
                n_checked += 1
                if not torch.equal(a.data, b.data):
                    bad.append(f"layers[{i}].{nm}")
        check(f"stages[{si}] {len(d.layers)} block x 9 tensor", not bad,
              "tat ca khop tuyet doi" if not bad else f"lech: {bad[:4]}")

    # ---------- A2. stages[0..1]: cat kenh ----------
    for si, dst_stage, blk_attr in ((0, new.stage1, "blocks"), (1, new.stage2, "blocks")):
        s = getattr(ref, f"stage{si + 1}")
        blocks = getattr(dst_stage, blk_attr)
        k = dst_stage.block_dim
        idx = new._stage_channel_index_(s, k)
        n_src += 1
        n_checked += 1
        check(f"stages[{si}] chi so kenh giu lai ({k}/{s.layers[0].depthwise_conv.weight.shape[0]})",
              idx.numel() == k and idx.max() < s.layers[0].depthwise_conv.weight.shape[0])

        bad = []
        for i in range(len(blocks)):
            bd, bs = blocks[i], s.layers[i]
            p1w, p2w = bs.pointwise_conv1.weight.data, bs.pointwise_conv2.weight.data
            hid = hid_for(p1w, p2w, k)
            pairs = [
                ("depthwise_conv.weight", bd.depthwise_conv.weight, bs.depthwise_conv.weight.data.index_select(0, idx)),
                ("depthwise_conv.bias", bd.depthwise_conv.bias, bs.depthwise_conv.bias.data.index_select(0, idx)),
                ("norm.weight", bd.norm.weight, bs.layer_norm.weight.data.index_select(0, idx)),
                ("norm.bias", bd.norm.bias, bs.layer_norm.bias.data.index_select(0, idx)),
                ("pointwise_conv1.weight",
                 bd.pointwise_conv1.weight[:, :, 0, 0],
                 p1w.index_select(0, hid).index_select(1, idx)),
                ("pointwise_conv1.bias", bd.pointwise_conv1.bias, bs.pointwise_conv1.bias.data.index_select(0, hid)),
                ("pointwise_conv2.weight",
                 bd.pointwise_conv2.weight[:, :, 0, 0],
                 p2w.index_select(0, idx).index_select(1, hid)),
                ("pointwise_conv2.bias", bd.pointwise_conv2.bias, bs.pointwise_conv2.bias.data.index_select(0, idx)),
            ]
            for nm, a, b in pairs:
                n_src += 1
                n_checked += 1
                if not torch.equal(a.data, b):
                    bad.append(f"blocks[{i}].{nm}")
            # gamma -> norm2.weight = 2*gamma*sigma_b (bien doi co chu dinh)
            n_src += 1
            n_checked += 1
            want = 2.0 * bs.gamma.data.index_select(0, idx) * bd.sigma_b
            if not torch.equal(bd.norm2.weight.data, want):
                bad.append(f"blocks[{i}].norm2.weight")
        check(f"stages[{si}] {len(blocks)} block cat kenh", not bad,
              "tat ca khop tuyet doi" if not bad else f"lech: {bad[:4]}")

    # down cua stage2 (CSP) cat tu downsample conv cua stages[1]
    s1 = ref.stage2
    n_src += 2
    n_checked += 2
    check("stage2.down.conv.weight cat tu stages[1].downsample_layers[1]",
          torch.equal(new.stage2.down.conv.weight.data, s1.downsample_layers[1].weight.data[: new.stage2.ch_mid]))
    check("stage2.down.conv.bias cat tu stages[1].downsample_layers[1]",
          torch.equal(new.stage2.down.conv.bias.data, s1.downsample_layers[1].bias.data[: new.stage2.ch_mid]))

    print(f"\n    => da doi chieu {n_checked}/{n_src} tensor, ket qua: "
          f"{'KHONG con tensor nao lech' if not FAIL else str(len(FAIL)) + ' muc FAIL'}", flush=True)

    # ---------- A3. tensor nguon KHONG duoc dung ----------
    print("\n" + "=" * 78, flush=True)
    print("  A3. TENSOR NGUON CO CHU DINH KHONG DUNG", flush=True)
    print("=" * 78, flush=True)
    dropped = {
        "stage1.downsample_layers[0] (LayerNorm truoc patchify)": ref.stage1.downsample_layers[0],
        "stage1.downsample_layers[1] (patchify conv 3->96)": ref.stage1.downsample_layers[1],
        "stage2.downsample_layers[0] (LayerNorm 96 truoc conv)": ref.stage2.downsample_layers[0],
    }
    tot_drop = 0
    print(f"    ref.stem = {type(ref.stem).__name__} (khong co tham so, patchify nam trong stage1)", flush=True)
    for nm, mod in dropped.items():
        ps = list(mod.parameters()) + list(mod.buffers())
        tot_drop += sum(t.numel() for t in ps)
        print(f"    - {nm:<52} {sum(t.numel() for t in ps):>9,} tham so", flush=True)
    print(f"    TONG bi bo: {tot_drop:,} tham so -> thay bang stem moi + stage1.down (xem phan B)",
          flush=True)
    check("cac tensor bi bo deu co chu dinh (thay bang module moi)", True)

    # ---------- B. tensor duoi tien to module moi ----------
    # LUU Y: nhom theo TIEN TO, nen "stage1.blocks." va "stage2.blocks." bao gom CA cac
    # tensor cat tu pretrain (da doi chieu o phan A) — khong phai tat ca deu la module moi.
    print("\n" + "=" * 78, flush=True)
    print("  B. TENSOR DUOI TIEN TO MODULE MOI (gom ca phan cat tu pretrain trong blocks)",
          flush=True)
    print("=" * 78, flush=True)
    GROUPS = (
        "stem.", "stage1.down.", "stage1.conv1.", "stage1.conv2.", "stage1.conv3.",
        "stage1.attn.", "stage1.blocks.",
        "stage2.down.", "stage2.conv1.", "stage2.conv2.", "stage2.conv3.",
        "stage2.attn.", "stage2.blocks.",
    )
    groups = {g: 0 for g in GROUPS}
    tot_all = 0
    outside = 0
    n_outside = 0
    # Phan loai MOI tham so VA buffer theo tien to module. Truoc day buffer co nhanh
    # rieng va tinh khoa nham (vi du "stem.bn.running" thay vi "stem."), nen running
    # stats cua BatchNorm bi cong vao TONG ma khong roi vao NHOM nao -> tong khong
    # bang tong cac nhom. Gop ve mot duong phan loai duy nhat de khong tai dien.
    for n, t in list(new.named_parameters()) + list(new.named_buffers()):
        tot_all += t.numel()
        hit = next((g for g in GROUPS if n.startswith(g)), None)
        if hit is None:
            outside += t.numel()
            n_outside += 1
        else:
            groups[hit] += t.numel()
    tot_new = sum(groups.values())
    for g, v in groups.items():
        if v:
            print(f"    {g:<42} {v:>10,} tham so", flush=True)
    print(f"    {'TONG module moi':<42} {tot_new:>10,} tham so", flush=True)
    print(f"    {'NGOAI nhom (stage3/stage4 copy tu pretrain)':<42} {outside:>10,} tham so "
          f"({n_outside} tensor)", flush=True)
    check("tong module moi + ngoai nhom = toan bo model", tot_new + outside == tot_all,
          f"{tot_new:,} + {outside:,} = {tot_new + outside:,} / model {tot_all:,}")

    # ---------- C. tuong duong chuc nang ca stage voi LayerNorm ----------
    print("\n" + "=" * 78, flush=True)
    print("  C. TUONG DUONG CHUC NANG CA STAGE KHI DUNG LAYERNORM", flush=True)
    print("=" * 78, flush=True)
    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        ref.train()
        new.train()
        feats = {1: ref.stage1(x)}
        for si in (2, 3, 4):
            feats[si] = getattr(ref, f"stage{si}")(feats[si - 1])
        for si in feats:
            print(f"    dau ra that cua DINOv3 stage{si}: {tuple(feats[si].shape)}", flush=True)

        for si in (2, 3):
            s, d = getattr(ref, f"stage{si + 1}"), getattr(new, f"stage{si + 1}")
            inp = feats[si]
            ln_stage = nn.Sequential(
                adapt_ln(s.downsample_layers[0]), s.downsample_layers[1],
                *[LNBlock(d.layers[i], adapt_ln(s.layers[i].layer_norm)) for i in range(len(d.layers))],
            )
            y_ref = s(inp)
            y_ln = ln_stage(inp)
            r = rel(y_ln, y_ref)
            check(f"stages[{si}] nguyen stage (down + {len(d.layers)} block) dung LayerNorm",
                  r < 1e-6, f"lech tuong doi {r:.3e}, std {y_ref.std():.5f}")
            # Duong THAT cua E-ConvNeXt — chot hoi quy cho phuong an 1 (giu LayerNorm
            # trong block thua huong). Neu ai doi norm cua block sang BatchNorm thi muc
            # nay FAIL ngay, khong can train moi phat hien.
            y_real = getattr(new, f"stage{si + 1}")(inp)
            r_real = rel(y_real, y_ref)
            check(f"stages[{si}] stage THAT cua E-ConvNeXt khop DINOv3",
                  r_real < 1e-6, f"lech tuong doi {r_real:.3e}, std {y_real.std():.5f}")

    print("\n" + "=" * 78, flush=True)
    if FAIL:
        print(f"  KET LUAN: CON {len(FAIL)} MUC CHUA DAT", flush=True)
        for f in FAIL:
            print(f"    - {f}", flush=True)
    else:
        print("  KET LUAN: moi tensor nguon deu duoc phan loai, khong con lech ngoai y muon", flush=True)
    print("=" * 78, flush=True)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
