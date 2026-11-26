"""Offline Stage 2 UAVReIDNet models, strict checkpoint auditing, and ONNX wrappers."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# --------------------------------------------------------------------------- #
# Gate Modules: RGA & BCGG (exact offline definitions matching Stage 1 & 2 ckpts)
# --------------------------------------------------------------------------- #
def _concat_relation(relation: Tensor) -> Tensor:
    return torch.cat([relation, relation.transpose(1, 2)], dim=1)


class RGASpatial(nn.Module):
    def __init__(
        self,
        channels: int,
        spatial_size: tuple[int, int],
        spatial_reduction: int = 8,
        channel_reduction: int = 8,
    ):
        super().__init__()
        self.spatial_size = spatial_size
        self.num_nodes = spatial_size[0] * spatial_size[1]
        self.spatial_reduction = spatial_reduction
        inter_channels = max(1, channels // channel_reduction)
        relation_features = max(1, self.num_nodes // spatial_reduction)
        self.theta = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.phi = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.g = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.relation = nn.Sequential(
            nn.Conv1d(self.num_nodes * 2, relation_features, kernel_size=1, bias=False),
            nn.BatchNorm1d(relation_features),
            nn.ReLU(inplace=True),
        )
        self.attn = nn.Sequential(
            nn.Conv1d(relation_features + 1, relation_features, kernel_size=1, bias=False),
            nn.BatchNorm1d(relation_features),
            nn.ReLU(inplace=True),
            nn.Conv1d(relation_features, 1, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
        self.last_attention: Optional[Tensor] = None

    def forward(self, x: Tensor) -> Tensor:
        b, _, h, w = x.shape
        if not torch.jit.is_tracing() and h * w != self.num_nodes:
            raise ValueError(f"Expected spatial {self.spatial_size}, got {(h, w)}")
        theta = self.theta(x).view(b, -1, self.num_nodes).transpose(1, 2)
        phi = self.phi(x).view(b, -1, self.num_nodes)
        spatial_relation = torch.bmm(theta, phi)
        relation = self.relation(_concat_relation(spatial_relation))
        channel_pooled = self.g(x).flatten(2).mean(dim=1, keepdim=True)
        attn = self.sigmoid(self.attn(torch.cat([relation, channel_pooled], dim=1))).view(b, 1, h, w)
        self.last_attention = attn
        return x * attn


class RGAChannel(nn.Module):
    def __init__(
        self,
        channels: int,
        spatial_size: tuple[int, int],
        spatial_reduction: int = 8,
        channel_reduction: int = 8,
    ):
        super().__init__()
        self.spatial_size = spatial_size
        self.num_nodes = spatial_size[0] * spatial_size[1]
        inter_spatial = max(1, self.num_nodes // spatial_reduction)
        relation_features = max(1, channels // channel_reduction)
        self.theta = nn.Sequential(
            nn.Conv1d(self.num_nodes, inter_spatial, kernel_size=1, bias=False),
            nn.BatchNorm1d(inter_spatial),
            nn.ReLU(inplace=True),
        )
        self.phi = nn.Sequential(
            nn.Conv1d(self.num_nodes, inter_spatial, kernel_size=1, bias=False),
            nn.BatchNorm1d(inter_spatial),
            nn.ReLU(inplace=True),
        )
        self.relation = nn.Sequential(
            nn.Conv1d(channels * 2, relation_features, kernel_size=1, bias=False),
            nn.BatchNorm1d(relation_features),
            nn.ReLU(inplace=True),
        )
        self.channel_embed = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.attn = nn.Sequential(
            nn.Conv1d(relation_features + 1, relation_features, kernel_size=1, bias=False),
            nn.BatchNorm1d(relation_features),
            nn.ReLU(inplace=True),
            nn.Conv1d(relation_features, 1, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        if not torch.jit.is_tracing() and h * w != self.num_nodes:
            raise ValueError(f"Expected spatial {self.spatial_size}, got {(h, w)}")
        x_perm = x.view(b, c, self.num_nodes).permute(0, 2, 1)
        theta = self.theta(x_perm).permute(0, 2, 1)
        phi = self.phi(x_perm)
        channel_relation = torch.bmm(theta, phi)
        relation = self.relation(_concat_relation(channel_relation))
        spatial_pooled = self.channel_embed(x).mean(dim=(2, 3)).unsqueeze(1)
        attn = self.sigmoid(self.attn(torch.cat([relation, spatial_pooled], dim=1))).view(b, c, 1, 1)
        return x * attn


class RGABlock(nn.Module):
    def __init__(self, channels: int, spatial_size: tuple[int, int]):
        super().__init__()
        self.rga_s = RGASpatial(channels=channels, spatial_size=spatial_size)
        self.rga_c = RGAChannel(channels=channels, spatial_size=spatial_size)

    @property
    def last_attention(self) -> Optional[Tensor]:
        return self.rga_s.last_attention

    def forward(self, x: Tensor) -> Tensor:
        return self.rga_c(self.rga_s(x))


def _fuse_conv_bn_params(conv: nn.Conv2d, bn: nn.modules.batchnorm._BatchNorm) -> tuple[Tensor, Tensor]:
    weight = conv.weight.detach().to(torch.float32)
    bias = (
        conv.bias.detach().to(torch.float32)
        if conv.bias is not None
        else torch.zeros(conv.out_channels, dtype=torch.float32, device=weight.device)
    )
    scale = bn.weight.detach().to(torch.float32) / torch.sqrt(
        bn.running_var.detach().to(torch.float32) + float(bn.eps)
    )
    fused_weight = (weight * scale.reshape(-1, *([1] * (weight.dim() - 1)))).to(conv.weight.dtype)
    fused_bias = (
        (bias - bn.running_mean.detach().to(torch.float32)) * scale
        + bn.bias.detach().to(torch.float32)
    ).to(conv.weight.dtype)
    return fused_weight, fused_bias


def _fold_conv_bn(conv: nn.Conv2d, bn: nn.modules.batchnorm._BatchNorm) -> nn.Conv2d:
    fused_weight, fused_bias = _fuse_conv_bn_params(conv, bn)
    fused = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
    ).to(device=fused_weight.device, dtype=fused_weight.dtype)
    with torch.no_grad():
        fused.weight.copy_(fused_weight)
        fused.bias.copy_(fused_bias)
    return fused


class ChannelLogits(nn.Module):
    def __init__(self, kernel_size: int = 5) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, self.kernel_size, padding=self.kernel_size // 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, c = x.shape[0], x.shape[1]
        y = self.pool(x).reshape(b, 1, c)
        return self.conv(y).reshape(b, c)


class _DepthwiseBranch(nn.Module):
    def __init__(self, channels: int, kernel_size: int, placement: str) -> None:
        super().__init__()
        if placement == "full":
            kernel, padding = kernel_size, (kernel_size // 2, kernel_size // 2)
        elif placement == "row":
            kernel, padding = (1, kernel_size), (0, kernel_size // 2)
        else:
            kernel, padding = (kernel_size, 1), (kernel_size // 2, 0)
        self.placement = placement
        self.conv = nn.Conv2d(channels, channels, kernel, padding=padding, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.bn(self.conv(x))

    def fused_params(self) -> tuple[Tensor, Tensor]:
        return _fuse_conv_bn_params(self.conv, self.bn)


class RepDepthwiseContext(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.branches = nn.ModuleList(
            [
                _DepthwiseBranch(self.channels, self.kernel_size, "full"),
                _DepthwiseBranch(self.channels, self.kernel_size, "row"),
                _DepthwiseBranch(self.channels, self.kernel_size, "col"),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.branches[0](x)
        for branch in self.branches[1:]:
            out = out + branch(x)
        return out

    @torch.no_grad()
    def fused_conv(self) -> nn.Conv2d:
        k = self.kernel_size
        ref_w, _ = self.branches[0].fused_params()
        fused_weight = torch.zeros(self.channels, 1, k, k, dtype=ref_w.dtype, device=ref_w.device)
        fused_bias = torch.zeros(self.channels, dtype=ref_w.dtype, device=ref_w.device)
        for branch in self.branches:
            w, b = branch.fused_params()
            if branch.placement == "full":
                fused_weight += w
            elif branch.placement == "row":
                fused_weight[:, :, k // 2, :] += w[:, :, 0, :]
            else:
                fused_weight[:, :, :, k // 2] += w[:, :, :, 0]
            fused_bias += b
        conv = nn.Conv2d(
            self.channels, self.channels, k, padding=k // 2, groups=self.channels, bias=True
        ).to(device=fused_weight.device, dtype=fused_weight.dtype)
        conv.weight.copy_(fused_weight)
        conv.bias.copy_(fused_bias)
        return conv


class BottleneckContextGroupGate(nn.Module):
    def __init__(
        self,
        channels: int = 384,
        hidden: int = 16,
        groups: int = 4,
        kernel_size: int = 7,
        channel_kernel: int = 5,
        inject_value: bool = False,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.hidden = int(hidden)
        self.groups = int(groups)
        self.kernel_size = int(kernel_size)
        self.channel_kernel = int(channel_kernel)
        self.inject_value = bool(inject_value)

        self.reduce = nn.Sequential(
            nn.Conv2d(self.channels, self.hidden, 1, bias=False),
            nn.BatchNorm2d(self.hidden),
            nn.ReLU(inplace=False),
        )
        self.context = RepDepthwiseContext(self.hidden, self.kernel_size)
        self.spatial = nn.Conv2d(self.hidden, self.groups, 1, bias=False)
        self.channel = ChannelLogits(self.channel_kernel)
        self.value = nn.Conv2d(self.hidden, self.channels, 1, bias=False) if self.inject_value else None

        self.attn_probe = nn.Identity()
        self.capture_attention = False
        self.last_attention: Optional[Tensor] = None
        self.deployed = False

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        u = self.context(self.reduce(x))
        spatial = self.spatial(u).unsqueeze(2)
        channel = self.channel(x).reshape(b, self.groups, c // self.groups, 1, 1)
        gate = 2.0 * torch.sigmoid(channel + spatial)
        if self.capture_attention:
            attn = (0.5 * gate).mean(dim=(1, 2), keepdim=True).reshape(b, 1, h, w)
            self.last_attention = attn
            self.attn_probe(attn)
        y = x * gate.reshape(b, c, h, w)
        if self.value is not None:
            y = y + self.value(u)
        return y

    @torch.no_grad()
    def to_deploy(self) -> "BottleneckContextGroupGate":
        self.capture_attention = False
        self.last_attention = None
        module = copy.deepcopy(self).eval()
        if isinstance(module.context, RepDepthwiseContext):
            module.context = module.context.fused_conv()
        if len(module.reduce) == 3 and isinstance(module.reduce[1], (nn.BatchNorm2d, nn.SyncBatchNorm)):
            fused = _fold_conv_bn(module.reduce[0], module.reduce[1])
            module.reduce = nn.Sequential(fused, module.reduce[2])
        module.deployed = True
        return module.eval()


# --------------------------------------------------------------------------- #
# Offline DINOv3 ConvNeXt-Small + GASNet
# --------------------------------------------------------------------------- #
class LayerNorm2d(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.normalized_shape = (dim,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class _DINOv3ConvNeXtLayer(nn.Module):
    def __init__(self, dim: int, layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), layer_scale_init_value))
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.pointwise_conv1 = nn.Linear(dim, 4 * dim)
        self.activation_fn = nn.GELU()
        self.pointwise_conv2 = nn.Linear(4 * dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.depthwise_conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        x = self.pointwise_conv1(x)
        x = self.activation_fn(x)
        x = self.pointwise_conv2(x)
        x = x * self.gamma
        x = x.permute(0, 3, 1, 2)
        return residual + x


class _DINOv3ConvNeXtStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int, is_stem: bool):
        super().__init__()
        if is_stem:
            self.downsample_layers = nn.ModuleList(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=4),
                    LayerNorm2d(out_channels, eps=1e-6),
                ]
            )
        else:
            self.downsample_layers = nn.ModuleList(
                [
                    LayerNorm2d(in_channels, eps=1e-6),
                    nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
                ]
            )
        self.layers = nn.ModuleList([_DINOv3ConvNeXtLayer(out_channels) for _ in range(depth)])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.downsample_layers:
            x = layer(x)
        for layer in self.layers:
            x = layer(x)
        return x


class OfflineDINOv3ConvNeXt(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Identity()
        self.stage1 = _DINOv3ConvNeXtStage(3, 96, depth=3, is_stem=True)
        self.stage2 = _DINOv3ConvNeXtStage(96, 192, depth=3, is_stem=False)
        self.stage3 = _DINOv3ConvNeXtStage(192, 384, depth=27, is_stem=False)
        self.stage4 = _DINOv3ConvNeXtStage(384, 768, depth=3, is_stem=False)


class Lite3x3(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ChannelGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.gate(x)


class OSBlockFS(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, reduction: int = 16):
        super().__init__()
        mid_ch = out_ch
        self.stream1 = nn.Sequential(Lite3x3(in_ch, mid_ch))
        self.stream2 = nn.Sequential(Lite3x3(in_ch, mid_ch), Lite3x3(mid_ch, mid_ch))
        self.stream3 = nn.Sequential(Lite3x3(in_ch, mid_ch), Lite3x3(mid_ch, mid_ch), Lite3x3(mid_ch, mid_ch))
        self.stream4 = nn.Sequential(
            Lite3x3(in_ch, mid_ch), Lite3x3(mid_ch, mid_ch), Lite3x3(mid_ch, mid_ch), Lite3x3(mid_ch, mid_ch)
        )
        self.gate = ChannelGate(mid_ch, reduction)
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        s1 = self.stream1(x)
        s2 = self.stream2(x)
        s3 = self.stream3(x)
        s4 = self.stream4(x)
        out = self.gate(s1) * s1 + self.gate(s2) * s2 + self.gate(s3) * s3 + self.gate(s4) * s4
        return self.relu(out + self.shortcut(x))


class BNNeck(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
        self.bn.bias.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        return self.bn(x)


class GeMPool(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = float(p)
        self.eps = float(eps)

    def forward(self, x: Tensor) -> Tensor:
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), (1, 1)).pow(1.0 / self.p)


class _FP16SafeGeMPool(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = float(p)
        self.eps = float(eps)

    def forward(self, x: Tensor) -> Tensor:
        x_clamped = x.clamp(min=self.eps)
        scale = x_clamped.amax(dim=(2, 3), keepdim=True)
        pooled = F.adaptive_avg_pool2d((x_clamped / scale).pow(self.p), (1, 1))
        return scale * pooled.pow(1.0 / self.p)


def _make_gate(kind: str, channels: int, spatial_size: tuple[int, int], gate_cfg: Dict[str, Any]) -> nn.Module:
    if kind == "rga":
        return RGABlock(channels, spatial_size=spatial_size)
    if kind == "bcgg":
        return BottleneckContextGroupGate(
            channels=channels,
            hidden=int(gate_cfg.get("gate_hidden", 16)),
            groups=int(gate_cfg.get("gate_groups", 4)),
            kernel_size=int(gate_cfg.get("gate_kernel", 7)),
            channel_kernel=int(gate_cfg.get("gate_channel_kernel", 5)),
            inject_value=bool(gate_cfg.get("gate_inject_value", False)),
        )
    raise ValueError(f"Unsupported gate kind: {kind!r}")


class GASNetConvNeXt(nn.Module):
    def __init__(self, num_classes: int = 502, gate: str = "bcgg", gate_cfg: Optional[Dict[str, Any]] = None, gem_p: float = 3.0):
        super().__init__()
        gate_cfg = gate_cfg or {}
        self.gate_kind = gate
        self.return_raw_features_eval = True
        self.convnext_backbone = OfflineDINOv3ConvNeXt()

        self.ga1 = _make_gate(gate, 96, (56, 56), gate_cfg)
        self.ga2 = _make_gate(gate, 192, (28, 28), gate_cfg)
        self.ga3 = _make_gate(gate, 384, (14, 14), gate_cfg)
        self.ga4 = _make_gate(gate, 768, (7, 7), gate_cfg)

        self.fs1 = OSBlockFS(384, 192)
        self.fs2 = OSBlockFS(192, 192)
        self.gap = GeMPool(p=gem_p)
        self.bnneck_global = BNNeck(768)
        self.bnneck_fs = BNNeck(192)
        self.classifier_global = nn.Linear(768, num_classes, bias=False)
        self.classifier_fs = nn.Linear(192, num_classes, bias=False)

    @torch.no_grad()
    def to_deploy(self) -> "GASNetConvNeXt":
        module = copy.deepcopy(self).eval()
        for name in ("ga1", "ga2", "ga3", "ga4"):
            blk = getattr(module, name, None)
            if blk is not None and hasattr(blk, "to_deploy"):
                setattr(module, name, blk.to_deploy())
        return module.eval()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.convnext_backbone.stem(x)
        x = self.convnext_backbone.stage1(x)
        x = self.ga1(x)
        x = self.convnext_backbone.stage2(x)
        x = self.ga2(x)
        x = self.convnext_backbone.stage3(x)
        fs = self.fs2(self.fs1(x))
        x = self.ga3(x)
        x = self.convnext_backbone.stage4(x)
        x = self.ga4(x)

        global_feat = self.gap(x).flatten(1)
        fs_feat = self.gap(fs).flatten(1)
        if self.return_raw_features_eval:
            return global_feat, fs_feat
        return self.bnneck_global(global_feat), self.bnneck_fs(fs_feat)


# --------------------------------------------------------------------------- #
# Temporal Encoders & Stage 2 ReID Head
# --------------------------------------------------------------------------- #
class AttentionPooling(nn.Module):
    def __init__(self, d_model: int, hidden: Optional[int] = None):
        super().__init__()
        hidden = hidden if hidden is not None else max(d_model // 2, 8)
        self.score = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)
        self.capture_weights: bool = False
        self.last_weights: Optional[Tensor] = None

    def forward(self, x: Tensor) -> Tensor:
        w = self.score(x).squeeze(-1)
        a = torch.softmax(w, dim=1)
        if self.capture_weights:
            self.last_weights = a.detach()
        return (a.unsqueeze(-1) * x).sum(dim=1)


class SimpleS6Block(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        _, L, _ = x.shape
        xz = self.in_proj(x)
        x_proj, z_gate = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_proj.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = self.act(x_conv)

        x_dbl = self.x_proj(x_conv)
        dt, B_mat, C_mat = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt)).float()
        B_mat = B_mat.float()
        C_mat = C_mat.float()
        x_conv_f32 = x_conv.float()
        A = -torch.exp(self.A_log.float())

        W = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        V = (dt.unsqueeze(-1) * B_mat.unsqueeze(2)) * x_conv_f32.unsqueeze(-1)
        a = torch.exp(W)

        a_steps = torch.unbind(a, dim=1)
        v_steps = torch.unbind(V, dim=1)
        h = torch.zeros_like(v_steps[0])
        h_list = []
        for t in range(L):
            h = a_steps[t] * h + v_steps[t]
            h_list.append(h)
        h_seq = torch.stack(h_list, dim=1)

        y = (h_seq * C_mat.unsqueeze(2)).sum(dim=-1).to(x.dtype)
        y = y + x_conv * self.D
        y = y * self.act(z_gate)
        return self.out_proj(y)


class TemporalMambaEncoder(nn.Module):
    def __init__(
        self,
        d_in: int = 960,
        d_model: int = 512,
        d_out: int = 512,
        max_seq_len: int = 64,
        num_layers: int = 2,
        pool: str = "attn",
        use_pe: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.pool = pool
        self.use_pe = use_pe
        self.max_seq_len = max_seq_len

        self.in_proj = nn.Linear(d_in, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model)) if use_pe else None
        self.attn_pool = AttentionPooling(d_model) if pool == "attn" else None
        self.layers = nn.ModuleList([SimpleS6Block(d_model=d_model) for _ in range(num_layers)])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.out_mlp = nn.Sequential(
            nn.Linear(d_model, d_out),
            nn.BatchNorm1d(d_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        _, N, _ = x.shape
        x = self.in_proj(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed[:, :N, :]
        for mamba_layer, norm in zip(self.layers, self.norm_layers):
            res = x
            x_fwd = mamba_layer(x)
            x_bwd = mamba_layer(x.flip(dims=[1])).flip(dims=[1])
            x = norm(x_fwd + x_bwd + res)
        temporal_seq = x
        pooled = self.attn_pool(x) if self.attn_pool is not None else x.mean(dim=1)
        return self.out_mlp(pooled), temporal_seq


class TemporalAttentionEncoder(nn.Module):
    def __init__(
        self,
        d_in: int = 960,
        d_model: int = 512,
        d_out: int = 512,
        max_seq_len: int = 64,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        pool: str = "attn",
        use_pe: bool = True,
    ):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model)) if use_pe else None
        self.attn_pool = AttentionPooling(d_model) if pool == "attn" else None
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_mlp = nn.Sequential(
            nn.Linear(d_model, d_out),
            nn.BatchNorm1d(d_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        _, N, _ = x.shape
        x = self.in_proj(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed[:, :N, :]
        x = self.transformer(x)
        temporal_seq = x
        pooled = self.attn_pool(x) if self.attn_pool is not None else x.mean(dim=1)
        return self.out_mlp(pooled), temporal_seq


class ReIDHead(nn.Module):
    def __init__(self, in_dim: int = 1472, num_identities: int = 502):
        super().__init__()
        self.bnneck = nn.BatchNorm1d(in_dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(in_dim, num_identities, bias=False)

    def forward(self, visual_feat: Tensor, temporal_token: Optional[Tensor] = None) -> Tensor:
        feat = visual_feat if temporal_token is None else torch.cat([visual_feat, temporal_token], dim=-1)
        return self.bnneck(feat)


class Stage2UAVReIDNet(nn.Module):
    def __init__(
        self,
        num_identities: int = 502,
        gate: str = "bcgg",
        gate_cfg: Optional[Dict[str, Any]] = None,
        temporal_type: str = "mamba",
        temporal_pool: str = "attn",
        temporal_pe: bool = True,
        temporal_d_model: int = 512,
        temporal_d_out: int = 512,
        temporal_layers: int = 2,
    ):
        super().__init__()
        self.gate = gate
        self.temporal_type = temporal_type
        self.temporal_pool = temporal_pool
        self.temporal_pe = temporal_pe
        self.visual_dim = 960

        self.backbone = GASNetConvNeXt(num_classes=num_identities, gate=gate, gate_cfg=gate_cfg)
        if temporal_type == "none":
            self.temporal_encoder = None
            head_in_dim = self.visual_dim
        elif temporal_type == "mamba":
            self.temporal_encoder = TemporalMambaEncoder(
                d_in=self.visual_dim, d_model=temporal_d_model, d_out=temporal_d_out,
                max_seq_len=64, num_layers=temporal_layers, pool=temporal_pool, use_pe=temporal_pe,
            )
            head_in_dim = self.visual_dim + temporal_d_out
        elif temporal_type == "attention":
            self.temporal_encoder = TemporalAttentionEncoder(
                d_in=self.visual_dim, d_model=temporal_d_model, d_out=temporal_d_out,
                max_seq_len=64, num_layers=temporal_layers, pool=temporal_pool, use_pe=temporal_pe,
            )
            head_in_dim = self.visual_dim + temporal_d_out
        else:
            raise ValueError(f"Unsupported temporal_type: {temporal_type!r}")
        self.head = ReIDHead(in_dim=head_in_dim, num_identities=num_identities)

    def set_temporal_attention_capture(self, enabled: bool) -> None:
        if self.temporal_encoder is not None and getattr(self.temporal_encoder, "attn_pool", None) is not None:
            ap = self.temporal_encoder.attn_pool
            ap.capture_weights = bool(enabled)
            if not enabled:
                ap.last_weights = None

    def extract_clip_visual(self, clips: Tensor) -> Tensor:
        B, N, C, H, W = clips.shape
        x = clips.view(B * N, C, H, W)
        g, fs = self.backbone(x)
        return torch.cat([g, fs], dim=-1).view(B, N, -1)

    def forward_all_spaces(self, clips: Tensor) -> Dict[str, Tensor]:
        seq_feats = self.extract_clip_visual(clips)
        visual_mean = seq_feats.mean(dim=1)
        if self.temporal_encoder is None:
            fused = self.head(visual_mean, None)
            return {
                "visual_mean": visual_mean,
                "fused": fused,
                "pre_bn": visual_mean,
            }
        temporal_token, _ = self.temporal_encoder(seq_feats)
        pre_bn = torch.cat([visual_mean, temporal_token], dim=-1)
        fused = self.head(visual_mean, temporal_token)
        return {
            "visual_mean": visual_mean,
            "temporal_token": temporal_token,
            "pre_bn": pre_bn,
            "fused": fused,
        }

    def forward(self, clips: Tensor) -> Tensor:
        seq_feats = self.extract_clip_visual(clips)
        visual_mean = seq_feats.mean(dim=1)
        if self.temporal_encoder is None:
            return self.head(visual_mean, None)
        temporal_token, _ = self.temporal_encoder(seq_feats)
        return self.head(visual_mean, temporal_token)

    @torch.no_grad()
    def to_deploy(self) -> "Stage2UAVReIDNet":
        mod = copy.deepcopy(self).eval()
        mod.backbone = mod.backbone.to_deploy()
        return mod.eval()


class Stage2BackboneWrapper(nn.Module):
    def __init__(self, uav_model: Stage2UAVReIDNet):
        super().__init__()
        self.backbone = copy.deepcopy(uav_model.backbone).cpu().eval().to_deploy()
        self.backbone.return_raw_features_eval = True
        if hasattr(self.backbone, "gap") and hasattr(self.backbone.gap, "p"):
            self.backbone.gap = _FP16SafeGeMPool(p=self.backbone.gap.p, eps=self.backbone.gap.eps)

    def forward(self, x: Tensor) -> Tensor:
        g, fs = self.backbone(x)
        return torch.cat([g, fs], dim=-1)


class Stage2TemporalWrapper(nn.Module):
    def __init__(self, uav_model: Stage2UAVReIDNet):
        super().__init__()
        self.temporal_encoder = copy.deepcopy(uav_model.temporal_encoder).cpu().eval()
        self.head = copy.deepcopy(uav_model.head).cpu().eval()

    def forward(self, seq_feats: Tensor) -> Tensor:
        visual_feat = seq_feats.mean(dim=1)
        temporal_token, _ = self.temporal_encoder(seq_feats)
        bn_feat = self.head(visual_feat, temporal_token)
        return F.normalize(bn_feat, p=2, dim=1)


# --------------------------------------------------------------------------- #
# Checkpoint Inspection & Strict Loader
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _strip_prefix(sd: Dict[str, Tensor]) -> Dict[str, Tensor]:
    return {
        (k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()
    }


def _is_classifier_key(name: str) -> bool:
    n = name.lower()
    return ("classifier" in n) or n.endswith(".fc.weight") or n.endswith(".fc.bias")


def auto_inspect_stage2_checkpoint(ckpt_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Tensor]]:
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"[FAIL] Missing Stage 2 checkpoint: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    provenance = dict(raw.get("provenance", {}) or {}) if isinstance(raw, dict) else {}
    sd_raw = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
    sd = _strip_prefix(sd_raw)

    # Infer gate
    is_bcgg = any(k.startswith("backbone.ga1.reduce.") for k in sd)
    gate = "bcgg" if is_bcgg else "rga"
    gate_cfg: Dict[str, Any] = {}
    if is_bcgg:
        sp = sd.get("backbone.ga1.spatial.weight")
        rd = sd.get("backbone.ga1.reduce.0.weight")
        ch = sd.get("backbone.ga1.channel.conv.weight")
        cx = sd.get("backbone.ga1.context.branches.0.conv.weight")
        gate_cfg = {
            "gate_groups": int(sp.shape[0]) if sp is not None else 4,
            "gate_hidden": int(rd.shape[0]) if rd is not None else 16,
            "gate_kernel": int(cx.shape[-1]) if cx is not None else 7,
            "gate_channel_kernel": int(ch.shape[-1]) if ch is not None else 5,
            "gate_inject_value": any(k.startswith("backbone.ga1.value.") for k in sd),
        }

    # Infer temporal type / pool / pe
    has_temp = any(k.startswith("temporal_encoder.") for k in sd)
    if not has_temp:
        temporal_type = "none"
        temporal_pool = "mean"
        temporal_pe = False
        temporal_impl = "none"
    else:
        if any(k.startswith("temporal_encoder.layers.0.x_proj") for k in sd):
            temporal_type = "mamba"
            x_shape = tuple(sd["temporal_encoder.layers.0.x_proj.weight"].shape)
            temporal_impl = "SimpleS6Block" if x_shape[0] == 33 else f"mamba_ssm{x_shape}"
        else:
            temporal_type = "attention"
            temporal_impl = "TemporalAttentionEncoder"
        temporal_pool = "attn" if any("temporal_encoder.attn_pool." in k for k in sd) else "mean"
        temporal_pe = "temporal_encoder.pos_embed" in sd

    cls_w = sd.get("head.classifier.weight")
    num_identities = int(cls_w.shape[0]) if cls_w is not None else 502

    meta = {
        "backbone": "dinov3_convnext",
        "gate": gate,
        "gate_cfg": gate_cfg,
        "temporal_type": temporal_type,
        "temporal_pool": temporal_pool,
        "temporal_pe": temporal_pe,
        "temporal_impl": temporal_impl,
        "num_frames_ckpt": provenance.get("num_frames", 8),
        "stride_ckpt": provenance.get("frame_stride", 4),
        "num_identities": num_identities,
        "best_val_rank1": float(raw["best_val_rank1"]) if isinstance(raw, dict) and "best_val_rank1" in raw else None,
        "epoch": raw.get("epoch") if isinstance(raw, dict) else None,
        "provenance": provenance,
    }
    return meta, raw, sd


def load_stage2_checkpoint_strict(
    model_key: str,
    ckpt_path: Path,
    deploy_bcgg: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[Stage2UAVReIDNet, Dict[str, Any]]:
    meta, _, sd = auto_inspect_stage2_checkpoint(ckpt_path)
    model = Stage2UAVReIDNet(
        num_identities=meta["num_identities"],
        gate=meta["gate"],
        gate_cfg=meta["gate_cfg"],
        temporal_type=meta["temporal_type"],
        temporal_pool=meta["temporal_pool"],
        temporal_pe=meta["temporal_pe"],
    )
    model_sd = model.state_dict()

    matched_keys: List[str] = []
    shape_mismatch: List[str] = []
    filtered: Dict[str, Tensor] = {}
    for k, v in sd.items():
        if k in model_sd:
            if tuple(v.shape) == tuple(model_sd[k].shape):
                filtered[k] = v
                matched_keys.append(k)
            else:
                shape_mismatch.append(f"{k}: ckpt={tuple(v.shape)} vs model={tuple(model_sd[k].shape)}")

    missing_all = [k for k in model_sd if k not in filtered]
    unexpected_all = [k for k in sd if k not in model_sd]

    missing_feature = [k for k in missing_all if not _is_classifier_key(k)]
    unexpected_feature = [k for k in unexpected_all if not _is_classifier_key(k)]
    ignored_classifier = [k for k in missing_all + unexpected_all if _is_classifier_key(k)]

    if missing_feature or unexpected_feature or shape_mismatch:
        raise RuntimeError(
            f"[FAIL] Strict Stage 2 checkpoint verification failed for {model_key} ({ckpt_path}): "
            f"missing_feature={missing_feature[:10]}, unexpected_feature={unexpected_feature[:10]}, "
            f"shape_mismatch={shape_mismatch[:5]}"
        )

    model.load_state_dict(filtered, strict=(len(missing_all) == 0 and len(unexpected_all) == 0))
    model.eval()
    if deploy_bcgg and meta["gate"] == "bcgg":
        model = model.to_deploy().eval()
    if device is not None:
        model = model.to(device)

    total_feature_keys = sum(1 for k in model_sd if not _is_classifier_key(k))
    matched_feature_keys = sum(1 for k in matched_keys if not _is_classifier_key(k))
    assert matched_feature_keys == total_feature_keys

    audit = {
        "model_key": model_key,
        "path": str(ckpt_path),
        "sha256": sha256_file(ckpt_path),
        "size_MB": round(ckpt_path.stat().st_size / (1024.0 * 1024.0), 3),
        "backbone": meta["backbone"],
        "gate": meta["gate"],
        "gate_cfg": meta["gate_cfg"],
        "temporal_type": meta["temporal_type"],
        "temporal_pool": meta["temporal_pool"],
        "temporal_pe": meta["temporal_pe"],
        "temporal_impl": meta["temporal_impl"],
        "num_frames_ckpt": meta["num_frames_ckpt"],
        "stride_ckpt": meta["stride_ckpt"],
        "num_identities": meta["num_identities"],
        "best_val_rank1": meta["best_val_rank1"],
        "matched_feature_keys": f"{matched_feature_keys}/{total_feature_keys}",
        "total_matched_keys": f"{len(matched_keys)}/{len(model_sd)}",
        "ignored_classifier_keys": ignored_classifier,
        "unexpected_feature_keys": unexpected_feature,
        "missing_feature_keys": missing_feature,
        "strict_feature_match": True,
    }
    return model, audit
