"""Hybrid EfficientNet-V2-S + Swin-V2-T model with cross-attention fusion.

Design:
  - Two timm backbones, forward_features() to obtain feature maps.
  - Global-avg-pool each stream to a compact token.
  - A single-layer bidirectional cross-attention block to fuse the two streams.
  - Concatenate the fused tokens and pass through a small MLP classifier head.

The fused vector serves as the target for the Grad-CAM analysis performed on
the EfficientNet-V2-S stream, which retains a 2D spatial feature map suitable
for CAM visualisation of the input image.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim_cnn: int, dim_tr: int, fused_dim: int = 512, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj_cnn = nn.Linear(dim_cnn, fused_dim)
        self.proj_tr = nn.Linear(dim_tr, fused_dim)
        self.norm_cnn = nn.LayerNorm(fused_dim)
        self.norm_tr = nn.LayerNorm(fused_dim)
        self.attn_cnn_from_tr = nn.MultiheadAttention(fused_dim, heads, dropout=dropout, batch_first=True)
        self.attn_tr_from_cnn = nn.MultiheadAttention(fused_dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_dim = fused_dim * 2

    def forward(self, f_cnn: torch.Tensor, f_tr: torch.Tensor) -> torch.Tensor:
        # f_cnn, f_tr are (B, D) vectors. Add a singleton token axis for MHA.
        c = self.norm_cnn(self.proj_cnn(f_cnn)).unsqueeze(1)  # (B,1,F)
        t = self.norm_tr(self.proj_tr(f_tr)).unsqueeze(1)     # (B,1,F)
        c2, _ = self.attn_cnn_from_tr(c, t, t)
        t2, _ = self.attn_tr_from_cnn(t, c, c)
        c_out = (c + c2).squeeze(1)
        t_out = (t + t2).squeeze(1)
        fused = torch.cat([c_out, t_out], dim=-1)
        fused = self.ffn(fused)
        return fused


class HybridLeafClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        cnn_name: str = "tf_efficientnetv2_s.in21k_ft_in1k",
        tr_name: str = "swinv2_tiny_window8_256.ms_in1k",
        img_size_tr: int = 256,
        pretrained: bool = True,
        img_size_cnn: int = None,
        freeze_stem_cnn: bool = True,
        freeze_stem_tr: bool = True,
    ):
        super().__init__()
        self.cnn = timm.create_model(cnn_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        self.tr = timm.create_model(tr_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        # feature dims (via a probe forward on cpu with tiny tensor)
        with torch.no_grad():
            probe = torch.zeros(1, 3, img_size_tr, img_size_tr)
            self.dim_cnn = self.cnn(probe).shape[-1]
            self.dim_tr = self.tr(probe).shape[-1]
        self.fusion = CrossAttentionFusion(self.dim_cnn, self.dim_tr, fused_dim=384)
        self.head = nn.Sequential(
            nn.LayerNorm(self.fusion.out_dim),
            nn.Dropout(0.15),
            nn.Linear(self.fusion.out_dim, num_classes),
        )
        if freeze_stem_cnn:
            self._freeze_first_stages_effv2()
        if freeze_stem_tr:
            self._freeze_first_stages_swin()

    def _freeze_first_stages_effv2(self):
        # freeze stem + first 2 blocks
        for name, p in self.cnn.named_parameters():
            if name.startswith("conv_stem") or name.startswith("bn1") or name.startswith("blocks.0") or name.startswith("blocks.1"):
                p.requires_grad = False

    def _freeze_first_stages_swin(self):
        for name, p in self.tr.named_parameters():
            if name.startswith("patch_embed") or name.startswith("layers.0"):
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_cnn = self.cnn(x)
        f_tr = self.tr(x)
        fused = self.fusion(f_cnn, f_tr)
        logits = self.head(fused)
        return logits

    # ------------- helpers for Grad-CAM on the CNN branch -----------------
    def cnn_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn.forward_features(x)


class SingleBackbone(nn.Module):
    """Baseline: one backbone + linear head. Used for ablation."""

    def __init__(self, num_classes: int, name: str, img_size: int = 256, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
        with torch.no_grad():
            probe = torch.zeros(1, 3, img_size, img_size)
            self.dim = self.backbone(probe).shape[-1]
        self.head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Dropout(0.15),
            nn.Linear(self.dim, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))
