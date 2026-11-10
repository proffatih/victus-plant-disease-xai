"""Hybrid EfficientNet-V2-S + Swin-V2-T classifiers.

R3 (Reviewer 5, E2). The R2 fusion block applied ``nn.MultiheadAttention`` to
key/value sequences of length one. With a single key the softmax returns weight
1 identically, so the block reduced to

    c_out = c + W_O W_V t ,   t_out = t + W_O' W_V' c ,

a residual *linear* cross-mixing with no query dependence: the heads and the
head dimension were inert, and the dropout acted as a stochastic gate on the
whole cross-term rather than as regularisation. That block is retained here
under its accurate name, ``GatedLinearLateFusion``, and the manuscript no
longer describes it as attention.

``TokenCrossAttentionFusion`` is the non-degenerate design the R2 text claimed:
genuine bidirectional multi-head cross-attention between the EfficientNetV2-S
final feature grid (8x8 = 64 tokens at 256 px) and the Swin V2-T stage-4 token
sequence (8x8 = 64 tokens). It is trained and reported as a like-for-like
ablation so that the null result can be attributed to the problem rather than
to the degeneracy.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import timm


class GatedLinearLateFusion(nn.Module):
    """The R2 block, named for what it actually computes (E2, option b).

    Pools each branch to a vector, applies a residual linear cross-mixing that
    is stochastically gated during training, concatenates and passes an MLP.
    Algebraically equivalent to concatenation plus an MLP. Kept bit-exact with
    the R2 implementation so the previously reported numbers remain valid.
    """

    def __init__(self, dim_cnn: int, dim_tr: int, fused_dim: int = 384,
                 heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj_cnn = nn.Linear(dim_cnn, fused_dim)
        self.proj_tr = nn.Linear(dim_tr, fused_dim)
        self.norm_cnn = nn.LayerNorm(fused_dim)
        self.norm_tr = nn.LayerNorm(fused_dim)
        # Retained as MultiheadAttention purely for weight compatibility with
        # the released R2 checkpoint; over a length-1 sequence it is linear.
        self.attn_cnn_from_tr = nn.MultiheadAttention(fused_dim, heads, dropout=dropout, batch_first=True)
        self.attn_tr_from_cnn = nn.MultiheadAttention(fused_dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_dim = fused_dim * 2

    def forward(self, f_cnn: torch.Tensor, f_tr: torch.Tensor) -> torch.Tensor:
        c = self.norm_cnn(self.proj_cnn(f_cnn)).unsqueeze(1)
        t = self.norm_tr(self.proj_tr(f_tr)).unsqueeze(1)
        c2, _ = self.attn_cnn_from_tr(c, t, t)
        t2, _ = self.attn_tr_from_cnn(t, c, c)
        fused = torch.cat([(c + c2).squeeze(1), (t + t2).squeeze(1)], dim=-1)
        return self.ffn(fused)


class TokenCrossAttentionFusion(nn.Module):
    """Genuine bidirectional cross-attention over spatial token sequences (E2a).

    Queries from one branch attend over the *sequence* of the other branch, so
    the softmax runs over 64 keys and the operation is query-dependent. Each
    direction is a pre-norm attention block with a residual, followed by a
    pre-norm MLP, then mean-pooling over tokens and concatenation.
    """

    def __init__(self, dim_cnn: int, dim_tr: int, fused_dim: int = 384,
                 heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj_cnn = nn.Linear(dim_cnn, fused_dim)
        self.proj_tr = nn.Linear(dim_tr, fused_dim)
        self.norm_c_q, self.norm_c_kv = nn.LayerNorm(fused_dim), nn.LayerNorm(fused_dim)
        self.norm_t_q, self.norm_t_kv = nn.LayerNorm(fused_dim), nn.LayerNorm(fused_dim)
        self.attn_cnn_from_tr = nn.MultiheadAttention(fused_dim, heads, dropout=dropout, batch_first=True)
        self.attn_tr_from_cnn = nn.MultiheadAttention(fused_dim, heads, dropout=dropout, batch_first=True)
        self.norm_c2, self.norm_t2 = nn.LayerNorm(fused_dim), nn.LayerNorm(fused_dim)
        self.mlp_c = nn.Sequential(nn.Linear(fused_dim, fused_dim * 2), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(fused_dim * 2, fused_dim))
        self.mlp_t = nn.Sequential(nn.Linear(fused_dim, fused_dim * 2), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(fused_dim * 2, fused_dim))
        self.out_dim = fused_dim * 2
        self.last_attn: dict[str, torch.Tensor] = {}

    def forward(self, tok_cnn: torch.Tensor, tok_tr: torch.Tensor,
                keep_attn: bool = False) -> torch.Tensor:
        # tok_cnn: (B, Nc, dim_cnn)   tok_tr: (B, Nt, dim_tr)
        c = self.proj_cnn(tok_cnn)
        t = self.proj_tr(tok_tr)
        c2, aw_c = self.attn_cnn_from_tr(self.norm_c_q(c), self.norm_t_kv(t), self.norm_t_kv(t),
                                         need_weights=keep_attn, average_attn_weights=True)
        t2, aw_t = self.attn_tr_from_cnn(self.norm_t_q(t), self.norm_c_kv(c), self.norm_c_kv(c),
                                         need_weights=keep_attn, average_attn_weights=True)
        c = c + c2
        t = t + t2
        c = c + self.mlp_c(self.norm_c2(c))
        t = t + self.mlp_t(self.norm_t2(t))
        if keep_attn:
            self.last_attn = {"cnn_from_tr": aw_c, "tr_from_cnn": aw_t}
        return torch.cat([c.mean(dim=1), t.mean(dim=1)], dim=-1)


class HybridLeafClassifier(nn.Module):
    """Two-backbone classifier. ``fusion`` selects the fusion block.

    fusion="gated_linear" reproduces the R2 model exactly (default, so the
        released checkpoint loads unchanged).
    fusion="token_cross_attn" is the genuine cross-attention variant (E2a).
    """

    def __init__(
        self,
        num_classes: int,
        cnn_name: str = "tf_efficientnetv2_s.in21k_ft_in1k",
        tr_name: str = "swinv2_tiny_window8_256.ms_in1k",
        img_size_tr: int = 256,
        pretrained: bool = True,
        freeze_stem_cnn: bool = True,
        freeze_stem_tr: bool = True,
        fusion: str = "gated_linear",
    ):
        super().__init__()
        self.fusion_kind = fusion
        pool = "" if fusion == "token_cross_attn" else "avg"
        self.cnn = timm.create_model(cnn_name, pretrained=pretrained, num_classes=0, global_pool=pool)
        self.tr = timm.create_model(tr_name, pretrained=pretrained, num_classes=0, global_pool=pool)

        with torch.no_grad():
            probe = torch.zeros(1, 3, img_size_tr, img_size_tr)
            if fusion == "token_cross_attn":
                self.dim_cnn = self.cnn.forward_features(probe).shape[1]   # NCHW -> C
                self.dim_tr = self.tr.forward_features(probe).shape[-1]    # NHWC -> C
            else:
                self.dim_cnn = self.cnn(probe).shape[-1]
                self.dim_tr = self.tr(probe).shape[-1]

        if fusion == "token_cross_attn":
            self.fusion = TokenCrossAttentionFusion(self.dim_cnn, self.dim_tr, fused_dim=384)
        elif fusion == "gated_linear":
            self.fusion = GatedLinearLateFusion(self.dim_cnn, self.dim_tr, fused_dim=384)
        else:
            raise ValueError(f"unknown fusion {fusion!r}")

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
        for name, p in self.cnn.named_parameters():
            if name.startswith(("conv_stem", "bn1", "blocks.0", "blocks.1")):
                p.requires_grad = False

    def _freeze_first_stages_swin(self):
        for name, p in self.tr.named_parameters():
            if name.startswith(("patch_embed", "layers.0")):
                p.requires_grad = False

    def forward(self, x: torch.Tensor, keep_attn: bool = False) -> torch.Tensor:
        if self.fusion_kind == "token_cross_attn":
            fc = self.cnn.forward_features(x)                 # (B, C, H, W)
            ft = self.tr.forward_features(x)                  # (B, H, W, C)
            tok_c = fc.flatten(2).transpose(1, 2)             # (B, HW, C)
            tok_t = ft.flatten(1, 2)                          # (B, HW, C)
            fused = self.fusion(tok_c, tok_t, keep_attn=keep_attn)
        else:
            fused = self.fusion(self.cnn(x), self.tr(x))
        return self.head(fused)

    def cnn_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Spatial feature map of the CNN branch, for Grad-CAM."""
        return self.cnn.forward_features(x)


class SingleBackbone(nn.Module):
    """Baseline: one backbone + linear head. Used for the ablation."""

    def __init__(self, num_classes: int, name: str, img_size: int = 256, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
        with torch.no_grad():
            self.dim = self.backbone(torch.zeros(1, 3, img_size, img_size)).shape[-1]
        self.head = nn.Sequential(
            nn.LayerNorm(self.dim), nn.Dropout(0.15), nn.Linear(self.dim, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))
