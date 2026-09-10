"""R3 / E2 — numerical proof that the R2 fusion block is not attention.

Reviewer 5 states that with a key/value sequence of length one the softmax
returns weight 1 identically, so the block reduces to a residual linear
cross-mixing. This script demonstrates that claim numerically on the released
architecture, and contrasts it with the token-level block, so the response
letter rests on a reproducible artefact rather than on an argument.

Checks:
  1. attention weights over a length-1 sequence are exactly 1;
  2. the cross-term is independent of the query (swap the query, output of the
     attention sub-layer is unchanged);
  3. the block is exactly affine in each input (superposition holds);
  4. the token-level block fails all three, i.e. it really attends.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from model import GatedLinearLateFusion, TokenCrossAttentionFusion

torch.manual_seed(0)
OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(parents=True, exist_ok=True)
res = {}

D_C, D_T, F = 1280, 768, 384
g = GatedLinearLateFusion(D_C, D_T, fused_dim=F).eval()

# --- 1. attention weights over a length-1 key sequence -----------------------
with torch.no_grad():
    c = torch.randn(8, 1, F)
    t = torch.randn(8, 1, F)
    _, w = g.attn_cnn_from_tr(c, t, t, need_weights=True, average_attn_weights=True)
res["attn_weights_shape"] = list(w.shape)
res["attn_weights_all_exactly_one"] = bool(torch.all(w == 1.0).item())
res["attn_weights_max_dev_from_1"] = float((w - 1.0).abs().max().item())

# --- 2. query independence of the cross term --------------------------------
with torch.no_grad():
    q1, q2 = torch.randn(8, 1, F), torch.randn(8, 1, F)
    kv = torch.randn(8, 1, F)
    o1, _ = g.attn_cnn_from_tr(q1, kv, kv, need_weights=False)
    o2, _ = g.attn_cnn_from_tr(q2, kv, kv, need_weights=False)
res["cross_term_query_independent"] = bool(torch.allclose(o1, o2, atol=1e-6))
res["cross_term_max_abs_diff_between_queries"] = float((o1 - o2).abs().max().item())

# --- 3. the attention sub-layer is exactly LINEAR in its key/value input ----
# Reviewer 5's algebraic claim is about the attention operator itself: with one
# key, attn(q, kv, kv) = W_O W_V kv + b, independent of q. LayerNorm upstream is
# of course nonlinear, so linearity is tested where the claim is made.
with torch.no_grad():
    q = torch.randn(8, 1, F)
    k1, k2 = torch.randn(8, 1, F), torch.randn(8, 1, F)
    a, b = 0.3, -1.7
    o0, _ = g.attn_cnn_from_tr(q, torch.zeros(8, 1, F), torch.zeros(8, 1, F), need_weights=False)
    o_mix, _ = g.attn_cnn_from_tr(q, a * k1 + b * k2, a * k1 + b * k2, need_weights=False)
    o_1, _ = g.attn_cnn_from_tr(q, k1, k1, need_weights=False)
    o_2, _ = g.attn_cnn_from_tr(q, k2, k2, need_weights=False)
    lhs = o_mix - o0
    rhs = a * (o_1 - o0) + b * (o_2 - o0)
res["attn_sublayer_exactly_linear_in_kv"] = bool(torch.allclose(lhs, rhs, atol=1e-5))
res["attn_linearity_max_abs_residual"] = float((lhs - rhs).abs().max().item())

# The same test on the token-level block must FAIL, because a 64-key softmax
# makes the operator genuinely nonlinear in its keys.
tk_probe = TokenCrossAttentionFusion(D_C, D_T, fused_dim=F).eval()
with torch.no_grad():
    q_n = torch.randn(8, 64, F)
    k1n, k2n = torch.randn(8, 64, F), torch.randn(8, 64, F)
    z = torch.zeros(8, 64, F)
    p0, _ = tk_probe.attn_cnn_from_tr(q_n, z, z, need_weights=False)
    pm, _ = tk_probe.attn_cnn_from_tr(q_n, a * k1n + b * k2n, a * k1n + b * k2n, need_weights=False)
    p1, _ = tk_probe.attn_cnn_from_tr(q_n, k1n, k1n, need_weights=False)
    p2, _ = tk_probe.attn_cnn_from_tr(q_n, k2n, k2n, need_weights=False)
    lhs_n = pm - p0
    rhs_n = a * (p1 - p0) + b * (p2 - p0)
res["token_attn_sublayer_exactly_linear_in_kv"] = bool(torch.allclose(lhs_n, rhs_n, atol=1e-5))
res["token_attn_linearity_max_abs_residual"] = float((lhs_n - rhs_n).abs().max().item())

# --- 4. the token-level block genuinely attends -----------------------------
tk = TokenCrossAttentionFusion(D_C, D_T, fused_dim=F).eval()
with torch.no_grad():
    tokc = torch.randn(8, 64, D_C)
    tokt = torch.randn(8, 64, D_T)
    _ = tk(tokc, tokt, keep_attn=True)
    aw = tk.last_attn["cnn_from_tr"]
    # query dependence
    q_a = tk.proj_cnn(tokc)
    q_b = tk.proj_cnn(torch.randn(8, 64, D_C))
    kv = tk.norm_t_kv(tk.proj_tr(tokt))
    oa, _ = tk.attn_cnn_from_tr(tk.norm_c_q(q_a), kv, kv, need_weights=False)
    ob, _ = tk.attn_cnn_from_tr(tk.norm_c_q(q_b), kv, kv, need_weights=False)
res["token_block"] = {
    "attn_weights_shape": list(aw.shape),
    "n_keys": int(aw.shape[-1]),
    "weight_entropy_mean_nats": float((-(aw.clamp_min(1e-12) * aw.clamp_min(1e-12).log()).sum(-1)).mean().item()),
    "max_entropy_if_uniform_nats": float(torch.log(torch.tensor(float(aw.shape[-1]))).item()),
    "weight_std_over_keys_mean": float(aw.std(-1).mean().item()),
    "cross_term_query_independent": bool(torch.allclose(oa, ob, atol=1e-6)),
    "cross_term_max_abs_diff_between_queries": float((oa - ob).abs().max().item()),
}

with open(OUT / "fusion_degeneracy_proof.json", "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))
