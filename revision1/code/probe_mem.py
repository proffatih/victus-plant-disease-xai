"""Peak GPU memory of one fully-trainable training step per configuration."""
import json, sys, time
from pathlib import Path
import torch, torch.nn as nn
from torch.amp import GradScaler, autocast
sys.path.insert(0, str(Path(__file__).parent))
from train import build_model

res = {}
for model_name, fusion in (("hybrid", "gated_linear"), ("hybrid", "token_cross_attn"), ("swinv2t", "gated_linear"), ("effv2s", "gated_linear")):
    for ckpt in (False, True):
        for mb in (8, 16, 32):
            key = f"{model_name}-{fusion}-ckpt{int(ckpt)}-mb{mb}"
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            try:
                m = build_model(model_name, 38, 256, True, fusion)
                for p in m.parameters():
                    p.requires_grad = True
                if ckpt:
                    if model_name == "hybrid":
                        m.tr.set_grad_checkpointing(True)
                    elif model_name == "swinv2t":
                        m.backbone.set_grad_checkpointing(True)
                m = m.cuda().train()
                opt = torch.optim.AdamW(m.parameters(), lr=1e-4); sc = GradScaler("cuda")
                x = torch.randn(mb, 3, 256, 256, device="cuda"); y = torch.randint(0, 38, (mb,), device="cuda")
                t0 = time.time()
                for _ in range(3):
                    with autocast("cuda", dtype=torch.float16):
                        l = nn.functional.cross_entropy(m(x), y)
                    sc.scale(l).backward(); sc.step(opt); sc.update(); opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                res[key] = {"ok": True, "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
                            "s_per_step": (time.time() - t0) / 3}
            except torch.OutOfMemoryError:
                res[key] = {"ok": False}
            print(key, res[key], flush=True)
            del m
            for v in ("opt", "sc", "x", "y", "l"):
                globals().pop(v, None)
            torch.cuda.empty_cache()
json.dump(res, open(Path(__file__).resolve().parents[1] / "logs" / "probe_mem.json", "w"), indent=1)
