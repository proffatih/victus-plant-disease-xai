"""R3 / E3 + E4 — real efficiency measurements.

Replaces the hard-coded constants behind the R2 Figure 9 and the R2 latency
number. Every quantity below is measured on this machine, at the resolution
actually used (256x256), for all three trained architectures plus the token
cross-attention variant:

  * parameters, counted from the instantiated model (no ImageNet head);
  * multiply-accumulate operations, counted with fvcore at 256x256;
  * batch-1 inference latency: 30 warm-up iterations, then >=100 timed
    iterations, each bracketed by torch.cuda.synchronize(); mean +/- SD;
  * batch-32 throughput, measured the same way;
  * peak GPU memory (torch.cuda.max_memory_allocated) for both batch sizes;
  * on-disk checkpoint size.

Model-only timing: a pre-loaded CUDA tensor is fed to the network, so no data
loading, JPEG decoding, host-to-device copy or post-processing is included.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import HybridLeafClassifier, SingleBackbone

RES = Path(os.environ.get("RESULTS_DIR", Path(__file__).resolve().parents[1] / "results"))
IMG = 256
N_CLASSES = 38
WARMUP = 30
REPEATS = 120


def build(kind: str):
    if kind == "hybrid_gated":
        return HybridLeafClassifier(N_CLASSES, img_size_tr=IMG, pretrained=False, fusion="gated_linear")
    if kind == "hybrid_tokenattn":
        return HybridLeafClassifier(N_CLASSES, img_size_tr=IMG, pretrained=False, fusion="token_cross_attn")
    if kind == "effv2s":
        return SingleBackbone(N_CLASSES, "tf_efficientnetv2_s.in21k_ft_in1k", IMG, False)
    if kind == "swinv2t":
        return SingleBackbone(N_CLASSES, "swinv2_tiny_window8_256.ms_in1k", IMG, False)
    raise ValueError(kind)


def count_macs(model, device):
    """MACs at the resolution actually used. fvcore reports MACs as 'flops'."""
    try:
        from fvcore.nn import FlopCountAnalysis
        m = model.eval()
        x = torch.zeros(1, 3, IMG, IMG, device=device)
        fca = FlopCountAnalysis(m, x)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        return float(fca.total()), "fvcore"
    except Exception as e:                                    # pragma: no cover
        print(f"  [warn] fvcore failed ({e}); trying thop")
        from thop import profile
        macs, _ = profile(model.eval(), inputs=(torch.zeros(1, 3, IMG, IMG, device=device),), verbose=False)
        return float(macs), "thop"


@torch.no_grad()
def time_model(model, device, batch: int):
    model.eval()
    x = torch.randn(batch, 3, IMG, IMG, device=device)        # already on device
    for _ in range(WARMUP):
        model(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(REPEATS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    t = np.array(times)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return {
        "batch": batch,
        "n_warmup": WARMUP,
        "n_timed": REPEATS,
        "batch_latency_ms_mean": float(t.mean() * 1e3),
        "batch_latency_ms_sd": float(t.std(ddof=1) * 1e3),
        "per_image_ms_mean": float(t.mean() * 1e3 / batch),
        "per_image_ms_sd": float(t.std(ddof=1) * 1e3 / batch),
        "throughput_img_per_s": float(batch / t.mean()),
        "peak_gpu_mem_MiB": float(peak),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "efficiency measurements require a GPU"
    print(f"[info] device={torch.cuda.get_device_name(0)} torch={torch.__version__}")
    out = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "img_size": IMG, "precision": "fp32", "models": {}}

    for kind in ["effv2s", "swinv2t", "hybrid_gated", "hybrid_tokenattn"]:
        print(f"[measure] {kind}", flush=True)
        model = build(kind).to(device)
        n_total = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        macs, macs_tool = count_macs(model, device)
        rec = {
            "params_total": n_total,
            "params_total_M": n_total / 1e6,
            "params_trainable_M": n_train / 1e6,
            "macs": macs,
            "macs_G": macs / 1e9,
            "macs_tool": macs_tool,
            "timing_batch1": time_model(model, device, 1),
            "timing_batch32": time_model(model, device, 32),
        }
        for tag in [kind, {"hybrid_gated": "hybrid_gated_s42",
                           "hybrid_tokenattn": "hybrid_tokenattn_s42",
                           "effv2s": "effv2s_s42", "swinv2t": "swinv2t_s42"}[kind]]:
            ck = RES / f"best_{tag}.pt"
            if ck.exists():
                rec["ckpt_file"] = ck.name
                rec["ckpt_size_MiB"] = ck.stat().st_size / (1024 ** 2)
                break
        out["models"][kind] = rec
        print(f"  params={rec['params_total_M']:.2f}M  MACs={rec['macs_G']:.2f}G  "
              f"b1={rec['timing_batch1']['per_image_ms_mean']:.2f}+/-"
              f"{rec['timing_batch1']['per_image_ms_sd']:.2f}ms  "
              f"b32={rec['timing_batch32']['per_image_ms_mean']:.2f}ms/img", flush=True)
        del model
        torch.cuda.empty_cache()

    with open(RES / "efficiency_measured.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {RES/'efficiency_measured.json'}")


if __name__ == "__main__":
    main()
