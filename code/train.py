"""Train a PlantVillage classifier (hybrid or single-backbone ablation).

R3 repairs (Reviewer 5):
  * minor-15 the final test evaluation reloads the retained best-validation
    checkpoint instead of scoring the last-epoch weights.
  * E11(b,c) the split is read from the deterministic manifests and every
    evaluation runs in FP32, so a checkpoint yields one test number.
  * E12(a) --seed fully seeds python/numpy/torch and is recorded in outputs.
  * E2 --fusion selects gated_linear (R2 model) or token_cross_attn.
"""
from __future__ import annotations
import argparse, csv, json, math, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset import (
    LeafDataset, index_plantvillage, stratified_split, list_plantvillage_classes,
    build_transforms, save_split_manifests, load_split_csv,
)
from model import HybridLeafClassifier, SingleBackbone

RESULTS = Path(os.environ.get("RESULTS_DIR", Path(__file__).resolve().parents[1] / "results"))


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_model(name: str, num_classes: int, img_size: int, pretrained=True, fusion="gated_linear"):
    if name == "hybrid":
        return HybridLeafClassifier(num_classes=num_classes, img_size_tr=img_size,
                                    pretrained=pretrained, fusion=fusion)
    if name == "effv2s":
        return SingleBackbone(num_classes, "tf_efficientnetv2_s.in21k_ft_in1k", img_size, pretrained)
    if name == "swinv2t":
        return SingleBackbone(num_classes, "swinv2_tiny_window8_256.ms_in1k", img_size, pretrained)
    raise ValueError(name)


@torch.no_grad()
def evaluate(model, loader, device):
    """FP32 evaluation (E11c): no autocast, so metrics are numerically stable."""
    model.eval()
    tot, correct, top5c, loss_sum = 0, 0, 0, 0.0
    ce = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x).float()
        loss_sum += ce(logits, y).item()
        correct += (logits.argmax(-1) == y).sum().item()
        top5 = logits.topk(min(5, logits.size(-1)), dim=-1).indices
        top5c += (top5 == y.unsqueeze(-1)).any(-1).sum().item()
        tot += y.size(0)
    return dict(loss=loss_sum / tot, acc=correct / tot, top5=top5c / tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hybrid", choices=["hybrid", "effv2s", "swinv2t"])
    ap.add_argument("--fusion", default="gated_linear", choices=["gated_linear", "token_cross_attn"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--img", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preproc", default="aspect", choices=["aspect", "square"])
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"{args.model}_{args.fusion}_s{args.seed}"

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classes = list_plantvillage_classes()
    c2i = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    # One canonical split, written once and reused by every run (E11a/b).
    if not (RESULTS / "split_train.csv").exists():
        tr, va, te = stratified_split(index_plantvillage(), seed=42)
        save_split_manifests(RESULTS, tr, va, te, c2i)
    tr = load_split_csv(RESULTS / "split_train.csv")
    va = load_split_csv(RESULTS / "split_val.csv")
    te = load_split_csv(RESULTS / "split_test.csv")
    print(f"[info] device={device} tag={tag} seed={args.seed} preproc={args.preproc}")
    print(f"[info] train={len(tr)} val={len(va)} test={len(te)} classes={num_classes}")

    tf_tr = build_transforms(args.img, train=True, preproc=args.preproc)
    tf_ev = build_transforms(args.img, train=False, preproc=args.preproc)
    mk = lambda items, tf, sh: DataLoader(
        LeafDataset(items, c2i, tf), batch_size=args.batch, shuffle=sh,
        num_workers=args.workers, pin_memory=True, drop_last=sh,
        persistent_workers=args.workers > 0)
    dl_tr, dl_va, dl_te = mk(tr, tf_tr, True), mk(va, tf_ev, False), mk(te, tf_ev, False)

    model = build_model(args.model, num_classes, args.img, True, args.fusion).to(device)
    n_tr_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all_p = sum(p.numel() for p in model.parameters())
    print(f"[info] params trainable={n_tr_p/1e6:.2f}M total={n_all_p/1e6:.2f}M")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.wd)
    total_steps = len(dl_tr) * args.epochs
    warmup = max(200, total_steps // 20)

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = GradScaler("cuda")
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)

    log_path = RESULTS / f"train_log_{tag}.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "step", "train_loss", "train_acc", "val_loss",
                                "val_acc", "val_top5", "lr", "time_s"])

    ckpt_path = RESULTS / f"best_{tag}.pt"
    best_val, step_global, t_start = 0.0, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        n, correct, loss_sum, t0 = 0, 0, 0.0, time.time()
        for x, y in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.float16):
                loss = ce(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            step_global += 1
            n += y.size(0); loss_sum += loss.item() * y.size(0)
            if step_global % 200 == 0:
                print(f"[e{epoch} s{step_global}] loss={loss_sum/n:.4f} "
                      f"lr={opt.param_groups[0]['lr']:.2e} {time.time()-t0:.0f}s", flush=True)
        tr_loss = loss_sum / max(1, n)
        m_val = evaluate(model, dl_va, device)
        el = time.time() - t_start
        print(f"[epoch {epoch}] train_loss={tr_loss:.4f} | val acc={m_val['acc']:.4f} "
              f"top5={m_val['top5']:.4f} | {el:.0f}s", flush=True)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, step_global, tr_loss, "", m_val["loss"],
                                    m_val["acc"], m_val["top5"],
                                    opt.param_groups[0]["lr"], el])
        if m_val["acc"] > best_val:
            best_val = m_val["acc"]
            torch.save({"model_state": model.state_dict(), "model_name": args.model,
                        "fusion": args.fusion, "img_size": args.img, "class_to_idx": c2i,
                        "epoch": epoch, "val_acc": best_val, "seed": args.seed,
                        "preproc": args.preproc}, ckpt_path)
            print(f"[info] best ckpt @ val_acc={best_val:.4f}", flush=True)

    # minor-15: score the retained best-validation checkpoint, not last-epoch weights.
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
    m_te = evaluate(model, dl_te, device)
    print(f"[test-final/best-val-ckpt] acc={m_te['acc']:.4f} top5={m_te['top5']:.4f}", flush=True)
    with open(RESULTS / f"test_final_{tag}.json", "w") as f:
        json.dump({"model": args.model, "fusion": args.fusion, "seed": args.seed,
                   "preproc": args.preproc, "test": m_te, "best_val": best_val,
                   "selected_epoch": torch.load(ckpt_path, map_location="cpu")["epoch"],
                   "epochs": args.epochs, "img": args.img, "batch": args.batch,
                   "params_trainable": n_tr_p, "params_total": n_all_p,
                   "eval_precision": "fp32"}, f, indent=2)


if __name__ == "__main__":
    main()
