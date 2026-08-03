"""Train the hybrid classifier (or a single-backbone ablation) on PlantVillage.

Saves per-epoch metrics as CSV and the best checkpoint under results/.
"""
from __future__ import annotations
import argparse, csv, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset import (
    LeafDataset, index_plantvillage, stratified_split,
    list_plantvillage_classes, build_transforms, save_split_manifests,
)
from model import HybridLeafClassifier, SingleBackbone

RESULTS = Path(os.environ.get("RESULTS_DIR", "/home/victus/Papers/Victus_Pardus_0011_Plant_Disease_XAI/Frontiers_Plant_Science/results"))


def build_model(name: str, num_classes: int, img_size: int, pretrained: bool = True) -> nn.Module:
    if name == "hybrid":
        return HybridLeafClassifier(num_classes=num_classes, img_size_tr=img_size, pretrained=pretrained)
    elif name == "effv2s":
        return SingleBackbone(num_classes, "tf_efficientnetv2_s.in21k_ft_in1k", img_size=img_size, pretrained=pretrained)
    elif name == "swinv2t":
        return SingleBackbone(num_classes, "swinv2_tiny_window8_256.ms_in1k", img_size=img_size, pretrained=pretrained)
    else:
        raise ValueError(name)


def evaluate(model, loader, device):
    model.eval()
    tot, correct, top5c, loss_sum = 0, 0, 0, 0.0
    ce = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast(dtype=torch.float16):
                logits = model(x)
            loss_sum += ce(logits.float(), y).item()
            p1 = logits.argmax(-1)
            correct += (p1 == y).sum().item()
            top5 = logits.topk(min(5, logits.size(-1)), dim=-1).indices
            top5c += (top5 == y.unsqueeze(-1)).any(-1).sum().item()
            tot += y.size(0)
    return dict(loss=loss_sum / tot, acc=correct / tot, top5=top5c / tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hybrid", choices=["hybrid", "effv2s", "swinv2t"])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--img", type=int, default=256, help="Fixed to 256 for Swin V2-T window8_256")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="hybrid")
    ap.add_argument("--train_subset", type=int, default=0, help="If >0, use only this many train images (debug)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}")

    classes = list_plantvillage_classes()
    c2i = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    items = index_plantvillage()
    tr, va, te = stratified_split(items, seed=42)
    if args.train_subset > 0:
        tr = tr[: args.train_subset]
    save_split_manifests(RESULTS, tr, va, te, c2i)
    print(f"[info] train={len(tr)} val={len(va)} test={len(te)} classes={num_classes}")

    tf_tr = build_transforms(args.img, train=True)
    tf_ev = build_transforms(args.img, train=False)
    ds_tr = LeafDataset(tr, c2i, tf_tr)
    ds_va = LeafDataset(va, c2i, tf_ev)
    ds_te = LeafDataset(te, c2i, tf_ev)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    dl_te = DataLoader(ds_te, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)

    model = build_model(args.model, num_classes, args.img).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[info] model={args.model} params: trainable={n_trainable/1e6:.2f}M total={n_total/1e6:.2f}M")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = len(dl_tr)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(200, total_steps // 20)

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = GradScaler()
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)

    log_path = RESULTS / f"train_log_{args.tag}.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "step", "train_loss", "train_acc", "val_loss", "val_acc", "val_top5", "lr", "time_s", "trainable_params", "total_params"])

    best_val = 0.0
    step_global = 0
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        n, correct, loss_sum = 0, 0, 0.0
        t0 = time.time()
        for x, y in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(dtype=torch.float16):
                logits = model(x)
                loss = ce(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step_global += 1
            with torch.no_grad():
                p1 = logits.argmax(-1)
                correct += (p1 == y).sum().item()
                n += y.size(0)
                loss_sum += loss.item() * y.size(0)
            if step_global % 50 == 0:
                el = time.time() - t0
                cur_lr = opt.param_groups[0]["lr"]
                print(f"[e{epoch} s{step_global}] loss={loss_sum/n:.4f} acc={correct/n:.4f} lr={cur_lr:.2e} {el:.1f}s")
        tr_loss, tr_acc = loss_sum / max(1, n), correct / max(1, n)
        m_val = evaluate(model, dl_va, device)
        elapsed = time.time() - t_start
        print(f"[epoch {epoch}] train loss={tr_loss:.4f} acc={tr_acc:.4f} | val loss={m_val['loss']:.4f} acc={m_val['acc']:.4f} top5={m_val['top5']:.4f} | {elapsed:.1f}s")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, step_global, tr_loss, tr_acc, m_val["loss"], m_val["acc"], m_val["top5"], opt.param_groups[0]["lr"], elapsed, n_trainable, n_total])
        if m_val["acc"] > best_val:
            best_val = m_val["acc"]
            ckpt = RESULTS / f"best_{args.tag}.pt"
            torch.save({
                "model_state": model.state_dict(),
                "model_name": args.model,
                "img_size": args.img,
                "class_to_idx": c2i,
                "epoch": epoch,
                "val_acc": best_val,
            }, ckpt)
            print(f"[info] saved best ckpt @ val_acc={best_val:.4f} -> {ckpt}")

    # Final test evaluation
    m_te = evaluate(model, dl_te, device)
    print(f"[test-final] loss={m_te['loss']:.4f} acc={m_te['acc']:.4f} top5={m_te['top5']:.4f}")
    with open(RESULTS / f"test_final_{args.tag}.json", "w") as f:
        json.dump({"model": args.model, "test": m_te, "best_val": best_val, "epochs": args.epochs, "img": args.img, "batch": args.batch, "params_trainable": n_trainable, "params_total": n_total}, f, indent=2)


if __name__ == "__main__":
    main()
