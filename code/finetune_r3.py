"""R3 / E5 — target-domain fine-tuning with honest model selection.

The R2 script had no validation split: it evaluated on the PlantDoc test split
after every epoch and kept the running maximum, so the reported 63.14% was a
selection statistic over thirty evaluations rather than a held-out estimate.

Here a validation split is carved out of the *training* partition, the epoch is
selected on that validation split alone, and the test partition is scored
exactly once, with the selected checkpoint. For reference the script also
records what the R2 protocol would have reported (the running test maximum), so
the size of the optimism can be quantified in the response.

Runs on PlantDoc (official train/test partitions) and on PlantWild, which ships
no partition file, so a seeded stratified 70/10/20 split is created and its
manifest is written alongside the results.
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
from dataset import (LeafDataset, build_transforms, index_plantdoc_shared,
                     PD_TRAIN_ROOT, PD_TEST_ROOT)
from plantwild import index_plantwild_shared
from model import HybridLeafClassifier, SingleBackbone

RES = Path(os.environ.get("RESULTS_DIR", Path(__file__).resolve().parents[1] / "results"))


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def stratify(items, fracs, seed):
    """Split (path, class) items per class by the given fractions."""
    rng = random.Random(seed)
    by = {}
    for p, c in items:
        by.setdefault(c, []).append(p)
    parts = [[] for _ in fracs]
    for c in sorted(by):
        ps = sorted(by[c]); rng.shuffle(ps)
        n = len(ps)
        cuts, acc = [], 0
        for fr in fracs[:-1]:
            acc += int(round(n * fr)); cuts.append(min(acc, n))
        bounds = [0] + cuts + [n]
        for i in range(len(fracs)):
            parts[i].extend((p, c) for p in ps[bounds[i]:bounds[i + 1]])
    return parts


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    T, P = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True)).float()
        P.append(logits.argmax(-1).cpu().numpy()); T.append(y.numpy())
    t, p = np.concatenate(T), np.concatenate(P)
    acc = float((t == p).mean())
    f1 = []
    for c in sorted(set(t.tolist())):
        tp = int(((p == c) & (t == c)).sum()); fp = int(((p == c) & (t != c)).sum())
        fn = int(((p != c) & (t == c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return acc, float(np.mean(f1)) if f1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", choices=["plantdoc", "plantwild"], default="plantdoc")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"ft_{args.dataset}_s{args.seed}"
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    c2i = ck["class_to_idx"]
    labels = [k for k, _ in sorted(c2i.items(), key=lambda kv: kv[1])]
    name, fusion = ck.get("model_name", "hybrid"), ck.get("fusion", "gated_linear")
    if name == "hybrid":
        model = HybridLeafClassifier(len(c2i), img_size_tr=ck["img_size"], pretrained=False, fusion=fusion)
    elif name == "effv2s":
        model = SingleBackbone(len(c2i), "tf_efficientnetv2_s.in21k_ft_in1k", ck["img_size"], False)
    else:
        model = SingleBackbone(len(c2i), "swinv2_tiny_window8_256.ms_in1k", ck["img_size"], False)
    model.load_state_dict(ck["model_state"]); model = model.to(device)

    # ---- partitions -------------------------------------------------------
    if args.dataset == "plantdoc":
        tr_all = [(p, c) for p, c in index_plantdoc_shared(c2i, splits=("train",)) if c in c2i]
        te = [(p, c) for p, c in index_plantdoc_shared(c2i, splits=("test",)) if c in c2i]
        (fit, val), split_note = stratify(tr_all, [1 - args.val_frac, args.val_frac], args.seed), \
            "official PlantDoc train partition split into fit/val; official test partition held out"
    else:
        allw = index_plantwild_shared(c2i)
        fit, val, te = stratify(allw, [0.70, 0.10, 0.20], args.seed)
        split_note = "PlantWild ships no partition file; seeded stratified 70/10/20 split created here"
        with open(RES / f"plantwild_split_{args.seed}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["path", "class", "split"])
            for nm, part in (("fit", fit), ("val", val), ("test", te)):
                w.writerows([(p, c, nm) for p, c in part])

    print(f"[info] {args.dataset}: fit={len(fit)} val={len(val)} test={len(te)} "
          f"classes={len({c for _,c in fit+val+te})}")
    print(f"[info] {split_note}")

    tf_tr = build_transforms(ck["img_size"], train=True, preproc="aspect")
    tf_ev = build_transforms(ck["img_size"], train=False, preproc="aspect")
    mk = lambda it, tf, sh: DataLoader(LeafDataset(it, c2i, tf), batch_size=args.batch,
                                       shuffle=sh, num_workers=4, pin_memory=True, drop_last=sh)
    dl_fit, dl_val, dl_te = mk(fit, tf_tr, True), mk(val, tf_ev, False), mk(te, tf_ev, False)

    zs_te_acc, zs_te_f1 = evaluate(model, dl_te, device)
    zs_val_acc, _ = evaluate(model, dl_val, device)
    print(f"[zero-shot] test acc={zs_te_acc:.4f} f1={zs_te_f1:.4f} | val acc={zs_val_acc:.4f}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)
    total = len(dl_fit) * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))
    scaler = GradScaler("cuda")
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    ckpt_path = RES / f"best_{tag}.pt"

    log = [["epoch", "train_loss", "val_acc", "val_f1", "test_acc_monitor_only"]]
    best_val, best_epoch, r2_style_running_max_test = zs_val_acc, 0, zs_te_acc
    torch.save({"model_state": model.state_dict(), "class_to_idx": c2i,
                "img_size": ck["img_size"], "model_name": name, "fusion": fusion,
                "epoch": 0, "val_acc": best_val}, ckpt_path)

    for ep in range(1, args.epochs + 1):
        model.train(); n, ls = 0, 0.0
        for x, y in dl_fit:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.float16):
                loss = ce(model(x), y)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            n += y.size(0); ls += loss.item() * y.size(0)
        v_acc, v_f1 = evaluate(model, dl_val, device)
        # Monitored only to quantify the optimism of the R2 protocol; never used
        # for selection.
        t_acc, _ = evaluate(model, dl_te, device)
        r2_style_running_max_test = max(r2_style_running_max_test, t_acc)
        log.append([ep, ls / max(1, n), v_acc, v_f1, t_acc])
        print(f"[ep {ep:2d}] loss={ls/max(1,n):.4f} val_acc={v_acc:.4f} "
              f"(test monitor {t_acc:.4f})", flush=True)
        if v_acc > best_val:
            best_val, best_epoch = v_acc, ep
            torch.save({"model_state": model.state_dict(), "class_to_idx": c2i,
                        "img_size": ck["img_size"], "model_name": name, "fusion": fusion,
                        "epoch": ep, "val_acc": best_val}, ckpt_path)

    # Score the test partition exactly once, with the validation-selected model.
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
    te_acc, te_f1 = evaluate(model, dl_te, device)
    print(f"[final] selected epoch={best_epoch} (val {best_val:.4f}) -> "
          f"test acc={te_acc:.4f} f1={te_f1:.4f}")
    print(f"[reference] R2-style running max over test = {r2_style_running_max_test:.4f} "
          f"(optimism {100*(r2_style_running_max_test-te_acc):+.2f} pp)")

    with open(RES / f"ftlog_{tag}.csv", "w", newline="") as f:
        csv.writer(f).writerows(log)
    with open(RES / f"ft_{tag}.json", "w") as f:
        json.dump({"dataset": args.dataset, "split_note": split_note,
                   "n_fit": len(fit), "n_val": len(val), "n_test": len(te),
                   "n_classes": len({c for _, c in fit + val + te}),
                   "zero_shot_test_acc": zs_te_acc, "zero_shot_test_macro_f1": zs_te_f1,
                   "selected_epoch": best_epoch, "selection_val_acc": best_val,
                   "finetuned_test_acc": te_acc, "finetuned_test_macro_f1": te_f1,
                   "r2_protocol_running_max_test_acc": r2_style_running_max_test,
                   "optimism_pp": 100 * (r2_style_running_max_test - te_acc),
                   "hyperparams": {"lr": args.lr, "epochs": args.epochs,
                                   "batch": args.batch, "optimizer": "AdamW",
                                   "weight_decay": 1e-4, "label_smoothing": 0.05,
                                   "schedule": "cosine", "val_frac": args.val_frac,
                                   "seed": args.seed, "stems_frozen": True,
                                   "eval_precision": "fp32"}}, f, indent=2)


if __name__ == "__main__":
    main()
