"""R3 / E10 — explainability evidence that supports the claims made about it.

The R2 evidence set was fourteen images drawn one per crop with a fixed seed:
twelve were healthy classes and all fourteen were correctly classified, so the
manuscript's statements about lesion localisation and about failure cases had no
supporting artefact. This script replaces it with:

  * disease-stratified sampling across all 38 classes, with an explicit quota of
    misclassified images so failure cases exist in the evidence set;
  * a quantitative faithfulness measurement -- deletion and insertion curves and
    their AUCs -- over a stated number of images, rather than a verbal claim;
  * the fraction of Grad-CAM energy falling outside the leaf, using a background
    mask, which is the quantity E8 makes relevant;
  * a branch-attribution control that tests whether a CAM on the CNN branch can
    legitimately explain a two-branch decision, by measuring how much each
    branch contributes to the fused representation.
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from dataset import LeafDataset, build_transforms, load_split_csv
from model import HybridLeafClassifier, SingleBackbone

RES = Path(os.environ.get("RESULTS_DIR", Path(__file__).resolve().parents[1] / "results"))
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ----------------------------------------------------------------- Grad-CAM
class GradCAM:
    """Grad-CAM on the final convolutional feature map of the CNN branch."""

    def __init__(self, model, device):
        self.model, self.device = model, device
        self.acts = self.grads = None
        # The backbone wrappers are built with global_pool="avg", so their own
        # output is a pooled vector. Hook the last convolution instead, whose
        # output is the (B, C, H, W) map Grad-CAM needs.
        branch = model.cnn if hasattr(model, "cnn") else model.backbone
        target = getattr(branch, "conv_head", None) or getattr(branch, "blocks", branch)
        self.h1 = target.register_forward_hook(self._fwd)
        self.h2 = target.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.acts = o.detach() if o.dim() == 4 else None

    def _bwd(self, m, gi, go):
        self.grads = go[0].detach() if go[0].dim() == 4 else None

    def __call__(self, x, cls_idx):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        sel = logits[range(x.size(0)), cls_idx].sum()
        sel.backward()
        a, g = self.acts, self.grads
        if a is None or g is None:
            raise RuntimeError("no 4-D activation captured on the CNN branch")
        w = g.mean(dim=(2, 3), keepdim=True)
        cam = Fn.relu((w * a).sum(1, keepdim=True))
        cam = Fn.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        flat = cam.flatten(1)
        lo = flat.min(1, keepdim=True).values
        hi = flat.max(1, keepdim=True).values
        cam = ((flat - lo) / (hi - lo + 1e-8)).view_as(cam)
        return cam.detach(), logits.detach()

    def close(self):
        self.h1.remove(); self.h2.remove()


def leaf_mask(x_norm: torch.Tensor) -> torch.Tensor:
    """Foreground mask: pixels far from the border colour. PlantVillage images
    sit on a near-uniform background, so this isolates the leaf well enough to
    measure how much CAM energy lands off the leaf."""
    img = (x_norm.cpu() * STD + MEAN).clamp(0, 1)
    B = img.shape[0]
    ring = torch.cat([img[:, :, :4, :].reshape(B, 3, -1), img[:, :, -4:, :].reshape(B, 3, -1),
                      img[:, :, :, :4].reshape(B, 3, -1), img[:, :, :, -4:].reshape(B, 3, -1)], dim=2)
    bg = ring.mean(dim=2).view(B, 3, 1, 1)
    d = (img - bg).pow(2).sum(1).sqrt()
    return (d > 0.15).float()


# ------------------------------------------------------- faithfulness curves
@torch.no_grad()
def del_ins_curves(model, x, cls_idx, cam, steps=20):
    """Deletion and insertion curves: progressively remove (or reveal) the
    highest-CAM pixels and track the softmax probability of the target class."""
    B, _, H, W = x.shape
    n = H * W
    order = cam.flatten(1).argsort(dim=1, descending=True)
    blur = Fn.avg_pool2d(x, 17, stride=1, padding=8)
    del_c = torch.zeros(B, steps + 1)
    ins_c = torch.zeros(B, steps + 1)
    for s in range(steps + 1):
        k = int(round(n * s / steps))
        m = torch.ones(B, n, device=x.device)
        if k:
            m.scatter_(1, order[:, :k], 0.0)
        m4 = m.view(B, 1, H, W)
        p_del = model(x * m4 + blur * (1 - m4)).softmax(-1)
        p_ins = model(x * (1 - m4) + blur * m4).softmax(-1)
        del_c[:, s] = p_del[range(B), cls_idx].cpu()
        ins_c[:, s] = p_ins[range(B), cls_idx].cpu()
    return del_c, ins_c


@torch.no_grad()
def branch_attribution(model, loader, device, n_batches=40):
    """How much does each branch contribute to the decision? Replace one
    branch's pooled features with the batch mean (its least informative value)
    and measure the accuracy drop. Justifies -- or refutes -- explaining a
    two-branch decision with a CAM computed on one branch."""
    if not hasattr(model, "fusion"):
        return None
    ok = {"full": 0, "cnn_only": 0, "tr_only": 0}
    tot = 0
    for bi, (x, y) in enumerate(loader):
        if bi >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        if model.fusion_kind == "token_cross_attn":
            fc = model.cnn.forward_features(x).flatten(2).transpose(1, 2)
            ft = model.tr.forward_features(x).flatten(1, 2)
        else:
            fc, ft = model.cnn(x), model.tr(x)
        # A zero baseline is used rather than the batch mean: with a batch of
        # eight the mean is itself informative, which made an earlier version of
        # this control report a higher accuracy for the ablated model than for
        # the full one.
        variants = {
            "full": (fc, ft),
            "cnn_only": (fc, torch.zeros_like(ft)),
            "tr_only": (torch.zeros_like(fc), ft),
        }
        for k, (a, b) in variants.items():
            ok[k] += (model.head(model.fusion(a, b)).argmax(-1) == y).sum().item()
        tot += y.size(0)
    return {k: v / tot for k, v in ok.items()} | {"n": tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--per_class", type=int, default=10, help="images sampled per class")
    ap.add_argument("--n_fail", type=int, default=40, help="misclassified images to include")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out_suffix", default="",
                    help="appended to output file names only, so evidence sets built "
                         "with different options never overwrite each other")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--field_failures", action="store_true",
                    help="also draw failure cases from PlantDoc, where the model "
                         "actually fails; in-domain accuracy leaves almost none")
    args = ap.parse_args()
    otag = args.tag + args.out_suffix   # outputs only; --tag still locates preds files

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
    model.load_state_dict(ck["model_state"])
    model = model.to(device).eval()

    te = load_split_csv(RES / "split_test.csv")
    if args.field_failures:
        from dataset import index_plantdoc_shared
        te = te + [(p, c) for p, c in index_plantdoc_shared(c2i) if c in c2i]
    tf = build_transforms(ck["img_size"], train=False, preproc="aspect")

    # ---- disease-stratified sample, plus an explicit failure quota ---------
    pred_files = [RES / f"preds_{args.tag}_plantvillage_test_aspect.csv"]
    if args.field_failures:
        pred_files.append(RES / f"preds_{args.tag}_plantdoc_aspect.csv")
    preds = {}
    for pf in pred_files:
        if pf.exists():
            with open(pf, newline="") as f:
                for r in csv.DictReader(f):
                    preds[r["path"]] = (r["true"], r["pred"])
        else:
            print(f"[warn] {pf.name} not found")
    rng = np.random.default_rng(args.seed)
    by_cls = {}
    for p, c in te:
        by_cls.setdefault(c, []).append(p)
    sample = []
    for c in sorted(by_cls):
        ps = sorted(by_cls[c])
        pick = rng.choice(len(ps), size=min(args.per_class, len(ps)), replace=False)
        sample += [(ps[i], c) for i in pick]
    fails = [(p, t) for p, (t, pr) in preds.items() if t != pr]
    n_fail_avail = len(fails)
    if fails:
        pick = rng.choice(len(fails), size=min(args.n_fail, len(fails)), replace=False)
        sample += [fails[i] for i in pick]
    seen, uniq = set(), []
    for p, c in sample:
        if p not in seen:
            seen.add(p); uniq.append((p, c))
    sample = uniq
    n_diseased = sum(1 for _, c in sample if not c.endswith("healthy"))
    print(f"[info] evidence set: {len(sample)} images, {n_diseased} diseased "
          f"({100*n_diseased/len(sample):.0f}%), {len({c for _,c in sample})} classes, "
          f"{min(args.n_fail, n_fail_avail)} known failures", flush=True)

    ds = LeafDataset(sample, c2i, tf)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=4)

    cam_engine = GradCAM(model, device)
    rows = []
    del_all, ins_all = [], []
    i0 = 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        cam, logits = cam_engine(x, y)
        pred = logits.argmax(-1)
        prob = logits.softmax(-1)[range(x.size(0)), y]
        mask = leaf_mask(x).to(device)
        e_tot = cam.flatten(1).sum(1)
        e_leaf = (cam * mask).flatten(1).sum(1)
        off = 1.0 - (e_leaf / (e_tot + 1e-8))
        d, ins = del_ins_curves(model, x, y, cam)
        del_all.append(d); ins_all.append(ins)
        for j in range(x.size(0)):
            p, c = sample[i0 + j]
            rows.append({
                "path": p, "true": c, "pred": labels[int(pred[j])],
                "correct": int(int(pred[j]) == int(y[j])),
                "target_prob": float(prob[j]),
                "cam_energy_off_leaf_frac": float(off[j]),
                "leaf_area_frac": float(mask[j].mean()),
                "deletion_auc": float(d[j].mean()), "insertion_auc": float(ins[j].mean()),
            })
        i0 += x.size(0)
        if i0 % 80 == 0:
            print(f"  processed {i0}/{len(sample)}", flush=True)
    cam_engine.close()

    with open(RES / f"gradcam_records_{otag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    D = torch.cat(del_all).numpy(); I = torch.cat(ins_all).numpy()
    np.save(RES / f"gradcam_deletion_curve_{otag}.npy", D)
    np.save(RES / f"gradcam_insertion_curve_{otag}.npy", I)

    ok = np.array([r["correct"] for r in rows], dtype=bool)
    off = np.array([r["cam_energy_off_leaf_frac"] for r in rows])
    dauc = np.array([r["deletion_auc"] for r in rows])
    iauc = np.array([r["insertion_auc"] for r in rows])
    dis = np.array([not r["true"].endswith("healthy") for r in rows])

    ba = branch_attribution(model, dl, device)
    summary = {
        "tag": args.tag, "n_images": len(rows),
        "n_classes": len({r["true"] for r in rows}),
        "n_diseased": int(dis.sum()), "frac_diseased": float(dis.mean()),
        "n_correct": int(ok.sum()), "n_misclassified": int((~ok).sum()),
        "sampling": f"{args.per_class}/class over all classes + up to {args.n_fail} "
                    f"misclassified images (of {n_fail_avail} available)",
        "faithfulness": {
            "deletion_auc_mean": float(dauc.mean()), "deletion_auc_sd": float(dauc.std(ddof=1)),
            "insertion_auc_mean": float(iauc.mean()), "insertion_auc_sd": float(iauc.std(ddof=1)),
            "steps": 21, "baseline": "17x17 average-pool blur",
            "note": "lower deletion AUC and higher insertion AUC indicate a more faithful map",
        },
        "cam_energy_off_leaf": {
            "mean": float(off.mean()), "sd": float(off.std(ddof=1)),
            "median": float(np.median(off)),
            "mean_correct": float(off[ok].mean()) if ok.any() else None,
            "mean_misclassified": float(off[~ok].mean()) if (~ok).any() else None,
            "mean_diseased": float(off[dis].mean()) if dis.any() else None,
            "mean_healthy": float(off[~dis].mean()) if (~dis).any() else None,
        },
        "branch_attribution": ba,
    }
    with open(RES / f"gradcam_summary_{otag}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
