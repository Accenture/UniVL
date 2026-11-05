#!/usr/bin/env python3
"""
Fit a scaling curve FID(N) = FID_inf + c/N + d/sqrt(N) by subsampling the 4000
panels at multiple sample sizes (100, 200, 500, 1000, 2000, 4000), computing
FID at each, and fitting via least-squares.

This lets you convert a small-N FID measurement to its asymptotic value.

Usage:
    python -m evaluation.fid.fit_fid_scaling \\
        --panels_dir output_inference/eval4k_Cn50 \\
        --gpu 0 \\
        --out output_inference/eval4k_Cn50/fid_scaling.json
"""
import argparse, glob, json, os, random
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

TARGET = (512, 512)
PANEL_W = 1536


def split_panel(img):
    return (img.crop((0, 0, 512, 512)),
            img.crop((512, 0, 1024, 512)),
            img.crop((1024, 0, 1536, 512)))


def collect_pairs(panels_dir):
    files = sorted(glob.glob(os.path.join(panels_dir, "*.jpg")))
    gens, gts = [], []
    for f in files:
        img = Image.open(f).convert("RGB")
        if img.size[0] != PANEL_W:
            continue
        _, g, gt = split_panel(img)
        gens.append(g)
        gts.append(gt)
    return gens, gts


def compute_fid(real_imgs, gen_imgs, device, batch_size=16):
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    to_tensor = T.ToTensor()

    def push(imgs, real):
        bb = []
        for img in imgs:
            bb.append(to_tensor(img.resize(TARGET)))
            if len(bb) == batch_size:
                fid.update(torch.stack(bb).to(device), real=real); bb = []
        if bb:
            fid.update(torch.stack(bb).to(device), real=real)
    push(real_imgs, True)
    push(gen_imgs, False)
    return fid.compute().item()


def fit_curves(Ns, fids):
    """Fit several FID(N) models."""
    Ns = np.asarray(Ns, dtype=np.float64)
    fids = np.asarray(fids, dtype=np.float64)
    ss_tot = np.sum((fids - fids.mean()) ** 2)

    def r2(pred):
        return float(1 - np.sum((fids - pred) ** 2) / ss_tot)

    # Model 1: FID = a + b/N  (canonical small-sample bias)
    X1 = np.stack([np.ones_like(Ns), 1.0 / Ns], axis=1)
    a1, b1 = np.linalg.lstsq(X1, fids, rcond=None)[0]
    pred1 = a1 + b1 / Ns

    # Model 2: FID = a + b/N + c/sqrt(N)
    X2 = np.stack([np.ones_like(Ns), 1.0 / Ns, 1.0 / np.sqrt(Ns)], axis=1)
    a2, b2, c2 = np.linalg.lstsq(X2, fids, rcond=None)[0]
    pred2 = a2 + b2 / Ns + c2 / np.sqrt(Ns)

    # Model 3 (power law, log-linear in log-log): log(FID) = log(A) + B * log(N)
    #   ⇒ FID = A * N^B
    Xlog = np.stack([np.ones_like(Ns), np.log(Ns)], axis=1)
    logA, B = np.linalg.lstsq(Xlog, np.log(fids), rcond=None)[0]
    A = float(np.exp(logA))
    pred3 = A * Ns ** B
    # R^2 in linear FID space (for honest comparison)
    r2_3_linear = r2(pred3)
    # R^2 in log space (typical for log-linear fit)
    log_pred = logA + B * np.log(Ns)
    log_fids = np.log(fids)
    r2_3_log = float(1 - np.sum((log_fids - log_pred) ** 2) /
                     np.sum((log_fids - log_fids.mean()) ** 2))

    # Model 4 (linear in log N): FID = a + b * log(N)  — alternative log-linear
    Xlin = np.stack([np.ones_like(Ns), np.log(Ns)], axis=1)
    a4, b4 = np.linalg.lstsq(Xlin, fids, rcond=None)[0]
    pred4 = a4 + b4 * np.log(Ns)

    return {
        "model1_inv_N":            {"form": "FID = a + b/N",
                                    "a": float(a1), "b": float(b1), "R2": r2(pred1)},
        "model2_inv_N_plus_inv_sqrtN": {"form": "FID = a + b/N + c/sqrt(N)",
                                        "a": float(a2), "b": float(b2), "c": float(c2),
                                        "R2": r2(pred2)},
        "model3_power_law":        {"form": "FID = A * N^B",
                                    "A": A, "B": float(B),
                                    "R2_linear": r2_3_linear,
                                    "R2_logspace": r2_3_log},
        "model4_log_N":            {"form": "FID = a + b*log(N)",
                                    "a": float(a4), "b": float(b4), "R2": r2(pred4)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels_dir", required=True)
    ap.add_argument("--Ns", type=int, nargs="+", default=[100, 200, 500, 1000, 2000, 4000])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)

    print(f"Loading panels from {args.panels_dir} ...")
    gens, gts = collect_pairs(args.panels_dir)
    n_total = len(gens)
    print(f"Total pairs: {n_total}")

    rng = random.Random(args.seed)
    indices_full = list(range(n_total))
    rng.shuffle(indices_full)

    results = []
    for N in args.Ns:
        if N > n_total:
            print(f"  N={N} > {n_total}, skip")
            continue
        sel = indices_full[:N]
        sub_gen = [gens[i] for i in sel]
        sub_gt  = [gts[i]  for i in sel]
        print(f"  N={N} ...", end=" ", flush=True)
        fid = compute_fid(sub_gt, sub_gen, device)
        print(f"FID = {fid:.4f}")
        results.append({"N": N, "FID": fid})

    Ns   = [r["N"]   for r in results]
    fids = [r["FID"] for r in results]
    fits = fit_curves(Ns, fids)

    print("\n=== Curve fits ===")
    m = fits["model1_inv_N"]
    print(f"  Model 1 (canonical): FID = {m['a']:.3f} + {m['b']:.1f}/N      R^2 = {m['R2']:.4f}")
    m = fits["model2_inv_N_plus_inv_sqrtN"]
    print(f"  Model 2 (1/N+1/sqrt N): FID = {m['a']:.3f} + {m['b']:.1f}/N + {m['c']:.2f}/sqrt(N)   R^2 = {m['R2']:.4f}")
    m = fits["model3_power_law"]
    print(f"  Model 3 (power law): FID = {m['A']:.3f} * N^{m['B']:.4f}     R^2_lin = {m['R2_linear']:.4f}, R^2_log = {m['R2_logspace']:.4f}")
    m = fits["model4_log_N"]
    print(f"  Model 4 (lin log N): FID = {m['a']:.3f} + {m['b']:.3f}*log(N)   R^2 = {m['R2']:.4f}")

    out = {
        "panels_dir": args.panels_dir,
        "seed": args.seed,
        "data": results,
        "fits": fits,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
